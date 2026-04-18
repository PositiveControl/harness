"""Textual chat app — Phase 3 adds a live metrics footer
(harness-17v) on top of Phase 2's model wiring (harness-29c).

Scope of Phase 3:
- The placeholder Static becomes a running ctx meter + an
  elapsed-time indicator that ticks while the model is generating.
- `set_interval(0.25, ...)` drives the tick so the display updates
  smoothly without the worker needing to push updates per second.
- Token counts are computed once per turn (before and after the
  adapter call) via `adapter.count_tokens` when the adapter exposes
  it; approx fallback otherwise.

Scope of earlier phases still applies: retrieval, adapter, and
transcript persistence run on a thread-backed exclusive worker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.widgets import Input, RichLog, Static

from harness.cli import (
    _format_ctx_meter,
    _render_fact_block,
    _render_memory_block,
    _retrieve_turn_context,
)
from harness.cli import _RetrievalState as _RetrievalHealth
from harness.model.adapter import ChatMessage, approx_token_count

if TYPE_CHECKING:
    from harness.character import Character
    from harness.model.adapter import ModelAdapter
    from harness.retrieval import VoiceRetriever
    from harness.store import EpisodicStore, SemanticStore
    from harness.store.transcript import Transcript


@dataclass
class _ChatAppState:
    """Per-app mutable state separated from the widget so Textual's
    reactive churn doesn't tempt anyone to reach into the App object
    directly. Held on the App instance but passed through worker
    callbacks by reference.

    `history` is the model-visible thread (system prompt is built
    fresh per turn; only user + assistant turns accumulate here).
    `retrieval_health` disables a retrieval source for the rest of
    the session once it raises so the user doesn't get the same
    warning on every turn.

    `turn_started_at` is a monotonic timestamp set when the worker
    kicks off and cleared when it finishes. `None` means 'idle' —
    the metrics tick renders nothing for the elapsed-time slot in
    that state.

    `ctx_used` is the most recent prompt+history token count,
    recomputed once per turn (after the reply is in) so the user
    sees the post-turn size without the metrics tick having to
    retokenize every 250ms."""

    history: list[ChatMessage] = field(default_factory=list)
    retrieval_health: _RetrievalHealth = field(default_factory=_RetrievalHealth)
    turn_started_at: float | None = None
    ctx_used: int = 0


class ChatApp(App[None]):
    """Persistent-input chat TUI.

    Layout (bottom-docked for stability across terminal resizes):

        +----------------------------+
        |        RichLog             |  scrollable history
        |  (user ›, airton ›, …)     |
        +----------------------------+
        | ctx — · elapsed —          |  metrics strip (Phase 3)
        +----------------------------+
        | > input prompt             |  persistent Input
        +----------------------------+

    One worker at a time: `run_worker(..., exclusive=True)` cancels
    any prior worker, so if the user somehow re-enables the input
    (future /cancel command, window refocus race, etc.) only the
    latest turn renders."""

    CSS = """
    RichLog {
        height: 1fr;
        border: none;
        padding: 1 2 0 2;
        background: $background;
    }

    #metrics {
        dock: bottom;
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 2;
    }

    Input {
        dock: bottom;
        border: tall $accent;
        margin: 0;
    }

    Input:disabled {
        border: tall $warning-muted;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "quit", "quit"),
        ("ctrl+d", "quit", "quit"),
    ]

    def __init__(
        self,
        *,
        character: Character,
        speaker: str,
        session: str,
        channel: str,
        adapter: ModelAdapter,
        transcript: Transcript,
        retriever: VoiceRetriever | None = None,
        top_k: int = 6,
        memory_store: EpisodicStore | None = None,
        memories: int = 0,
        memories_threshold: float = 0.5,
        semantic_store: SemanticStore | None = None,
        facts: int = 0,
        facts_threshold: float = 0.45,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self._character = character
        self._speaker = speaker
        self._session = session
        self._channel = channel
        self._adapter = adapter
        self._transcript = transcript
        self._retriever = retriever
        self._top_k = top_k
        self._memory_store = memory_store
        self._memories = memories
        self._memories_threshold = memories_threshold
        self._semantic_store = semantic_store
        self._facts = facts
        self._facts_threshold = facts_threshold
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._state = _ChatAppState()

    # ---------- compose / mount ----------

    def compose(self) -> ComposeResult:
        # Textual stacks docked widgets in reverse declaration order,
        # so yield the Input *before* the metrics Static to get
        # metrics on top of input.
        yield RichLog(id="output", wrap=True, markup=True, highlight=False)
        yield Input(id="prompt", placeholder="type a message… (ctrl+c to quit)")
        yield Static("ctx — · elapsed —", id="metrics")

    def on_mount(self) -> None:
        log = self.query_one("#output", RichLog)
        log.write(
            f"[dim]Chat with {self._character.name}. "
            f"adapter={self._adapter.id} session={self._session}. "
            f"Phase 3: live metrics footer wired; tools land in "
            f"harness-1r4.[/dim]"
        )
        self.query_one("#prompt", Input).focus()
        # Baseline ctx count — just the system prompt framing is not
        # known before the first turn, so seed the meter at 0. It
        # updates after each turn completes.
        self._refresh_metrics()
        # 250ms tick matches the classic CLI's thinking-spinner cadence
        # so the elapsed-time field feels identical. A faster tick is
        # wasted work; a slower one makes the counter feel stuck.
        self.set_interval(0.25, self._refresh_metrics)

    # ---------- input path ----------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        log = self.query_one("#output", RichLog)
        # Build a Text object so the user's content can't be parsed as
        # Rich markup — `[echo]` etc. in free-form text would otherwise
        # get eaten as an unknown style tag.
        line = Text()
        line.append(f"{self._speaker} ›", style="bold cyan")
        line.append(" ")
        line.append(text)
        log.write(line)
        event.input.value = ""
        self._set_input_enabled(False)
        # Start the elapsed-time clock on the UI thread so the metrics
        # tick sees it immediately — no 'thinking 0.0s' → 'thinking 0.3s'
        # gap where the spinner hasn't caught up yet.
        self._state.turn_started_at = time.monotonic()
        self._refresh_metrics()
        # `thread=True` runs the sync adapter off the UI thread so
        # token generation doesn't block the event loop. `exclusive`
        # ensures a stray second submission cancels the older worker
        # rather than queueing — we don't want two concurrent
        # transcripts writes to the same row.
        self.run_worker(
            lambda: self._run_turn_sync(text),
            thread=True,
            exclusive=True,
        )

    # ---------- worker (thread; no UI access except call_from_thread) ----------

    def _run_turn_sync(self, user_input: str) -> None:
        """Executed on the worker thread. All UI updates go through
        call_from_thread so Textual's reactive tree stays single-
        threaded. Any exception becomes a red log line — we don't
        tear down the app on a per-turn failure."""
        try:
            # Retrieval + system prompt + messages. Any single source
            # raising disables it on _state.retrieval_health; the
            # warn callback is invoked inline and routed to the log.
            examples, recalled, known_facts = _retrieve_turn_context(
                user_input=user_input,
                speaker=self._speaker,
                retriever=self._retriever,
                memory_store=self._memory_store,
                semantic_store=self._semantic_store,
                top_k=self._top_k,
                memories=self._memories,
                memories_threshold=self._memories_threshold,
                facts=self._facts,
                facts_threshold=self._facts_threshold,
                state=self._state.retrieval_health,
                warn=self._emit_warning,
            )

            system_content = (
                self._character.system_prompt(include_samples=examples)
                if examples
                else self._character.system_prompt()
            )
            if recalled:
                system_content = f"{system_content}\n\n{_render_memory_block(recalled)}"
            if known_facts:
                system_content = f"{system_content}\n\n{_render_fact_block(known_facts)}"

            system = ChatMessage(role="system", content=system_content)
            user_msg = ChatMessage(role="user", content=user_input)
            messages = [system, *self._state.history, user_msg]

            # Persist the user turn before the model runs so a crash
            # mid-generation still preserves the prompt in the
            # transcript (symmetric with the classic CLI).
            self._transcript.append(
                session=self._session,
                channel=self._channel,
                speaker=self._speaker,
                role="user",
                content=user_input,
            )

            reply = self._adapter.complete(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            # Persist + update history only on success; a failed call
            # leaves history as-is so the user can retry without the
            # thread getting a ghost assistant turn.
            self._transcript.append(
                session=self._session,
                channel=self._channel,
                speaker=self._character.name,
                role="assistant",
                content=reply,
            )
            self._state.history.append(user_msg)
            self._state.history.append(ChatMessage(role="assistant", content=reply))

            self.call_from_thread(self._render_reply, reply)
        except Exception as exc:
            self.call_from_thread(self._render_error, exc)
        finally:
            # Stop the elapsed-time counter and re-enable input from
            # the UI thread. Order matters: clear the timer BEFORE
            # refreshing metrics so the next tick shows 'idle' rather
            # than a stale final elapsed-time string.
            self.call_from_thread(self._finish_turn)

    # ---------- UI-thread helpers (all run via call_from_thread) ----------

    def _render_reply(self, reply: str) -> None:
        log = self.query_one("#output", RichLog)
        line = Text()
        line.append(f"{self._character.name} ›", style="bold green")
        line.append(" ")
        # Reply goes in as plain text so the model can't inject markup.
        line.append(reply)
        log.write(line)

    def _render_error(self, exc: BaseException) -> None:
        log = self.query_one("#output", RichLog)
        line = Text()
        line.append("error running turn: ", style="red")
        line.append(f"{type(exc).__name__}: {exc}", style="dim")
        log.write(line)

    def _emit_warning(self, msg: str) -> None:
        """Retrieval-source warning — the sources pass this as a
        sync callback from the worker thread, so hop onto the UI
        thread before touching the log."""
        self.call_from_thread(self._render_warning, msg)

    def _render_warning(self, msg: str) -> None:
        log = self.query_one("#output", RichLog)
        log.write(f"[yellow]⚠ {msg}[/yellow]")

    def _set_input_enabled(self, enabled: bool) -> None:
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = not enabled
        if enabled:
            prompt.focus()

    def _finish_turn(self) -> None:
        """Worker-completion hook on the UI thread. Stops the elapsed
        clock, recomputes the ctx meter against the now-updated
        history, re-enables input, and pushes a metrics refresh so
        the user sees the final state immediately instead of waiting
        for the next tick."""
        self._state.turn_started_at = None
        self._recompute_ctx_used()
        self._set_input_enabled(True)
        self._refresh_metrics()

    def _recompute_ctx_used(self) -> None:
        """Cached per-turn token count for the metrics footer.
        Prefer the adapter's tokenizer (exact) when it exposes one;
        fall back to the char-heuristic used by approx_token_count.
        Any exception (e.g. a tokenizer that panics on bad input) is
        swallowed — we don't want a metrics hiccup to take out the
        whole turn."""
        count_fn = getattr(self._adapter, "count_tokens", None)
        try:
            if callable(count_fn):
                self._state.ctx_used = int(count_fn(self._state.history))
            else:
                self._state.ctx_used = approx_token_count(self._state.history)
        except Exception:
            self._state.ctx_used = approx_token_count(self._state.history)

    def _refresh_metrics(self) -> None:
        """Render the metrics strip. Idle state shows just the ctx
        meter; active turn appends 'thinking N.Ns' so the user can
        see the model is alive even when streaming tokens hasn't
        started yet. Elapsed time is formatted to one decimal place
        so the ticker visibly advances at the 0.25s tick cadence."""
        metrics = self.query_one("#metrics", Static)
        meter = _format_ctx_meter(self._state.ctx_used, self._adapter.context_window) or "ctx —"
        if self._state.turn_started_at is not None:
            elapsed = time.monotonic() - self._state.turn_started_at
            metrics.update(Text.from_markup(f"{meter} · thinking {elapsed:.1f}s"))
        else:
            metrics.update(Text.from_markup(f"{meter} · idle"))
