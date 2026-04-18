"""Textual chat app — Phase 4 adds tool-loop integration
(harness-1r4) on top of Phase 3's metrics footer (harness-17v).

Scope of Phase 4:
- When a `registry: ToolRegistry` is passed, the worker drives
  `run_tool_loop` instead of calling `adapter.complete` directly.
- Observer callback translates ToolLoopEvent → rich.Text lines and
  hops onto the UI thread via `call_from_thread` to append them to
  the RichLog in real time (router_intent, tool_call_start/end/
  failed/declined/deduped).
- Write-tier tools always decline for now — the confirmation modal
  lands in harness-mz2. This keeps Phase 4 observable without
  introducing a modal-screen dependency.

Scope of earlier phases still applies: retrieval, transcript
persistence, metrics footer.
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
    _build_tool_grounding_block,
    _format_ctx_meter,
    _persist_tool_exchange,
    _render_fact_block,
    _render_memory_block,
    _retrieve_turn_context,
)
from harness.cli import _RetrievalState as _RetrievalHealth
from harness.model.adapter import ChatMessage, approx_token_count
from harness.orchestrator import ToolLoopEvent, run_tool_loop

if TYPE_CHECKING:
    from pathlib import Path

    from harness.character import Character
    from harness.model.adapter import ModelAdapter
    from harness.retrieval import VoiceRetriever
    from harness.router import Router
    from harness.store import EpisodicStore, SemanticStore
    from harness.store.transcript import Transcript
    from harness.tools import ToolCall, ToolRegistry


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
        registry: ToolRegistry | None = None,
        router: Router | None = None,
        workspace_path: Path | None = None,
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
        self._tool_registry = registry
        self._router = router
        self._workspace_path = workspace_path
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
        tools_note = (
            f"tools={self._tool_registry.names()}"
            if self._tool_registry is not None
            else "no tools"
        )
        log.write(
            f"[dim]Chat with {self._character.name}. "
            f"adapter={self._adapter.id} session={self._session}. "
            f"{tools_note}. Phase 4: tool loop wired; streaming "
            f"lands in harness-lrg, write-tier confirmation in "
            f"harness-mz2.[/dim]"
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
            if self._tool_registry is not None and self._workspace_path is not None:
                system_content = (
                    f"{system_content}\n\n"
                    f"{_build_tool_grounding_block(self._tool_registry, self._workspace_path)}"
                )
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

            if self._tool_registry is not None:
                # Tools active: drive the orchestrator loop. Observer
                # hops every ToolLoopEvent onto the UI thread so the
                # RichLog shows router_intent / 🔧 calls / results
                # inline. Write-tier tools always decline for now —
                # the modal-screen confirmation lives in harness-mz2.
                initial_messages = [system, *self._state.history]
                loop_result = run_tool_loop(
                    self._adapter,  # type: ignore[arg-type]
                    [*initial_messages, user_msg],
                    self._tool_registry,
                    confirm=self._decline_write_tools,
                    observe=self._observe_tool_event,
                    router=self._router,
                )
                reply = loop_result.content or "(no reply)"
                # Persist the full tool exchange + update in-memory
                # history so subsequent turns see the tool results.
                _persist_tool_exchange(
                    self._transcript,
                    session=self._session,
                    channel=self._channel,
                    character_name=self._character.name,
                    initial_count=len(initial_messages),
                    loop_messages=loop_result.messages,
                )
                # initial_messages[0] is system; drop it from the
                # history carry-over since we rebuild the system
                # prompt fresh each turn.
                self._state.history = list(loop_result.messages[1:])
            else:
                reply = self._adapter.complete(
                    messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                self._state.history.append(user_msg)
                self._state.history.append(ChatMessage(role="assistant", content=reply))
                self._transcript.append(
                    session=self._session,
                    channel=self._channel,
                    speaker=self._character.name,
                    role="assistant",
                    content=reply,
                )

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

    # ---------- tool loop plumbing ----------

    def _tool_label(self, name: str) -> str:
        """Friendly label for a tool — the spec's display_name when
        registered, the raw name otherwise (e.g. for a tool the model
        hallucinated that isn't actually in the registry)."""
        if self._tool_registry is None:
            return name
        if name in self._tool_registry:
            return self._tool_registry.get(name).spec.label
        return name

    def _decline_write_tools(self, call: ToolCall) -> bool:
        """Phase 4 placeholder: write-tier tools always decline until
        harness-mz2 replaces this with a modal confirmation screen.
        Emit a log line so the user can see the call was rejected.

        Returning False here tells run_tool_loop to inject a
        'user declined' tool-role message; the model adapts and
        (usually) reports inability to the user."""
        self.call_from_thread(
            self._emit_decline_notice,
            call.name,
        )
        return False

    def _emit_decline_notice(self, tool_name: str) -> None:
        log = self.query_one("#output", RichLog)
        line = Text()
        line.append(
            f"   ⚠ {self._tool_label(tool_name)} declined automatically — "
            f"write-tier confirmation modal lands in harness-mz2.",
            style="yellow",
        )
        log.write(line)

    def _observe_tool_event(self, event: ToolLoopEvent) -> None:
        """Observer callback passed to run_tool_loop. Runs on the
        worker thread — must hop to the UI thread before touching
        RichLog. Text-only events (model_call_start/end, round_start/
        complete, token_delta) are ignored here; streaming is
        harness-lrg's job."""
        self.call_from_thread(self._render_tool_event, event)

    def _render_tool_event(self, event: ToolLoopEvent) -> None:
        """UI-thread handler: translate a ToolLoopEvent into one or
        two lines in the RichLog. Uses rich.Text (not markup strings)
        so tool arguments / output snippets can't inject styles."""
        log = self.query_one("#output", RichLog)
        if event.kind == "router_intent":
            call = event.call
            assert call is not None
            line = Text(f"→ routed to {call.name}", style="dim magenta")
            log.write(line)
        elif event.kind == "tool_call_start":
            call = event.call
            assert call is not None
            line = Text()
            line.append("🔧 ", style="cyan")
            line.append(self._tool_label(call.name), style="bold cyan")
            line.append(f" {call.arguments}", style="dim")
            log.write(line)
        elif event.kind in ("tool_call_end", "tool_call_failed"):
            result = event.result
            assert result is not None
            success = event.kind == "tool_call_end"
            mark = "✓" if success else "✗"
            style = "green" if success else "red"
            snippet = result.output[:120].replace("\n", " ")
            more = "…" if len(result.output) > 120 else ""
            line = Text()
            line.append(f"   {mark} ", style=style)
            line.append(f"{snippet}{more}", style="dim")
            log.write(line)
        elif event.kind == "tool_call_declined":
            log.write(Text("   ✗ declined", style="yellow"))
        elif event.kind == "tool_call_deduped":
            call = event.call
            assert call is not None
            label = self._tool_label(call.name)
            log.write(
                Text(
                    f"⇢ {label} {call.arguments} — duplicate call skipped",
                    style="dim",
                )
            )
        # Other event kinds (round_start, model_call_start/end,
        # token_delta, round_complete) are internal book-keeping —
        # the metrics footer covers 'model is thinking'; streaming
        # lands in harness-lrg.

    # ---------- metrics ----------

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
