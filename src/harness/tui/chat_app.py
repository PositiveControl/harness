"""Textual chat app — Phase 7 adds history replay on mount +
slash-command parity with the classic REPL (harness-01o).

Scope of Phase 7:
- On mount, tail the current session's transcript (capped at
  `max_history_replay`) and replay user + assistant turns into
  the RichLog. `_state.history` is populated in the same pass so
  the model sees continuity across app restarts.
- Slash commands handled in on_input_submitted before the worker
  is kicked off: `/exit`, `/quit`, `:q` all call `self.exit()`.
  Empty/whitespace submissions stay a no-op (unchanged from
  earlier phases).

Scope of earlier phases still applies: retrieval, transcript
persistence, metrics footer, tool observer, streaming, write-tier
confirmation modal.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
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
from harness.tui.confirm_screen import ALWAYS, APPROVE, ConfirmToolScreen

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
    # Per-turn streaming buffer. Reset at turn start; written to
    # from `_feed_stream` (UI thread only) as token deltas arrive;
    # flushed at turn end. `stream_first_chunk` tracks whether the
    # 'airton ›' prefix has been emitted yet.
    stream_buffer: str = ""
    stream_first_chunk: bool = True
    # Always-on input (harness-c93): while a worker is running,
    # Enter enqueues into `pending_prompts` instead of kicking a
    # second worker. `is_busy` is the single source of truth for
    # 'a turn is running' — toggled on the UI thread in _start_turn
    # and _finish_turn so reads are race-free.
    pending_prompts: deque[str] = field(default_factory=deque)
    is_busy: bool = False
    # Persistent elapsed-time (harness-jsu): the most recent turn's
    # wall-clock duration, retained so the idle metrics strip reads
    # '· last N.Ns' instead of blanking back to '· idle'. Cleared
    # only at app start — overwritten at every turn completion.
    last_elapsed: float | None = None
    # Interrupt plumbing (harness-xuh): every start-of-turn bumps
    # `turn_seq` and the worker captures its value at kickoff.
    # Cancellation bumps it again so any trailing call_from_thread
    # updates from the doomed worker (stream deltas, _finish_turn)
    # compare against the new value and no-op. Textual's thread
    # workers can't be killed mid-execution; the seq gate is how we
    # make their late updates harmless.
    turn_seq: int = 0


# Sentence boundary: `.!?` followed by whitespace / closing quote /
# paren, OR a literal newline. Same heuristic as cli._StreamRenderer;
# tuned so URLs like www.example.com/path don't split at the dot
# inside the host.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?][\s)\]'\"]+|\n)")


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
        padding: 1 2 1 2;
        background: $background;
    }

    #metrics {
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 2;
    }

    Input {
        border: tall $accent;
        margin: 0 0 1 0;
    }

    Input:disabled {
        border: tall $warning-muted;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "quit", "quit"),
        ("ctrl+d", "quit", "quit"),
        # priority=True so the focused Input doesn't swallow the
        # keypress — interrupt has to be reachable mid-turn when the
        # prompt is where the user's hands already are.
        Binding("ctrl+x", "interrupt", "interrupt", priority=True),
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
        max_history_replay: int = 20,
        startup_warnings: tuple[str, ...] = (),
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
        self._max_history_replay = max_history_replay
        # Session-scoped always-approve set. The modal writes into
        # this when the user picks 'always' so subsequent calls to
        # the same tool skip the modal. Cleared on app exit — no
        # persistence across sessions, matching the classic REPL's
        # behavior.
        self._approved_tools: set[str] = set()
        self._startup_warnings = startup_warnings
        self._state = _ChatAppState()

    # ---------- compose / mount ----------

    def compose(self) -> ComposeResult:
        # Natural vertical flow: RichLog takes remaining space, metrics
        # sits as a 1-row strip above the input, input at the bottom.
        # Previous dock-bottom on both overlapped visually — prompt
        # border got clipped by the stat bar.
        yield RichLog(id="output", wrap=True, markup=True, highlight=False)
        yield Static("ctx — · elapsed —", id="metrics")
        yield Input(id="prompt", placeholder="type a message… (ctrl+c to quit)")

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
            f"{tools_note}. /exit /quit :q to leave.[/dim]"
        )
        for w in self._startup_warnings:
            log.write(f"[yellow]⚠ {w}[/yellow]")
        self._replay_history(log)
        # Baseline ctx count — just the system prompt framing is not
        # known before the first turn, so seed the meter at 0. It
        # updates after each turn completes.
        self._refresh_metrics()
        # 250ms tick matches the classic CLI's thinking-spinner cadence
        # so the elapsed-time field feels identical. A faster tick is
        # wasted work; a slower one makes the counter feel stuck.
        self.set_interval(0.25, self._refresh_metrics)
        # MLX adapters lazy-load on first complete(). On a 32GB box with
        # --router that's two cold loads on the first turn (router 2GB +
        # main 4-5GB) with nothing but a silent 'thinking N.Ns' counter
        # — looks hung for 1-3 min. Kick a worker to preload both now so
        # the user sees progress in the log and the first turn's latency
        # is just generation, not load. Input stays disabled until warm
        # to avoid a race between the load and an eager first prompt.
        if self._needs_preload():
            self._set_input_enabled(False)
            self.run_worker(self._preload_sync, thread=True, exclusive=False, group="warmup")
        else:
            self.query_one("#prompt", Input).focus()

    def _needs_preload(self) -> bool:
        """True iff at least one adapter we're about to use exposes a
        load() worth doing off the UI thread. Echo / Ollama fall through
        — their load paths are either no-ops or trivially fast, so
        firing a worker just to move focus is overhead."""
        main_has_load = callable(getattr(self._adapter, "load", None))
        router_adapter = (
            getattr(self._router, "adapter", None) if self._router is not None else None
        )
        router_has_load = callable(getattr(router_adapter, "load", None))
        return main_has_load or router_has_load

    def _preload_sync(self) -> None:
        """Worker-thread. Load the main (and router, if present) MLX
        model so the first real turn doesn't eat the cold-load cost
        invisibly. Adapters that don't expose `load()` (echo, ollama)
        are fine — the getattr just returns None and we skip. Any
        load failure is surfaced as a warning line but does NOT block
        the session: on OOM or download hiccup the first turn will
        retry via the normal lazy path."""
        try:
            self.call_from_thread(self._render_loading, f"loading {self._adapter.id}…")
            load_main = getattr(self._adapter, "load", None)
            if callable(load_main):
                load_main()
            self.call_from_thread(self._render_loading, f"✓ {self._adapter.id} ready")
            if self._router is not None:
                router_adapter = getattr(self._router, "adapter", None)
                router_id = getattr(router_adapter, "id", "router")
                self.call_from_thread(self._render_loading, f"loading {router_id}…")
                load_router = getattr(router_adapter, "load", None)
                if callable(load_router):
                    load_router()
                self.call_from_thread(self._render_loading, f"✓ {router_id} ready")
        except Exception as exc:
            self.call_from_thread(
                self._render_warning,
                f"preload failed ({type(exc).__name__}: {exc}); first turn will retry",
            )
        finally:
            self.call_from_thread(self._finish_warmup)

    def _render_loading(self, msg: str) -> None:
        log = self.query_one("#output", RichLog)
        log.write(f"[dim]{msg}[/dim]")

    def _finish_warmup(self) -> None:
        self._set_input_enabled(True)
        self.query_one("#prompt", Input).focus()

    def _replay_history(self, log: RichLog) -> None:
        """Tail the session transcript and render up to
        `max_history_replay` user + assistant turns into the log on
        mount. Populates `_state.history` in the same pass so the
        model sees continuity across app restarts.

        Tool-role turns are intentionally skipped here: the log
        already contained the 🔧 / ✓ lines when those tools ran in
        a prior session, and replaying the raw tool outputs would
        clutter the scroll-back. Historical tool *results* are
        already condensed into the assistant replies that follow
        them — the model's next turn sees that context via the
        assistant message."""
        tail = self._transcript.tail(self._session, limit=self._max_history_replay)
        if not tail:
            return
        log.write(
            Text(
                f"— replaying {len(tail)} prior turns from session '{self._session}' —",
                style="dim",
            )
        )
        for msg in tail:
            if msg.role == "user":
                line = Text()
                line.append(f"{msg.speaker} ›", style="bold cyan")
                line.append(" ")
                line.append(msg.content)
                log.write(line)
                self._state.history.append(ChatMessage(role="user", content=msg.content))
            elif msg.role == "assistant":
                line = Text()
                line.append(f"{msg.speaker} ›", style="bold green")
                line.append(" ")
                line.append(msg.content)
                log.write(line)
                self._state.history.append(ChatMessage(role="assistant", content=msg.content))
            # tool-role rows are not replayed — see docstring.

    # ---------- input path ----------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        # Slash-command intercept. Parity with the classic REPL:
        # /exit, /quit, :q all exit; unrecognized commands fall
        # through to the model so a user who types '/anything' into
        # a prompt isn't silently dropped. More commands (e.g.
        # /edit) land in later follow-ups — they need
        # app.suspend() for $EDITOR and feel out of scope here.
        if text.lower() in {"/exit", "/quit", ":q"}:
            event.input.value = ""
            self.exit()
            return
        event.input.value = ""
        # Always-on prompt: if a turn is in flight, enqueue instead of
        # starting a second worker. _finish_turn drains the queue.
        if self._state.is_busy:
            self._enqueue_prompt(text)
            return
        self._start_turn(text)

    def _enqueue_prompt(self, text: str) -> None:
        """UI-thread. Append a prompt to the pending queue, echo a dim
        placeholder in the log so the user sees it was captured, and
        refresh metrics to show the updated queue count."""
        self._state.pending_prompts.append(text)
        log = self.query_one("#output", RichLog)
        line = Text()
        line.append(f"queued [{len(self._state.pending_prompts)}] ", style="dim yellow")
        line.append(text, style="dim")
        log.write(line)
        self._refresh_metrics()

    def _start_turn(self, text: str) -> None:
        """UI-thread. Echo the user turn into the log and kick the
        worker. Separated from on_input_submitted so _finish_turn can
        drain the pending queue without re-entering the event handler."""
        log = self.query_one("#output", RichLog)
        # Build a Text object so the user's content can't be parsed as
        # Rich markup — `[echo]` etc. in free-form text would otherwise
        # get eaten as an unknown style tag.
        line = Text()
        line.append(f"{self._speaker} ›", style="bold cyan")
        line.append(" ")
        line.append(text)
        log.write(line)
        # Reset per-turn streaming state on the UI thread so the first
        # token arrives into a fresh buffer. stream_first_chunk flips
        # to False after the first sentence is rendered with the
        # 'airton ›' prefix; subsequent sentences are continuations.
        self._state.stream_buffer = ""
        self._state.stream_first_chunk = True
        self._state.is_busy = True
        self._state.turn_seq += 1
        seq = self._state.turn_seq
        # Start the elapsed-time clock on the UI thread so the metrics
        # tick sees it immediately — no 'thinking 0.0s' → 'thinking 0.3s'
        # gap where the spinner hasn't caught up yet.
        self._state.turn_started_at = time.monotonic()
        self._refresh_metrics()
        # `thread=True` runs the sync adapter off the UI thread so
        # token generation doesn't block the event loop. `exclusive=False`
        # because we serialize manually via is_busy + pending_prompts;
        # there is never a second worker we'd want to cancel. `group="turn"`
        # so interrupt can target turn workers without touching warmup.
        self.run_worker(
            lambda: self._run_turn_sync(text, seq),
            thread=True,
            exclusive=False,
            group="turn",
        )

    # ---------- worker (thread; no UI access except call_from_thread) ----------

    def _run_turn_sync(self, user_input: str, seq: int) -> None:
        """Executed on the worker thread. All UI updates go through
        call_from_thread so Textual's reactive tree stays single-
        threaded. Any exception becomes a red log line — we don't
        tear down the app on a per-turn failure.

        `seq` is the turn sequence captured at kickoff. UI updates
        and transcript writes compare against the live
        `_state.turn_seq`; on a mismatch (the turn was interrupted
        or a newer turn started) they no-op so the dying worker
        can't mutate state belonging to a later turn."""

        def hop(fn: Any, *args: Any) -> None:
            """Hop to the UI thread only if this turn is still the
            current one. The guard runs on the UI thread too — safe
            because turn_seq is only ever mutated there."""

            def guarded() -> None:
                if self._state.turn_seq == seq:
                    fn(*args)

            self.call_from_thread(guarded)

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
                # inline. token_delta events feed the stream buffer
                # so the final wrap-up reply streams in live. Write-
                # tier tools always decline for now — the modal
                # confirmation lives in harness-mz2.
                initial_messages = [system, *self._state.history]

                def observe(event: ToolLoopEvent) -> None:
                    hop(self._render_tool_event, event)

                loop_result = run_tool_loop(
                    self._adapter,  # type: ignore[arg-type]
                    [*initial_messages, user_msg],
                    self._tool_registry,
                    confirm=self._confirm_write_tool,
                    observe=observe,
                    router=self._router,
                )
                if self._state.turn_seq != seq:
                    return  # interrupted; drop partial reply + skip persistence
                # The observer streamed tokens as they arrived; the
                # tail may still be in the buffer if the final reply
                # didn't end on a sentence boundary.
                hop(self._flush_stream_buffer)
                # If the stream observer didn't surface any tokens
                # (e.g. the loop exhausted without tool calls, or the
                # adapter doesn't support stream_with_tools), fall
                # back to rendering the blocking reply content so the
                # user isn't staring at a silent log.
                if self._state.stream_first_chunk and loop_result.content:
                    hop(self._feed_stream, loop_result.content)
                    hop(self._flush_stream_buffer)
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
                # Stream when the adapter exposes stream(); fall back
                # to blocking complete() for adapters that don't.
                # Every built-in adapter (echo, mlx, ollama) supplies
                # stream(), so this is the hot path.
                stream_fn = getattr(self._adapter, "stream", None)
                if callable(stream_fn):
                    reply_parts: list[str] = []
                    for delta in stream_fn(
                        messages,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                    ):
                        if self._state.turn_seq != seq:
                            break  # interrupted: stop pulling tokens from the generator
                        reply_parts.append(delta)
                        hop(self._feed_stream, delta)
                    reply = "".join(reply_parts)
                else:
                    reply = self._adapter.complete(
                        messages,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                    )
                    hop(self._feed_stream, reply)
                if self._state.turn_seq != seq:
                    return  # interrupted; drop partial reply + skip persistence
                # Push any trailing tail (last non-sentence fragment)
                # into the log so nothing is swallowed between the
                # final period and the next user turn.
                hop(self._flush_stream_buffer)
                self._state.history.append(user_msg)
                self._state.history.append(ChatMessage(role="assistant", content=reply))
                self._transcript.append(
                    session=self._session,
                    channel=self._channel,
                    speaker=self._character.name,
                    role="assistant",
                    content=reply,
                )
        except Exception as exc:
            hop(self._render_error, exc)
        finally:
            # Stop the elapsed-time counter and re-enable input from
            # the UI thread. Order matters: clear the timer BEFORE
            # refreshing metrics so the next tick shows 'idle' rather
            # than a stale final elapsed-time string. If this turn was
            # interrupted, the UI-thread action_interrupt already did
            # the teardown — the seq-gated hop no-ops here.
            hop(self._finish_turn)

    # ---------- UI-thread helpers (all run via call_from_thread) ----------

    def _feed_stream(self, delta: str) -> None:
        """UI-thread only. Append a token delta to the per-turn
        buffer and emit every completed sentence as a RichLog line.
        Called from the no-tools stream path via call_from_thread,
        and from _render_tool_event when a token_delta observer
        event fires."""
        if not delta:
            return
        self._state.stream_buffer += delta
        while True:
            match = _SENTENCE_BOUNDARY_RE.search(self._state.stream_buffer)
            if match is None:
                return
            end = match.end()
            sentence = self._state.stream_buffer[:end]
            self._state.stream_buffer = self._state.stream_buffer[end:]
            self._emit_stream_sentence(sentence)

    def _flush_stream_buffer(self) -> None:
        """UI-thread only. Emit whatever is left in the buffer even
        without a trailing sentence boundary — the model may stop
        mid-sentence, or the reply may end with an incomplete quote
        the regex won't match."""
        if self._state.stream_buffer:
            self._emit_stream_sentence(self._state.stream_buffer)
            self._state.stream_buffer = ""

    def _emit_stream_sentence(self, sentence: str) -> None:
        """UI-thread only. Write one sentence of the assistant's
        reply to the log, prefixing the first emission of this turn
        with 'airton ›' so the user can see the model started
        speaking. Subsequent sentences land as continuation lines."""
        log = self.query_one("#output", RichLog)
        line = Text()
        if self._state.stream_first_chunk:
            line.append(f"{self._character.name} ›", style="bold green")
            line.append(" ")
            self._state.stream_first_chunk = False
        # rstrip the trailing newline the regex captured — RichLog
        # adds its own line break and double newlines look off.
        line.append(sentence.rstrip("\n"))
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

    def action_interrupt(self) -> None:
        """Ctrl-X binding: cancel the running turn (harness-xuh).

        Bumps `turn_seq` so the doomed worker's trailing
        call_from_thread updates no-op. Cancels the `turn` worker
        group (Textual thread workers can't be killed mid-syscall —
        the generator loop checks the seq each iteration and breaks
        voluntarily; the seq gate makes any late UI updates harmless).
        Renders an '⏹ interrupted' marker, drops the in-flight stream
        buffer so partial output doesn't leak into the next turn,
        and drains the pending queue the same way _finish_turn does
        so queued prompts aren't stranded."""
        if not self._state.is_busy:
            return
        self._state.turn_seq += 1
        self._state.stream_buffer = ""
        self._state.stream_first_chunk = True
        if self._state.turn_started_at is not None:
            self._state.last_elapsed = time.monotonic() - self._state.turn_started_at
        self._state.turn_started_at = None
        self._state.is_busy = False
        # Cancellation is best-effort — Textual's API surface has
        # shifted between versions. A missed cancel is fine because
        # the seq gate neutralises trailing updates.
        with contextlib.suppress(Exception):
            self.workers.cancel_group(self, "turn")
        log = self.query_one("#output", RichLog)
        log.write(Text("⏹ interrupted", style="bold red"))
        self.query_one("#prompt", Input).focus()
        if self._state.pending_prompts:
            next_text = self._state.pending_prompts.popleft()
            self._start_turn(next_text)
            return
        self._refresh_metrics()

    def _finish_turn(self) -> None:
        """Worker-completion hook on the UI thread. Stops the elapsed
        clock, snapshots the final duration into last_elapsed so the
        idle metrics strip can keep showing it, recomputes the ctx
        meter against the now-updated history, clears the busy flag,
        and either drains the pending queue into the next turn or
        just refreshes metrics so the user sees the final state
        immediately instead of waiting for the next tick."""
        if self._state.turn_started_at is not None:
            self._state.last_elapsed = time.monotonic() - self._state.turn_started_at
        self._state.turn_started_at = None
        self._state.is_busy = False
        self._recompute_ctx_used()
        self.query_one("#prompt", Input).focus()
        if self._state.pending_prompts:
            next_text = self._state.pending_prompts.popleft()
            self._start_turn(next_text)
            return
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

    def _confirm_write_tool(self, call: ToolCall) -> bool:
        """Worker-thread confirm callback for write-tier tool calls.
        If the tool has been marked always-allowed for this session,
        approve silently. Otherwise block the worker until the user
        dismisses the modal on the UI thread.

        `call_from_thread(coro)` schedules the coroutine on the
        Textual event loop and blocks this thread until it returns
        — which is exactly what we need to give run_tool_loop the
        sync `bool` it expects."""
        if call.name in self._approved_tools:
            return True
        # Textual stubs Callable[..., Awaitable[Never]] for
        # call_from_thread's arg, which doesn't line up with an async
        # method that returns str | None — even though the runtime
        # supports exactly this. Cast via Any locally so the calling
        # site stays readable.
        call_from_thread: Any = self.call_from_thread
        decision: str | None = call_from_thread(self._prompt_for_confirm, call)
        if decision == APPROVE:
            return True
        if decision == ALWAYS:
            self._approved_tools.add(call.name)
            return True
        # DECLINE, None (modal cancelled), or any unexpected value.
        return False

    async def _prompt_for_confirm(self, call: ToolCall) -> str | None:
        """UI-thread coroutine: push the modal and await its result.
        Returns the dismiss payload verbatim ('approve' / 'decline'
        / 'always'), or None if the screen was dismissed without a
        value (shouldn't happen with the normal bindings but we
        handle it defensively)."""
        screen = ConfirmToolScreen(call, label=self._tool_label(call.name))
        return await self.push_screen_wait(screen)

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
        elif event.kind == "token_delta":
            # Orchestrator / adapter already masked tool-call tag
            # spans; whatever survives is visible reply text. Feed
            # through the sentence buffer so the wrap-up reply
            # streams in live.
            if event.delta:
                self._feed_stream(event.delta)
        # Other event kinds (round_start, model_call_start/end,
        # round_complete) are internal book-keeping — the metrics
        # footer already covers 'model is thinking'.

    # ---------- metrics ----------

    def _refresh_metrics(self) -> None:
        """Render the metrics strip. Idle state shows just the ctx
        meter; active turn appends 'thinking N.Ns' so the user can
        see the model is alive even when streaming tokens hasn't
        started yet. Elapsed time is formatted to one decimal place
        so the ticker visibly advances at the 0.25s tick cadence.
        When the always-on prompt has queued submissions, append
        'queued N' so the user can see the backlog depth."""
        metrics = self.query_one("#metrics", Static)
        meter = _format_ctx_meter(self._state.ctx_used, self._adapter.context_window) or "ctx —"
        parts = [meter]
        if self._state.turn_started_at is not None:
            elapsed = time.monotonic() - self._state.turn_started_at
            parts.append(f"thinking {elapsed:.1f}s")
        elif self._state.last_elapsed is not None:
            parts.append(f"last {self._state.last_elapsed:.1f}s")
        else:
            parts.append("idle")
        if self._state.pending_prompts:
            parts.append(f"queued {len(self._state.pending_prompts)}")
        metrics.update(Text.from_markup(" · ".join(parts)))
