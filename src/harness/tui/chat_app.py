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

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, ClassVar

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, Input, RichLog, Static

from harness.cli import (
    _TOOL_CALLS_SENTINEL,
    _build_tool_grounding_block,
    _persist_tool_exchange,
    _render_fact_block,
    _render_memory_block,
    _retrieve_turn_context,
)
from harness.cli import _RetrievalState as _RetrievalHealth
from harness.model.adapter import ChatMessage
from harness.orchestrator import ToolLoopEvent, run_tool_loop
from harness.tui.confirm import (
    ALWAYS,
    APPROVE,
    DECLINE,
    ConfirmController,
)
from harness.tui.metrics import MetricsView
from harness.tui.slash_ops import SlashOps
from harness.tui.stream import StreamView

if TYPE_CHECKING:
    from pathlib import Path

    from harness.character import Character
    from harness.compaction import CompactionStore
    from harness.model.adapter import ModelAdapter
    from harness.retrieval import VoiceRetriever
    from harness.router import Router
    from harness.store import EpisodicStore, SemanticStore
    from harness.store.bd_adapter import BeadsAdapter
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
    # flushed at round end to a rich.Markdown block in the main log.
    # `stream_first_chunk` tracks whether the 'airton ›' badge has
    # been emitted yet this turn (shared across tool-loop rounds).
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
    # Inline write-tier confirmation (harness-drd). Future is created
    # on the UI event loop when a write-tier tool call arrives and
    # resolved by action_confirm_{approve,decline,always} or escape.
    # The worker thread awaits it via call_from_thread. None means
    # the confirm strip is hidden and no call is pending.
    confirm_future: asyncio.Future[str] | None = None


# Speaker-badge styles (harness-nrx). Reverse-video bold colored pads
# draw the eye to turn boundaries without touching the message body —
# code blocks, bullet lists, and other structure stay untouched.
_USER_BADGE_STYLE = "reverse bold cyan"
_ASSISTANT_BADGE_STYLE = "reverse bold green"


def _make_user_line(speaker: str, content: str) -> Text:
    """Build `[badge] speaker › [/badge] content` as a single rich
    Text. Extracted so live turns, replayed turns, and tests all
    produce identical shape."""
    line = Text()
    line.append(f" {speaker} › ", style=_USER_BADGE_STYLE)
    line.append(" ")
    line.append(content)
    return line


def _make_assistant_badge(speaker: str) -> Text:
    """Return a Text seeded with the assistant badge. Caller appends
    the body, either a full line (replay) or a streamed sentence
    (live turn)."""
    line = Text()
    line.append(f" {speaker} › ", style=_ASSISTANT_BADGE_STYLE)
    line.append(" ")
    return line


# Slash-command registry for the palette. Alpha order is the contract
# the palette relies on — keep it sorted by name. Descriptions are
# rendered dim next to the name. `:q` and `/capture` are NOT in the
# palette: `:q` is a hidden vim-muscle-memory alias for /exit and
# `/capture` is a hidden alias for /edit; both stay routable from the
# submit handler so typing them directly still works. harness-kg9.
_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/clear", "wipe model-visible history; persisted stores untouched"),
    ("/compact", "summarize older turns into a session summary"),
    ("/consolidate", "merge near-duplicate memories + facts"),
    ("/edit", "edit Airton's last reply as a new voice sample"),
    ("/exit", "leave chat"),
    ("/quit", "leave chat"),
    ("/retro", "ab's thought-graph retrospective (summary)"),
    ("/scribe", "extract memory candidates from recent turns"),
    ("/sessions", "list every recorded chat session, newest first"),
    ("/session", "dump current session (or `<id>`) as paste-ready markdown"),
    ("/session-compact-reset", "drop folded summary for current (or `<id>`)"),
    ("/session-reset", "full clean of current (or `<id>`): summary + clear + working memory"),
)


class SlashPalette(Static):
    """Inline picker for slash commands (harness-kg9).

    Shown directly above the Input when the Input's value starts with
    `/`. Hidden otherwise. Filters `_SLASH_COMMANDS` by case-insensitive
    prefix as the user types; arrow keys move the highlight, Enter
    fills the Input with the highlighted command and triggers submit.
    Escape closes without a selection.

    Rendered as a Static (not an OptionList) so the Input keeps focus
    — no focus juggling mid-keystroke, and the user's cursor position
    stays where they expect. Navigation + selection go through App
    bindings with `check_action` gating them to palette-visible state
    so Enter/Up/Down behave normally when the palette is closed."""

    def __init__(self, commands: tuple[tuple[str, str], ...]) -> None:
        # Seed with a plain string (not Text) so Textual's visual cache
        # doesn't trip over an empty rich Text during the initial layout
        # pass — which measures hidden widgets too.
        super().__init__(" ", id="slash_palette", markup=True)
        self._all = tuple(sorted(commands, key=lambda c: c[0]))
        self._filtered: list[tuple[str, str]] = []
        self._highlight: int = -1
        self.display = False

    @property
    def is_open(self) -> bool:
        """True while the palette is visible. Named `is_open` (not
        `visible`) to avoid shadowing Widget.visible, which is a
        read/write reactive the base class owns."""
        return bool(self.display)

    def filter_to(self, prefix: str) -> None:
        """Update filtered list for `prefix` and re-render. Called from
        on_input_changed — cheap enough to run on every keystroke."""
        key = prefix.lower()
        self._filtered = [(n, d) for (n, d) in self._all if n.lower().startswith(key)]
        self._highlight = 0 if self._filtered else -1
        self._refresh_list()
        self.display = bool(self._filtered)

    def move(self, delta: int) -> None:
        if not self._filtered:
            return
        self._highlight = (self._highlight + delta) % len(self._filtered)
        self._refresh_list()

    def selected_name(self) -> str | None:
        if 0 <= self._highlight < len(self._filtered):
            return self._filtered[self._highlight][0]
        return None

    def close(self) -> None:
        self.display = False
        self._filtered = []
        self._highlight = -1

    def _refresh_list(self) -> None:
        lines: list[str] = []
        for i, (name, desc) in enumerate(self._filtered):
            if i == self._highlight:
                lines.append(f"[bold cyan reverse]▸ {name}[/] [dim]— {desc}[/dim]")
            else:
                lines.append(f"[cyan]  {name}[/cyan] [dim]— {desc}[/dim]")
        # Fall back to a space so the Static always has non-empty
        # content; empty-Text content has tripped Textual's visual
        # cache during layout in 8.x.
        self.update("\n".join(lines) if lines else " ")


class ConfirmStrip(Widget):
    """Inline write-tier confirmation strip (harness-drd).

    Docks to the left of the prompt Input inside `#prompt_row`. When a
    write-tier tool call arrives the strip is shown, receives focus,
    and offers three verdicts via keybindings (y/n/a, plus Escape as
    a decline alias) and three real Buttons so mouse users aren't
    stranded. The Input stays visible with its caret and value intact
    — it's just temporarily not the focused widget so letter bindings
    can fire without being swallowed by Input's character path.

    Verdicts are posted upward to the App via its `_resolve_confirm`
    hook so a single code path fulfils the pending confirm future
    regardless of whether it came from a key or a click."""

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "approve", "approve", priority=True),
        Binding("n", "decline", "decline", priority=True),
        Binding("a", "always", "always", priority=True),
        Binding("escape", "decline", "cancel", priority=True),
        # Left/right walk between the three Buttons so keyboard-only
        # users can tab-navigate without going through the Input.
        Binding("left", "focus_previous", show=False),
        Binding("right", "focus_next", show=False),
    ]

    def __init__(self) -> None:
        super().__init__(id="confirm_strip")

    def compose(self) -> ComposeResult:
        yield Static("", id="confirm_summary", markup=True)
        with Horizontal(id="confirm_buttons"):
            yield Button("[y] approve", id="btn_confirm_approve", variant="success")
            yield Button("[n] decline", id="btn_confirm_decline", variant="error")
            yield Button("[a] always", id="btn_confirm_always", variant="warning")

    def show_for(self, summary_markup: str) -> None:
        self.query_one("#confirm_summary", Static).update(summary_markup)
        self.add_class("-visible")
        # Focus a button (not the wrapper) so Enter/Space can activate
        # the default verdict and mouse/keyboard navigation feels
        # coherent. `approve` is the highlighted default.
        self.query_one("#btn_confirm_approve", Button).focus()

    def hide(self) -> None:
        self.remove_class("-visible")
        self.query_one("#confirm_summary", Static).update("")

    # Widget-level actions forward to the app resolver so the pending
    # future is fulfilled once; the app also hides the strip and
    # returns focus to the Input.
    def action_approve(self) -> None:
        app = self.app
        if isinstance(app, ChatApp):
            app._resolve_confirm(APPROVE)

    def action_decline(self) -> None:
        app = self.app
        if isinstance(app, ChatApp):
            app._resolve_confirm(DECLINE)

    def action_always(self) -> None:
        app = self.app
        if isinstance(app, ChatApp):
            app._resolve_confirm(ALWAYS)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "btn_confirm_approve": APPROVE,
            "btn_confirm_decline": DECLINE,
            "btn_confirm_always": ALWAYS,
        }
        decision = mapping.get(event.button.id or "")
        if decision is None:
            return
        app = self.app
        if isinstance(app, ChatApp):
            app._resolve_confirm(decision)


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

    #stream_preview {
        height: auto;
        max-height: 8;
        padding: 0 2;
        background: $background;
        color: $text-muted;
        display: none;
    }

    #stream_preview.-visible {
        display: block;
    }

    #prompt_row {
        height: auto;
    }

    Input {
        border: tall $accent;
        margin: 0 0 1 0;
        width: 1fr;
    }

    Input:disabled {
        border: tall $warning-muted;
    }

    #confirm_strip {
        width: auto;
        max-width: 70;
        height: auto;
        margin: 0 1 1 0;
        padding: 0 1;
        border: tall $warning;
        background: $panel;
        color: $text;
        display: none;
    }

    #confirm_strip.-visible {
        display: block;
    }

    #confirm_summary {
        width: auto;
        padding: 0 0 1 0;
    }

    #confirm_buttons {
        height: auto;
        width: auto;
    }

    #confirm_buttons Button {
        min-width: 14;
        margin: 0 1 0 0;
        height: 1;
        border: none;
    }

    #slash_palette {
        padding: 0 2;
        background: $boost;
        color: $text;
        border-top: wide $primary;
        max-height: 8;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "quit", "quit"),
        ("ctrl+d", "quit", "quit"),
        # priority=True so the focused Input doesn't swallow the
        # keypress — interrupt has to be reachable mid-turn when the
        # prompt is where the user's hands already are.
        Binding("ctrl+x", "interrupt", "interrupt", priority=True),
        # Slash-palette navigation (harness-kg9). priority=True so
        # Enter is intercepted before Input's Submitted fires, letting
        # us replace the input value with the highlighted command.
        # check_action gates these to palette-visible state so Enter /
        # Up / Down / Escape behave normally when the palette is hidden.
        Binding("up", "palette_prev", show=False, priority=True),
        Binding("down", "palette_next", show=False, priority=True),
        Binding("enter", "palette_select", show=False, priority=True),
        # Escape at the app level only dismisses the slash palette —
        # confirm-strip escape lives on that widget so it fires when
        # the strip has focus (harness-drd).
        Binding("escape", "palette_close", show=False, priority=True),
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
        retrieval_health: _RetrievalHealth | None = None,
        compaction_store: CompactionStore | None = None,
        scribe_lock_dir: Path | None = None,
        scribe_user_id: str | None = None,
        auto_scribe: bool = True,
        ab_adapter: BeadsAdapter | None = None,
        hooks: object | None = None,
        allowed_sessions: tuple[str, ...] | None = None,
        recency_ranks: dict[str, int] | None = None,
        recency_weight: float = 0.0,
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
        self._startup_warnings = startup_warnings
        # Slash-command ops plumbing (harness-kg9). Each may be None —
        # in which case the matching /compact /scribe command reports
        # "not configured" instead of 500-ing. /consolidate wants the
        # memory + semantic stores already held above; /edit uses the
        # transcript alone.
        self._compaction_store = compaction_store
        self._scribe_lock_dir = scribe_lock_dir
        self._scribe_user_id = scribe_user_id
        # When True, /compact first scribes unprocessed turns into
        # episodic + semantic memory, then folds older turns into a
        # summary. Default on so the user doesn't have to remember to
        # /scribe before /compact. See harness-0kw.
        self._auto_scribe = auto_scribe
        # Ab's bd adapter — only populated when character=airton_b and
        # the isolated bd dir verifies. Feeds /retro and the
        # per-turn reset_turn_counter hook. None for every other
        # character; the /retro handler reports 'not configured'.
        self._ab_adapter = ab_adapter
        # Optional HookPipeline override — CLI injects one when
        # --summarize-tool-results is set so a post_tool hook can
        # compress grep / list_dir / search_web dumps before they
        # hit the main model's context.
        self._hooks = hooks
        # Resolved `--memory-scope` filter (harness-w3mo). None = no
        # session filter (default 'all'). Tuple = NULL OR session_id
        # IN (...). Threaded into `_retrieve_turn_context` per turn so
        # the TUI matches the classic REPL's session-bounded retrieval.
        self._allowed_sessions = allowed_sessions
        # Recency-RRF gate (harness-w3mo step 5). When weight > 0,
        # the store's hybrid search fuses a third RRF tier ordered
        # by session recency. None ranks + zero weight means the
        # gate is fully off and the store skips the third ranking.
        self._recency_ranks = recency_ranks
        self._recency_weight = recency_weight
        # Accept an external retrieval_health reference so the
        # IntrospectTool (harness-8is) can see live voice/episodic/
        # semantic health without a callback plumbing. When None the
        # default factory produces a fresh all-ok state.
        self._state = _ChatAppState()
        if retrieval_health is not None:
            self._state.retrieval_health = retrieval_health
        self._stream = StreamView(self, self._state, self._character.name)
        self._metrics = MetricsView(self, self._state, self._adapter)
        self._confirm = ConfirmController(self, self._state)
        self._ops = SlashOps(self)

    # ---------- compose / mount ----------

    def compose(self) -> ComposeResult:
        # Natural vertical flow: RichLog takes remaining space, metrics
        # sits as a 1-row strip above the input, input at the bottom.
        # Previous dock-bottom on both overlapped visually — prompt
        # border got clipped by the stat bar.
        yield RichLog(id="output", wrap=True, markup=True, highlight=False)
        # Live stream preview (harness-fup): in-flight assistant tokens
        # render here as plain Text for responsiveness. At round end the
        # buffered text commits to the main RichLog as a rich.Markdown
        # block so headers, lists, and fenced code render properly.
        yield Static("", id="stream_preview", markup=False)
        yield Static("ctx — · elapsed —", id="metrics")
        yield SlashPalette(_SLASH_COMMANDS)
        # Prompt row is a Horizontal so the transient write-tier
        # confirm strip can dock to the left and visually shift the
        # Input without clearing its value or moving the caret inside
        # the Input itself (harness-drd).
        yield Horizontal(
            ConfirmStrip(),
            Input(id="prompt", placeholder="type a message… (ctrl+c to quit)"),
            id="prompt_row",
        )

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
        assistant message.

        Assistant turns carrying the `__TOOL_CALLS_V1__` sentinel
        (harness-4fc) are also skipped. Those are persisted with the
        structured tool-call payload appended to the content; without
        their paired tool-role result rows (which we already skip) the
        history is orphaned, and rendering the encoded content dumps
        raw JSON into the log. Dropping them keeps replay coherent at
        the cost of an occasional gap in the scroll-back when a prior
        session ended mid-tool-exchange."""
        # Honor any persisted /clear watermark so a prior `/clear`
        # in this session stays cut after a chat restart (harness-rrkj).
        # Compaction store may be unwired (e.g. compact_at=0); fall
        # back to the plain tail in that case. When a watermark
        # hydrates, also flip retrieval_health.muted so the next
        # turn sees the same retrieval-suppression an in-process
        # /clear would have set (harness-eftf).
        clear_after = (
            self._compaction_store.latest_clear_after_id(self._session)
            if self._compaction_store is not None
            else None
        )
        if clear_after is not None:
            self._state.retrieval_health.muted = True
            rows = self._transcript.fetch_after(self._session, after_id=clear_after)
            tail = rows[-self._max_history_replay :] if rows else []
        else:
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
                # Blank line before each user turn groups the prior
                # assistant turn's work (reply + any tool calls) into
                # one block and gives the next pair breathing room.
                # harness-nrx.
                log.write("")
                log.write(_make_user_line(msg.speaker, msg.content))
                self._state.history.append(ChatMessage(role="user", content=msg.content))
            elif msg.role == "assistant":
                if _TOOL_CALLS_SENTINEL in msg.content:
                    continue  # tool-call turn — skip to mirror tool-role skipping
                log.write(_make_assistant_badge(msg.speaker))
                log.write(Markdown(msg.content, code_theme="monokai"))
                self._state.history.append(ChatMessage(role="assistant", content=msg.content))
            # tool-role rows are not replayed — see docstring.

    # ---------- input path ----------

    def on_input_changed(self, event: Input.Changed) -> None:
        """Keep the slash-palette in sync with the input. Any value
        starting with `/` opens / filters the palette; anything else
        closes it. Runs on every keystroke — cheap because the filter
        is a prefix scan over a handful of entries."""
        palette = self.query_one("#slash_palette", SlashPalette)
        value = event.value
        if value.startswith("/"):
            palette.filter_to(value)
        elif palette.is_open:
            palette.close()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        # Close the palette on every submit — if the user typed a
        # command that isn't in the registry, we fall through to the
        # model and the palette no longer applies to the next turn.
        self.query_one("#slash_palette", SlashPalette).close()
        # User-turn boundary — refresh the per-turn ab-bead create
        # budget so each turn starts with a fresh 3-slot allowance.
        # Runs before slash dispatch so even slash commands bump the
        # counter (they're user actions, and never spawn ab-captures,
        # so the cost is free). harness-4ate.
        if self._ab_adapter is not None:
            self._ab_adapter.reset_turn_counter()
        # Split off the verb (lowercased so case doesn't matter for
        # the command word) from any positional args. Session ids are
        # case-sensitive and pass through verbatim.
        parts = text.strip().split()
        cmd = parts[0].lower() if parts else ""
        cmd_args = parts[1:]
        # Slash-command intercept. Parity with the classic REPL:
        # /exit, /quit, :q all exit; /edit (+ /capture alias) opens
        # $EDITOR on Airton's last reply for voice capture; /compact,
        # /scribe, /consolidate invoke the corresponding memory op;
        # /retro runs ab's retrospective summary; /sessions + /session
        # family inspect / reset persisted session state.
        # Unrecognized slash commands fall through to the model so a
        # user who types '/anything' isn't silently dropped.
        if cmd in {"/exit", "/quit", ":q"}:
            event.input.value = ""
            self.exit()
            return
        if cmd in {"/edit", "/capture"}:
            event.input.value = ""
            self._ops.run_edit_capture()
            return
        if cmd == "/clear":
            event.input.value = ""
            self._ops.run_clear()
            return
        if cmd == "/compact":
            event.input.value = ""
            self._ops.kick("compact", "run_compact_sync")
            return
        if cmd == "/scribe":
            event.input.value = ""
            self._ops.kick("scribe", "run_scribe_sync")
            return
        if cmd == "/consolidate":
            event.input.value = ""
            self._ops.kick("consolidate", "run_consolidate_sync")
            return
        if cmd == "/retro":
            event.input.value = ""
            self._ops.run_retro()
            return
        if cmd == "/sessions":
            event.input.value = ""
            self._ops.run_session_list()
            return
        if cmd == "/session":
            event.input.value = ""
            self._ops.run_session_show(cmd_args[0] if cmd_args else None)
            return
        if cmd == "/session-compact-reset":
            event.input.value = ""
            self._ops.run_session_compact_reset(cmd_args[0] if cmd_args else None)
            return
        if cmd == "/session-reset":
            event.input.value = ""
            self._ops.run_session_reset(cmd_args[0] if cmd_args else None)
            return
        event.input.value = ""
        # Always-on prompt: if a turn is in flight, enqueue instead of
        # starting a second worker. _finish_turn drains the queue.
        if self._state.is_busy:
            self._enqueue_prompt(text)
            return
        self._start_turn(text)

    # ---------- slash-palette actions (harness-kg9) ----------

    def _palette_visible(self) -> bool:
        """True iff the slash-palette is currently shown. Used by
        check_action so Enter / Up / Down / Escape only intercept when
        the palette is active — otherwise they must fall through to
        their default handlers (Input submit, no-op, no-op)."""
        try:
            palette = self.query_one("#slash_palette", SlashPalette)
        except Exception:
            return False
        return palette.is_open

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Gate palette-scoped bindings by visibility. Returning False
        disables the binding so the keypress propagates to its default
        handler (Input.Submitted for Enter, cursor motion otherwise)."""
        if action in {"palette_prev", "palette_next", "palette_select", "palette_close"}:
            return self._palette_visible()
        return True

    def action_palette_prev(self) -> None:
        self.query_one("#slash_palette", SlashPalette).move(-1)

    def action_palette_next(self) -> None:
        self.query_one("#slash_palette", SlashPalette).move(1)

    def action_palette_close(self) -> None:
        self.query_one("#slash_palette", SlashPalette).close()

    def action_palette_select(self) -> None:
        """Fill the Input with the highlighted command and submit.
        Runs only while the palette is visible (check_action gate).
        If nothing is highlighted — user typed `/` but then kept typing
        past any match so `_filtered` is empty — just close and let the
        user keep typing; nothing to select."""
        palette = self.query_one("#slash_palette", SlashPalette)
        name = palette.selected_name()
        palette.close()
        prompt = self.query_one("#prompt", Input)
        if name is None:
            return
        prompt.value = name
        # Fire Submitted so all the usual slash-command routing runs.
        prompt.post_message(Input.Submitted(prompt, name))

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
        # Blank line between turns + reverse-video speaker badge gives
        # the eye a clear boundary without touching the message body.
        # Body text goes through a rich Text (not markup) so '[echo]'
        # etc. in user input can't be parsed as style tags. harness-nrx.
        log.write("")
        log.write(_make_user_line(self._speaker, text))
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
                allowed_sessions=self._allowed_sessions,
                recency_ranks=self._recency_ranks,
                recency_weight=self._recency_weight,
            )

            system_content = (
                self._character.system_prompt(include_samples=examples, now=date.today())
                if examples
                else self._character.system_prompt(now=date.today())
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

            # Topic-boundary signal (harness-eftf + harness-w3mo).
            # Mirror of the classic-REPL injection.
            from harness.cli import _topic_boundary_suffix

            system_content = (
                f"{system_content}"
                f"{_topic_boundary_suffix(self._state.retrieval_health, self._allowed_sessions)}"
            )

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
                    hooks=self._hooks,  # type: ignore[arg-type]  # typed `object` to skip import
                    memory_block_attached=bool(recalled),
                    force_search_memory=self._character.require_search_memory,
                    scope_redirect_template=self._character.scope_redirect_template,
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
        self._stream.feed(delta)

    def _flush_stream_buffer(self) -> None:
        self._stream.flush()

    def _hide_stream_preview(self) -> None:
        self._stream.hide_preview()

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
        # If the worker is parked on a write-tier confirm, unblock it
        # with a decline so the turn can tear down cleanly; the
        # seq-gated hop will no-op any trailing UI updates.
        if self._confirm_pending():
            self._resolve_confirm(DECLINE)
        self._state.turn_seq += 1
        self._stream.reset()
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
        self._metrics.recompute_ctx()

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
        return self._confirm.request(call)

    def _confirm_pending(self) -> bool:
        return self._confirm.is_pending()

    def _resolve_confirm(self, decision: str) -> None:
        self._confirm.resolve(decision)

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
        # harness-fup: non-token events mean the current streaming
        # round is done (or never started). Flush the live preview
        # into the log as Markdown before rendering the tool-event
        # line so visual order matches emission order. Skip the flush
        # for truncated_retry / bail_retry — those branches drop the
        # partial instead of committing it, since the retry supersedes
        # the draft.
        if event.kind not in ("token_delta", "truncated_retry", "bail_retry"):
            self._flush_stream_buffer()
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
        elif event.kind == "truncated_retry":
            # Wrap-up round stopped at the token cap; orchestrator is
            # re-running with a doubled budget. Drop any unflushed
            # partial in the stream buffer and flag the break so the
            # user knows the next reply replaces the partial above,
            # not appends to it (harness-6rl).
            self._stream.reset()
            log.write(Text("⋯ truncated, retrying with wider budget…", style="dim"))
        elif event.kind == "bail_retry":
            # 0-tool-calls reply tripped a fabrication / teaser
            # catcher; orchestrator appended a nudge and is re-running.
            # Same drop-partial contract as truncated_retry so the
            # fabricated draft doesn't stack above the next retry
            # (harness-24xj). `event.catcher` names the hook so the
            # user can diagnose WHY the retry happened.
            self._stream.reset()
            suffix = f" ({event.catcher})" if event.catcher else ""
            log.write(Text(f"⋯ discarding draft, retrying{suffix}…", style="dim"))
        # Other event kinds (round_start, model_call_start/end,
        # round_complete) are internal book-keeping — the metrics
        # footer already covers 'model is thinking'.

    # ---------- metrics ----------

    def _refresh_metrics(self) -> None:
        self._metrics.refresh()
