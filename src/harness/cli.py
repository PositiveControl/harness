from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import cast

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status
from rich.table import Table

import harness._quiet  # noqa: F401 — side-effect import: silences HF/transformers/sentence-transformers noise before they load
from harness.character import Character, VoiceSample, load_character
from harness.compaction import (
    CompactionOutcome,
    CompactionStore,
    run_compaction,
    should_compact,
)
from harness.config import settings
from harness.consolidate import run_consolidation
from harness.evals.router import (
    RouterEvalResult,
    default_fixture_path,
    load_fixture,
    run_router_eval,
)
from harness.evals.voice import run_voice_eval
from harness.model import AdapterName, ChatMessage, ModelAdapter, make_adapter
from harness.model.adapter import Role, count_tokens
from harness.orchestrator import (
    _FABRICATED_SEARCH_RE,
    _FALSE_SUCCESS_RE,
    _META_CONFIRM_RE,
    _TOOL_INTENT_RE,
    ToolLoopEvent,
    run_tool_loop,
)
from harness.persona import PersonaAdapter
from harness.persona.rewriter import build_rewriter_messages
from harness.retrieval import VoiceRetriever
from harness.router import GrammarRouter, ModelRouter, Router
from harness.scribe import run_scribe
from harness.store import (
    EpisodicRecord,
    EpisodicStore,
    SemanticFact,
    SemanticStore,
    ensure_seeds_ingested,
)
from harness.store.transcript import Transcript, TranscriptMessage
from harness.tools import (
    DEFAULT_PROFILE,
    TOOL_PROFILES,
    ConsolidateMemoryTool,
    EditFileTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    RememberEventTool,
    RememberFactTool,
    ScribeSessionTool,
    SearchFactsTool,
    SearchMemoryTool,
    SearchWebTool,
    ShellTool,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    WriteFileTool,
    resolve_tool_names,
)

_EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit", ":q", ":quit"})
_EDIT_COMMANDS = frozenset({"/edit", "/capture"})

# Sentinel used to encode structured tool_calls onto an assistant turn's
# content when persisting to the transcript. Two-line format: human-readable
# content, then the sentinel, then a single JSON line with the tool_calls.
_TOOL_CALLS_SENTINEL = "\n__TOOL_CALLS_V1__\n"


def _encode_assistant_with_tool_calls(content: str, tool_calls: tuple[ToolCall, ...]) -> str:
    if not tool_calls:
        return content
    payload = json.dumps(
        {"tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls]}
    )
    return f"{content}{_TOOL_CALLS_SENTINEL}{payload}"


def _decode_transcript_message(m: TranscriptMessage) -> ChatMessage:
    """Convert a persisted transcript row back into a ChatMessage so the
    next turn's history reconstructs tool_calls on assistant turns and
    carries the tool name on tool-role turns."""
    role = cast(Role, m.role)
    content = m.content
    tool_calls: tuple[ToolCall, ...] = ()
    if role == "assistant" and _TOOL_CALLS_SENTINEL in content:
        head, _, tail_json = content.partition(_TOOL_CALLS_SENTINEL)
        try:
            payload = json.loads(tail_json)
            parsed = payload.get("tool_calls") or []
            tool_calls = tuple(
                ToolCall(name=p["name"], arguments=p.get("arguments", {}))
                for p in parsed
                if isinstance(p, dict) and isinstance(p.get("name"), str)
            )
            content = head
        except (json.JSONDecodeError, KeyError, TypeError):
            tool_calls = ()
    name = m.speaker if role == "tool" else None
    return ChatMessage(role=role, content=content, name=name, tool_calls=tool_calls)


def _persist_tool_exchange(
    transcript: Transcript,
    *,
    session: str,
    channel: str,
    character_name: str,
    initial_count: int,
    loop_messages: list[ChatMessage],
) -> None:
    """Append the assistant tool-call and tool-result turns from a tool
    loop to the transcript so the next user turn sees them in history.
    `initial_count` is the number of messages that were already in the
    working list before the loop added any (system + history length)."""
    for msg in loop_messages[initial_count:]:
        if msg.role == "assistant":
            transcript.append(
                session=session,
                channel=channel,
                speaker=character_name,
                role="assistant",
                content=_encode_assistant_with_tool_calls(msg.content, msg.tool_calls),
            )
        elif msg.role == "tool":
            transcript.append(
                session=session,
                channel=channel,
                speaker=msg.name or "tool",
                role="tool",
                content=msg.content,
            )


app = typer.Typer(add_completion=False, no_args_is_help=True)
eval_app = typer.Typer(help="Evaluations against the current character.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
memory_app = typer.Typer(help="Inspect and manage episodic memory.", no_args_is_help=True)
app.add_typer(memory_app, name="memory")
voice_app = typer.Typer(help="Voice suite — capture and manage samples.", no_args_is_help=True)
app.add_typer(voice_app, name="voice")
console = Console()


_EMBEDDER_SENTINEL: object = object()
_cached_embedder: object = _EMBEDDER_SENTINEL


def _load_embedder() -> object | None:
    """Lazy-import and memoize the default embedder. Returns None (with
    a warning) if the retrieval extra isn't installed.

    The result is cached process-wide — `cmd_chat` wires retriever +
    episodic store + semantic store from the same instance, so the 1.3
    GB embedder model loads once instead of three times."""
    global _cached_embedder
    if _cached_embedder is not _EMBEDDER_SENTINEL:
        return None if _cached_embedder is None else _cached_embedder
    try:
        from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    except ImportError:
        console.print(
            "[yellow]retrieval extra not installed. "
            "Run `uv sync --extra retrieval` to enable retrieval + memory.[/yellow]"
        )
        _cached_embedder = None
        return None
    _cached_embedder = SentenceTransformersEmbedder()
    return _cached_embedder


def _maybe_retriever(character: Character, top_k: int) -> VoiceRetriever | None:
    """Build a VoiceRetriever if retrieval is requested and the optional
    sentence-transformers dep is installed. Returns None to signal the
    caller to fall back to full-set few-shot."""
    if top_k <= 0:
        return None
    embedder = _load_embedder()
    if embedder is None:
        return None
    with Status(f"warming embedder ({embedder.id})…", console=console):  # type: ignore[attr-defined]
        return VoiceRetriever(embedder=embedder, character=character)  # type: ignore[arg-type]


def _open_episodic_store(character: Character, *, ingest: bool = True) -> EpisodicStore | None:
    """Open the episodic store, ingesting seeds on first run. Returns
    None if the retrieval extra isn't installed."""
    embedder = _load_embedder()
    if embedder is None:
        return None
    store = EpisodicStore(settings.db_path, embedder=embedder)  # type: ignore[arg-type]
    if ingest:
        inserted = ensure_seeds_ingested(character, store)
        if inserted > 0:
            console.print(f"[dim]seeded {inserted} episodic memories from character.[/dim]")
    return store


def _open_semantic_store() -> SemanticStore | None:
    embedder = _load_embedder()
    if embedder is None:
        return None
    return SemanticStore(settings.db_path, embedder=embedder)  # type: ignore[arg-type]


def _format_ctx_meter(used: int, total: int) -> str:
    """Render 'ctx 4.2k / 32k (13%)' with color thresholds: dim under
    75%, yellow 75-90%, red above 90%. Only shown when `total > 0`."""
    if total <= 0:
        return ""
    pct = used / total
    if pct >= 0.9:
        color = "red"
    elif pct >= 0.75:
        color = "yellow"
    else:
        color = "dim"
    return f"[{color}]ctx {used / 1000:.1f}k / {total / 1000:.0f}k ({pct * 100:.0f}%)[/{color}]"


def _pre_validate_write_call(call: ToolCall, workspace: Path) -> str | None:
    """Run cheap sanity checks on a write-tier tool call BEFORE asking
    the user to approve. Returns None when the call looks sane; returns
    a human-readable reason when we should refuse outright and redirect
    the model.

    Currently guards only the write_file(overwrite=True) shrink-clobber
    pattern (see harness-2tq) — the tool itself has the same check as
    a defense-in-depth, but catching here keeps the 'approve?' prompt
    out of the user's face for calls that would just fail anyway."""
    if call.name != "write_file":
        return None
    args = call.arguments
    if not args.get("overwrite"):
        return None
    rel = str(args.get("path", ""))
    new_content = args.get("content", "") or ""
    if not rel:
        return None
    try:
        target = (workspace / rel).resolve()
        target.relative_to(workspace.resolve())
    except (ValueError, OSError):
        # Path issues — let the tool itself surface the error.
        return None
    if not target.exists() or not target.is_file():
        return None
    try:
        existing_size = target.stat().st_size
    except OSError:
        return None
    if len(new_content) < existing_size // 2 and len(new_content) < 1024:
        return (
            f"new content is {len(new_content)} bytes but {rel} is "
            f"{existing_size} bytes — looks like an append disguised "
            f'as overwrite. Use edit_file(path={rel!r}, old_string="", '
            f"new_string=...) to append."
        )
    return None


def _describe_call(call: ToolCall, workspace: Path) -> str:
    """Render a one-line intent summary for the approve prompt, so the
    user doesn't have to read through a raw {args} dict to decide."""
    args = call.arguments
    name = call.name
    if name == "write_file":
        rel = str(args.get("path", ""))
        size = len(args.get("content", "") or "")
        overwrite = bool(args.get("overwrite"))
        target = (workspace / rel).resolve() if rel else None
        pre_existed = bool(target and target.exists())
        if pre_existed and overwrite:
            try:
                existing_size = target.stat().st_size if target else 0
            except OSError:
                existing_size = 0
            return f"overwrite {rel} ({existing_size}B → {size}B)"
        if pre_existed:
            return f"write {rel} ({size}B) — BLOCKED: already exists"
        return f"create {rel} ({size}B)"
    if name == "edit_file":
        rel = str(args.get("path", ""))
        old = args.get("old_string", "")
        new = args.get("new_string", "") or ""
        if not old:
            return f"append {len(new)}B to {rel}"
        replace_all = bool(args.get("replace_all"))
        scope = "all matches" if replace_all else "1 match"
        return f"edit {rel} ({scope}, -{len(old)}B / +{len(new)}B)"
    if name == "shell":
        cmd = str(args.get("cmd", ""))
        trimmed = cmd if len(cmd) <= 80 else cmd[:77] + "..."
        return f"run: {trimmed}"
    if name in ("remember_fact",):
        return f"{args.get('subject', '?')} {args.get('predicate', '?')} {args.get('object', '?')}"
    if name in ("remember_event",):
        title = str(args.get("title", ""))[:60]
        return f'record event: "{title}"'
    if name in ("scribe_session", "consolidate_memory"):
        return " ".join(f"{k}={v}" for k, v in args.items()) or "(no args)"
    # Fallback: the raw args dict.
    return str(args)


def _open_in_editor(initial_text: str) -> str | None:
    """Launch $EDITOR (fallback: vi) on a temp file pre-loaded with
    `initial_text`. Returns the edited text on successful exit, or None
    if the user quit without saving / left the file unchanged / the
    editor failed to launch."""
    import os
    import shutil
    import subprocess
    import tempfile

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    editor_bin = shutil.which(editor.split()[0])
    if editor_bin is None:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".md",
        delete=False,
    ) as tmp:
        tmp.write(initial_text)
        tmp_path = Path(tmp.name)
    try:
        # Split the editor env var so "code --wait" etc. still work.
        cmd = [*editor.split(), str(tmp_path)]
        try:
            subprocess.run(cmd, check=False)  # noqa: S603 — command comes from $EDITOR
        except OSError:
            return None
        edited = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)

    if edited.strip() == initial_text.strip():
        return None
    return edited


def _render_chat_header(
    *,
    console: Console,
    character_name: str,
    session: str,
    speaker: str,
    adapter_id: str,
    lora_path: str | None,
    persona: bool,
    top_k: int,
    retriever_active: bool,
    memories: int,
    memories_threshold: float,
    memories_active: bool,
    facts: int,
    facts_threshold: float,
    facts_active: bool,
    tools_enabled: bool,
    tool_set: str,
    tool_names: list[str],
    workspace_path: Path | None,
    rewrite_on_tools: bool,
    router_enabled: bool,
    router_repo: str | None,
    compact_at: float,
    compact_keep_recent: int,
    dev: bool,
) -> None:
    """Render the chat-session loading header as an aligned key-value grid.

    The top line is the character's name rendered as a pseudo-logo (a
    single-glyph mark today; a proper ASCII logo can slot in when it
    lands). Every flag that meaningfully changes behavior gets its own
    row so the user can see at a glance what's on: persona state,
    retrieval knobs, tool profile, workspace sandbox, compaction cap,
    dev-mode toggle. Missing / disabled features render as `off` in
    dim text, so the eye skips them."""
    from rich.table import Table

    # Header. One unicode glyph keeps enough room for a multi-line ASCII
    # logo later without needing to reflow the grid.
    console.print(f"\n[bold green]◈ {character_name}[/bold green]\n")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column()

    grid.add_row("session", f"[cyan]{session}[/cyan]  · speaker: [cyan]{speaker}[/cyan]")

    model_value = f"[bold]{adapter_id}[/bold]"
    if lora_path:
        model_value += f"  +lora: [dim]{lora_path}[/dim]"
    grid.add_row("model", model_value)

    grid.add_row("persona", "[green]on[/green]" if persona else "[dim]off[/dim]")

    retrieval_bits: list[str] = []
    if retriever_active and top_k > 0:
        retrieval_bits.append(f"voice×{top_k}")
    if memories_active and memories > 0:
        retrieval_bits.append(f"memories×{memories} [dim](≥{memories_threshold:.2f})[/dim]")
    if facts_active and facts > 0:
        retrieval_bits.append(f"facts×{facts} [dim](≥{facts_threshold:.2f})[/dim]")
    grid.add_row(
        "retrieval",
        " · ".join(retrieval_bits) if retrieval_bits else "[dim]off[/dim]",
    )

    if tools_enabled and tool_names:
        tools_summary = (
            f"[green]{tool_set}[/green] · {len(tool_names)} tools "
            f"[dim]({', '.join(tool_names[:6])}"
            + (f", …+{len(tool_names) - 6}" if len(tool_names) > 6 else "")
            + ")[/dim]"
        )
        grid.add_row("tools", tools_summary)
        if workspace_path is not None:
            try:
                ws_display = "~/" + str(workspace_path.relative_to(Path.home()))
            except ValueError:
                ws_display = str(workspace_path)
            grid.add_row("workspace", ws_display)
        if rewrite_on_tools:
            grid.add_row("rewrite-on-tools", "[green]on[/green]")
        if router_enabled and router_repo:
            # Strip the HF org prefix for a tighter display — full repo is in --help.
            short_repo = router_repo.rsplit("/", 1)[-1]
            grid.add_row("router", f"[green]on[/green] · [dim]{short_repo}[/dim]")
    else:
        grid.add_row("tools", "[dim]off[/dim]")

    if compact_at > 0:
        grid.add_row(
            "compact",
            f"{int(compact_at * 100)}% of window · keep {compact_keep_recent}",
        )
    else:
        grid.add_row("compact", "[dim]off[/dim]")

    if dev:
        grid.add_row("mode", "[yellow]dev[/yellow]")

    console.print(grid)
    console.print()


def _render_fact_block(facts: list[SemanticFact]) -> str:
    """Render retrieved semantic facts as a compact block for the system
    prompt. One line per fact — subject, predicate, object, confidence."""
    lines = ["Relevant facts I know:"]
    for f in facts:
        lines.append(f"- {f.subject} {f.predicate} {f.object} (conf={f.confidence:.2f})")
    return "\n".join(lines)


class _ThinkingSpinner:
    """Live 'thinking… Ns' spinner. Rich's Status animates the spinner
    glyph; a small daemon thread updates the elapsed-seconds suffix
    every 250 ms so the user sees the timer tick.

    Both start() and stop() are idempotent: start() is a no-op when
    already running, stop() a no-op when already stopped. This lets the
    chat loop kick the spinner on as soon as the user presses Enter
    (so the prompt isn't silent), have the tool-loop observer bounce it
    per model call, and stop it cleanly before any console.input() or
    final Markdown print — without any caller needing to track state."""

    def __init__(self, console: Console, label: str = "thinking") -> None:
        self._console = console
        self._label = label
        self._running = False
        self._status: Status | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_at = time.monotonic()
        self._status = self._console.status(
            f"[dim]⋯ {self._label}… 0s[/dim]",
            spinner="dots",
        )
        self._status.__enter__()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._status is not None:
            self._status.__exit__(None, None, None)
        self._status = None
        self._stop_event = None
        self._thread = None

    def _tick(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.wait(0.25):
            if self._status is None:
                return
            elapsed = int(time.monotonic() - self._started_at)
            self._status.update(f"[dim]⋯ {self._label}… {elapsed}s[/dim]")

    def __enter__(self) -> _ThinkingSpinner:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


# A sentence ends at .?! followed by whitespace / end-of-string, or at
# any newline. Requiring whitespace after the period means URLs like
# 'www.example.com/page' don't get split at the dot inside the host
# (regression from harness-q27 where that split hid the fabrication
# pattern from the suppression regexes).
_SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?][\s)\]'\"]+|\n)")


def _is_suppressible(text: str) -> bool:
    """True when `text` matches any of the stream-level filter rules
    (meta-confirm, false-success, fabricated search output, or bare
    tool-intent statements). Single entry point so the renderer's
    three call sites stay in lock-step as the rule set grows.

    Tool-intent statements ('I will search…', 'let me check…') get
    suppressed because either (a) the model actually calls the tool,
    in which case the tool-call status line replaces the preamble, or
    (b) the model doesn't call it, in which case the preamble is a
    misleading lead-in to fabricated output. Dropping in both cases
    is the right trade."""
    return bool(
        _META_CONFIRM_RE.search(text)
        or _FALSE_SUCCESS_RE.search(text)
        or _FABRICATED_SEARCH_RE.search(text)
        or _TOOL_INTENT_RE.search(text)
    )


class _StreamRenderer:
    """Plain-text streaming region for token deltas, with sentence-level
    meta-confirm / false-success suppression.

    Earlier versions wrapped a `rich.live.Live` around a re-rendered
    `Markdown` block. That repainted the full buffer at 10 Hz, and when
    the buffer exceeded terminal height Rich could not clear the prior
    frames — each tick leaked into scrollback as a growing-prefix
    duplicate. Streaming is now plain-text append, sentence-buffered:
    each completed sentence is checked against the meta-confirm /
    false-success regexes and dropped if it matches, so the user never
    sees 'Would you like me to…?' / 'has been added' narrative the
    orchestrator is about to strip anyway.

    Cost: sentence-level latency instead of token-level. The user sees
    one sentence appear at a time rather than token-by-token. Worth it
    to keep small-model noise off the screen. Guard against degenerate
    no-punctuation loops by force-flushing the buffer at _MAX_BUFFER
    chars — the user still sees something going wrong instead of a
    silent terminal that looks stuck."""

    # Cap on buffered characters before we force a flush. Long enough
    # to include a whole paragraph; short enough that a runaway
    # 'would you like me to would you like me to…' loop surfaces within
    # ~a second rather than piling up invisibly until max_tokens fires.
    _MAX_BUFFER = 400

    def __init__(self, console: Console, *, show_suppressions: bool = False) -> None:
        self._console = console
        # Dev flag — when True, print the '⋯ suppressed N line(s)…' footer
        # so you can see the filter working. Off by default: users don't
        # need to know about the model's self-inflicted noise.
        self._show_suppressions = show_suppressions
        self._visible = ""  # emitted to console
        self._pending = ""  # tokens not yet at a sentence boundary
        self._suppressed_count = 0
        self._active = False

    def start(self) -> None:
        self._visible = ""
        self._pending = ""
        self._suppressed_count = 0
        self._active = True

    def append(self, delta: str) -> None:
        if not self._active:
            self.start()
        self._pending += delta
        self._flush_complete_sentences()
        # No sentence boundary yet? Bail if the buffer is getting huge —
        # that's either a very long paragraph or a degenerate loop. Either
        # way the user wants tokens on screen, not silence.
        if len(self._pending) >= self._MAX_BUFFER:
            self._force_flush_pending()

    def _flush_complete_sentences(self) -> None:
        while True:
            match = _SENTENCE_BOUNDARY_RE.search(self._pending)
            if match is None:
                return
            end = match.end()
            sentence = self._pending[:end]
            self._pending = self._pending[end:]
            if _is_suppressible(sentence):
                # Drop — do not print. The orchestrator will feed a nudge
                # back to the model on the next round.
                self._suppressed_count += 1
                continue
            self._emit(sentence)

    def _force_flush_pending(self) -> None:
        """Emit the pending buffer even without a sentence boundary.
        Still runs the meta-confirm / false-success / fabrication
        regexes so a runaway hallucination gets dropped instead of
        spilling to screen; the counter tells the user something was
        suppressed."""
        if _is_suppressible(self._pending):
            self._suppressed_count += 1
        else:
            self._emit(self._pending)
        self._pending = ""

    def _emit(self, text: str) -> None:
        self._visible += text
        self._console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    def stop(self) -> str:
        # Flush any trailing partial sentence — the regex check still
        # runs so a model that trailed off mid-meta-confirm ("Would
        # you like me to") doesn't leak in the final chunk either.
        if self._pending:
            if _is_suppressible(self._pending):
                self._suppressed_count += 1
            else:
                self._emit(self._pending)
            self._pending = ""
        if self._active and self._suppressed_count > 0 and self._show_suppressions:
            self._console.print(
                f"[dim]⋯ suppressed {self._suppressed_count} line(s) of "
                f"meta-confirm / hallucinated-success narrative[/dim]"
            )
        out = self._visible
        show_trailing_newline = out or (self._suppressed_count > 0 and self._show_suppressions)
        if self._active and show_trailing_newline:
            self._console.print()
        self._visible = ""
        self._pending = ""
        self._suppressed_count = 0
        self._active = False
        return out

    @property
    def active(self) -> bool:
        return self._active


def _render_tool_event(
    event: ToolLoopEvent,
    *,
    console: Console,
    thinking: _ThinkingSpinner,
    stream_renderer: _StreamRenderer,
    tool_label: Callable[[str], str],
) -> None:
    """Render a single ToolLoopEvent to the console. Lifted out of the
    chat command closure so tests can capture the per-event output and
    verify that every tool call in a turn produces its own 🔧 line
    (harness-cx2 regression — the description suspected a 'first call
    only' guard; this makes absence-of-guard testable).

    Each event is independent: no dedup, no once-per-turn gating. A
    tool_call_start event always prints, a tool_call_end always prints
    a ✓/✗ line. The spinner + stream_renderer state-machine lives here
    because the renderer is the only thing that knows when the model
    is thinking vs. streaming vs. done."""
    if event.kind == "router_intent":
        call = event.call
        assert call is not None
        console.print(f"[dim magenta]→ routed to {call.name}[/dim magenta]")
    elif event.kind == "model_call_start":
        thinking.start()
    elif event.kind == "token_delta":
        # First token received — drop the spinner, open a Live region
        # (if not already) and append. Subsequent deltas just append.
        thinking.stop()
        if event.delta:
            stream_renderer.append(event.delta)
    elif event.kind == "model_call_end":
        thinking.stop()
        stream_renderer.stop()
    elif event.kind == "tool_call_start":
        call = event.call
        assert call is not None
        label = tool_label(call.name)
        console.print(f"[cyan]🔧 {label}[/cyan] [dim]({call.arguments})[/dim]")
    elif event.kind in ("tool_call_end", "tool_call_failed"):
        result = event.result
        assert result is not None
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        snippet = result.output[:120].replace("\n", " ")
        more = "…" if len(result.output) > 120 else ""
        console.print(f"   {status} [dim]{snippet}{more}[/dim]")
    elif event.kind == "tool_call_declined":
        console.print("   [yellow]✗ declined[/yellow]")
    elif event.kind == "tool_call_deduped":
        call = event.call
        assert call is not None
        label = tool_label(call.name)
        # One dim line noting the dedup — enough to show the user the
        # model tried to re-call the same tool, but not enough to
        # clutter the transcript. Result is the stock nudge; no need
        # to echo it.
        console.print(f"[dim]⇢ {label} {call.arguments} — duplicate call skipped[/dim]")


def _stream_or_complete(
    adapter: object,
    messages: list[ChatMessage],
    *,
    stream_renderer: _StreamRenderer,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> tuple[str, bool]:
    """Stream via `adapter.stream(...)` when available, otherwise fall
    back to the blocking `adapter.complete(...)`. Returns the reply
    text and whether streaming actually happened — the caller uses the
    streamed flag to skip a duplicate final Markdown print (Live
    already rendered the content)."""
    stream_fn = getattr(adapter, "stream", None)
    if callable(stream_fn):
        stream_renderer.start()
        for delta in stream_fn(messages, max_tokens=max_tokens, temperature=temperature):
            stream_renderer.append(delta)
        text = stream_renderer.stop()
        return text, True
    complete_fn = adapter.complete  # type: ignore[attr-defined]
    text = complete_fn(messages, max_tokens=max_tokens, temperature=temperature)
    assert isinstance(text, str)
    return text, False


@dataclass
class _RetrievalState:
    """Per-chat-session health of the three retrieval sources. Once a
    source raises we disable it for the rest of the session so the user
    doesn't get a warning on every turn. The chat still works — just
    without that source's prompt context."""

    voice_ok: bool = True
    episodic_ok: bool = True
    semantic_ok: bool = True


def _retrieve_turn_context(
    *,
    user_input: str,
    speaker: str,
    retriever: VoiceRetriever | None,
    memory_store: EpisodicStore | None,
    semantic_store: SemanticStore | None,
    top_k: int,
    memories: int,
    memories_threshold: float,
    facts: int,
    facts_threshold: float,
    state: _RetrievalState,
    warn: Callable[[str], None],
) -> tuple[list[VoiceSample], list[EpisodicRecord], list[SemanticFact]]:
    """Run the three retrieval sources for one turn. Any that raise are
    disabled for the rest of the session (flagged on `state`) and a
    one-time `warn(msg)` fires. Returns the hits from the sources that
    are still healthy — empty lists for the ones that aren't."""
    examples: list[VoiceSample] = []
    if retriever is not None and state.voice_ok and top_k > 0:
        try:
            examples = retriever.top_k(user_input, k=top_k)
        except Exception as exc:
            state.voice_ok = False
            warn(f"voice retrieval disabled for this session: {exc}")

    recalled: list[EpisodicRecord] = []
    if memory_store is not None and state.episodic_ok and memories > 0:
        try:
            hits = memory_store.search(
                user_input,
                k=memories,
                min_score=memories_threshold,
                user_id=speaker,
            )
            recalled = [rec for rec, _score in hits]
        except Exception as exc:
            state.episodic_ok = False
            warn(f"episodic memory disabled for this session: {exc}")

    known_facts: list[SemanticFact] = []
    if semantic_store is not None and state.semantic_ok and facts > 0:
        try:
            fact_hits = semantic_store.search(
                user_input,
                k=facts,
                min_score=facts_threshold,
                user_id=speaker,
            )
            known_facts = [f for f, _score in fact_hits]
        except Exception as exc:
            state.semantic_ok = False
            warn(f"semantic facts disabled for this session: {exc}")

    return examples, recalled, known_facts


def _render_memory_block(memories: list[EpisodicRecord]) -> str:
    """Render retrieved memories as a section of the system prompt. One
    block per memory, title as heading, principle italicized, body as
    prose. Kept close to the on-disk seed format so the model sees
    familiar shape."""
    lines = ["Relevant past experience — things I remember from before:"]
    for m in memories:
        lines.append("")
        lines.append(f"## {m.title}")
        if m.principle:
            lines.append(f"*Lesson: {m.principle}*")
        lines.append("")
        lines.append(m.body)
    return "\n".join(lines)


def _resolve_adapter(
    name: str,
    *,
    persona: bool = False,
    character: Character | None = None,
    model_repo: str | None = None,
    lora_path: str | None = None,
) -> ModelAdapter:
    # Custom configs bypass the factory and instantiate the adapter
    # directly. --lora-path is MLX-only; --model-repo works for MLX
    # (HF repo) and Ollama (model tag like "gemma4:latest").
    if lora_path and name != "mlx":
        raise typer.BadParameter("--lora-path requires --model mlx.")

    adapter: ModelAdapter
    if model_repo or lora_path:
        if name == "mlx":
            from harness.model.mlx import MLXAdapter

            mlx_kwargs: dict[str, object] = {}
            if model_repo:
                mlx_kwargs["repo"] = model_repo
            if lora_path:
                mlx_kwargs["adapter_path"] = lora_path
            adapter = MLXAdapter(**mlx_kwargs)  # type: ignore[arg-type]
        elif name == "ollama":
            from harness.model.ollama import OllamaAdapter

            adapter = OllamaAdapter(model=model_repo) if model_repo else OllamaAdapter()
        else:
            raise typer.BadParameter(
                f"--model-repo not supported for --model {name}; use mlx or ollama."
            )
    else:
        try:
            adapter = make_adapter(cast(AdapterName, name))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if persona:
        if character is None:
            raise typer.BadParameter("persona=True requires a character")
        adapter = PersonaAdapter(adapter, character)

    # Honor an optional eager `.load()` method without making it part of
    # the ModelAdapter Protocol — only some adapters need it.
    loader = getattr(adapter, "load", None)
    if callable(loader):
        with Status(f"loading {adapter.id}…", console=console):
            loader()
    return adapter


@app.command()
def chat(
    session: str = typer.Option("local", help="Session identifier"),
    channel: str = typer.Option("cli", help="Channel name"),
    speaker: str = typer.Option("mark", help="Your handle"),
    model: str = typer.Option("echo", help="Adapter: echo | mlx | ollama"),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help="Override the model identifier for the selected adapter. "
        "For mlx: HF repo (default mlx-community/Qwen2.5-7B-Instruct-4bit). "
        "For ollama: model tag (default gemma4:latest). Ignored for echo.",
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="Path to a DIRECTORY produced by `mlx_lm.lora` training (contains "
        "adapter_config.json plus weight files). Applied on top of the base MLX "
        "model. Requires --model mlx.",
    ),
    persona: bool = typer.Option(
        False,
        "--persona/--no-persona",
        help="Wrap the model with a voice-rewrite post-pass (Airton's register).",
    ),
    top_k: int = typer.Option(
        6,
        help="Retrieve top-K voice samples by similarity to the user message "
        "(default 6). Set 0 to show every sample.",
    ),
    memories: int = typer.Option(
        3,
        "--memories",
        help="Max episodic memories to retrieve per user turn (default 3). "
        "Set 0 to disable memory retrieval.",
    ),
    memories_threshold: float = typer.Option(
        0.5,
        "--memories-threshold",
        help="Cosine-similarity floor for memory retrieval. Memories below "
        "this are dropped even if there are fewer than --memories of them. "
        "Prevents irrelevant memories from polluting the prompt.",
    ),
    facts: int = typer.Option(
        5,
        "--facts",
        help="Max semantic facts to retrieve per user turn (default 5). "
        "Set 0 to disable fact retrieval.",
    ),
    facts_threshold: float = typer.Option(
        0.45,
        "--facts-threshold",
        help="Cosine-similarity floor for fact retrieval. Lower than the "
        "memory floor because facts are much shorter strings and score lower.",
    ),
    tools: bool = typer.Option(
        False,
        "--tools/--no-tools",
        help="Enable tool use. Which tools are registered depends on "
        "--tool-set (default: 'core'). Write-tier tools prompt for "
        "confirmation the first time they're called each session.",
    ),
    tool_set: str = typer.Option(
        DEFAULT_PROFILE,
        "--tool-set",
        help=(
            f"Named profile of tools to enable with --tools. One of: "
            f"{sorted(TOOL_PROFILES)}. Schema cost targets kept under "
            f"~1500 tokens per profile."
        ),
    ),
    tools_add: str | None = typer.Option(
        None,
        "--tools-add",
        help="Comma-separated tool names to add on top of the --tool-set.",
    ),
    tools_drop: str | None = typer.Option(
        None,
        "--tools-drop",
        help="Comma-separated tool names to drop from the --tool-set.",
    ),
    rewrite_on_tools: bool = typer.Option(
        False,
        "--rewrite-on-tools/--no-rewrite-on-tools",
        help="When tools ran in a turn, also apply the persona rewriter to the "
        "final reply. Off by default — the rewriter is trained to compress, "
        "which is wrong for summarize / investigate tasks that need prose. Turn "
        "on for casual tooled chat where you want Airton-voice on every reply.",
    ),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Directory read_file / write_file / shell operate inside. "
        "Default: the harness repo root. Only takes effect with --tools. "
        "Memory and transcripts still live under the harness data dir.",
    ),
    compact_at: float = typer.Option(
        0.8,
        "--compact-at",
        help="Fraction of the context window at which to auto-summarize "
        "older turns (0 to disable). When the context meter crosses this, "
        "every turn older than --compact-keep-recent is folded into a "
        "single session summary. The transcript is unchanged — only the "
        "prompt the model sees shrinks.",
    ),
    compact_keep_recent: int = typer.Option(
        10,
        "--compact-keep-recent",
        help="Number of most-recent turns to leave verbatim when "
        "compaction fires. Older turns become summary.",
    ),
    dev: bool = typer.Option(
        False,
        "--dev/--no-dev",
        help="Dev mode — surface internal signals like the stream filter's "
        "'⋯ suppressed N line(s)…' markers. Off by default so users don't "
        "see the model's self-inflicted noise; on for developers tuning "
        "the filter or debugging small-model behavior.",
    ),
    router_enabled: bool = typer.Option(
        False,
        "--router/--no-router",
        help="Front the tool loop with a small intent-router model. When "
        "it classifies the user turn into a known read-tier tool with "
        "valid args, the orchestrator executes the tool itself and the "
        "main model only does a wrap-up round — no fabricate-and-nudge "
        "rounds. Advisory: unparseable / null / write-tier intents fall "
        "through to the normal loop. See harness-ut3.",
    ),
    router_repo: str = typer.Option(
        "mlx-community/Hermes-3-Llama-3.2-3B-4bit",
        "--router-repo",
        help="HF repo for the router model. Default is "
        "Hermes-3-Llama-3.2-3B-4bit (~2GB RAM, function-call-tuned). "
        "Only used when --router is on. Router is MLX-only for now.",
    ),
    router_mode: str = typer.Option(
        "free",
        "--router-mode",
        help="Routing strategy. 'free' = free-form JSON + tolerant parse "
        "(current behavior). 'grammar' = JSON-schema-constrained decoding "
        "that guarantees valid output + valid tool name by construction "
        "(requires the `grammar` extra, adds ~1GB RAM for outlines' FSM "
        "machinery).",
    ),
    tui: bool = typer.Option(
        False,
        "--tui/--no-tui",
        help="Launch the Textual chat app instead of the classic REPL. "
        "Persistent input at the bottom, scrolling output above, live "
        "ctx + elapsed metrics. Phase 1 is a scaffold (echo only); "
        "model wiring lands in harness-29c. Requires the `tui` extra: "
        "uv sync --extra tui.",
    ),
) -> None:
    """CLI chat loop. Swap model runtimes with --model."""
    if tui:
        # Phase 2: adapter + persona + retrieval wired. Tools,
        # router, compaction, and voice-capture land in later
        # phases — reject flags the TUI can't honor yet so the
        # user isn't surprised when they're silently ignored.
        if tools:
            raise typer.BadParameter(
                "--tui does not support --tools yet (lands in harness-1r4). "
                "Drop --tools or use the classic REPL for now."
            )
        if router_enabled:
            raise typer.BadParameter(
                "--tui does not support --router yet (lands in harness-1r4). "
                "Drop --router or use the classic REPL for now."
            )
        try:
            from harness.tui import ChatApp
        except ImportError as exc:
            raise typer.BadParameter(
                "--tui requires the `tui` extra. Install it with: uv sync --extra tui"
            ) from exc
        character_for_tui = load_character(settings.character_path)
        tui_adapter = _resolve_adapter(
            model,
            persona=persona,
            character=character_for_tui,
            model_repo=model_repo,
            lora_path=lora_path,
        )
        tui_retriever = _maybe_retriever(character_for_tui, top_k)
        tui_memory_store = _open_episodic_store(character_for_tui) if memories > 0 else None
        tui_semantic_store = _open_semantic_store() if facts > 0 else None
        tui_transcript = Transcript(settings.db_path)
        ChatApp(
            character=character_for_tui,
            speaker=speaker,
            session=session,
            channel=channel,
            adapter=tui_adapter,
            transcript=tui_transcript,
            retriever=tui_retriever,
            top_k=top_k,
            memory_store=tui_memory_store,
            memories=memories,
            memories_threshold=memories_threshold,
            semantic_store=tui_semantic_store,
            facts=facts,
            facts_threshold=facts_threshold,
        ).run()
        return

    character = load_character(settings.character_path)
    workspace_path = Path(workspace).expanduser().resolve() if workspace else settings.root
    if tools and not workspace_path.is_dir():
        raise typer.BadParameter(f"workspace {workspace_path} is not a directory")
    # When tools are active we run persona manually *after* the tool loop,
    # so we resolve the base adapter unwrapped. Without tools, persona
    # wraps the base adapter as before.
    adapter = _resolve_adapter(
        model,
        persona=persona and not tools,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
    )
    router: Router | None = None
    if router_enabled:
        if not tools:
            raise typer.BadParameter("--router requires --tools (nothing to route to otherwise).")
        if router_mode not in {"free", "grammar"}:
            raise typer.BadParameter(
                f"--router-mode must be 'free' or 'grammar' (got {router_mode!r})."
            )
        from harness.model.mlx import MLXAdapter

        router_adapter = MLXAdapter(repo=router_repo)
        router = (
            GrammarRouter(adapter=router_adapter)
            if router_mode == "grammar"
            else ModelRouter(adapter=router_adapter)
        )
    retriever = _maybe_retriever(character, top_k)
    memory_store = _open_episodic_store(character) if memories > 0 else None
    semantic_store = _open_semantic_store() if facts > 0 else None
    transcript = Transcript(settings.db_path)
    compaction_store = CompactionStore(settings.db_path) if compact_at > 0 else None

    registry: ToolRegistry | None = None
    approved_tools: set[str] = set()
    if tools:
        try:
            wanted_names = resolve_tool_names(
                tool_set,
                add=tuple((tools_add or "").split(",")),
                drop=tuple((tools_drop or "").split(",")),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        # Map names → builders. Memory tools return None when their store
        # isn't available (--memories 0 / --facts 0). Unknown names fall
        # through to the warning path so future-tool profiles stay loadable.
        builders: dict[str, Callable[[], Tool | None]] = {
            "read_file": lambda: ReadFileTool(root=workspace_path),
            "edit_file": lambda: EditFileTool(root=workspace_path),
            "write_file": lambda: WriteFileTool(root=workspace_path),
            "shell": lambda: ShellTool(cwd=workspace_path),
            "list_dir": lambda: ListDirTool(root=workspace_path),
            "grep": lambda: GrepTool(root=workspace_path),
            "glob": lambda: GlobTool(root=workspace_path),
            "git_status": lambda: GitStatusTool(root=workspace_path),
            "git_diff": lambda: GitDiffTool(root=workspace_path),
            "git_log": lambda: GitLogTool(root=workspace_path),
            "search_memory": (
                lambda: (
                    SearchMemoryTool(store=memory_store, user_id=speaker)
                    if memory_store is not None
                    else None
                )
            ),
            "search_facts": (
                lambda: (
                    SearchFactsTool(store=semantic_store, user_id=speaker)
                    if semantic_store is not None
                    else None
                )
            ),
            "search_web": lambda: SearchWebTool(),
            "remember_fact": (
                lambda: (
                    RememberFactTool(store=semantic_store, user_id=speaker, session_id=session)
                    if semantic_store is not None
                    else None
                )
            ),
            "remember_event": (
                lambda: (
                    RememberEventTool(store=memory_store, user_id=speaker, session_id=session)
                    if memory_store is not None
                    else None
                )
            ),
            "scribe_session": (
                lambda: (
                    ScribeSessionTool(
                        adapter=adapter,
                        character=character,
                        transcript=transcript,
                        episodic_store=memory_store,
                        semantic_store=semantic_store,
                        default_user_id=speaker,
                    )
                    if memory_store is not None and semantic_store is not None
                    else None
                )
            ),
            "consolidate_memory": (
                lambda: (
                    ConsolidateMemoryTool(
                        episodic_store=memory_store,
                        semantic_store=semantic_store,
                    )
                    if memory_store is not None and semantic_store is not None
                    else None
                )
            ),
        }

        registry = ToolRegistry()
        for name in wanted_names:
            builder = builders.get(name)
            if builder is None:
                console.print(f"[yellow]⚠ tool {name!r} not yet implemented — skipping[/yellow]")
                continue
            tool = builder()
            if tool is None:
                console.print(
                    f"[yellow]⚠ tool {name!r} needs a store that isn't enabled "
                    f"(check --memories / --facts)[/yellow]"
                )
                continue
            registry.register(tool)

        if not registry.names():
            registry = None  # empty profile → same as --no-tools

    retrieval_state = _RetrievalState()
    thinking = _ThinkingSpinner(console)
    stream_renderer = _StreamRenderer(console, show_suppressions=dev)

    def _warn_once(msg: str) -> None:
        console.print(f"[yellow]⚠ {msg}[/yellow]")

    def _tool_label(name: str) -> str:
        if registry is not None and name in registry:
            return registry.get(name).spec.label
        return name

    def confirm_write_tool(call: ToolCall) -> bool:
        # Pre-validate: some calls are so obviously wrong that we refuse
        # them without even asking the user. The tool's own call() has
        # the same guard as a safety net, but catching it here keeps the
        # approve prompt out of the user's face for doomed calls.
        refusal = _pre_validate_write_call(call, workspace_path)
        if refusal is not None:
            console.print(f"[red]🚫 refusing {call.name}: {refusal}[/red]")
            return False
        if call.name in approved_tools:
            return True
        label = _tool_label(call.name)
        summary = _describe_call(call, workspace_path)
        console.print(f"[yellow]🔧 Airton wants to [bold]{label}[/bold] — {summary}[/yellow]")
        answer = console.input("   approve? [y/N/always]: ").strip().lower()
        if answer == "always":
            approved_tools.add(call.name)
            return True
        return answer.startswith("y")

    def render_tool_event(event: ToolLoopEvent) -> None:
        _render_tool_event(
            event,
            console=console,
            thinking=thinking,
            stream_renderer=stream_renderer,
            tool_label=_tool_label,
        )

    _render_chat_header(
        console=console,
        character_name=character.name,
        session=session,
        speaker=speaker,
        adapter_id=adapter.id,
        lora_path=lora_path,
        persona=persona,
        top_k=top_k,
        retriever_active=retriever is not None,
        memories=memories,
        memories_threshold=memories_threshold,
        memories_active=memory_store is not None,
        facts=facts,
        facts_threshold=facts_threshold,
        facts_active=semantic_store is not None,
        tools_enabled=registry is not None,
        tool_set=tool_set,
        tool_names=registry.names() if registry is not None else [],
        workspace_path=workspace_path if registry is not None else None,
        rewrite_on_tools=rewrite_on_tools,
        router_enabled=router is not None,
        router_repo=router_repo if router is not None else None,
        compact_at=compact_at,
        compact_keep_recent=compact_keep_recent,
        dev=dev,
    )
    console.print(
        "[dim](ctrl-c, /exit, /quit, or :q to exit · "
        "/edit to capture a corrected reply as a voice sample)[/dim]\n"
    )

    def _load_history() -> tuple[ChatMessage | None, list[ChatMessage]]:
        """Return (optional summary-system-message, turns-since-pointer).
        When a compaction summary exists, turns before the pointer are
        represented by the summary only; the raw rows stay in the
        transcript for audit but never hit the model."""
        record = compaction_store.latest_for_session(session) if compaction_store else None
        if record is not None:
            summary_msg = ChatMessage(
                role="system",
                content=(
                    "Earlier conversation in this session (summarized; "
                    f"{record.covered_turns} turns folded in):\n\n{record.summary}"
                ),
            )
            rows = transcript.fetch_after(session, after_id=record.up_to_turn_id)
            return summary_msg, [_decode_transcript_message(m) for m in rows]
        rows = transcript.tail(session, limit=50)
        return None, [_decode_transcript_message(m) for m in rows]

    def _measure_ctx() -> int:
        """Estimate tokens for what the NEXT turn will start with:
        character.system_prompt() (cheap fallback — no retrieval yet),
        plus any compaction summary, plus history since the pointer.
        Undercounts slightly because retrieved memories/facts add text
        per turn, but tracks transcript growth accurately."""
        baseline_system = ChatMessage(role="system", content=character.system_prompt())
        summary_msg, history_msgs = _load_history()
        msgs: list[ChatMessage] = [baseline_system]
        if summary_msg is not None:
            msgs.append(summary_msg)
        msgs.extend(history_msgs)
        return count_tokens(adapter, msgs)

    def _print_ctx_meter() -> None:
        used = _measure_ctx()
        meter = _format_ctx_meter(used, adapter.context_window)
        if meter:
            console.print(meter)

    def _maybe_compact() -> None:
        if compaction_store is None:
            return
        used = _measure_ctx()
        if not should_compact(
            used_tokens=used,
            context_window=adapter.context_window,
            threshold_pct=compact_at,
        ):
            return
        console.print(
            f"[dim]compacting history (ctx {used / 1000:.1f}k, threshold "
            f"{compact_at * 100:.0f}%)…[/dim]"
        )
        thinking.start()
        try:
            outcome: CompactionOutcome = run_compaction(
                adapter,
                transcript,
                compaction_store,
                session_id=session,
                keep_recent=compact_keep_recent,
            )
        finally:
            thinking.stop()
        if outcome.wrote:
            console.print(
                f"[dim]compacted {outcome.covered_turns} turns "
                f"(pointer → #{outcome.new_up_to_turn_id})[/dim]"
            )
        else:
            console.print(
                "[yellow]compaction skipped — nothing qualified "
                "(fewer turns than keep-recent, or model returned empty).[/yellow]"
            )

    try:
        while True:
            _maybe_compact()
            _print_ctx_meter()
            user_input = console.input("[bold cyan]you › [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in _EXIT_COMMANDS:
                break
            if user_input.lower() in _EDIT_COMMANDS:
                # Slash command: open $EDITOR on Airton's last reply.
                # Saving writes a new captured voice sample paired with
                # the preceding user prompt. Closes the loop between
                # 'reply was off-register' and 'new training sample'
                # without leaving chat.
                history_tail = transcript.tail(session, limit=50)
                user_turns = [m for m in history_tail if m.role == "user"]
                assistant_turns = [m for m in history_tail if m.role == "assistant"]
                if not user_turns or not assistant_turns:
                    console.print(
                        "[yellow]no exchange to capture yet — have a turn "
                        "first, then run /edit.[/yellow]"
                    )
                    continue
                prev_prompt = user_turns[-1].content
                prev_reply = assistant_turns[-1].content
                edited = _open_in_editor(prev_reply)
                if edited is None:
                    console.print("[dim](no changes — nothing captured)[/dim]")
                    continue
                captured_path, sample_id, total = _write_voice_capture(
                    prompt=prev_prompt,
                    gold=edited,
                    session=session,
                    original=prev_reply,
                )
                console.print(
                    f"[green]captured[/green] id={sample_id!r} "
                    f"→ {captured_path.relative_to(settings.root)} "
                    f"(now {total} captured sample(s))"
                )
                continue
            transcript.append(
                session=session,
                channel=channel,
                speaker=speaker,
                role="user",
                content=user_input,
            )
            # Start the spinner immediately so the user sees acknowledgement
            # of their submission, not a blank cursor, while retrieval warms
            # up and the model runs. The tool-loop observer drops/restarts it
            # as needed across rounds; we stop it unconditionally before any
            # interactive prompt or the final reply render.
            thinking.start()

            examples, recalled, known_facts = _retrieve_turn_context(
                user_input=user_input,
                speaker=speaker,
                retriever=retriever,
                memory_store=memory_store,
                semantic_store=semantic_store,
                top_k=top_k,
                memories=memories,
                memories_threshold=memories_threshold,
                facts=facts,
                facts_threshold=facts_threshold,
                state=retrieval_state,
                warn=_warn_once,
            )

            if examples:
                system_content = character.system_prompt(include_samples=examples)
            else:
                system_content = character.system_prompt()

            if recalled:
                system_content = f"{system_content}\n\n{_render_memory_block(recalled)}"

            if registry is not None:
                tool_names = ", ".join(registry.names())
                system_content = (
                    f"{system_content}\n\n"
                    f"Workspace grounding — you are a real process on Mark's Mac. "
                    f"The tool sandbox root is `{workspace_path}`. Available tools: "
                    f"{tool_names}. Paths passed to `read_file` / `write_file` / "
                    f"`edit_file` are relative to the sandbox root; `shell` runs "
                    f"with it as cwd.\n\n"
                    "TOOL-USE RULES (follow these EVERY turn):\n"
                    "- The user's request IS the instruction. Act on it immediately.\n"
                    "- FORBIDDEN PHRASES — never emit any of these in your reply:\n"
                    '    • "Would you like to / Would you like me to"\n'
                    '    • "Should I proceed / Shall I / Do you want me to"\n'
                    '    • "Please confirm / Let\'s confirm / confirm your approval"\n'
                    '    • "we need to make sure the user confirms"\n'
                    "  If you catch yourself typing any of these, STOP — delete "
                    "the sentence and call the tool instead. The tool layer runs "
                    "its own approve/decline UX for write-tier tools; chat-level "
                    "meta-confirm just wastes the user's time.\n"
                    "- NEVER claim you did something (added/updated/created/wrote/"
                    "edited/appended a file, ran a command, etc.) unless you actually "
                    "called the corresponding write-tier tool on this turn AND the "
                    "tool's result message says it succeeded. If you don't have a "
                    "tool for the action the user asked for, say so plainly.\n"
                    "- ADDING a line or block to an existing file (e.g. 'add scratch "
                    "to .gitignore', 'append an import', 'add this to the config') → "
                    "use `edit_file` with an EMPTY `old_string` and `new_string` = the "
                    "text to append. Example: "
                    'edit_file(path=".gitignore", old_string="", new_string="scratch\\n"). '
                    "Do NOT use `write_file` for this — `write_file` replaces the "
                    "ENTIRE file and will destroy the existing content.\n"
                    "- CHANGING an existing line → `edit_file(path=..., old_string=..., "
                    "new_string=...)` with enough context in `old_string` to make it "
                    "unique.\n"
                    "- CREATING a brand-new file → `write_file(path=..., content=...)`. "
                    "Only use `overwrite=true` when the user explicitly asked you to "
                    "regenerate the file from scratch.\n"
                    "- Never describe the contents of the workspace from memory. If "
                    "the user asks what's in a directory, what a file contains, or "
                    "what this project does, you MUST call a tool first (`list_dir`, "
                    "`read_file`, `grep`) and base your answer on the tool's output.\n"
                    "- NEVER fabricate tool output. If the user asks you to search "
                    "the web, fetch a URL, read a file, or look up a fact in memory, "
                    "you MUST call the corresponding tool first. Do NOT invent "
                    "URLs, titles, snippets, file contents, or search results — "
                    "placeholder domains (example.com, your-site.com, localhost) "
                    "are forbidden. If the right tool isn't available this turn, "
                    "say so plainly.\n"
                    "- After tool results come back, respond with a substantive "
                    "reply that uses them. Never return an empty reply — the user "
                    "is waiting for your conclusion, not just the tool output.\n"
                    "- The user CANNOT see raw tool output — only your final reply. "
                    "Restate the key findings (names, numbers, quoted lines) in your "
                    "reply. Do not answer with meta-phrases like 'awaiting input' or "
                    "'the content is available'."
                )

            if known_facts:
                system_content = f"{system_content}\n\n{_render_fact_block(known_facts)}"

            system = ChatMessage(role="system", content=system_content)

            summary_msg, history = _load_history()
            history_messages: list[ChatMessage] = []
            if summary_msg is not None:
                history_messages.append(summary_msg)
            history_messages.extend(history)

            console.print(f"[bold green]{character.name} ›[/bold green]")
            streamed = False
            if registry is not None:
                # Tools active: drive the tool loop (observer handles live
                # rendering per model call), then optionally apply the
                # voice rewriter to the final text.
                initial_messages: list[ChatMessage] = [system, *history_messages]
                loop_result = run_tool_loop(
                    adapter,  # type: ignore[arg-type]
                    initial_messages,
                    registry,
                    confirm=confirm_write_tool,
                    observe=render_tool_event,
                    router=router,
                )
                streamed = True
                # Persist the tool exchange (assistant tool-call turns +
                # tool-role result turns) so the next user turn can see
                # what was read / run. Without this, every turn is amnesia.
                _persist_tool_exchange(
                    transcript,
                    session=session,
                    channel=channel,
                    character_name=character.name,
                    initial_count=len(initial_messages),
                    loop_messages=loop_result.messages,
                )
                draft = loop_result.content
                # Small models (gemma4 8B) sometimes bail after a tool
                # result — empty content AND no further tool calls. Nudge
                # them once with an explicit follow-up asking for the
                # final answer before falling back to the sentinel.
                if not draft.strip():
                    nudge_msgs = [
                        *loop_result.messages,
                        ChatMessage(
                            role="user",
                            content=(
                                "Your last reply was empty. Give me a final answer "
                                "based on what the tools already returned. Restate "
                                "the key findings in prose. Do not return empty."
                            ),
                        ),
                    ]
                    retry = adapter.complete_with_tools(  # type: ignore[attr-defined]
                        nudge_msgs,
                        tools=registry.specs(),
                        max_tokens=2048,
                        temperature=0.3,
                    )
                    if retry.content.strip():
                        draft = retry.content
                # Skip the rewriter when (a) rewrite-on-tools is off (default)
                # because the rewriter compresses prose that summarize /
                # investigate tasks need, or (b) the tool loop left no
                # substantive draft — otherwise the rewriter sees an empty
                # "Draft:" block and hallucinates "paste the text."
                if persona and rewrite_on_tools and draft.strip():
                    console.print("\n[dim]*— voice pass —*[/dim]")
                    rewrite_msgs = build_rewriter_messages(character, draft)
                    reply, _ = _stream_or_complete(
                        adapter,
                        rewrite_msgs,
                        stream_renderer=stream_renderer,
                        temperature=0.2,
                        max_tokens=2048,
                    )
                else:
                    reply = draft or "(no reply — model returned empty text after tool calls)"
            else:
                reply, streamed = _stream_or_complete(
                    adapter,
                    [system, *history_messages],
                    stream_renderer=stream_renderer,
                )

            thinking.stop()
            stream_renderer.stop()
            transcript.append(
                session=session,
                channel=channel,
                speaker=character.name,
                role="assistant",
                content=reply,
            )
            if not streamed:
                console.print(Markdown(reply))
            console.print()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]bye.[/dim]")
    finally:
        # Order matters: stop anything that could still be painting the
        # terminal first (spinner, stream) so a later exception doesn't
        # leave a live region hanging. Store closes last — they're
        # idempotent and safe under exceptions.
        thinking.stop()
        if stream_renderer.active:
            stream_renderer.stop()
        transcript.close()
        if compaction_store is not None:
            compaction_store.close()
        if memory_store is not None:
            memory_store.close()
        if semantic_store is not None:
            semantic_store.close()
        # MLX and sentence-transformers hold their weights in Python
        # attributes; normal GC releases them on process exit. No
        # explicit unload call is needed and mlx_lm doesn't expose one.


@app.command()
def describe() -> None:
    """Print Airton's resolved character sheet (sanity check)."""
    character = load_character(settings.character_path)
    console.print(f"[bold]{character.name}[/bold] — {character.premise}\n")
    console.print("[bold]Values[/bold]")
    for v in character.values:
        console.print(f"  • {v.rule}")
    console.print("\n[bold]Taboos[/bold]")
    for t in character.taboos:
        console.print(f"  • {t}")
    console.print("\n[bold]Seed memories[/bold]")
    for s in character.seed_memories:
        console.print(f"  • {s.title} — {s.principle}")
    console.print("\n[bold]Voice samples[/bold]")
    for sample in character.voice_samples:
        console.print(f"  • {sample.id}")


@eval_app.command("voice")
def eval_voice(
    model: str = typer.Option("mlx", help="Adapter: echo | mlx | ollama"),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help="Override the model id. MLX: HF repo. Ollama: model tag.",
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="LoRA adapter directory (from `mlx_lm.lora` training). Requires --model mlx.",
    ),
    sample: list[str] | None = typer.Option(
        None, "--sample", help="Limit to a specific sample id (repeatable)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    temperature: float = typer.Option(0.5, help="Sampling temperature"),
    leave_one_out: bool = typer.Option(
        True,
        "--leave-one-out/--no-leave-one-out",
        help="Exclude each sample from its own few-shot examples (default on). "
        "Disable to measure the ceiling with the full example set in view.",
    ),
    persona: bool = typer.Option(
        False,
        "--persona/--no-persona",
        help="Run the voice-rewrite post-pass after the substance pass.",
    ),
    top_k: int = typer.Option(
        6,
        help="Retrieve top-K voice samples by similarity to each prompt "
        "(default 6). Set 0 to show every sample (Phase 1a.2 baseline).",
    ),
    chain_rewrites: bool = typer.Option(
        False,
        "--chain-rewrites/--no-chain-rewrites",
        help="Add a second 'concrete substitution' rewrite pass on top of the "
        "style pass. Requires --persona.",
    ),
    use_judge: bool = typer.Option(
        False,
        "--judge/--no-judge",
        help="After heuristic scoring, ask the adapter to rate each response "
        "1-10 against gold. Circular (same model) but catches register drift "
        "the regex scorer misses.",
    ),
) -> None:
    """Run the canonical voice prompts and show model-vs-gold side by side."""
    character = load_character(settings.character_path)
    # eval runs persona inline in run_voice_eval so both passes stay
    # leave-one-out-consistent — do not wrap adapter here.
    adapter = _resolve_adapter(model, model_repo=model_repo, lora_path=lora_path)
    retriever = _maybe_retriever(character, top_k)

    results = run_voice_eval(
        character,
        adapter,
        temperature=temperature,
        sample_ids=sample if sample else None,
        leave_one_out=leave_one_out,
        persona=persona,
        retriever=retriever,
        top_k=top_k,
        use_judge=use_judge,
        chain_rewrites=chain_rewrites,
    )

    if as_json:
        payload = [
            {
                "sample_id": r.sample_id,
                "prompt": r.prompt,
                "gold": r.gold,
                "actual": r.actual,
                "score": {
                    "aggregate": r.score.aggregate,
                    "length_match": r.score.length_match,
                    "no_banned_openers": r.score.no_banned_openers,
                    "bullet_discipline": r.score.bullet_discipline,
                    "bullet_density": r.score.bullet_density,
                    "filler_discipline": r.score.filler_discipline,
                    "judge_score": r.score.judge_score,
                    "notes": list(r.score.notes),
                },
                **({"draft": r.draft} if r.draft is not None else {}),
            }
            for r in results
        ]
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"Voice eval — {adapter.id}", show_lines=True)
    table.add_column("id", style="bold")
    table.add_column("prompt")
    table.add_column("gold", style="green")
    table.add_column("actual", style="yellow")
    table.add_column("score", style="cyan")
    for r in results:
        judge_line = f"\njudge={r.score.judge_score}/10" if r.score.judge_score is not None else ""
        score_cell = (
            f"{r.score.aggregate:.2f}\n"
            f"len={r.score.length_match:.2f}\n"
            f"open={r.score.no_banned_openers:.0f}\n"
            f"bul={r.score.bullet_discipline:.1f}\n"
            f"den={r.score.bullet_density:.2f}\n"
            f"fil={r.score.filler_discipline:.2f}"
            f"{judge_line}"
        )
        table.add_row(r.sample_id, r.prompt, r.gold.strip(), r.actual.strip(), score_cell)
    aggregate = sum(r.score.aggregate for r in results) / max(len(results), 1)
    console.print(table)
    console.print(
        f"[bold]aggregate voice score:[/bold] {aggregate:.3f} across {len(results)} sample(s)"
    )
    judge_scores = [r.score.judge_score for r in results if r.score.judge_score is not None]
    if judge_scores:
        judge_mean = sum(judge_scores) / len(judge_scores)
        console.print(
            f"[bold]judge mean:[/bold] {judge_mean:.2f}/10 across {len(judge_scores)} sample(s)"
        )


def _resolve_router_tool_specs(tool_names: Sequence[str], workspace: Path) -> list[ToolSpec]:
    """Instantiate the minimum-viable set of tools needed to read their
    ToolSpecs for the router eval. Memory-dependent tools (search_memory,
    search_facts, remember_*) are skipped with a warning — the eval
    fixture can still cover them via the same tool name, but the router
    will see the spec from a dummy no-op tool below."""
    builders: dict[str, Callable[[], Tool]] = {
        "read_file": lambda: ReadFileTool(root=workspace),
        "list_dir": lambda: ListDirTool(root=workspace),
        "grep": lambda: GrepTool(root=workspace),
        "glob": lambda: GlobTool(root=workspace),
        "search_web": lambda: SearchWebTool(),
        "git_status": lambda: GitStatusTool(root=workspace),
        "git_diff": lambda: GitDiffTool(root=workspace),
        "git_log": lambda: GitLogTool(root=workspace),
    }
    # Memory tools need live stores; for eval we only need the schema,
    # so substitute a no-op stub that carries the same ToolSpec shape.
    memory_schemas: dict[str, ToolSpec] = {
        "search_memory": ToolSpec(
            name="search_memory",
            description="Search Airton's episodic memory for past events, decisions, or exchanges.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Semantic search query"}},
                "required": ["query"],
            },
            tier="read",
        ),
        "search_facts": ToolSpec(
            name="search_facts",
            description="Search Airton's semantic facts (subject/predicate/object triples).",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Semantic search query"}},
                "required": ["query"],
            },
            tier="read",
        ),
    }
    specs: list[ToolSpec] = []
    for name in tool_names:
        if name in builders:
            specs.append(builders[name]().spec)
        elif name in memory_schemas:
            specs.append(memory_schemas[name])
        else:
            console.print(f"[yellow]⚠ tool {name!r} has no eval-time spec — skipping[/yellow]")
    return specs


@eval_app.command("router")
def eval_router(
    router_repo: str = typer.Option(
        "mlx-community/Hermes-3-Llama-3.2-3B-4bit",
        "--router-repo",
        help="HF repo for the router model under test.",
    ),
    router_mode: str = typer.Option(
        "free",
        "--router-mode",
        help="'free' (default) or 'grammar' (JSON-schema-constrained).",
    ),
    tool_set: str = typer.Option(
        "research",
        "--tool-set",
        help="Tool profile whose specs the router sees. Default 'research' "
        "(read/list/grep/glob + search_memory/facts + search_web).",
    ),
    tools_add: str | None = typer.Option(
        None, "--tools-add", help="Comma-separated tool names to add on top of --tool-set."
    ),
    tools_drop: str | None = typer.Option(
        None, "--tools-drop", help="Comma-separated tool names to drop from --tool-set."
    ),
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to a router-eval YAML file. Defaults to `character/<name>/router_eval.yaml`.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Replay the router-eval fixture through the configured router and
    score tool-selection accuracy. Lock in quality before swapping
    models or tweaking prompts."""
    character = load_character(settings.character_path)
    path = fixture_path or default_fixture_path(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"router eval fixture not found: {path}")
    fixture = load_fixture(path)

    try:
        wanted_names = resolve_tool_names(
            tool_set,
            add=tuple((tools_add or "").split(",")),
            drop=tuple((tools_drop or "").split(",")),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    tool_specs = _resolve_router_tool_specs(wanted_names, settings.root)

    if router_mode not in {"free", "grammar"}:
        raise typer.BadParameter(
            f"--router-mode must be 'free' or 'grammar' (got {router_mode!r})."
        )
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter(repo=router_repo)
    router = (
        GrammarRouter(adapter=adapter) if router_mode == "grammar" else ModelRouter(adapter=adapter)
    )
    result = run_router_eval(router, tool_specs, fixture)

    if as_json:
        payload = {
            "router_repo": router_repo,
            "fixture": str(path),
            "accuracy": result.accuracy,
            "tool_accuracy": result.tool_accuracy,
            "cases": [
                {
                    "prompt": c.prompt,
                    "expected_tool": c.expected_tool,
                    "actual_tool": c.actual_tool,
                    "expected_args": list(c.expected_args),
                    "actual_args": c.actual_args,
                    "passed": c.passed,
                    "tool_correct": c.tool_correct,
                    "args_correct": c.args_correct,
                }
                for c in result.cases
            ],
        }
        console.print_json(json.dumps(payload))
        return

    _print_router_eval_table(result, router_repo, character.name)


def _print_router_eval_table(
    result: RouterEvalResult, router_repo: str, character_name: str
) -> None:
    table = Table(title=f"Router eval — {router_repo} · {character_name}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("prompt")
    table.add_column("expected", style="green")
    table.add_column("actual", style="yellow")
    table.add_column("args", style="dim")
    for c in result.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        exp = c.expected_tool if c.expected_tool is not None else "[dim]null[/dim]"
        act = c.actual_tool if c.actual_tool is not None else "[dim]null[/dim]"
        if c.args_correct:
            args_note = ""
        else:
            missing = sorted(set(c.expected_args) - c.actual_args.keys())
            args_note = f"missing {missing}"
        table.add_row(mark, c.prompt, exp, act, args_note)
    console.print(table)
    passed = sum(1 for c in result.cases if c.passed)
    console.print(
        f"[bold]{passed}/{len(result.cases)} passed · "
        f"{result.accuracy * 100:.1f}% full · "
        f"{result.tool_accuracy * 100:.1f}% tool-only[/bold]"
    )


@memory_app.command("list")
def memory_list(
    tier: str | None = typer.Option(None, help="Filter by tier: seed | consolidated | working"),
) -> None:
    """List every record in the episodic store."""
    character = load_character(settings.character_path)
    store = _open_episodic_store(character)
    if store is None:
        raise typer.Exit(code=1)
    try:
        records = store.all(tier=tier)
        if not records:
            console.print("[dim](empty)[/dim]")
            return
        table = Table(title=f"Episodic memory ({len(records)} records)", show_lines=True)
        table.add_column("id", style="bold")
        table.add_column("tier")
        table.add_column("title")
        table.add_column("principle", style="dim")
        for r in records:
            table.add_row(str(r.id), r.tier, r.title, r.principle or "")
        console.print(table)
    finally:
        store.close()


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="What to search for"),
    k: int = typer.Option(3, help="How many matches to return"),
) -> None:
    """Search episodic memory by semantic similarity."""
    character = load_character(settings.character_path)
    store = _open_episodic_store(character)
    if store is None:
        raise typer.Exit(code=1)
    try:
        hits = store.search(query, k=k)
        if not hits:
            console.print("[dim](no matches — store empty?)[/dim]")
            return
        for record, score in hits:
            console.print(f"[bold cyan]{score:.3f}[/bold cyan]  [bold]{record.title}[/bold]")
            if record.principle:
                console.print(f"  [dim italic]{record.principle}[/dim italic]")
            snippet = record.body[:220]
            ellipsis = "…" if len(record.body) > 220 else ""
            console.print(f"  [dim]{snippet}{ellipsis}[/dim]")
            console.print()
    finally:
        store.close()


@memory_app.command("fact-list")
def memory_fact_list(
    tier: str | None = typer.Option(None, help="Filter by tier: seed | consolidated | working"),
    subject: str | None = typer.Option(None, help="Filter by subject"),
) -> None:
    """List facts in the semantic store."""
    store = _open_semantic_store()
    if store is None:
        raise typer.Exit(code=1)
    try:
        facts = store.all(tier=tier, subject=subject)
        if not facts:
            console.print("[dim](empty)[/dim]")
            return
        table = Table(title=f"Semantic facts ({len(facts)})", show_lines=False)
        table.add_column("id", style="bold")
        table.add_column("tier")
        table.add_column("subject")
        table.add_column("predicate")
        table.add_column("object")
        table.add_column("conf", style="cyan")
        for f in facts:
            table.add_row(
                str(f.id), f.tier, f.subject, f.predicate, f.object, f"{f.confidence:.2f}"
            )
        console.print(table)
    finally:
        store.close()


@memory_app.command("fact-search")
def memory_fact_search(
    query: str = typer.Argument(..., help="Query text"),
    k: int = typer.Option(5, help="How many matches to return"),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence to include"),
) -> None:
    """Semantic-search the fact store."""
    store = _open_semantic_store()
    if store is None:
        raise typer.Exit(code=1)
    try:
        hits = store.search(query, k=k, min_confidence=min_confidence)
        if not hits:
            console.print("[dim](no matches)[/dim]")
            return
        for fact, score in hits:
            console.print(
                f"[bold cyan]{score:.3f}[/bold cyan]  "
                f"[bold]{fact.subject}[/bold] {fact.predicate} {fact.object} "
                f"[dim](conf={fact.confidence:.2f}, tier={fact.tier})[/dim]"
            )
    finally:
        store.close()


@memory_app.command("fact-add")
def memory_fact_add(
    subject: str = typer.Argument(..., help="Subject of the fact"),
    predicate: str = typer.Argument(..., help="Predicate (relation)"),
    object_: str = typer.Argument(..., metavar="OBJECT", help="Object / value"),
    confidence: float = typer.Option(0.9, help="Confidence 0-1"),
    tier: str = typer.Option("working", help="Tier: seed | consolidated | working"),
    source: str = typer.Option("user", help="Provenance label"),
    user: str | None = typer.Option(
        None,
        "--user",
        help="Relationship scope. Leave unset (or pass empty) to write a "
        "shared fact visible to everyone.",
    ),
) -> None:
    """Add a single fact to the semantic store."""
    store = _open_semantic_store()
    if store is None:
        raise typer.Exit(code=1)
    try:
        fact = store.add(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence,
            source=source,
            user_id=user if user else None,
            tier=tier,
        )
        console.print(
            f"[green]added[/green] id={fact.id}: "
            f"{fact.subject} {fact.predicate} {fact.object} (conf={fact.confidence:.2f})"
        )
    finally:
        store.close()


@memory_app.command("scribe")
def memory_scribe(
    session: str = typer.Option("local", help="Session id to scribe"),
    user: str = typer.Option(
        "mark",
        "--user",
        help="User to tag scribed memories with (relationship scope). "
        "Use --shared to write character-level shared memory instead.",
    ),
    shared: bool = typer.Option(
        False,
        "--shared",
        help="Write scribed memories as shared (user_id = NULL) rather "
        "than scoped to --user. Intended for character-level extractions.",
    ),
    model: str = typer.Option("mlx", help="Adapter for extraction: echo | mlx | ollama"),
    model_repo: str | None = typer.Option(
        None, "--model-repo", help="Override the model id. MLX: HF repo. Ollama: model tag."
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="LoRA adapter directory (from `mlx_lm.lora` training). Requires --model mlx.",
    ),
    window_size: int = typer.Option(20, help="Turns per extraction window"),
) -> None:
    """Walk unprocessed transcript turns and extract candidate memories."""
    character = load_character(settings.character_path)
    adapter = _resolve_adapter(model, model_repo=model_repo, lora_path=lora_path)
    episodic = _open_episodic_store(character)
    semantic = _open_semantic_store()
    if episodic is None or semantic is None:
        raise typer.Exit(code=1)
    transcript = Transcript(settings.db_path)
    try:
        user_id = None if shared else user
        with Status(f"scribe running on session={session}…", console=console):
            summary = run_scribe(
                adapter,
                character,
                transcript,
                episodic,
                semantic,
                session_id=session,
                user_id=user_id,
                window_size=window_size,
            )
        console.print(
            f"processed [bold]{summary.turns_processed}[/bold] turns across "
            f"[bold]{summary.windows}[/bold] window(s). "
            f"wrote [bold]{summary.episodic_written}[/bold] episodic, "
            f"[bold]{summary.semantic_written}[/bold] semantic."
        )
        if summary.parse_errors:
            console.print(f"[yellow]{len(summary.parse_errors)} parse error(s):[/yellow]")
            for err in summary.parse_errors:
                console.print(f"  - {err}")
    finally:
        transcript.close()
        episodic.close()
        semantic.close()


@memory_app.command("consolidate")
def memory_consolidate(
    episodic_threshold: float = typer.Option(
        0.80,
        help="Cosine-similarity threshold for clustering near-duplicate episodes.",
    ),
) -> None:
    """Promote working-tier memories to consolidated. Clusters similar
    episodes; groups semantic facts by (subject, predicate); marks
    superseded rows so they drop out of retrieval. Idempotent."""
    character = load_character(settings.character_path)
    episodic = _open_episodic_store(character, ingest=False)
    semantic = _open_semantic_store()
    if episodic is None or semantic is None:
        raise typer.Exit(code=1)
    try:
        with Status("running consolidation…", console=console):
            summary = run_consolidation(episodic, semantic, episodic_threshold=episodic_threshold)
        console.print(
            f"episodic: considered [bold]{summary.episodic_considered}[/bold] working, "
            f"merged [bold]{summary.episodic_clusters_merged}[/bold] cluster(s), "
            f"promoted [bold]{summary.episodic_promoted}[/bold], "
            f"superseded [bold]{summary.episodic_superseded}[/bold]"
        )
        console.print(
            f"semantic: considered [bold]{summary.semantic_considered}[/bold] working, "
            f"merged [bold]{summary.semantic_groups_merged}[/bold] group(s), "
            f"promoted [bold]{summary.semantic_promoted}[/bold], "
            f"superseded [bold]{summary.semantic_superseded}[/bold]"
        )
    finally:
        episodic.close()
        semantic.close()


@memory_app.command("rebuild-embeddings")
def memory_rebuild_embeddings() -> None:
    """Re-embed every active episodic and semantic record with the
    current embedder. Use after switching embedder models so existing
    data participates in search again."""
    character = load_character(settings.character_path)
    episodic = _open_episodic_store(character, ingest=False)
    semantic = _open_semantic_store()
    if episodic is None or semantic is None:
        raise typer.Exit(code=1)
    try:
        ep_mismatched = episodic.count_mismatched_embeddings()
        sem_mismatched = semantic.count_mismatched_embeddings()
        console.print(
            f"episodic: {ep_mismatched} mismatched; semantic: {sem_mismatched} mismatched."
        )
        with Status("re-embedding episodic…", console=console):
            ep_updated, _ = episodic.rebuild_embeddings()
        with Status("re-embedding semantic…", console=console):
            sem_updated, _ = semantic.rebuild_embeddings()
        console.print(
            f"[green]rebuilt {ep_updated} episodic and {sem_updated} semantic "
            f"embeddings with {episodic.embedder.id}.[/green]"
        )
    finally:
        episodic.close()
        semantic.close()


@memory_app.command("wipe")
def memory_wipe(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Clear episodic, semantic, and scribe-watermark data. Transcripts
    and character data are preserved. Use after switching embedder
    dimensions, or when you want to re-ingest from scratch."""
    if not yes:
        typer.confirm(
            "This wipes all episodic, semantic, and scribe-watermark data. Proceed?",
            abort=True,
        )
    import contextlib
    import sqlite3

    # Hardcoded whitelist — not user input, so S608 is a false positive
    # for this interpolation, but we keep the table names fixed anyway.
    tables = ("episodic", "semantic", "scribe_watermark")
    conn = sqlite3.connect(settings.db_path)
    try:
        for table in tables:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"DELETE FROM {table}")  # noqa: S608
        conn.commit()
    finally:
        conn.close()
    console.print("[yellow]wiped episodic, semantic, and scribe_watermark.[/yellow]")


@memory_app.command("ingest")
def memory_ingest() -> None:
    """Force an ingestion pass of the character's seed memories.
    Idempotent — existing records with matching external_id are
    preserved."""
    character = load_character(settings.character_path)
    store = _open_episodic_store(character, ingest=False)
    if store is None:
        raise typer.Exit(code=1)
    try:
        inserted = ensure_seeds_ingested(character, store)
        total = len(store.all())
        console.print(f"[bold]{inserted}[/bold] new, [bold]{total}[/bold] total in episodic store.")
    finally:
        store.close()


@voice_app.command("capture")
def _write_voice_capture(
    *,
    prompt: str,
    gold: str,
    session: str,
    original: str | None,
    sample_id: str | None = None,
) -> tuple[Path, str, int]:
    """Shared between `harness voice capture` and the in-chat /edit
    slash command. Appends a sample to `voice/captured.yaml` and
    returns (path, sample_id, total_sample_count)."""
    import yaml

    captured_path = settings.character_path / "voice" / "captured.yaml"
    captured_path.parent.mkdir(parents=True, exist_ok=True)

    if captured_path.exists():
        doc = yaml.safe_load(captured_path.read_text()) or {"samples": []}
    else:
        doc = {"version": 1, "samples": []}

    if sample_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        sample_id = f"captured-{stamp}"

    new_sample: dict[str, object] = {
        "id": sample_id,
        "prompt": prompt,
        "gold": gold.strip(),
        "captured_at": datetime.now(UTC).isoformat(),
        "captured_from": f"session={session}",
    }
    if original is not None:
        new_sample["original"] = original

    doc.setdefault("samples", []).append(new_sample)
    captured_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return captured_path, sample_id, len(doc["samples"])


def voice_capture(
    session: str = typer.Option("local", help="Session id to pull the exchange from."),
    gold: str = typer.Option(
        ...,
        "--gold",
        help="The corrected reply — what Airton should have said in response "
        "to the last user prompt in the session.",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Override the user prompt this sample is paired with. Defaults "
        "to the last user turn in the session.",
    ),
    sample_id: str | None = typer.Option(
        None,
        "--id",
        help="Custom sample id. Defaults to captured-<UTC timestamp>.",
    ),
) -> None:
    """Capture a user edit of Airton's reply as a new voice sample.

    The captured sample goes into `character/<name>/voice/captured.yaml`,
    a separate file from the curated canonical set, and will be loaded
    alongside canonical samples on the next character load. Over time
    this is how the voice corpus compounds from real use."""
    transcript = Transcript(settings.db_path)
    try:
        history = transcript.tail(session, limit=200)
    finally:
        transcript.close()

    if prompt is None:
        user_turns = [m for m in history if m.role == "user"]
        if not user_turns:
            console.print(
                f"[red]no user turns in session '{session}'. "
                "Pass --prompt to supply one explicitly.[/red]"
            )
            raise typer.Exit(code=1)
        prompt = user_turns[-1].content

    original: str | None = None
    assistant_turns = [m for m in history if m.role == "assistant"]
    if assistant_turns:
        original = assistant_turns[-1].content

    captured_path, final_id, total = _write_voice_capture(
        prompt=prompt,
        gold=gold,
        session=session,
        original=original,
        sample_id=sample_id,
    )

    console.print(
        f"[green]captured[/green] id={final_id!r} "
        f"→ {captured_path.relative_to(settings.root)} "
        f"(now {total} captured sample(s))"
    )


@voice_app.command("list-captured")
def voice_list_captured() -> None:
    """List every captured sample in the current character."""
    import yaml

    captured_path = settings.character_path / "voice" / "captured.yaml"
    if not captured_path.exists():
        console.print("[dim](no captured samples yet)[/dim]")
        return
    doc = yaml.safe_load(captured_path.read_text()) or {}
    samples = doc.get("samples", []) or []
    if not samples:
        console.print("[dim](no captured samples yet)[/dim]")
        return
    table = Table(title=f"Captured samples ({len(samples)})", show_lines=True)
    table.add_column("id", style="bold")
    table.add_column("captured_at")
    table.add_column("prompt")
    table.add_column("gold", style="green")
    for s in samples:
        table.add_row(
            str(s.get("id", "?")),
            str(s.get("captured_at", "?")),
            str(s.get("prompt", "?")),
            str(s.get("gold", "?")),
        )
    console.print(table)


if __name__ == "__main__":
    app()
