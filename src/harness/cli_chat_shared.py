"""Shared chat-surface helpers: transcript codec, approve-prompt
rendering, the streaming filter, and per-turn retrieval composition.

Step 2 of docs/cli-extraction-plan.md. These are the exact symbols the
sibling UIs (`cli_classic`, `cli_repl`, `cli_tui`) used to reach back
into `cli.py` for through deferred in-function imports; giving them a
real home fixes the dependency direction instead of dancing around a
circular import.

Nothing here owns state beyond `_RetrievalState` (per-session mute
flags, handed in by the caller). Every renderer takes the Console it
should print to, so the module never reaches for a module-level one.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import cast

from rich.console import Console
from rich.status import Status

from harness.character import VoiceSample
from harness.config import settings
from harness.model.adapter import ChatMessage, Role
from harness.orchestrator import (
    _FABRICATED_SEARCH_RE,
    _FALSE_SUCCESS_RE,
    _META_CONFIRM_RE,
    _TOOL_INTENT_RE,
    ToolLoopEvent,
    format_truncated_retry_suffix,
)
from harness.retrieval import VoiceRetriever
from harness.store.bd_adapter import BeadsAdapter, BeadsAdapterError
from harness.store.episodic import EpisodicRecord, EpisodicStore
from harness.store.semantic import SemanticFact, SemanticStore
from harness.store.transcript import Transcript, TranscriptMessage
from harness.tools import ToolCall

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
    if name in ("transcript_ingest",):
        turns = args.get("turns") or []
        n = len(turns) if isinstance(turns, list) else 0
        sid = str(args.get("session_id", "?"))
        return f"ingest {n} turn(s) into session {sid!r}"
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
    auto_scribe: bool,
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
        auto_scribe_bit = (
            " · [green]auto-scribe[/green]"
            if auto_scribe and memories_active and facts_active
            else ""
        )
        grid.add_row(
            "compact",
            f"{int(compact_at * 100)}% of window · keep {compact_keep_recent}{auto_scribe_bit}",
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
    elif event.kind == "tool_call_blocked":
        call = event.call
        assert call is not None
        label = tool_label(call.name)
        # A pre_tool guard refused a first-time call (policy block, not a
        # duplicate). Surface the guard's own corrective message so the
        # user sees WHY it was blocked, not a misleading "duplicate" line.
        result = event.result
        reason = result.error if result is not None and result.error else "blocked"
        console.print(f"[dim]⇢ {label} {call.arguments} — blocked ({reason})[/dim]")
    elif event.kind == "truncated_retry":
        # Wrap-up round hit the token cap mid-reply; orchestrator
        # widened the budget and is about to re-run. Drop the
        # in-flight stream buffer so we don't keep a partial-then-
        # full double and flag the break so the user knows the
        # upcoming reply supersedes the partial they just saw.
        # Budget progression annotation (harness-738f) lets the
        # user diagnose runaway-preamble vs. healthy-tail-clip
        # without re-running with trace logging.
        stream_renderer.stop()
        suffix = format_truncated_retry_suffix(event.budget_before, event.budget_after)
        console.print(f"[dim]⋯ truncated, retrying with wider budget{suffix}…[/dim]")
    elif event.kind == "bail_retry":
        # 0-tool-calls reply tripped a fabrication / teaser catcher;
        # orchestrator appended a nudge and is re-running. Drop the
        # partial stream so the fabricated draft doesn't stay
        # stacked above the next retry (harness-24xj).
        stream_renderer.stop()
        suffix = f" ({event.catcher})" if event.catcher else ""
        console.print(f"[dim]⋯ discarding draft, retrying{suffix}…[/dim]")


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
    without that source's prompt context.

    `muted` is a separate axis flipped by `/clear` (harness-zpe): when
    True, all three sources return empty without querying so prior-
    session memories can't leak back into the fresh start. Stored data
    is untouched — restart the process to re-enable retrieval (or, when
    the persisted /clear watermark hydrates on init, ContextMeter sets
    muted there too — harness-rrkj + harness-eftf)."""

    voice_ok: bool = True
    episodic_ok: bool = True
    semantic_ok: bool = True
    muted: bool = False


_TOPIC_BOUNDARY_NOTE_RESET = (
    "[NEW TOPIC] The user just reset this conversation. "
    "Earlier exchanges in this session may not apply to "
    "the current turn unless the user explicitly references them."
)


_TOPIC_BOUNDARY_NOTE_SCOPE = (
    "[SESSION-SCOPED] Retrieval for this turn is restricted to the "
    "current conversation. Memories or facts from earlier sessions "
    "are not part of this thread unless the user explicitly references them."
)


def _topic_boundary_suffix(
    retrieval_state: _RetrievalState,
    allowed_sessions: tuple[str, ...] | None = None,
) -> str:
    """Return the topic-boundary system-prompt suffix (with leading
    separator). Fires on two triggers (harness-eftf + harness-w3mo):

    1. `retrieval_state.muted` — the user ran /clear, either in this
       process or via a hydrated watermark from a prior session.
    2. `allowed_sessions is not None` — `--memory-scope` is bounding
       retrieval to a session subset. Different note wording
       reflects the different cause: scope-bounded retrieval is
       not 'fresh start within this session' but 'this thread
       excludes other sessions'.

    Both signals shorten to a single note when they coincide
    (mute wins; the cleared-conversation framing is stronger)."""
    if retrieval_state.muted:
        return f"\n\n{_TOPIC_BOUNDARY_NOTE_RESET}"
    if allowed_sessions is not None:
        return f"\n\n{_TOPIC_BOUNDARY_NOTE_SCOPE}"
    return ""


def _append_retrieval_error_log(*, source: str, user_input: str, exc: BaseException) -> None:
    """Append a retrieval-pipeline failure to the shared log file.

    Episodic / semantic / voice search all swallow exceptions with a
    one-time warn() and a state flag that disables the source. That's
    right for the chat surface but it loses the stack — observed
    2026-05-15 with three sessions in a row tripping 'bad value(s) in
    fds_to_keep' across all three sources at session start (embedder
    first-load path). Writes to the SAME file the assemble_context
    tool uses so a single `tail` covers the whole retrieval surface.
    Silent on disk-write failure — losing the log is strictly better
    than crashing the chat turn on top of an already-failed search."""
    import sys as _sys
    import traceback as _tb
    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    log_path = settings.data_path / "logs" / "assemble_context_errors.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"--- {_datetime.now(_UTC).isoformat()} ---\n")
            f.write(f"source={source!r}\n")
            f.write(f"user_input_prefix={user_input[:120]!r}\n")
            f.write(f"exception={type(exc).__name__}: {exc}\n")
            f.write(_tb.format_exc())
            f.write("\n")
    except OSError:
        pass
    # Mirror to stderr too — classic-REPL users will see it inline.
    _tb.print_exc(file=_sys.stderr)


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
    allowed_sessions: tuple[str, ...] | None = None,
    recency_ranks: dict[str, int] | None = None,
    recency_weight: float = 0.0,
) -> tuple[list[VoiceSample], list[EpisodicRecord], list[SemanticFact]]:
    """Run the three retrieval sources for one turn. Any that raise are
    disabled for the rest of the session (flagged on `state`) and a
    one-time `warn(msg)` fires. Returns the hits from the sources that
    are still healthy — empty lists for the ones that aren't.

    When `state.muted` is set (by `/clear`), all three sources return
    empty immediately — a cleared session must feel cleared, and prior-
    session memories landing in the system prompt via retrieval is
    exactly what causes the 'why is the agent still asking about Brad
    Hintze' symptom."""
    if state.muted:
        return [], [], []
    examples: list[VoiceSample] = []
    if retriever is not None and state.voice_ok and top_k > 0:
        try:
            examples = retriever.top_k(user_input, k=top_k)
        except Exception as exc:
            state.voice_ok = False
            _append_retrieval_error_log(source="voice", user_input=user_input, exc=exc)
            warn(f"voice retrieval disabled for this session: {exc}")

    recalled: list[EpisodicRecord] = []
    if memory_store is not None and state.episodic_ok and memories > 0:
        try:
            hits = memory_store.search(
                user_input,
                k=memories,
                min_score=memories_threshold,
                user_id=speaker,
                allowed_sessions=allowed_sessions,
                recency_ranks=recency_ranks,
                recency_weight=recency_weight,
            )
            recalled = [rec for rec, _score in hits]
        except Exception as exc:
            state.episodic_ok = False
            _append_retrieval_error_log(source="episodic", user_input=user_input, exc=exc)
            warn(f"episodic memory disabled for this session: {exc}")

    known_facts: list[SemanticFact] = []
    if semantic_store is not None and state.semantic_ok and facts > 0:
        try:
            fact_hits = semantic_store.search(
                user_input,
                k=facts,
                min_score=facts_threshold,
                user_id=speaker,
                allowed_sessions=allowed_sessions,
                recency_ranks=recency_ranks,
                recency_weight=recency_weight,
            )
            known_facts = [f for f, _score in fact_hits]
        except Exception as exc:
            state.semantic_ok = False
            _append_retrieval_error_log(source="semantic", user_input=user_input, exc=exc)
            warn(f"semantic facts disabled for this session: {exc}")

    return examples, recalled, known_facts


def _render_ab_memories_block(adapter: BeadsAdapter) -> str | None:
    """Fetch ab's bd-owned memories and wrap them as a system-prompt
    block. Returns None when the store is empty or bd is transiently
    unavailable — the caller should skip the injection rather than
    emitting an empty section (harness-hc9k).

    The chat pipeline's existing `_render_memory_block` only surfaces
    the harness-local EpisodicStore; persisted `bd remember` insights
    stayed dormant across sessions until a tool round called
    `memories` explicitly. Auto-injecting them makes durable
    preferences take effect the very next turn."""
    try:
        out = adapter.memories().strip()
    except BeadsAdapterError:
        return None
    if not out or out.startswith("No memories stored"):
        return None
    return "Durable preferences and notes from earlier sessions — apply automatically:\n\n" + out


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
