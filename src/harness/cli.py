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
from rich.status import Status
from rich.table import Table

import harness._quiet  # noqa: F401 — side-effect import: silences HF/transformers/sentence-transformers noise before they load
from harness.character import Character, VoiceSample, load_character
from harness.cli_introspect import list_cli_commands
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
from harness.model.adapter import Role
from harness.orchestrator import (
    _FABRICATED_SEARCH_RE,
    _FALSE_SUCCESS_RE,
    _META_CONFIRM_RE,
    _TOOL_INTENT_RE,
    ToolLoopEvent,
)
from harness.persona import PersonaAdapter
from harness.persona.caveman_rewriter import CavemanRewriter, load_register_map
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
from harness.store.audit import AuditStore
from harness.store.bd_adapter import BeadsAdapter, BeadsAdapterError
from harness.store.transcript import Transcript, TranscriptMessage
from harness.tools import (
    DEFAULT_PROFILE,
    TOOL_PROFILES,
    EditFileTool,
    FetchUrlTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    GlobTool,
    GrepTool,
    IntrospectContext,
    IntrospectTool,
    ListDirTool,
    ReadFileTool,
    RememberEventTool,
    RememberFactTool,
    SearchFactsTool,
    SearchMemoryTool,
    SearchWebTool,
    ShellTool,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    TranscriptIngestTool,
    WriteFileTool,
    resolve_tool_names,
)
from harness.tools.ab_ops import (
    CaptureTool,
    CloseTool,
    CommentsTool,
    DeferTool,
    DeleteTool,
    DepTool,
    DriftTool,
    FindDuplicatesTool,
    ForgetTool,
    LabelTool,
    ListTool,
    MemoriesTool,
    PersistFocusNoteTool,
    PlanTool,
    RememberTool,
    ReopenTool,
    ReprioritizeTool,
    RetroTool,
    SearchTool,
    StatusTool,
    UpdateTool,
)

_EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit", ":q", ":quit"})
_EDIT_COMMANDS = frozenset({"/edit", "/capture"})
_RETRO_COMMANDS = frozenset({"/retro"})

# atc-family fetch_url host allowlist (harness-xbk.3). Applied when the
# active character is any atc-archetype persona (airton_c and its
# narrower variants airton_c1, airton_c2, …). Keeps the tutoring
# surface to authoritative aviation sources (current weather, NOTAMs,
# the pilot-facing FAA portals). Non-atc characters pass None and
# fetch_url stays unrestricted. Hostnames are lowercased, netloc-only
# (no scheme, no path). Grow this list as atc's research needs widen;
# keep it conservative by default.
_ATC_FETCH_URL_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "aviationweather.gov",
        "www.aviationweather.gov",
        "notams.aim.faa.gov",
        "1800wxbrief.com",
        "www.1800wxbrief.com",
        "faa.gov",
        "www.faa.gov",
    }
)

# Characters that inherit the atc archetype (generalist + narrower
# document-scoped variants) all get the aviation allowlist. Extend
# here when you spin up a new airton_c* persona; scripts/character_
# from_template.py handles the on-disk scaffold.
_ATC_FAMILY_NAMES: frozenset[str] = frozenset({"airton_c", "airton_c1"})

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
    store = EpisodicStore(settings.character_db_path, embedder=embedder)  # type: ignore[arg-type]
    if ingest:
        inserted = ensure_seeds_ingested(character, store)
        if inserted > 0:
            console.print(f"[dim]seeded {inserted} episodic memories from character.[/dim]")
    return store


def _open_semantic_store() -> SemanticStore | None:
    embedder = _load_embedder()
    if embedder is None:
        return None
    return SemanticStore(settings.character_db_path, embedder=embedder)  # type: ignore[arg-type]


def _open_audit_store() -> AuditStore:
    """Open the per-turn audit log on the character's shared SQLite.
    Always returns a live store — the audit log has no embedder
    dependency and the feature is on for every chat session
    (harness-ywp.2). Callers close() at session teardown."""
    return AuditStore(settings.character_db_path)


def _print_session_end_retro(ab_adapter: BeadsAdapter | None) -> None:
    """Render RetroTool's summary view at graceful session exit
    (/exit, :q, Ctrl-C). Skipped for non-ab sessions and on retro
    failure — the retro is a convenience, not a blocker on shutdown.
    Crash/kill paths intentionally don't hit this (no atexit); an
    unreliable retro is worse than none."""
    if ab_adapter is None:
        return
    try:
        summary = RetroTool(ab_adapter).call(mode="summary")
    except Exception:  # exit path; never raise on shutdown
        return
    console.print(f"[dim]{summary}[/dim]")


def _maybe_bd_adapter(
    character: Character,
    *,
    include_internal: bool = False,
) -> BeadsAdapter | None:
    """Construct a bd adapter for the active character's ops plane.
    Returns None (with a yellow warning) when bd isn't runnable or the
    character's bd dir hasn't been bootstrapped — ops tools then skip
    registration with a hint. Works for any character: per-character
    bd dirs (~/.harness/<name>/) mirror the memory-store isolation,
    so airton and airton_b never cross thought-graphs.

    `include_internal=False` (default) hides ab-owned thought-graph
    beads (assignee=airton_b) from read views. `--dev` or explicit
    `--include-internal` on the CLI flip this on. The filter is a
    no-op for characters with no airton_b-assigned beads; keeping it
    uniform avoids branching on character name here."""
    bd_dir = settings.bd_dir_for(character.name)
    exclude = None if include_internal else "airton_b"
    # airton_b shares the project bd dir (Path 2, harness-55y) so the
    # adapter sees every bead in the graph. Restrict ab's read surface
    # to items scoped to its domain (professional/personal) — dev /
    # maintenance beads never carry those labels and so fall out
    # (harness-j7y). Other characters stay unconstrained.
    scope_allowlist = ("professional", "personal") if character.name == "airton_b" else None
    adapter = BeadsAdapter(
        bd_dir,
        default_exclude_assignee=exclude,
        ab_assignee="airton_b",
        default_scope_allowlist=scope_allowlist,
        turn_cap=settings.ab_turn_cap,
        inflight_cap=settings.ab_inflight_cap,
    )
    try:
        adapter.verify()
    except BeadsAdapterError as exc:
        console.print(f"[yellow]⚠ ops tools unavailable: {exc}[/yellow]")
        return None
    # Surface which path the adapter landed on so misconfigured
    # HARNESS_AB_BD_DIR (or missing bootstrap) is visible at session
    # start rather than silently writing to the wrong DB.
    console.print(f"[dim]{character.name} bd → {bd_dir}[/dim]")
    return adapter


# Back-compat alias — external callers (tests, scripts) still import
# the old name. Remove once all call sites are migrated.
_maybe_ab_bd_adapter = _maybe_bd_adapter


def _maybe_harvest_skills(
    ab_adapter: BeadsAdapter | None,
    episodic: EpisodicStore | None,
    *,
    enabled: bool = True,
) -> None:
    """Run the bd → episodic skill harvester at session start, when
    both halves of the substrate are available.

    Idempotent on external_id (bead id), so the steady-state cost is
    one `bd list` call + zero embeds. The first run after new
    decisions / observations close batches the new rows through the
    embedder — still fast (<1 s) for realistic working-set sizes.

    Failures log a yellow warning but never raise: self-improvement is
    a comfort, not a correctness requirement, and a flaky bd
    subprocess must not block the user from opening chat."""
    if not enabled or ab_adapter is None or episodic is None:
        return
    try:
        from harness.skills import harvest_bd_skills

        report = harvest_bd_skills(ab_adapter=ab_adapter, episodic=episodic)
    except Exception as exc:
        console.print(f"[yellow]⚠ skill harvest skipped: {exc}[/yellow]")
        return
    if report.newly_ingested > 0:
        console.print(
            f"[dim]harvested {report.newly_ingested} new skill(s) from bd "
            f"({report.already_present} already present)[/dim]"
        )


def _maybe_harvest_bd_memories(
    ab_adapter: BeadsAdapter | None,
    episodic: EpisodicStore | None,
    *,
    enabled: bool = True,
) -> None:
    """Mirror bd's persistent memories into the episodic store at
    session start so the main retrieval path can surface them on
    identity / biographical questions (harness-9yd).

    Same failure discipline as `_maybe_harvest_skills`: warn on
    failure, never raise — a flaky bd subprocess must not block chat
    startup. Idempotent on `external_id='bd-mem:<key>'`, so steady-
    state cost is one `bd memories --json` call plus zero embeds."""
    if not enabled or ab_adapter is None or episodic is None:
        return
    try:
        from harness.skills import harvest_bd_memories

        report = harvest_bd_memories(ab_adapter=ab_adapter, episodic=episodic)
    except Exception as exc:
        console.print(f"[yellow]⚠ bd memory harvest skipped: {exc}[/yellow]")
        return
    if report.newly_ingested > 0:
        console.print(
            f"[dim]harvested {report.newly_ingested} new bd memorie(s) "
            f"({report.already_present} already present)[/dim]"
        )


# Names of the ops (bd-backed) tools. Used by the registry warning
# path to distinguish "unknown tool" from "ops tool whose bd adapter
# didn't bind" — the user-facing hint is very different.
OPS_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "plan",
        "capture",
        "status",
        "drift",
        "reprioritize",
        "close",
        "defer",
        "retro",
        "reopen",
        "delete",
        "update",
        "search",
        "list",
        "memories",
        "remember",
        "forget",
        "dep",
        "label",
        "comments",
        "find_duplicates",
        "persist_focus_note",
    }
)


def _missing_builder_reason(name: str, character: Character | None) -> str:
    """Explain why a requested tool name isn't in the registry. Splits
    three cases the old 'not yet implemented' blanket message lumped
    together: genuinely unknown tools, ops tools whose bd adapter
    didn't bind, and everything else (forward-compat placeholders)."""
    if name in OPS_TOOL_NAMES:
        char = character.name if character is not None else "<character>"
        bd_dir = settings.bd_dir_for(char) if character is not None else "<bd dir>"
        return (
            f"tool {name!r} needs a working bd dir at {bd_dir} "
            f"(run `cd {bd_dir} && bd init` if none, or `bd dolt start` "
            f"to bring the Dolt server back up)"
        )
    return f"tool {name!r} not yet implemented — skipping"


def _ab_tool_builders(
    ab_adapter: BeadsAdapter | None,
) -> dict[str, Callable[[], Tool | None]]:
    """Return tool-name → builder map for the ops surface. When the
    adapter is None (bd not runnable or dir not bootstrapped), returns
    an empty dict so the caller's merge is a no-op. When present,
    every builder is unconditional — ops tools don't depend on
    episodic or semantic stores the way the memory tools do.

    Any character can get an ops surface if their per-character bd
    dir is bootstrapped; the builders don't care which character the
    adapter was constructed for."""
    if ab_adapter is None:
        return {}
    return {
        "plan": lambda: PlanTool(ab_adapter),
        "capture": lambda: CaptureTool(ab_adapter),
        "status": lambda: StatusTool(ab_adapter),
        "drift": lambda: DriftTool(ab_adapter),
        "reprioritize": lambda: ReprioritizeTool(ab_adapter),
        "close": lambda: CloseTool(ab_adapter),
        "defer": lambda: DeferTool(ab_adapter),
        "retro": lambda: RetroTool(ab_adapter),
        "reopen": lambda: ReopenTool(ab_adapter),
        "delete": lambda: DeleteTool(ab_adapter),
        "update": lambda: UpdateTool(ab_adapter),
        "search": lambda: SearchTool(ab_adapter),
        "list": lambda: ListTool(ab_adapter),
        "memories": lambda: MemoriesTool(ab_adapter),
        "remember": lambda: RememberTool(ab_adapter),
        "forget": lambda: ForgetTool(ab_adapter),
        "dep": lambda: DepTool(ab_adapter),
        "label": lambda: LabelTool(ab_adapter),
        "comments": lambda: CommentsTool(ab_adapter),
        "find_duplicates": lambda: FindDuplicatesTool(ab_adapter),
        "persist_focus_note": lambda: PersistFocusNoteTool(ab_adapter),
    }


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
    elif event.kind == "truncated_retry":
        # Wrap-up round hit the token cap mid-reply; orchestrator
        # widened the budget and is about to re-run. Drop the
        # in-flight stream buffer so we don't keep a partial-then-
        # full double and flag the break so the user knows the
        # upcoming reply supersedes the partial they just saw.
        stream_renderer.stop()
        console.print("[dim]⋯ truncated, retrying with wider budget…[/dim]")
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
    is untouched — restart the process to re-enable retrieval."""

    voice_ok: bool = True
    episodic_ok: bool = True
    semantic_ok: bool = True
    muted: bool = False


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


def _build_tool_registry_for_tui(
    *,
    tools: bool,
    tool_set: str,
    tools_add: str | None,
    tools_drop: str | None,
    workspace_path: Path,
    memory_store: EpisodicStore | None,
    semantic_store: SemanticStore | None,
    speaker: str,
    session: str,
    adapter: ModelAdapter | None = None,
    character: Character | None = None,
    retrieval_health: object | None = None,
    persona_active: bool = False,
    router_id: str | None = None,
    transcript: Transcript | None = None,
    warnings_out: list[str] | None = None,
    include_internal: bool = False,
    ab_adapter: BeadsAdapter | None = None,
    router: Router | None = None,
) -> ToolRegistry | None:
    """Build a ToolRegistry for the Textual app. Subset of the
    classic REPL's setup — skips scribe_session and
    consolidate_memory (they require the resolved adapter +
    character + transcript; the TUI can add them later if they show
    up in real use). Unknown or store-dependent tools that get dropped
    push a line into `warnings_out` so ChatApp can render them on mount
    — silent drops bit a user who'd passed profile names to --tools-add
    and got nothing back (harness-akq)."""
    if not tools:
        return None
    try:
        wanted_names = resolve_tool_names(
            tool_set,
            add=tuple((tools_add or "").split(",")),
            drop=tuple((tools_drop or "").split(",")),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    # Character-local synonym expansion for search_memory (harness-ajn).
    # Loads corpus/synonyms.yaml + corpus/query_synonyms.yaml (query-only
    # additive entries). NullQueryExpander when both are absent.
    from harness.retrieval.query_expander import (
        default_query_only_synonyms_path,
        default_synonyms_path,
        load_query_expander,
    )

    query_expander = load_query_expander(
        default_synonyms_path(settings.character_path),
        query_only_path=default_query_only_synonyms_path(settings.character_path),
    )

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
        "search_memory": lambda: (
            SearchMemoryTool(store=memory_store, user_id=speaker, expander=query_expander)
            if memory_store is not None
            else None
        ),
        "search_facts": lambda: (
            SearchFactsTool(store=semantic_store, user_id=speaker)
            if semantic_store is not None
            else None
        ),
        "search_web": lambda: SearchWebTool(),
        "fetch_url": lambda: FetchUrlTool(
            allowed_hosts=(
                _ATC_FETCH_URL_ALLOWED_HOSTS
                if character is not None and character.name in _ATC_FAMILY_NAMES
                else None
            )
        ),
        "remember_fact": lambda: (
            RememberFactTool(store=semantic_store, user_id=speaker, session_id=session)
            if semantic_store is not None
            else None
        ),
        "remember_event": lambda: (
            RememberEventTool(store=memory_store, user_id=speaker, session_id=session)
            if memory_store is not None
            else None
        ),
        "transcript_ingest": lambda: (
            TranscriptIngestTool(store=memory_store) if memory_store is not None else None
        ),
    }
    # Same ab-ops injection as the REPL builder. When an ab_adapter is
    # passed in (TUI hoists the construction so ChatApp can share the
    # reference), reuse it; otherwise construct lazily so non-TUI
    # callers don't need to know about the plumbing.
    if character is not None:
        effective_ab = (
            ab_adapter
            if ab_adapter is not None
            else _maybe_bd_adapter(character, include_internal=include_internal)
        )
        builders.update(_ab_tool_builders(effective_ab))

    # `introspect` and `spawn_subagent` both inspect / operate on the
    # already-populated registry, so they're constructed in a second
    # pass after the concrete tools are in place.
    _deferred = {"introspect", "spawn_subagent"}

    registry = ToolRegistry()
    for name in wanted_names:
        if name in _deferred:
            continue
        builder = builders.get(name)
        if builder is None:
            if warnings_out is not None:
                warnings_out.append(_missing_builder_reason(name, character))
            continue
        tool = builder()
        if tool is None:
            if warnings_out is not None:
                warnings_out.append(
                    f"tool {name!r} needs a store that isn't enabled (check --memories / --facts)"
                )
            continue
        registry.register(tool)

    if "introspect" in wanted_names:
        if adapter is None or character is None:
            if warnings_out is not None:
                warnings_out.append("tool 'introspect' needs adapter + character — skipping")
        else:
            registry.register(
                _make_introspect_tool(
                    registry,
                    adapter,
                    character,
                    workspace_path,
                    episodic=memory_store,
                    semantic=semantic_store,
                    user_id=speaker,
                    retrieval_health=retrieval_health,
                    persona_active=persona_active,
                    router_id=router_id,
                    transcript=transcript,
                    session_id=session,
                )
            )

    if "spawn_subagent" in wanted_names:
        if adapter is None:
            if warnings_out is not None:
                warnings_out.append("tool 'spawn_subagent' needs an adapter — skipping")
        else:
            from harness.orchestrator.hooks import default_hook_pipeline
            from harness.tools import SpawnSubagentTool

            registry.register(
                SpawnSubagentTool(
                    adapter=adapter,  # type: ignore[arg-type]  # narrower _ToolCapableAdapter, checked at runtime
                    registry=registry,
                    hooks=default_hook_pipeline(),
                    router=router,
                )
            )

    if not registry.names():
        return None

    from harness.tools.profiles import apply_profile_descriptions

    apply_profile_descriptions(registry, tool_set)
    return registry


def _make_introspect_tool(
    registry: ToolRegistry,
    adapter: ModelAdapter,
    character: Character,
    workspace_path: Path | None,
    *,
    episodic: EpisodicStore | None = None,
    semantic: SemanticStore | None = None,
    user_id: str | None = None,
    retrieval_health: object | None = None,
    persona_active: bool = False,
    router_id: str | None = None,
    transcript: Transcript | None = None,
    session_id: str | None = None,
) -> IntrospectTool:
    """Construct an IntrospectTool bound to the already-populated
    registry. Pre-enumerates the CLI commands from the Typer app so
    the tool doesn't have to import harness.cli at runtime.

    `retrieval_health` is passed by reference so scope=memory reads
    the live state — sources flip to disabled mid-session when they
    raise, and introspect should reflect that. persona_active /
    router_id are snapshots captured at session start (they don't
    change mid-session)."""
    ctx = IntrospectContext(
        registry=registry,
        adapter=adapter,
        character=character,
        settings=settings,
        episodic=episodic,
        semantic=semantic,
        workspace=workspace_path,
        user_id=user_id,
        commands=tuple(list_cli_commands(app)),
        retrieval_health=retrieval_health,
        persona_active=persona_active,
        router_id=router_id,
        transcript=transcript,
        session_id=session_id,
    )
    return IntrospectTool(context=ctx)


def _router_id_label(router: Router | None) -> str | None:
    """Short human-readable label for introspect's model scope.
    Returns 'grammar:Hermes-3-3B-4bit' style string or None when the
    router isn't loaded this session."""
    if router is None:
        return None
    mode = "grammar" if isinstance(router, GrammarRouter) else "free"
    adapter_id = getattr(getattr(router, "adapter", None), "id", "router")
    return f"{mode}:{adapter_id}"


def _build_tool_grounding_block(registry: ToolRegistry, workspace_path: Path) -> str:
    """The long tool-use grounding block the system prompt grows when
    tools are active. Lifted out of the chat command so the Textual
    app (harness-1r4) can reuse it verbatim — small model behavior
    is sensitive enough that maintaining two copies would drift."""
    tool_names = ", ".join(registry.names())
    base = (
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
        "'the content is available'.\n"
        "- SCOPE of the reply = THIS turn only. Describe only the "
        "action you just took on this turn and the findings from the "
        "tools you just called. Do NOT recap, restate, or summarize "
        "prior turns' actions, tool calls, or results — the user saw "
        "them already. Prior context surfaces only when the user "
        "explicitly asks for it ('recap', 'what did we cover', "
        "'status of X'). If a prior-turn fact is strictly required "
        "for this turn's conclusion, cite it in one clause, not a "
        "paragraph."
    )
    # Follow-up fetch nudge (harness-f5x). When fetch_url is loaded and
    # the user references an item from an earlier fetch ("more details
    # on 8", "tell me about the second one"), the ONLY correct path is
    # to re-call fetch_url on THAT item's URL — not to paraphrase from
    # context. Small models default to regenerating a fake summary list
    # when the index→URL lookup is too much work; this directive +
    # the FabricatedItemizationHook together catch the class.
    if "fetch_url" in registry:
        base += (
            "\n- FOLLOW-UP DETAIL ASKS. When the user references an "
            "item from an earlier tool result ('more details on 8', "
            "'expand the second one', 'what's story 3 about'), you "
            "MUST call `fetch_url` on THAT item's URL from the prior "
            "tool output. Do NOT regenerate a summary list, "
            "paraphrase, or invent one-line 'details' from context. "
            "If you can't find the URL for the item the user named, "
            "tell them you need the URL pasted."
        )
    if "introspect" in registry:
        # Small belt-and-suspenders nudge (harness-u71). The tool's own
        # schema already describes it, but models prone to hallucinating
        # their own abilities benefit from an explicit directive to call
        # it instead of guessing.
        base += (
            "\n- When asked what you can do, what tools you have, what "
            "model you are running, what your context window is, how "
            "much you remember, or what CLI commands exist, call the "
            "`introspect` tool with the matching scope. Do not guess "
            "your capabilities from the character sheet or training."
        )
    # Preference-capture nudge (harness-7jda). When the remember tool is
    # loaded, the model must route durable-preference turns through it
    # BEFORE replying; otherwise the preference is acknowledged in chat
    # and lost at session end. Listing concrete trigger phrases gives
    # small models the pattern-matches they need.
    if "remember" in registry:
        base += (
            "\n- PREFERENCE CAPTURE. When the user states a durable "
            "preference or rule — phrases like 'from now on', 'always', "
            "'remember to', 'keep in mind', 'note to self', 'never again' "
            "— call the `remember` tool with the verbatim rule BEFORE "
            "replying. Then apply it. Acknowledging a preference only in "
            "chat does NOT persist it across sessions."
        )
    # Backlog grounding (harness-dxv). When ab's ops tools are wired,
    # any question about the user's tasks / plans / priorities /
    # blockers MUST route through bd via plan/status/drift — not
    # fabricated from memory of the plan format or prior turns.
    if any(t in registry for t in ("plan", "status", "drift")):
        base += (
            "\n- BACKLOG is tool-routed. Never describe the user's tasks, "
            "plan, priorities, tiers, blockers, deadlines, or project "
            "state from memory. Every ask like 'do we have plans', "
            "'what's up today', 'what am I working on', 'what's "
            "blocked', 'status of X', 'priorities', 'top tasks' MUST "
            "call the matching tool first (`plan` for tiered today's "
            "path, `drift` for blocked/at-risk items, `status` for a "
            "single issue). Do NOT invent issue ids, scopes, tiers, "
            "titles, reasons, or deadlines — they live in bd and are "
            "read-only to you until a tool call returns them. If none "
            "of these tools is loaded this turn, say so plainly."
        )
    return base


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


def _resolve_adapter(
    name: str,
    *,
    persona: bool = False,
    character: Character | None = None,
    model_repo: str | None = None,
    lora_path: str | None = None,
    draft_repo: str | None = None,
    chain_rewrites: bool = False,
    rewriter_temperature: float | None = None,
) -> ModelAdapter:
    # Custom configs bypass the factory and instantiate the adapter
    # directly. --lora-path is MLX-only; --model-repo works for MLX
    # (HF repo) and Ollama (model tag like "gemma4:latest").
    if lora_path and name != "mlx":
        raise typer.BadParameter("--lora-path requires --model mlx.")
    if draft_repo and name != "mlx":
        raise typer.BadParameter("--draft-repo requires --model mlx.")

    adapter: ModelAdapter
    if model_repo or lora_path or draft_repo:
        if name == "mlx":
            from harness.model.mlx import MLXAdapter

            mlx_kwargs: dict[str, object] = {}
            if model_repo:
                mlx_kwargs["repo"] = model_repo
            if lora_path:
                mlx_kwargs["adapter_path"] = lora_path
            if draft_repo:
                mlx_kwargs["draft_repo"] = draft_repo
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
        if character.name == "airton_b":
            # ab ships its own voice layer — caveman compression with a
            # per-surface intensity map — instead of Airton's style
            # rewrite. The register_map lives next to the character so
            # it ships and evolves with the persona data.
            register_map = load_register_map(
                settings.root / "character" / "airton_b" / "register_map.yaml"
            )
            adapter = CavemanRewriter(
                adapter,
                intensity=settings.ab_register,
                register_map=register_map,
                rewrite_on_tools=settings.ab_rewrite_on_tools,
            )
        else:
            persona_kwargs: dict[str, object] = {"chain_rewrites": chain_rewrites}
            if rewriter_temperature is not None:
                persona_kwargs["rewriter_temperature"] = rewriter_temperature
            adapter = PersonaAdapter(adapter, character, **persona_kwargs)  # type: ignore[arg-type]

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
    draft_repo: str | None = typer.Option(
        None,
        "--draft-repo",
        help="HF repo of a smaller draft model for MLX speculative decoding "
        "(e.g. mlx-community/Qwen2.5-0.5B-Instruct-4bit). Must share the "
        "target model's tokenizer vocab. Typical uplift: 1.5-2x tok/s on "
        "7B/32B targets. Zero quality loss — output is distribution-identical. "
        "Defaults to HARNESS_MLX_DRAFT_MODEL_REPO. Requires --model mlx.",
    ),
    summarize_tool_results: bool = typer.Option(
        False,
        "--summarize-tool-results/--no-summarize-tool-results",
        help="Compress high-noise tool outputs (grep / list_dir / search_web) "
        "above ~1KB before the main model sees them. Preserves identifiers / "
        "paths / line numbers verbatim. Uses the router's adapter when "
        "--router is on; otherwise builds a small MLX adapter from "
        "--router-repo. Attacks context drift from bulk tool output "
        "(sota punch #3).",
    ),
    harvest_skills: bool = typer.Option(
        True,
        "--harvest-skills/--no-harvest-skills",
        help="At session start, harvest closed thought:decision / "
        "thought:observation beads from ab's bd store into the episodic "
        "memory as tier='procedural'. Idempotent — only new beads cost "
        "embedding work. Lets relevant past decisions surface on future "
        "user turns via the normal retrieval path (sota punch #7).",
    ),
    harvest_memories: bool = typer.Option(
        True,
        "--harvest-memories/--no-harvest-memories",
        help="At session start, mirror bd memories (bd remember / "
        "retro record) into the episodic store as tier='procedural' "
        "so identity / biographical questions land through the "
        "normal retrieval path instead of hallucinating. Idempotent "
        "on external_id='bd-mem:<key>' (harness-9yd).",
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
    chain_rewrites: bool = typer.Option(
        False,
        "--chain-rewrites/--no-chain-rewrites",
        help="Add a second 'concrete substitution' rewrite pass on top of the "
        "style pass. Requires --persona. Doubles persona latency but pulls "
        "the reply further toward Airton's register for drift-prone prompts.",
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
    auto_scribe: bool = typer.Option(
        True,
        "--auto-scribe/--no-auto-scribe",
        help="When compaction fires, scribe unprocessed transcript "
        "turns into episodic + semantic memory *before* the summarizer "
        "folds them. Default on — keeps 'what have we talked about' "
        "answerable via memory search after the transcript compresses. "
        "Needs --memories > 0 and --facts > 0; otherwise no-op. "
        "Watermark-gated, so repeat compactions only scribe new turns.",
    ),
    dev: bool = typer.Option(
        False,
        "--dev/--no-dev",
        help="Dev mode — surface internal signals like the stream filter's "
        "'⋯ suppressed N line(s)…' markers. Off by default so users don't "
        "see the model's self-inflicted noise; on for developers tuning "
        "the filter or debugging small-model behavior. Also implies "
        "--include-internal so ab's own thought-graph beads are visible.",
    ),
    include_internal: bool = typer.Option(
        False,
        "--include-internal/--no-include-internal",
        help="Show ab-internal beads (assignee=airton_b) in plan / list / "
        "drift / search views. Off by default so user-captured work isn't "
        "buried under ab's scratchpad thought-graph. --dev implies on.",
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
        settings.router_repo,
        "--router-repo",
        help="HF repo for the router model. Default from Settings."
        "router_repo (currently Hermes-3-Llama-3.2-3B-4bit, ~2 GB, "
        "function-call-tuned — generic small routers under-route). "
        "Override via HARNESS_ROUTER_REPO or this flag. Only used "
        "when --router is on. Router is MLX-only for now.",
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
        from harness.cli_tui import run_tui

        run_tui(
            session=session,
            channel=channel,
            speaker=speaker,
            model=model,
            model_repo=model_repo,
            lora_path=lora_path,
            draft_repo=draft_repo,
            summarize_tool_results=summarize_tool_results,
            harvest_skills=harvest_skills,
            harvest_memories=harvest_memories,
            persona=persona,
            chain_rewrites=chain_rewrites,
            top_k=top_k,
            memories=memories,
            memories_threshold=memories_threshold,
            facts=facts,
            facts_threshold=facts_threshold,
            tools=tools,
            tool_set=tool_set,
            tools_add=tools_add,
            tools_drop=tools_drop,
            workspace=workspace,
            compact_at=compact_at,
            auto_scribe=auto_scribe,
            router_enabled=router_enabled,
            router_repo=router_repo,
            router_mode=router_mode,
            include_internal=include_internal,
            dev=dev,
        )
        return

    from harness.cli_classic import run_classic_chat

    run_classic_chat(
        console=console,
        session=session,
        channel=channel,
        speaker=speaker,
        model=model,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
        summarize_tool_results=summarize_tool_results,
        harvest_skills=harvest_skills,
        harvest_memories=harvest_memories,
        persona=persona,
        chain_rewrites=chain_rewrites,
        top_k=top_k,
        memories=memories,
        memories_threshold=memories_threshold,
        facts=facts,
        facts_threshold=facts_threshold,
        tools=tools,
        tool_set=tool_set,
        tools_add=tools_add,
        tools_drop=tools_drop,
        rewrite_on_tools=rewrite_on_tools,
        workspace=workspace,
        compact_at=compact_at,
        compact_keep_recent=compact_keep_recent,
        auto_scribe=auto_scribe,
        dev=dev,
        include_internal=include_internal,
        router_enabled=router_enabled,
        router_repo=router_repo,
        router_mode=router_mode,
    )
    return


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
    draft_repo: str | None = typer.Option(
        None,
        "--draft-repo",
        help="HF repo of a smaller MLX draft model for speculative decoding. "
        "Distribution-preserving throughput boost on 7B/32B targets. "
        "Defaults to HARNESS_MLX_DRAFT_MODEL_REPO.",
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
    adapter = _resolve_adapter(
        model, model_repo=model_repo, lora_path=lora_path, draft_repo=draft_repo
    )
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
        "transcript_ingest": ToolSpec(
            name="transcript_ingest",
            description=(
                "Ingest a recorded session transcript as JSON turns into episodic working memory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "turns": {"type": "array", "items": {"type": "object"}},
                    "session_id": {"type": "string"},
                    "user_id": {"type": "string"},
                    "source_tag": {"type": "string"},
                },
                "required": ["turns", "session_id", "user_id", "source_tag"],
            },
            tier="write",
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
        settings.router_repo,
        "--router-repo",
        help="HF repo for the router model under test (default from "
        "Settings.router_repo / HARNESS_ROUTER_REPO).",
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

    # Apply profile-scoped description overrides so the router sees the
    # same reframed specs it would see in a live chat session (e.g. atc's
    # search_memory → rulebook framing). Without this the eval scores
    # the router against the wrong descriptions.
    from dataclasses import replace as _replace

    from harness.tools.profiles import TOOL_PROFILE_DESCRIPTIONS

    _overrides = TOOL_PROFILE_DESCRIPTIONS.get(tool_set, {})
    if _overrides:
        tool_specs = [
            _replace(spec, description=_overrides[spec.name]) if spec.name in _overrides else spec
            for spec in tool_specs
        ]

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


@eval_app.command("session-resume")
def eval_session_resume(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to a session-resume eval YAML file. Defaults to "
        "`character/<name>/session_resume_eval.yaml`.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Replay the session-resume fixture against build_resume_summary
    and score contains / not_contains assertions per scenario. Locks
    in quality so changes to the resume protocol don't silently drop
    a load-bearing section."""
    from harness.evals.session_resume import (
        default_fixture_path as _sr_default_fixture,
    )
    from harness.evals.session_resume import (
        load_fixture as _sr_load,
    )
    from harness.evals.session_resume import (
        run_session_resume_eval as _sr_run,
    )

    character = load_character(settings.character_path)
    path = fixture_path or _sr_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"session-resume eval fixture not found: {path}")
    fixtures = _sr_load(path)
    result = _sr_run(fixtures)

    if as_json:
        payload = {
            "character": character.name,
            "pass_rate": result.pass_rate,
            "cases": [
                {
                    "id": c.id,
                    "passed": c.passed,
                    "missing_contains": list(c.missing_contains),
                    "unexpected_contains": list(c.unexpected_contains),
                }
                for c in result.cases
            ],
        }
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"Session-resume eval — {character.name}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("id")
    table.add_column("missing contains", style="yellow")
    table.add_column("unexpected", style="red")
    for c in result.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        missing = ", ".join(c.missing_contains) if c.missing_contains else ""
        unexpected = ", ".join(c.unexpected_contains) if c.unexpected_contains else ""
        table.add_row(mark, c.id, missing, unexpected)
    console.print(table)
    passed = sum(1 for c in result.cases if c.passed)
    console.print(
        f"[bold]{passed}/{len(result.cases)} passed · {result.pass_rate * 100:.1f}%[/bold]"
    )


@eval_app.command("atc")
def eval_atc(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to an atc eval YAML. Defaults to `character/<name>/atc_eval.yaml`.",
    ),
    model: str = typer.Option("mlx", help="Adapter: echo | mlx | ollama"),
    model_repo: str | None = typer.Option(None, "--model-repo"),
    lora_path: str | None = typer.Option(None, "--lora-path"),
    draft_repo: str | None = typer.Option(None, "--draft-repo"),
    temperature: float = typer.Option(
        0.0,
        help=(
            "Sampling temperature. Defaults to 0 (greedy) so run-to-run "
            "pass-rate is stable — see harness-ald. Raise only when you "
            "want to sample variance explicitly."
        ),
    ),
    rewriter_temperature: float = typer.Option(
        0.0,
        help=(
            "Temperature for the PersonaAdapter's voice-rewrite pass. "
            "Defaults to 0 for the same reason — pass-2 style variance "
            "can swing whether a citation survives the rewrite."
        ),
    ),
    memories: int = typer.Option(3, help="Top-K episodic memories per turn"),
    facts: int = typer.Option(5, help="Top-K semantic facts per turn"),
    top_k: int = typer.Option(6, help="Top-K voice samples per turn"),
    audience: str | None = typer.Option(
        None,
        "--audience",
        help="Filter fixture to one audience (ppl|ifr|…). Default: all.",
    ),
    persona: bool = typer.Option(
        True,
        "--persona/--no-persona",
        help="Wrap the base adapter in PersonaAdapter (default on).",
    ),
    holdout: bool = typer.Option(
        False,
        "--holdout",
        help=(
            "Exclude voice samples listed in character/<name>/voice/"
            "holdout.yaml from retrieval for this eval run. Score delta "
            "vs. the default (no flag) is the generalization signal "
            "(harness-w49p)."
        ),
    ),
    holdout_ids: str | None = typer.Option(
        None,
        "--holdout-ids",
        help=(
            "Comma-separated sample IDs to exclude at retrieval time for "
            "this run. Overrides --holdout and the on-disk manifest — "
            "useful for round-robin per-sample memorization probes "
            "without mutating voice/holdout.yaml."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Run atc's domain eval: replay PPL/IFR Q&A cases through the full
    persona + retrieval stack and score citation presence + keyword
    recall. Phase-1 target: ≥80% pass. Becomes the gate for Phase-2
    voice changes and Phase-3 LoRA (harness-xbk.7)."""
    from harness.evals.atc import (
        default_fixture_path as _atc_default_fixture,
    )
    from harness.evals.atc import (
        load_fixture as _atc_load_fixture,
    )
    from harness.evals.atc import (
        run_atc_eval as _atc_run,
    )

    character = load_character(settings.character_path)
    path = fixture_path or _atc_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"atc eval fixture not found: {path}")
    fixture = _atc_load_fixture(path)
    if audience is not None:
        fixture = tuple(row for row in fixture if row.audience == audience)
    if not fixture:
        console.print("[yellow](no cases in fixture after filter — nothing to score)[/yellow]")
        raise typer.Exit(code=0)

    adapter = _resolve_adapter(
        model,
        persona=persona,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
        rewriter_temperature=rewriter_temperature,
    )

    # Retrieval stack. Off-the-shelf defaults match `harness chat`
    # with --memories 3 --facts 5 — same numbers the eval target
    # calibrates against.
    retriever = _maybe_retriever(character, top_k=top_k)
    memory_store = _open_episodic_store(character, ingest=False)
    semantic_store = _open_semantic_store()
    speaker = "eval"  # shared persona-wide content is user_id=None; a
    # literal speaker keeps the retrieval API consistent while never
    # matching a user-siloed row.

    # Holdout IDs (harness-w49p): samples listed in voice/holdout.yaml are
    # excluded from the retriever's returns when `--holdout` is on. The
    # set is empty when the flag is off or the character has no holdout
    # file, so the default retrieval path is unchanged.
    #
    # `--holdout-ids CSV` overrides both `--holdout` and the manifest
    # for ad-hoc per-sample probes (round-robin memorization map).
    if holdout_ids is not None:
        _holdout_ids: frozenset[str] = frozenset(
            part.strip() for part in holdout_ids.split(",") if part.strip()
        )
    elif holdout:
        _holdout_ids = frozenset(s.id for s in character.holdout_voice_samples)
    else:
        _holdout_ids = frozenset()

    def run_turn(question: str) -> str:
        examples: list[VoiceSample] = []
        if retriever is not None and top_k > 0:
            try:
                # Ask for a wider slate when holdout is on so the post-
                # filter doesn't shrink below top_k on characters with
                # many canonical samples (airton has 20+).
                request_k = top_k + len(_holdout_ids)
                voice_hits = retriever.top_k(question, k=request_k)
                examples = [s for s in voice_hits if s.id not in _holdout_ids][:top_k]
            except Exception:  # eval is read-only; surface score only
                examples = []

        recalled: list[EpisodicRecord] = []
        if memory_store is not None and memories > 0:
            try:
                hits = memory_store.search(question, k=memories, user_id=speaker)
                recalled = [rec for rec, _score in hits]
            except Exception:
                recalled = []

        known_facts: list[SemanticFact] = []
        if semantic_store is not None and facts > 0:
            try:
                fact_hits = semantic_store.search(question, k=facts, user_id=speaker)
                known_facts = [f for f, _score in fact_hits]
            except Exception:
                known_facts = []

        sys_prompt = character.system_prompt(include_samples=examples)
        extra: list[str] = []
        if recalled:
            extra.append(_render_memory_block(recalled))
        if known_facts:
            extra.append(_render_fact_block(known_facts))
        if extra:
            sys_prompt = sys_prompt + "\n\n" + "\n\n".join(extra)

        messages = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=question),
        ]
        return adapter.complete(messages, temperature=temperature).strip()

    try:
        result = _atc_run(fixture, run_turn)
    finally:
        if memory_store is not None:
            memory_store.close()
        if semantic_store is not None:
            semantic_store.close()

    if as_json:
        payload = {
            "character": character.name,
            "adapter": adapter.id,
            "pass_rate": result.pass_rate,
            "pass_rate_by_audience": result.pass_rate_by_audience(),
            "cases": [
                {
                    "id": c.id,
                    "audience": c.audience,
                    "passed": c.passed,
                    "citations_pass": c.citations_pass,
                    "keywords_pass": c.keywords_pass,
                    "missing_citations": list(c.missing_citations),
                    "matched_keywords": list(c.matched_keywords),
                    "keyword_hits": c.keyword_hits,
                    "min_keyword_hits": c.min_keyword_hits,
                    "reply": c.actual_reply,
                }
                for c in result.cases
            ],
        }
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"atc eval — {character.name} · {adapter.id}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("id")
    table.add_column("aud.", width=4)
    table.add_column("cite", style="cyan")
    table.add_column("kw hits", style="cyan")
    table.add_column("missing citations", style="yellow")
    for c in result.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        cite = "[green]✓[/green]" if c.citations_pass else "[red]✗[/red]"
        kw = f"{c.keyword_hits}/{c.min_keyword_hits}"
        missing = ", ".join(c.missing_citations) if c.missing_citations else ""
        table.add_row(mark, c.id, c.audience, cite, kw, missing)
    console.print(table)
    passed = sum(1 for c in result.cases if c.passed)
    console.print(
        f"[bold]{passed}/{len(result.cases)} passed · {result.pass_rate * 100:.1f}%[/bold]"
    )
    rates = result.pass_rate_by_audience()
    if len(rates) > 1:
        detail = " · ".join(f"{aud}: {r * 100:.1f}%" for aud, r in sorted(rates.items()))
        console.print(f"[dim]by audience — {detail}[/dim]")


@eval_app.command("atc-retrieval")
def eval_atc_retrieval(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to an atc eval YAML. Defaults to `character/<name>/atc_eval.yaml`.",
    ),
    k: int = typer.Option(10, help="Top-K depth ceiling for retrieval"),
    audience: str | None = typer.Option(
        None,
        "--audience",
        help="Filter fixture to one audience. Default: all.",
    ),
    expand_queries: bool = typer.Option(
        True,
        "--expand-queries/--no-expand-queries",
        help=(
            "Apply corpus/synonyms.yaml query-side expansion (harness-ajn) "
            "before running each fixture case through the store. Default on "
            "to match real-chat behaviour (SearchMemoryTool uses the same "
            "expander). Disable for A/B baselines measuring expander lift."
        ),
    ),
    llm_expand: bool = typer.Option(
        False,
        "--llm-expand/--no-llm-expand",
        help=(
            "Add the LLMQueryExpander pre-pass (harness-hvu1) — small "
            "model rewrites the user query into 3-5 doc-style keyword "
            "phrases that get appended before retrieval. Chains in front "
            "of the static synonym expander. Default off; flip on to "
            "measure recall lift vs the static-only baseline. Adds one "
            "small-model call per case (~100-200ms p50)."
        ),
    ),
    llm_expand_repo: str | None = typer.Option(
        None,
        "--llm-expand-repo",
        help=(
            "HF repo for the LLM-expander adapter when --llm-expand is "
            "set. Defaults to HARNESS_ROUTER_MODEL_REPO so the same "
            "small-model footprint serves routing + query expansion. "
            "Shared adapter, separate calls."
        ),
    ),
    save_baseline: bool = typer.Option(
        False,
        "--save-baseline",
        help=(
            "Write the run to character/<name>/atc_retrieval_baseline.json. "
            "Intended for snapshotting post-change so future runs can diff "
            "against the frozen rank-of-first-expected per case."
        ),
    ),
    compare_baseline: bool = typer.Option(
        False,
        "--compare-baseline",
        help=(
            "Diff this run against character/<name>/atc_retrieval_baseline.json "
            "(or --baseline-path). Exits non-zero on regression: aggregate "
            "recall@N drop OR per-case rank worsening past --regression-budget. "
            "The gate that turns the baseline JSON from a snapshot into a "
            "contract (harness-sb6r)."
        ),
    ),
    baseline_path: Path | None = typer.Option(
        None,
        "--baseline-path",
        help=(
            "Override the baseline file location for --save-baseline / "
            "--compare-baseline. Defaults to "
            "character/<name>/atc_retrieval_baseline.json."
        ),
    ),
    regression_budget: int = typer.Option(
        0,
        "--regression-budget",
        min=0,
        help=(
            "Allow up to N per-case rank regressions WHEN aggregate recall "
            "holds. Use sparingly — chunker changes that rebalance top-K "
            "without losing recall are the only legit case."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Retrieval-only atc eval: runs each fixture case through the
    episodic store and reports rank-of-first-expected + aggregate
    recall@1/@3/@5/@K. Skips the model entirely — decouples retrieval
    quality measurement from reply quality (harness-dfa)."""
    import json as _json_mod

    from harness.evals.atc import (
        default_fixture_path as _atc_default_fixture,
    )
    from harness.evals.atc import (
        load_fixture as _atc_load_fixture,
    )
    from harness.evals.atc_retrieval import (
        RetrievalHit,
        compare_baselines,
        default_baseline_path,
        load_baseline,
        run_atc_retrieval,
    )

    if save_baseline and compare_baseline:
        raise typer.BadParameter(
            "--save-baseline and --compare-baseline are mutually exclusive: "
            "compare first to confirm no regression, then re-run with "
            "--save-baseline to snapshot the new known-good state."
        )

    character = load_character(settings.character_path)
    path = fixture_path or _atc_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"atc eval fixture not found: {path}")
    fixture = _atc_load_fixture(path)
    if audience is not None:
        fixture = tuple(row for row in fixture if row.audience == audience)
    if not fixture:
        console.print("[yellow](no cases in fixture after filter — nothing to score)[/yellow]")
        raise typer.Exit(code=0)

    store = _open_episodic_store(character, ingest=False)
    if store is None:
        raise typer.BadParameter(
            "retrieval eval needs the `retrieval` extra — re-run `uv sync --extra retrieval`."
        )

    # Build the query expander the same way SearchMemoryTool does, so
    # eval recall@k numbers reflect the retrieval path a real chat turn
    # would take. --no-expand-queries gives the A/B baseline.
    from harness.retrieval.query_expander import (
        LLMQueryExpander,
        NullQueryExpander,
        QueryExpander,
        _load_llm_expand_prompt,
        default_llm_expand_prompt_path,
        default_query_only_synonyms_path,
        default_synonyms_path,
        load_query_expander,
    )

    base_expander: QueryExpander = (
        load_query_expander(
            default_synonyms_path(settings.character_path),
            query_only_path=default_query_only_synonyms_path(settings.character_path),
        )
        if expand_queries
        else NullQueryExpander()
    )

    expander: QueryExpander
    if llm_expand:
        from harness.model.mlx import MLXAdapter

        repo = llm_expand_repo or settings.router_repo
        llm_adapter = MLXAdapter(repo=repo)
        prompt_template = _load_llm_expand_prompt(
            default_llm_expand_prompt_path(settings.character_path)
        )
        expander = LLMQueryExpander(
            llm_adapter,
            chain_to=base_expander,
            prompt_template=prompt_template,
        )
        console.print(f"[dim]llm-expand: {repo}[/dim]")
    else:
        expander = base_expander

    def _search(query: str, depth: int) -> list[RetrievalHit]:
        raw = store.search(expander.expand(query), k=depth, mode="hybrid")
        return [
            RetrievalHit(principle=rec.principle or "", score=float(score)) for rec, score in raw
        ]

    result = run_atc_retrieval(fixture, _search, k=k)

    resolved_baseline_path = baseline_path or default_baseline_path(settings.character_path)

    comparison = None
    if compare_baseline:
        if not resolved_baseline_path.exists():
            raise typer.BadParameter(
                f"no baseline at {resolved_baseline_path}; run with "
                f"--save-baseline first to snapshot a known-good state."
            )
        comparison = compare_baselines(load_baseline(resolved_baseline_path), result)

    if as_json:
        envelope: dict[str, object] = {
            "character": character.name,
            "fixture": str(path),
            "k": result.k,
            "recall_at_1": result.recall_at_1,
            "recall_at_3": result.recall_at_3,
            "recall_at_5": result.recall_at_5,
            "recall_at_k": result.recall_at_k,
            "median_rank": result.median_rank,
            "cases": [
                {
                    "id": c.id,
                    "audience": c.audience,
                    "query": c.query,
                    "expected_anchors": list(c.expected_anchors),
                    "rank_of_first_expected": c.rank_of_first_expected,
                    "score_of_first_expected": c.score_of_first_expected,
                    "found": c.found,
                    "top_hits": [{"principle": h.principle, "score": h.score} for h in c.hits],
                }
                for c in result.cases
            ],
        }
        if comparison is not None:
            envelope["comparison"] = {
                "baseline_path": str(resolved_baseline_path),
                "regression_budget": regression_budget,
                "has_regression": comparison.has_regression(regression_budget=regression_budget),
                "aggregate_deltas": [
                    {"metric": d.metric, "old": d.old, "new": d.new}
                    for d in comparison.aggregate_deltas
                ],
                "case_regressions": [
                    {"id": d.id, "old_rank": d.old_rank, "new_rank": d.new_rank}
                    for d in comparison.case_regressions
                ],
                "case_improvements": [
                    {"id": d.id, "old_rank": d.old_rank, "new_rank": d.new_rank}
                    for d in comparison.case_improvements
                ],
                "new_cases": list(comparison.new_cases),
                "dropped_cases": list(comparison.dropped_cases),
            }
        if save_baseline:
            resolved_baseline_path.write_text(_json_mod.dumps(envelope, indent=2))
        console.print_json(data=envelope)
        if comparison is not None and comparison.has_regression(
            regression_budget=regression_budget
        ):
            raise typer.Exit(code=1)
        return

    from rich.table import Table

    table = Table(title=f"atc retrieval eval (k={result.k})", show_lines=False)
    table.add_column("pass", justify="center")
    table.add_column("id")
    table.add_column("expected")
    table.add_column("rank", justify="right")
    table.add_column("score", justify="right")
    for c in result.cases:
        mark = (
            "[green]✓[/green]"
            if c.recall_at(3)
            else ("[yellow]~[/yellow]" if c.found else "[red]✗[/red]")
        )
        rank = str(c.rank_of_first_expected) if c.rank_of_first_expected is not None else "—"
        score = f"{c.score_of_first_expected:.4f}" if c.score_of_first_expected is not None else "—"
        expected = ", ".join(c.expected_anchors)
        table.add_row(mark, c.id, expected, rank, score)
    console.print(table)
    console.print(
        f"[bold]recall@1: {result.recall_at_1 * 100:.1f}%  · "
        f"recall@3: {result.recall_at_3 * 100:.1f}%  · "
        f"recall@5: {result.recall_at_5 * 100:.1f}%  · "
        f"recall@{result.k}: {result.recall_at_k * 100:.1f}%[/bold]"
    )
    median = result.median_rank
    if median is not None:
        console.print(f"[dim]median rank of first expected (among found): {median:g}[/dim]")
    misses = result.hard_misses()
    if misses:
        console.print(
            f"[dim]hard misses (expected not in top-{result.k}): "
            f"{', '.join(m.id for m in misses)}[/dim]"
        )
    if save_baseline:
        envelope = {
            "character": character.name,
            "fixture": str(path),
            "k": result.k,
            "recall_at_1": result.recall_at_1,
            "recall_at_3": result.recall_at_3,
            "recall_at_5": result.recall_at_5,
            "recall_at_k": result.recall_at_k,
            "median_rank": result.median_rank,
            "cases": [
                {
                    "id": c.id,
                    "audience": c.audience,
                    "query": c.query,
                    "expected_anchors": list(c.expected_anchors),
                    "rank_of_first_expected": c.rank_of_first_expected,
                    "score_of_first_expected": c.score_of_first_expected,
                    "found": c.found,
                }
                for c in result.cases
            ],
        }
        resolved_baseline_path.write_text(_json_mod.dumps(envelope, indent=2))
        console.print(f"[dim]baseline written → {resolved_baseline_path}[/dim]")

    if comparison is not None:
        diff_table = Table(
            title=f"baseline diff (vs {resolved_baseline_path.name})",
            show_lines=False,
        )
        diff_table.add_column("metric")
        diff_table.add_column("old", justify="right")
        diff_table.add_column("new", justify="right")
        diff_table.add_column("Δ", justify="right")
        for agg in comparison.aggregate_deltas:
            delta = agg.new - agg.old
            color = "red" if delta < 0 else ("green" if delta > 0 else "dim")
            diff_table.add_row(
                agg.metric,
                f"{agg.old * 100:.1f}%",
                f"{agg.new * 100:.1f}%",
                f"[{color}]{delta * 100:+.1f} pp[/{color}]",
            )
        console.print(diff_table)

        if comparison.case_regressions:
            console.print(f"[red]case regressions ({len(comparison.case_regressions)}):[/red]")
            for case in comparison.case_regressions:
                old_s = "—" if case.old_rank is None else str(case.old_rank)
                new_s = "—" if case.new_rank is None else str(case.new_rank)
                console.print(f"  [red]✗[/red] {case.id}: rank {old_s} → {new_s}")
        if comparison.case_improvements:
            console.print(
                f"[green]case improvements ({len(comparison.case_improvements)}):[/green]"
            )
            for case in comparison.case_improvements:
                old_s = "—" if case.old_rank is None else str(case.old_rank)
                new_s = "—" if case.new_rank is None else str(case.new_rank)
                console.print(f"  [green]✓[/green] {case.id}: rank {old_s} → {new_s}")
        if comparison.new_cases:
            console.print(
                f"[dim]new cases (no baseline entry): {', '.join(comparison.new_cases)}[/dim]"
            )
        if comparison.dropped_cases:
            console.print(
                f"[dim]dropped cases (in baseline, not in run): "
                f"{', '.join(comparison.dropped_cases)}[/dim]"
            )

        if comparison.has_regression(regression_budget=regression_budget):
            console.print(
                f"[bold red]✗ regression detected[/bold red] (budget={regression_budget})"
            )
            raise typer.Exit(code=1)
        console.print("[bold green]✓ no regressions[/bold green]")


@eval_app.command("tool-loop")
def eval_tool_loop(
    fixture_path: Path | None = typer.Option(
        None,
        "--fixture",
        help="Path to a tool-loop eval YAML file. Defaults to "
        "`character/<name>/tool_loop_eval.yaml`.",
    ),
    attribute: bool = typer.Option(
        False,
        "--attribute",
        help="Also run the per-catcher attribution harness (disable each "
        "catcher in turn and report which scenarios it uniquely saves).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Replay the tool-loop failure corpus through scripted adapters and
    score contains / not_contains / events / messages assertions per
    scenario. With --attribute, also measure which orchestrator catcher
    uniquely saves which scenario."""
    from harness.evals.tool_loop import (
        default_fixture_path as _tl_default_fixture,
    )
    from harness.evals.tool_loop import (
        load_fixture as _tl_load,
    )
    from harness.evals.tool_loop import (
        run_attribution as _tl_attribution,
    )
    from harness.evals.tool_loop import (
        run_tool_loop_eval as _tl_run,
    )

    character = load_character(settings.character_path)
    path = fixture_path or _tl_default_fixture(settings.character_path)
    if not path.exists():
        raise typer.BadParameter(f"tool-loop eval fixture not found: {path}")
    fixtures = _tl_load(path)

    if attribute:
        attr = _tl_attribution(fixtures)
        baseline = attr.baseline
    else:
        baseline = _tl_run(fixtures)
        attr = None

    if as_json:
        payload: dict[str, object] = {
            "character": character.name,
            "pass_rate": baseline.pass_rate,
            "cases": [
                {
                    "id": c.id,
                    "label": c.label,
                    "passed": c.passed,
                    "rounds": c.rounds,
                    "missing_contains": list(c.missing_contains),
                    "unexpected_contains": list(c.unexpected_contains),
                    "missing_events": list(c.missing_events),
                    "unexpected_events": list(c.unexpected_events),
                    "missing_message_substrings": list(c.missing_message_substrings),
                    "unexpected_message_substrings": list(c.unexpected_message_substrings),
                    "expected_fallback": c.expected_fallback,
                    "fallback_triggered": c.fallback_triggered,
                }
                for c in baseline.cases
            ],
        }
        if attr is not None:
            payload["attributions"] = [
                {
                    "catcher": a.catcher,
                    "unique_saves": list(a.unique_saves),
                    "also_breaks": list(a.also_breaks),
                    "no_effect": a.no_effect,
                }
                for a in attr.attributions
            ]
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"Tool-loop eval — {character.name}", show_lines=False)
    table.add_column("✓", style="bold", width=2)
    table.add_column("id")
    table.add_column("label", style="cyan")
    table.add_column("rounds", style="dim", justify="right")
    table.add_column("failures", style="red")
    for c in baseline.cases:
        mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
        failures: list[str] = []
        if c.missing_contains:
            failures.append(f"missing: {list(c.missing_contains)}")
        if c.unexpected_contains:
            failures.append(f"unexpected: {list(c.unexpected_contains)}")
        if c.missing_events:
            failures.append(f"missing events: {list(c.missing_events)}")
        if c.unexpected_events:
            failures.append(f"unexpected events: {list(c.unexpected_events)}")
        if c.missing_message_substrings:
            failures.append(f"missing msg: {list(c.missing_message_substrings)}")
        if c.unexpected_message_substrings:
            failures.append(f"unexpected msg: {list(c.unexpected_message_substrings)}")
        if c.expected_fallback != c.fallback_triggered:
            failures.append(f"fallback expected={c.expected_fallback} got={c.fallback_triggered}")
        table.add_row(mark, c.id, c.label, str(c.rounds), " · ".join(failures))
    console.print(table)
    passed = sum(1 for c in baseline.cases if c.passed)
    console.print(
        f"[bold]{passed}/{len(baseline.cases)} passed · {baseline.pass_rate * 100:.1f}%[/bold]"
    )

    if attr is not None:
        attr_table = Table(title="Per-catcher attribution", show_lines=False, title_style="bold")
        attr_table.add_column("catcher", style="cyan")
        attr_table.add_column("uniquely saves", style="green")
        attr_table.add_column("shares coverage with (also_breaks)", style="dim")
        attr_table.add_column("no effect", style="red")
        for a in attr.attributions:
            attr_table.add_row(
                a.catcher,
                ", ".join(a.unique_saves) or "-",
                ", ".join(a.also_breaks) or "-",
                "yes" if a.no_effect else "",
            )
        console.print(attr_table)


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


def _parse_as_of(value: str) -> datetime:
    """Accept ISO 8601 ('2024-03-15T00:00:00+00:00') or plain date
    ('2024-03-15'). Plain dates are interpreted at UTC midnight so
    the temporal filter has a deterministic boundary regardless of
    the user's local time zone."""
    import typer

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"--as-of must be ISO 8601 or YYYY-MM-DD (got {value!r})") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@memory_app.command("fact-search")
def memory_fact_search(
    query: str = typer.Argument(..., help="Query text"),
    k: int = typer.Option(5, help="How many matches to return"),
    min_confidence: float = typer.Option(0.0, help="Minimum confidence to include"),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="Shift the temporal lens to this date (ISO 8601 or YYYY-MM-DD). "
        "Facts whose validity window doesn't contain this moment are filtered out. "
        "Defaults to now.",
    ),
) -> None:
    """Semantic-search the fact store."""
    store = _open_semantic_store()
    if store is None:
        raise typer.Exit(code=1)
    try:
        as_of_dt = _parse_as_of(as_of) if as_of is not None else None
        hits = store.search(query, k=k, min_confidence=min_confidence, as_of=as_of_dt)
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
    valid_from: str | None = typer.Option(
        None,
        "--valid-from",
        help="Earliest time the fact was true (ISO 8601 or YYYY-MM-DD). Unset = unbounded past.",
    ),
    valid_to: str | None = typer.Option(
        None,
        "--valid-to",
        help="Time after which the fact ceased to be true. Unset = still valid.",
    ),
    asserted_at: str | None = typer.Option(
        None,
        "--asserted-at",
        help="When the fact was communicated (vs. when the row was created). "
        "Useful for backfill: user describes something that happened last year.",
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
            valid_from=_parse_as_of(valid_from) if valid_from else None,
            valid_to=_parse_as_of(valid_to) if valid_to else None,
            asserted_at=_parse_as_of(asserted_at) if asserted_at else None,
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
    transcript = Transcript(settings.character_db_path)
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
    conn = sqlite3.connect(settings.character_db_path)
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


@memory_app.command("harvest-skills")
def memory_harvest_skills(
    status: str = typer.Option(
        "closed",
        "--status",
        help="bd status filter: 'closed' (default — decisions / observations "
        "that have resolved) or 'all' (include open thoughts; speculative, "
        "not recommended for routine runs).",
    ),
    labels: str | None = typer.Option(
        None,
        "--labels",
        help="Comma-separated list of thought labels to harvest. Defaults to "
        "'thought:decision,thought:observation'. Use this to run a one-off "
        "sweep over hypotheses / questions for diagnostic purposes.",
    ),
) -> None:
    """Harvest ab's bd thought-graph into the episodic store as
    tier='procedural' records so relevant decisions / observations
    surface on future user turns. Idempotent — re-running after new
    beads close picks up only the new ones."""
    from harness.skills import DEFAULT_HARVEST_LABELS, harvest_bd_skills

    character = load_character(settings.character_path)
    ab_adapter = _maybe_bd_adapter(character, include_internal=True)
    if ab_adapter is None:
        console.print(
            "[red]no bd adapter available — set HARNESS_AB_BD_DIR to ab's "
            "bd working directory first[/red]"
        )
        raise typer.Exit(code=1)
    store = _open_episodic_store(character)
    if store is None:
        console.print("[red]episodic store not enabled[/red]")
        raise typer.Exit(code=1)
    try:
        label_filter = (
            tuple(lbl.strip() for lbl in labels.split(",") if lbl.strip())
            if labels
            else DEFAULT_HARVEST_LABELS
        )
        report = harvest_bd_skills(
            ab_adapter=ab_adapter,
            episodic=store,
            labels=label_filter,
            status=status,
        )
        console.print(
            f"[bold]{report.newly_ingested}[/bold] new, "
            f"[bold]{report.already_present}[/bold] already present, "
            f"{report.scanned} matched filter "
            f"({', '.join(report.labels_filtered)}, status={report.status_filter})."
        )
        if report.ingested_ids:
            console.print(f"[dim]ingested: {', '.join(report.ingested_ids)}[/dim]")
    finally:
        store.close()


@memory_app.command("harvest-memories")
def memory_harvest_bd_memories() -> None:
    """Mirror bd's persistent memories (bd remember / retro record)
    into the episodic store as tier='procedural' records so the main
    retrieval path surfaces them on identity / biographical questions.
    Idempotent on external_id='bd-mem:<key>' — re-running after new
    `bd remember` calls picks up only the new keys (harness-9yd)."""
    from harness.skills import harvest_bd_memories

    character = load_character(settings.character_path)
    ab_adapter = _maybe_bd_adapter(character, include_internal=True)
    if ab_adapter is None:
        console.print(
            "[red]no bd adapter available — set HARNESS_AB_BD_DIR to ab's "
            "bd working directory first[/red]"
        )
        raise typer.Exit(code=1)
    store = _open_episodic_store(character)
    if store is None:
        console.print("[red]episodic store not enabled[/red]")
        raise typer.Exit(code=1)
    try:
        report = harvest_bd_memories(ab_adapter=ab_adapter, episodic=store)
        console.print(
            f"[bold]{report.newly_ingested}[/bold] new, "
            f"[bold]{report.already_present}[/bold] already present, "
            f"{report.scanned} total bd memorie(s) scanned."
        )
        if report.ingested_keys:
            console.print(f"[dim]ingested: {', '.join(report.ingested_keys)}[/dim]")
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
    transcript = Transcript(settings.character_db_path)
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
