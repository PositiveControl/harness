from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.table import Table

import harness._quiet
from harness.character import load_character
from harness.config import settings
from harness.model import AdapterName, make_adapter
from harness.store.audit import AuditStore
from harness.store.bd_adapter import BeadsAdapter, BeadsAdapterError
from harness.store.transcript import Transcript, TranscriptMessage
from harness.tools import (
    DEFAULT_PROFILE,
    TOOL_PROFILES,
    CalcTool,
    DateMathTool,
    NowTool,
    PythonEvalTool,
    StatsTool,
    SunTool,
    ToolRegistry,
    TzConvertTool,
)

_EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit", ":q", ":quit"})
_EDIT_COMMANDS = frozenset({"/edit", "/capture"})
_RETRO_COMMANDS = frozenset({"/retro"})


def _build_recency_ranks() -> dict[str, int]:
    """One-shot build of the session_id → recency-rank map for the
    chat process. Read from the live transcript ordered by `last_at
    DESC`; stale within a process is fine because the only new
    session per chat run is the running one and it'd be rank 1
    anyway. NULL-session rows aren't in the dict — they skip the
    recency tier entirely (seeds + procedural + cross-session
    consolidated). Used by `_search_hybrid` when
    `HARNESS_RETRIEVAL_RECENCY_WEIGHT > 0`. harness-w3mo step 5."""
    from harness.store._hybrid import build_session_recency_ranks
    from harness.store.transcript import Transcript

    ts = Transcript(settings.character_db_path)
    try:
        sessions = ts.list_sessions()
    finally:
        ts.close()
    return build_session_recency_ranks([s.session for s in sessions])


def _resolve_memory_scope(
    *,
    memory_scope: str,
    session: str,
    window: int,
    transcript_db_path: Path,
) -> tuple[str, ...] | None:
    """Translate a `--memory-scope` value into the `allowed_sessions`
    tuple the stores expect. None = no filter (the 'all' default).
    Tuple = NULL OR session_id IN (...).

    'current-session' returns just the running session id; consolidated
    rows tagged with prior sessions stay reachable via the NULL=eligible
    rule only when the consolidator already attached this session — for
    cross-session consolidated rows, scope='recent' or 'all' is the
    answer. 'recent' needs the transcript to enumerate sessions by
    last activity; we open it briefly and close it. harness-w3mo."""
    if memory_scope == "all":
        return None
    if memory_scope == "current-session":
        return (session,)
    if memory_scope == "recent":
        from harness.store.transcript import Transcript

        ts = Transcript(transcript_db_path)
        try:
            rows = ts.list_sessions()
        finally:
            ts.close()
        # Most recent N by last activity. Always include the current
        # session even if no rows have been written yet (first turn
        # of a fresh session).
        recent = [r.session for r in rows[:window]]
        if session not in recent:
            recent.insert(0, session)
        return tuple(recent)
    raise typer.BadParameter(
        f"--memory-scope must be one of: all, current-session, recent. Got {memory_scope!r}."
    )


def _coin_session_id(*, now: datetime | None = None) -> str:
    """Mint a fresh, daily-rotated, launch-unique session id.

    Format: `cli-YYYY-MM-DD-HHMMSS` (UTC). Daily prefix groups
    same-day sessions in `harness session list`; the timestamp
    suffix makes every launch a clean break from the prior run's
    compaction summary + scribed memory. Used by `cmd_chat` when
    the user does not pass `--session`. `now` is parameterized so
    tests can pin the timestamp."""
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d-%H%M%S")
    return f"cli-{stamp}"


# `tool` / `plan` / `denylist` / `web` / `phraseology` handlers live in
# cli_misc.py (step 5d of docs/cli-extraction-plan.md). Imported for the
# registration side effect.
# `harness eval` handlers live in cli_eval.py (step 5c of
# docs/cli-extraction-plan.md). Imported for the registration side
# effect; nothing else in cli.py calls into them.
import harness.cli_eval  # noqa: E402
import harness.cli_misc  # noqa: E402,F401

# `harness memory` and `harness voice` handlers live in cli_memory.py /
# cli_voice.py (step 5b of docs/cli-extraction-plan.md). Imported for
# the registration side effect — the @memory_app.command() decorators
# run at import time — and re-exported for the call sites that address
# the helpers as harness.cli.<name>.
# The Typer app objects live in `cli_apps.py` (step 5a of
# docs/cli-extraction-plan.md) so handler modules can register against
# them without importing this module. Re-exported here: the
# `harness.cli:app` entry point and every existing
# `from harness.cli import app` call site keep working.
from harness.cli_apps import (  # noqa: E402
    app as app,
)
from harness.cli_apps import (  # noqa: E402
    denylist_app as denylist_app,
)
from harness.cli_apps import (  # noqa: E402
    drive_app as drive_app,
)
from harness.cli_apps import (  # noqa: E402
    eval_app as eval_app,
)
from harness.cli_apps import (  # noqa: E402
    memory_app as memory_app,
)
from harness.cli_apps import (  # noqa: E402
    phraseology_app as phraseology_app,
)
from harness.cli_apps import (  # noqa: E402
    plan_app as plan_app,
)
from harness.cli_apps import (  # noqa: E402
    session_app as session_app,
)
from harness.cli_apps import (  # noqa: E402
    tool_app as tool_app,
)
from harness.cli_apps import (  # noqa: E402
    voice_app as voice_app,
)
from harness.cli_apps import (  # noqa: E402
    web_app as web_app,
)
from harness.cli_memory import (  # noqa: E402
    _resolve_memory_target as _resolve_memory_target,
)
from harness.cli_voice import (  # noqa: E402
    _write_voice_capture as _write_voice_capture,
)

console = Console()


# Embedder + store construction lives in `cli_store.py` (step 1 of
# docs/cli-extraction-plan.md). Re-exported here because the sibling
# UIs and the subcommand handlers still reach for `harness.cli.<name>`;
# `cli_store` is the single owner of the process-wide embedder cache.
# Adapter composition lives in `cli_adapter.py` (step 6 of
# docs/cli-extraction-plan.md).
from harness.cli_adapter import (  # noqa: E402
    _resolve_adapter as _resolve_adapter,
)

# Tool-registry construction lives in `cli_tools.py` (step 4 of
# docs/cli-extraction-plan.md). Explicit re-exports, same reason as
# the two blocks above.
# bd adapter lifecycle lives in `cli_bd.py` (step 3 of
# docs/cli-extraction-plan.md). Explicit re-exports for the same
# reason as the cli_chat_shared block above.
from harness.cli_bd import (  # noqa: E402
    _maybe_ab_bd_adapter as _maybe_ab_bd_adapter,
)
from harness.cli_bd import (  # noqa: E402
    _maybe_bd_adapter as _maybe_bd_adapter,
)
from harness.cli_bd import (  # noqa: E402
    _maybe_harvest_bd_memories as _maybe_harvest_bd_memories,
)
from harness.cli_bd import (  # noqa: E402
    _maybe_harvest_skills as _maybe_harvest_skills,
)
from harness.cli_bd import (  # noqa: E402
    _print_session_end_retro as _print_session_end_retro,
)

# Chat-surface helpers live in `cli_chat_shared.py` (step 2 of
# docs/cli-extraction-plan.md). Explicit re-exports: the sibling UIs,
# the subcommand handlers and the tests all still address these as
# `harness.cli.<name>`, and mypy strict requires the `as` form to
# treat a re-export as public.
from harness.cli_chat_shared import (  # noqa: E402
    _TOOL_CALLS_SENTINEL as _TOOL_CALLS_SENTINEL,
)
from harness.cli_chat_shared import (  # noqa: E402
    _append_retrieval_error_log as _append_retrieval_error_log,
)
from harness.cli_chat_shared import (  # noqa: E402
    _decode_transcript_message as _decode_transcript_message,
)
from harness.cli_chat_shared import (  # noqa: E402
    _describe_call as _describe_call,
)
from harness.cli_chat_shared import (  # noqa: E402
    _encode_assistant_with_tool_calls as _encode_assistant_with_tool_calls,
)
from harness.cli_chat_shared import (  # noqa: E402
    _format_ctx_meter as _format_ctx_meter,
)
from harness.cli_chat_shared import (  # noqa: E402
    _is_suppressible as _is_suppressible,
)
from harness.cli_chat_shared import (  # noqa: E402
    _open_in_editor as _open_in_editor,
)
from harness.cli_chat_shared import (  # noqa: E402
    _persist_tool_exchange as _persist_tool_exchange,
)
from harness.cli_chat_shared import (  # noqa: E402
    _pre_validate_write_call as _pre_validate_write_call,
)
from harness.cli_chat_shared import (  # noqa: E402
    _render_ab_memories_block as _render_ab_memories_block,
)
from harness.cli_chat_shared import (  # noqa: E402
    _render_chat_header as _render_chat_header,
)
from harness.cli_chat_shared import (  # noqa: E402
    _render_fact_block as _render_fact_block,
)
from harness.cli_chat_shared import (  # noqa: E402
    _render_memory_block as _render_memory_block,
)
from harness.cli_chat_shared import (  # noqa: E402
    _render_tool_event as _render_tool_event,
)
from harness.cli_chat_shared import (  # noqa: E402
    _RetrievalState as _RetrievalState,
)
from harness.cli_chat_shared import (  # noqa: E402
    _retrieve_turn_context as _retrieve_turn_context,
)
from harness.cli_chat_shared import (  # noqa: E402
    _stream_or_complete as _stream_or_complete,
)
from harness.cli_chat_shared import (  # noqa: E402
    _StreamRenderer as _StreamRenderer,
)
from harness.cli_chat_shared import (  # noqa: E402
    _ThinkingSpinner as _ThinkingSpinner,
)
from harness.cli_chat_shared import (  # noqa: E402
    _topic_boundary_suffix as _topic_boundary_suffix,
)
from harness.cli_store import (  # noqa: E402
    _load_embedder as _load_embedder,
)
from harness.cli_store import (  # noqa: E402
    _maybe_retriever as _maybe_retriever,
)
from harness.cli_store import (  # noqa: E402
    _open_episodic_store as _open_episodic_store,
)
from harness.cli_store import (  # noqa: E402
    _open_semantic_store as _open_semantic_store,
)
from harness.cli_tools import (  # noqa: E402
    _ATC_LINT_SOURCE_FILTER as _ATC_LINT_SOURCE_FILTER,
)
from harness.cli_tools import (  # noqa: E402
    OPS_TOOL_NAMES as OPS_TOOL_NAMES,
)
from harness.cli_tools import (  # noqa: E402
    _ab_tool_builders as _ab_tool_builders,
)
from harness.cli_tools import (  # noqa: E402
    _build_assemble_context_tool as _build_assemble_context_tool,
)
from harness.cli_tools import (  # noqa: E402
    _build_document_tree_store_for_session as _build_document_tree_store_for_session,
)
from harness.cli_tools import (  # noqa: E402
    _build_tabular_store_for_session as _build_tabular_store_for_session,
)
from harness.cli_tools import (  # noqa: E402
    _build_tool_grounding_block as _build_tool_grounding_block,
)
from harness.cli_tools import (  # noqa: E402
    _build_tool_registry_for_tui as _build_tool_registry_for_tui,
)
from harness.cli_tools import (  # noqa: E402
    _hot_reload_synthesized as _hot_reload_synthesized,
)
from harness.cli_tools import (  # noqa: E402
    _make_introspect_tool as _make_introspect_tool,
)
from harness.cli_tools import (  # noqa: E402
    _missing_builder_reason as _missing_builder_reason,
)
from harness.cli_tools import (  # noqa: E402
    _open_fetch_denylist as _open_fetch_denylist,
)
from harness.cli_tools import (  # noqa: E402
    _resolve_catalog_path as _resolve_catalog_path,
)
from harness.cli_tools import (  # noqa: E402
    _resolve_router_tool_specs as _resolve_router_tool_specs,
)
from harness.cli_tools import (  # noqa: E402
    _router_id_label as _router_id_label,
)
from harness.cli_tools import (  # noqa: E402
    _session_tool_catalog as _session_tool_catalog,
)


def _open_audit_store() -> AuditStore:
    """Open the per-turn audit log on the character's shared SQLite.
    Always returns a live store — the audit log has no embedder
    dependency and the feature is on for every chat session
    (harness-ywp.2). Callers close() at session teardown."""
    return AuditStore(settings.character_db_path)


@app.command()
def chat(
    session: str | None = typer.Option(
        None,
        help=(
            "Session identifier. When omitted, every launch coins a "
            "fresh id of the form `cli-YYYY-MM-DD-HHMMSS` (UTC) so a "
            "new chat invocation never inherits the prior run's "
            "compaction summary or scribed memory. Pass an explicit "
            "name (e.g. --session local) to resume a prior session."
        ),
    ),
    channel: str = typer.Option("cli", help="Channel name"),
    speaker: str = typer.Option("mark", help="Your handle"),
    model: str = typer.Option("echo", help="Adapter: echo | mlx | ollama | vllm"),
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
    memory_scope: str = typer.Option(
        "all",
        "--memory-scope",
        help=(
            "Session-scope filter on episodic + semantic retrieval. "
            "'all' = no filter (default — every shared / per-user row "
            "the user can see is eligible). 'current-session' = only "
            "rows tagged with this chat's session id, plus untagged "
            "(seeds, procedural, consolidated cross-session) rows. "
            "'recent' = the last --memory-scope-window sessions by "
            "last activity, plus untagged rows. Use current-session "
            "after a /clear when you want the next turn to learn "
            "from this conversation only. harness-w3mo."
        ),
    ),
    memory_scope_window: int = typer.Option(
        3,
        "--memory-scope-window",
        help=(
            "How many recent sessions count as 'recent' under "
            "--memory-scope=recent. Includes the current session. "
            "Ignored for other scope values."
        ),
        min=1,
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
        "uv sync --extra all.",
    ),
) -> None:
    """CLI chat loop. Swap model runtimes with --model."""
    if session is None:
        session = _coin_session_id()
        console.print(f"[dim]session: {session} (auto)[/dim]")
    allowed_sessions = _resolve_memory_scope(
        memory_scope=memory_scope,
        session=session,
        window=memory_scope_window,
        transcript_db_path=settings.character_db_path,
    )
    if allowed_sessions is not None:
        console.print(
            f"[dim]memory-scope: {memory_scope} → "
            f"{len(allowed_sessions)} session(s) + untagged[/dim]"
        )
    # Recency-RRF gate (harness-w3mo step 5). Build the session_id →
    # rank map once at chat boot when the env-driven weight is
    # positive; keep it None otherwise so the store skips the third
    # ranking entirely.
    recency_weight = settings.retrieval_recency_weight
    recency_ranks: dict[str, int] | None = None
    if recency_weight > 0.0:
        recency_ranks = _build_recency_ranks()
        console.print(
            f"[dim]retrieval-recency: weight={recency_weight:.2f} "
            f"over {len(recency_ranks)} session(s)[/dim]"
        )
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
            compact_keep_recent=compact_keep_recent,
            auto_scribe=auto_scribe,
            router_enabled=router_enabled,
            router_repo=router_repo,
            router_mode=router_mode,
            include_internal=include_internal,
            dev=dev,
            allowed_sessions=allowed_sessions,
            recency_ranks=recency_ranks,
            recency_weight=recency_weight,
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
        allowed_sessions=allowed_sessions,
        recency_ranks=recency_ranks,
        recency_weight=recency_weight,
    )
    return


@app.command()
def daemon(
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character to run the daemon for. Default: HARNESS_CHARACTER_NAME or 'airton'.",
    ),
    interval_default: float = typer.Option(
        60.0,
        "--interval-default",
        help="Default heartbeat interval in seconds. Registered tasks may override.",
    ),
    compaction_interval: float = typer.Option(
        600.0,
        "--compaction-interval",
        help="Periodic-compaction tick interval in seconds. 0 disables the task.",
    ),
    compaction_model: str = typer.Option(
        "echo",
        "--compaction-model",
        help="Adapter for compaction summarization: echo | mlx | ollama | vllm. "
        "Defaults to echo so the daemon stays cheap until you opt in to a real model.",
    ),
    consolidation_interval: float = typer.Option(
        3600.0,
        "--consolidation-interval",
        help="Periodic-consolidation tick interval in seconds. 0 disables the task.",
    ),
    consolidation_min_working: int = typer.Option(
        5,
        "--consolidation-min-working",
        help="Skip the consolidator unless the working tier has at least this many "
        "episodic records (avoids wasted scans on empty stores).",
    ),
    drift_interval: float = typer.Option(
        3600.0,
        "--drift-interval",
        help="Periodic bd-drift-check tick interval in seconds. 0 disables the task.",
    ),
    drift_assignee: str = typer.Option(
        "mark",
        "--drift-assignee",
        help="Bd assignee whose in_progress work the drift check inspects "
        "(use 'airton_b' for the ab data-plane daemon).",
    ),
    drift_max_in_progress: int = typer.Option(
        3,
        "--drift-max-in-progress",
        help="Overload threshold: flag drift when in_progress count for the "
        "watched assignee exceeds this.",
    ),
    drift_stale_days: float = typer.Option(
        7.0,
        "--drift-stale-days",
        help="Staleness threshold: flag in_progress beads with no updated_at "
        "activity within this many days.",
    ),
    schedule_interval: float = typer.Option(
        300.0,
        "--schedule-interval",
        help="Periodic scheduled-tool-calls tick interval in seconds. 0 disables the task.",
    ),
    schedule_path: Path | None = typer.Option(
        None,
        "--schedule-path",
        help="YAML file with scheduled tool calls. "
        "Default: <character>/data/heartbeat_schedule.yaml.",
    ),
    plan_revision_interval: float = typer.Option(
        0.0,
        "--plan-revision-interval",
        help="Periodic plan-revision tick interval in seconds. 0 disables the task "
        "(default: off — run `harness plan bootstrap` first, then opt in).",
    ),
    plan_revision_id: str = typer.Option(
        "bd:mark",
        "--plan-revision-id",
        help="Plan id the revision task loads + saves. Matches the bootstrap "
        "default 'bd:<assignee>'.",
    ),
    plan_revision_assignee: str = typer.Option(
        "mark",
        "--plan-revision-assignee",
        help="Bd assignee whose open/closed bead sets feed the WorldSnapshot.",
    ),
    plan_revision_writeback: bool = typer.Option(
        False,
        "--plan-revision-writeback",
        help="Apply bd writeback after each revision (active→achieved closes the "
        "bead, etc.). Default: off — watch a few ticks first.",
    ),
    plans_dir: Path | None = typer.Option(
        None,
        "--plans-dir",
        help="Directory the plan-revision task reads PlanStore from. "
        "Default: <character>/data/plans/.",
    ),
    state_path: Path | None = typer.Option(
        None,
        "--state-path",
        help="JSON sidecar where heartbeat state (last-fire timestamps, error counts, "
        "quarantine flags) is persisted. Default: <character>/data/heartbeat_state.json. "
        "Survives daemon restarts so a quarantined task stays quarantined.",
    ),
    grace_period: float = typer.Option(
        10.0,
        "--grace-period",
        help="Seconds to wait for the loop to exit cleanly after SIGINT/SIGTERM "
        "before force-exiting. A task hung past this is hard-killed.",
    ),
    tick_once: bool = typer.Option(
        False,
        "--tick-once",
        help="Fire each registered task once and exit. For integration tests.",
    ),
) -> None:
    """Start the heartbeat daemon — harness-swvf.

    Built-in tasks (registered when their interval > 0):
      * heartbeat_alive  — logs an ISO timestamp every tick (sanity).
      * compaction       — scans recently-active sessions and folds
                            ones over threshold (harness-klwg).

    Real consolidation / drift / scheduled-tool-call tasks land in
    separate sub-beads (srus / c32m / 6dnf).
    """
    import asyncio
    import signal

    from harness.runtime import Heartbeat
    from harness.runtime.tasks import (
        build_compaction_task,
        build_consolidation_task,
        build_drift_task,
        build_plan_revision_task,
        build_scheduled_tools_task,
    )

    char_path = settings.character_path
    if character:
        char_path = char_path.parent / character
    char = load_character(char_path)

    def _err(name: str, exc: BaseException) -> None:
        console.print(f"[red]heartbeat error[/red] [{name}]: {exc!r}")

    # State path defaults to <character>/data/heartbeat_state.json,
    # alongside the SQLite stores. Override via --state-path.
    resolved_state_path = state_path or (char_path / "data" / "heartbeat_state.json")
    hb = Heartbeat(on_error=_err, state_path=resolved_state_path)

    def heartbeat_alive() -> None:
        ts = datetime.now(UTC).isoformat(timespec="seconds")
        console.print(f"[dim]heartbeat[/dim] tick {ts}")

    hb.register("heartbeat_alive", heartbeat_alive, interval_s=interval_default)

    # Compaction task — opt-in via --compaction-interval > 0. Builds
    # the adapter once at daemon startup so subsequent ticks reuse it.
    if compaction_interval > 0:
        from harness.compaction import CompactionStore

        transcript = Transcript(settings.character_db_path)
        compaction_store = CompactionStore(settings.character_db_path)
        compaction_adapter = make_adapter(cast("AdapterName", compaction_model))

        def _compaction_sink(outcome: object) -> None:
            console.print(f"[dim]heartbeat[/dim] compaction {outcome!r}")

        hb.register(
            "compaction",
            build_compaction_task(
                adapter=compaction_adapter,
                transcript=transcript,
                compaction_store=compaction_store,
                sink=_compaction_sink,
            ),
            interval_s=compaction_interval,
        )

    # Scheduled-tool-calls task — opt-in via --schedule-interval > 0.
    # Builds a minimal read-tier registry (the reckon primitives) and
    # routes the entry's tool+args through it. Schedule file defaults
    # to <character>/data/heartbeat_schedule.yaml; state sidecar lives
    # next to it. A missing schedule file is fine (empty schedule);
    # malformed schedule surfaces via the outcome's load_error field.
    if schedule_interval > 0:
        from harness.runtime.tasks.scheduled_tools import ScheduledToolsTaskOutcome

        sched_data_dir = char_path / "data"
        sched_yaml = schedule_path or (sched_data_dir / "heartbeat_schedule.yaml")
        # Co-locate state with the schedule file so a --schedule-path
        # override produces a self-contained pair on disk (test
        # isolation depends on this — a tmp_path schedule shouldn't
        # consult the character-default state file).
        sched_state = sched_yaml.parent / (sched_yaml.stem + "_state.json")

        # Read-tier reckon registry only — daemon mode should never
        # exercise write-tier tools without the per-session confirm UX.
        sched_registry = ToolRegistry()
        for tool in (
            NowTool(),
            DateMathTool(),
            CalcTool(),
            PythonEvalTool(),
            TzConvertTool(),
            StatsTool(),
            SunTool(),
        ):
            sched_registry.register(tool)

        def _schedule_executor(tool_name: str, args: dict[str, object]) -> str:
            result = sched_registry.call(tool_name, args)
            return result.output

        def _schedule_sink(outcome: ScheduledToolsTaskOutcome) -> None:
            console.print(f"[dim]heartbeat[/dim] schedule {outcome!r}")

        hb.register(
            "schedule",
            build_scheduled_tools_task(
                schedule_path=sched_yaml,
                state_path=sched_state,
                tool_executor=_schedule_executor,
                sink=_schedule_sink,
            ),
            interval_s=schedule_interval,
        )

    # Drift task — bd-based heuristics; opt-in via --drift-interval > 0.
    # Constructs a fresh BeadsAdapter pointing at the character's bd
    # directory. If bd isn't initialized in that dir (no .beads),
    # disable drift with a console warning instead of crashing.
    if drift_interval > 0:
        drift_bd_dir = settings.bd_dir_for(char.name)
        drift_adapter = BeadsAdapter(
            drift_bd_dir,
            default_exclude_assignee=char.bd_exclude_assignee,
            ab_assignee=char.bd_assignee,
        )
        try:
            drift_adapter.verify()
        except BeadsAdapterError as exc:
            console.print(
                f"[yellow]heartbeat[/yellow] drift disabled: bd not "
                f"available at {drift_bd_dir} ({exc})"
            )
        else:

            def _drift_sink(outcome: object) -> None:
                console.print(f"[dim]heartbeat[/dim] drift {outcome!r}")

            hb.register(
                "drift",
                build_drift_task(
                    bd_adapter=drift_adapter,
                    assignee=drift_assignee,
                    max_in_progress=drift_max_in_progress,
                    stale_after_days=drift_stale_days,
                    sink=_drift_sink,
                ),
                interval_s=drift_interval,
            )

    # Consolidation task — opt-in via --consolidation-interval > 0.
    # Uses the character's existing episodic + semantic stores (same
    # DB the chat loop writes to). Embedder loaded once via the shared
    # helper so multiple ticks don't reinstantiate it.
    if consolidation_interval > 0:
        ep_store = _open_episodic_store(character=char, ingest=False)
        sem_store = _open_semantic_store()
        if ep_store is None or sem_store is None:
            console.print(
                "[yellow]heartbeat[/yellow] consolidation disabled: "
                "stores unavailable (embedder load failed?)"
            )
        else:

            def _consolidation_sink(outcome: object) -> None:
                console.print(f"[dim]heartbeat[/dim] consolidation {outcome!r}")

            hb.register(
                "consolidation",
                build_consolidation_task(
                    episodic_store=ep_store,
                    semantic_store=sem_store,
                    min_working_records=consolidation_min_working,
                    sink=_consolidation_sink,
                ),
                interval_s=consolidation_interval,
            )

    # Plan-revision task — opt-in via --plan-revision-interval > 0.
    # Loads a Plan, builds a WorldSnapshot from bd, advances subgoal
    # statuses, saves. Optional bd writeback (off by default — operator
    # opts in after watching a few ticks). Requires `harness plan
    # bootstrap` to have run first; missing plan logs cleanly and tick
    # is a no-op (harness-rbj9).
    if plan_revision_interval > 0:
        from harness.plan import JsonPlanStore

        plan_revision_dir = plans_dir or (char_path / "data" / "plans")
        plan_revision_store = JsonPlanStore(plan_revision_dir)
        plan_revision_adapter = BeadsAdapter(
            settings.bd_dir_for(char.name),
            default_exclude_assignee=char.bd_exclude_assignee,
            ab_assignee=char.bd_assignee,
        )
        try:
            plan_revision_adapter.verify()
        except BeadsAdapterError as exc:
            console.print(
                f"[yellow]heartbeat[/yellow] plan-revision disabled: bd not available ({exc})"
            )
        else:

            def _plan_revision_sink(outcome: object) -> None:
                console.print(f"[dim]heartbeat[/dim] plan-revision {outcome!r}")

            hb.register(
                "plan_revision",
                build_plan_revision_task(
                    plan_store=plan_revision_store,
                    plan_id=plan_revision_id,
                    bd_adapter=plan_revision_adapter,
                    assignee=plan_revision_assignee,
                    apply_bd_writeback=plan_revision_writeback,
                    sink=_plan_revision_sink,
                ),
                interval_s=plan_revision_interval,
            )

    # Restore persisted state from prior daemon runs (quarantine flags,
    # last-fire timestamps). Idempotent — no file = first launch.
    hb.restore_from_disk()
    if hb.quarantined_tasks():
        console.print(
            f"[yellow]heartbeat[/yellow] resumed with quarantined tasks: "
            f"{', '.join(hb.quarantined_tasks())}. Inspect via 'harness daemon-status'."
        )

    console.print(
        f"[bold]heartbeat[/bold] daemon starting for {char.name} "
        f"(interval_default={interval_default}s, "
        f"{len(hb.names())} task(s) registered, "
        f"state_path={resolved_state_path})"
    )

    if tick_once:
        asyncio.run(hb.tick_once())
        console.print("[bold]heartbeat[/bold] tick-once complete")
        return

    import os as _os

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown() -> None:
        hb.stop()

        def _force_exit() -> None:
            console.print(
                f"[yellow]heartbeat[/yellow] grace period {grace_period}s expired — forcing exit"
            )
            _os._exit(1)

        loop.call_later(grace_period, _force_exit)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)
    try:
        loop.run_until_complete(hb.run_forever())
    finally:
        loop.close()
    console.print("[bold]heartbeat[/bold] daemon stopped.")


@app.command("daemon-status")
def daemon_status(
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose daemon state to inspect. "
        "Default: HARNESS_CHARACTER_NAME or 'airton'.",
    ),
    state_path: Path | None = typer.Option(
        None,
        "--state-path",
        help="Override path to the heartbeat state JSON. "
        "Default: <character>/data/heartbeat_state.json.",
    ),
) -> None:
    """Read the heartbeat state sidecar and print a table of per-task
    last-success / last-error / consecutive_errors / quarantined flags.

    Use this to verify the daemon is alive and inspect any failing
    tasks without tailing logs. The state file is written by the
    running daemon on every tick (harness-m64i)."""
    from harness.runtime.state import load_state

    char_path = settings.character_path
    if character:
        char_path = char_path.parent / character
    char = load_character(char_path)
    resolved_path = state_path or (char_path / "data" / "heartbeat_state.json")

    if not resolved_path.exists():
        console.print(
            f"[yellow]no heartbeat state file at {resolved_path}[/yellow] — "
            f"daemon may not have run for character {char.name!r} yet."
        )
        return

    state = load_state(resolved_path)
    console.print(f"[bold]heartbeat state[/bold] for {char.name} ({resolved_path})")
    if not state.tasks:
        console.print("  (no tasks recorded yet)")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("task")
    table.add_column("last_success")
    table.add_column("last_error")
    table.add_column("errs", justify="right")
    table.add_column("quarantined")
    for name in sorted(state.tasks):
        rec = state.tasks[name]
        table.add_row(
            name,
            rec.last_success_at or "(never)",
            rec.last_error_at or "(none)",
            str(rec.consecutive_errors),
            "yes" if rec.quarantined else "no",
        )
    console.print(table)


# --- harness tool ... ---------------------------------------------------


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


def _emit_session_message(msg: TranscriptMessage, *, json_out: bool) -> None:
    """Emit one transcript row to stdout in markdown (default) or
    JSONL form. Plain `print()` keeps output paste-friendly when
    redirected to a file or piped into Claude Code — no ANSI."""
    if json_out:
        print(
            json.dumps(
                {
                    "id": msg.id,
                    "session": msg.session,
                    "channel": msg.channel,
                    "speaker": msg.speaker,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                }
            )
        )
        return
    ts = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
    print(f"### {msg.role} ({msg.speaker}) — {ts}")
    print()
    print(msg.content)
    print()


@session_app.command("list")
def session_list_cmd() -> None:
    """List every recorded chat session, newest first."""
    transcript = Transcript(settings.character_db_path)
    try:
        rows = transcript.list_sessions()
        if not rows:
            console.print("[dim](no sessions yet)[/dim]")
            return
        table = Table(title=f"Sessions ({len(rows)})", show_lines=False)
        table.add_column("session", style="bold")
        table.add_column("channel")
        table.add_column("turns", justify="right")
        table.add_column("user", justify="right")
        table.add_column("airton", justify="right")
        table.add_column("first")
        table.add_column("last")
        for r in rows:
            table.add_row(
                r.session,
                r.channel,
                str(r.total_rows),
                str(r.user_turns),
                str(r.assistant_turns),
                r.first_at.strftime("%Y-%m-%d %H:%M"),
                r.last_at.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(table)
    finally:
        transcript.close()


@session_app.command("show")
def session_show_cmd(
    session_id: str | None = typer.Argument(
        None,
        help="Session id to dump. Defaults to most recent.",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Tail the session — keep printing turns as they land.",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit JSONL (one message per line) instead of markdown.",
    ),
    poll_interval: float = typer.Option(
        0.5,
        "--poll",
        help="Seconds between fetches in --follow mode.",
        min=0.1,
    ),
) -> None:
    """Dump a session as paste-ready markdown (default) or JSONL.

    Defaults to the most recent session — pass an id to pick another.
    Use --follow for live streaming as Airton replies."""
    transcript = Transcript(settings.character_db_path)
    try:
        if session_id is None:
            sessions = transcript.list_sessions()
            if not sessions:
                typer.echo("(no sessions yet)", err=True)
                raise typer.Exit(code=1)
            session_id = sessions[0].session
            typer.echo(f"session: {session_id}", err=True)

        rows = transcript.fetch_after(session_id, after_id=0)
        if not rows and not follow:
            typer.echo(f"(session {session_id!r} has no turns)", err=True)
            return
        last_id = 0
        for msg in rows:
            _emit_session_message(msg, json_out=json_out)
            last_id = msg.id
        if not follow:
            return

        typer.echo(f"-- following {session_id} (Ctrl+C to stop) --", err=True)
        try:
            while True:
                time.sleep(poll_interval)
                fresh = transcript.fetch_after(session_id, after_id=last_id)
                for msg in fresh:
                    _emit_session_message(msg, json_out=json_out)
                    last_id = msg.id
        except KeyboardInterrupt:
            return
    finally:
        transcript.close()


@session_app.command("reset")
def session_reset_cmd(
    session_id: str = typer.Argument(
        ...,
        help="Session id to reset.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    """Full reset for one session: drop the compaction summary, set
    a persistent /clear watermark at the current transcript tip,
    and prune the working-tier scribed memory written from this
    session.

    The transcript table is preserved for audit + scribe re-runs +
    `harness session show`. Shared seeds, consolidated memory,
    other sessions' working-tier rows, and the procedural bd-harvest
    tier are untouched (harness-k7m9).

    Use `session reset` when a session has accumulated weeks of
    mixed-topic context and you want a clean slate without losing
    the raw transcript. For a lighter touch — drop the summary
    only, keep raw history visible — use `session compact-reset`.
    For an in-process cut — same effect but only for the running
    chat — use `/clear` inside chat (harness-rrkj makes that cut
    durable too)."""
    from harness.compaction import CompactionStore
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore

    transcript = Transcript(settings.character_db_path)
    compaction = CompactionStore(settings.character_db_path)
    embedder = _load_embedder()
    if embedder is None:
        # Without embeddings the stores can still open and DELETE,
        # but skipping construction here avoids loading the model
        # for a pure delete pass — pass None and let each store's
        # init lazily error if a downstream call needs vectors.
        episodic = EpisodicStore(settings.character_db_path, embedder=None)  # type: ignore[arg-type]
        semantic = SemanticStore(settings.character_db_path, embedder=None)  # type: ignore[arg-type]
    else:
        episodic = EpisodicStore(settings.character_db_path, embedder=embedder)  # type: ignore[arg-type]
        semantic = SemanticStore(settings.character_db_path, embedder=embedder)  # type: ignore[arg-type]
    try:
        tail = transcript.tail(session_id, limit=1)
        tip = tail[-1].id if tail else 0
        existing_summary = compaction.latest_for_session(session_id)
        if not yes:
            covered = (
                f"{existing_summary.covered_turns} folded turns + "
                if existing_summary is not None
                else ""
            )
            typer.confirm(
                f"Reset session {session_id!r}? Drops {covered}"
                f"compaction summary, sets /clear watermark at row {tip}, "
                f"and removes working-tier scribed memory tagged to this "
                f"session. Transcript stays.",
                abort=True,
            )
        summaries_dropped = compaction.invalidate_summaries(session_id)
        compaction.record_clear(session_id=session_id, after_id=tip)
        episodic_dropped = episodic.delete_working_for_session(session_id)
        semantic_dropped = semantic.delete_working_for_session(session_id)
        console.print(
            f"[yellow]reset session {session_id!r}: "
            f"summary={summaries_dropped} "
            f"clear_after_id={tip} "
            f"episodic_working={episodic_dropped} "
            f"semantic_working={semantic_dropped}[/yellow]"
        )
    finally:
        episodic.close()
        semantic.close()
        compaction.close()
        transcript.close()


@session_app.command("compact-reset")
def session_compact_reset_cmd(
    session_id: str = typer.Argument(
        ...,
        help="Session id whose compaction summary should be invalidated.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt.",
    ),
) -> None:
    """Drop the compaction summary for a session so future chat
    invocations don't reattach folded-turn context.

    Compaction summaries are otherwise permanent: once turns fold,
    every future load_history call prepends 'Earlier conversation:
    [old threads]' even on totally unrelated topics. This command
    forces load_history to fall through to the raw-transcript tail
    path. The transcript itself is untouched — scribe + retro +
    `harness session show` still see every turn (harness-xf8d).

    Pairs with `harness-rrkj` (durable /clear): `/clear` cuts BOTH
    the summary and the raw history; `compact-reset` only drops
    the summary, keeping the raw history available for the next
    chat to roll forward."""
    from harness.compaction import CompactionStore

    store = CompactionStore(settings.character_db_path)
    try:
        existing = store.latest_for_session(session_id)
        if existing is None:
            typer.echo(f"(session {session_id!r} has no compaction summary)", err=True)
            return
        if not yes:
            typer.confirm(
                f"Drop compaction summary for session {session_id!r} "
                f"({existing.covered_turns} turns folded)? Transcript stays.",
                abort=True,
            )
        deleted = store.invalidate_summaries(session_id)
        console.print(
            f"[yellow]dropped {deleted} compaction summary "
            f"row(s) for session {session_id!r}.[/yellow]"
        )
    finally:
        store.close()


if __name__ == "__main__":
    app()
