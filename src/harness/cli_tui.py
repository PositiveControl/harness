"""TUI launch path for `harness chat --tui`.

Extracted from cli.chat() (harness-0n1r). Owns the branch that
builds the Textual ChatApp: resolves adapter + retriever + stores +
router + tool registry + compaction store, then runs the app and
closes the compaction store on exit.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from harness.character import load_character
from harness.compaction import CompactionStore
from harness.config import settings
from harness.router import GrammarRouter, ModelRouter
from harness.store.transcript import Transcript

if TYPE_CHECKING:
    from harness.router import Router
    from harness.tools import ToolRegistry


def run_tui(
    *,
    session: str,
    channel: str,
    speaker: str,
    model: str,
    model_repo: str | None,
    lora_path: str | None,
    persona: bool,
    top_k: int,
    memories: int,
    memories_threshold: float,
    facts: int,
    facts_threshold: float,
    tools: bool,
    tool_set: str,
    tools_add: str | None,
    tools_drop: str | None,
    workspace: str | None,
    compact_at: float,
    router_enabled: bool,
    router_repo: str,
    router_mode: str,
    include_internal: bool,
    dev: bool,
) -> None:
    from harness.cli import (
        _build_tool_registry_for_tui,
        _maybe_bd_adapter,
        _maybe_retriever,
        _open_episodic_store,
        _open_semantic_store,
        _resolve_adapter,
        _RetrievalState,
        _router_id_label,
    )

    try:
        from harness.tui import ChatApp
    except ImportError as exc:
        raise typer.BadParameter(
            "--tui requires the `tui` extra. Install it with: uv sync --extra tui"
        ) from exc

    character = load_character(settings.character_path)
    workspace_path = Path(workspace).expanduser().resolve() if workspace else settings.root
    if tools and not workspace_path.is_dir():
        raise typer.BadParameter(f"workspace {workspace_path} is not a directory")

    # Tools-active path runs persona *after* the loop in the classic REPL;
    # the TUI currently skips post-loop persona rewrite (the rewriter
    # tends to compress investigate-style replies). So persona goes to
    # _resolve_adapter only when tools are off.
    adapter = _resolve_adapter(
        model,
        persona=persona and not tools,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
    )
    retriever = _maybe_retriever(character, top_k)
    memory_store = _open_episodic_store(character) if memories > 0 else None
    semantic_store = _open_semantic_store() if facts > 0 else None
    transcript = Transcript(settings.character_db_path)

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

    registry_warnings: list[str] = []
    # Shared between ChatApp (which mutates it when retrieval raises)
    # and the introspect tool (which reads live status).
    retrieval_health = _RetrievalState()
    ab_adapter = _maybe_bd_adapter(character, include_internal=include_internal or dev)
    registry: ToolRegistry | None = _build_tool_registry_for_tui(
        tools=tools,
        tool_set=tool_set,
        tools_add=tools_add,
        tools_drop=tools_drop,
        workspace_path=workspace_path,
        memory_store=memory_store,
        semantic_store=semantic_store,
        speaker=speaker,
        session=session,
        adapter=adapter,
        character=character,
        retrieval_health=retrieval_health,
        persona_active=persona and not tools,
        router_id=_router_id_label(router),
        transcript=transcript,
        warnings_out=registry_warnings,
        include_internal=include_internal or dev,
        ab_adapter=ab_adapter,
        router=router,
    )

    compaction_store = CompactionStore(settings.character_db_path) if compact_at > 0 else None
    ChatApp(
        character=character,
        speaker=speaker,
        session=session,
        channel=channel,
        adapter=adapter,
        transcript=transcript,
        retriever=retriever,
        top_k=top_k,
        memory_store=memory_store,
        memories=memories,
        memories_threshold=memories_threshold,
        semantic_store=semantic_store,
        facts=facts,
        facts_threshold=facts_threshold,
        registry=registry,
        router=router,
        workspace_path=workspace_path,
        startup_warnings=tuple(registry_warnings),
        retrieval_health=retrieval_health,
        compaction_store=compaction_store,
        scribe_user_id=speaker,
        ab_adapter=ab_adapter,
    ).run()
    if compaction_store is not None:
        compaction_store.close()
