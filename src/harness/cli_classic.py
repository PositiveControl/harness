"""Classic REPL driver for `harness chat` (harness-0n1r).

Extracted from cli.chat() to separate the classic-mode turn loop
from typer arg-parsing + TUI-mode dispatch. Entry point:
`run_classic_chat(...)` owns setup, REPL loop, and teardown.

`ClassicChatSession` bundles all per-session state the turn body
used to capture as closures (adapter, character, stores, registry,
router, retrieval, approved-tools set, renderers). Methods on it
replace the nested defs in the old chat() function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markdown import Markdown

from harness.character import Character, load_character
from harness.cli_repl import (
    ContextMeter,
    handle_clear_slash,
    handle_edit_slash,
    handle_retro_slash,
)
from harness.compaction import CompactionStore
from harness.config import settings
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.orchestrator import ToolLoopEvent, run_tool_loop
from harness.persona.rewriter import build_rewriter_messages
from harness.retrieval import VoiceRetriever
from harness.router import GrammarRouter, ModelRouter, Router
from harness.store import EpisodicStore, SemanticStore
from harness.store.bd_adapter import BeadsAdapter
from harness.store.transcript import Transcript
from harness.tools import (
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
    WriteFileTool,
    resolve_tool_names,
)
from harness.tools.ab_ops import build_resume_summary

if TYPE_CHECKING:
    from harness.cli import _RetrievalState, _StreamRenderer, _ThinkingSpinner
    from harness.orchestrator.hooks import HookPipeline


_EXIT_COMMANDS = frozenset({"/exit", "/quit", ":q"})
_RETRO_COMMANDS = frozenset({"/retro"})
_EDIT_COMMANDS = frozenset({"/edit", "/capture"})
_CLEAR_COMMANDS = frozenset({"/clear"})


def _build_hook_pipeline(
    *,
    summarize_tool_results: bool,
    router: Router | None,
    router_repo: str,
    console: Console,
) -> HookPipeline | None:
    """Return a HookPipeline override when the summarizer is requested,
    else None so the orchestrator uses its module default.

    The summarizer reuses the router's adapter when available — it's
    a small MLX model already loaded into the process. Otherwise we
    build a fresh MLX adapter from `router_repo`. Either way the
    extra model cost is bounded (~1 GB for a 3B-4bit router)."""
    if not summarize_tool_results:
        return None
    from harness.orchestrator.hooks import (
        ToolResultSummarizerHook,
        default_hook_pipeline,
    )

    summarizer_adapter: Any
    if router is not None and hasattr(router, "adapter"):
        summarizer_adapter = router.adapter  # duck-typed; ModelRouter + GrammarRouter both carry it
        console.print(
            f"[dim]tool-result summarizer: reusing router adapter ({router_repo})[/dim]"
        )
    else:
        from harness.model.mlx import MLXAdapter

        summarizer_adapter = MLXAdapter(repo=router_repo)
        console.print(
            f"[dim]tool-result summarizer: loading {router_repo} (first turn is slower)[/dim]"
        )

    pipeline = default_hook_pipeline()
    pipeline.post_tool.append(ToolResultSummarizerHook(summarizer=summarizer_adapter))
    return pipeline


def build_classic_registry(
    *,
    tools: bool,
    tool_set: str,
    tools_add: str | None,
    tools_drop: str | None,
    workspace_path: Path,
    adapter: ModelAdapter,
    character: Character,
    transcript: Transcript,
    memory_store: EpisodicStore | None,
    semantic_store: SemanticStore | None,
    speaker: str,
    session: str,
    ab_adapter: BeadsAdapter | None,
    console: Console,
    retrieval_state: _RetrievalState,
    persona: bool,
    router: Router | None,
) -> ToolRegistry | None:
    """Assemble the classic-mode tool registry. Returns None when
    --no-tools or the resolved profile comes back empty."""
    import typer

    from harness.cli import (
        _ab_tool_builders,
        _make_introspect_tool,
        _missing_builder_reason,
        _router_id_label,
    )

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
    builders.update(_ab_tool_builders(ab_adapter))

    # `introspect` and `spawn_subagent` are meta-tools that reflect on
    # the already-populated registry, so they get registered after the
    # first pass. Listing their names here keeps the deferral logic in
    # one place and mirrors the TUI builder.
    _deferred = {"introspect", "spawn_subagent"}

    registry = ToolRegistry()
    for name in wanted_names:
        if name in _deferred:
            continue
        builder = builders.get(name)
        if builder is None:
            console.print(f"[yellow]⚠ {_missing_builder_reason(name, character)}[/yellow]")
            continue
        tool = builder()
        if tool is None:
            console.print(
                f"[yellow]⚠ tool {name!r} needs a store that isn't enabled "
                f"(check --memories / --facts)[/yellow]"
            )
            continue
        registry.register(tool)

    if "introspect" in wanted_names:
        registry.register(
            _make_introspect_tool(
                registry,
                adapter,
                character,
                workspace_path,
                episodic=memory_store,
                semantic=semantic_store,
                user_id=speaker,
                retrieval_health=retrieval_state,
                persona_active=persona and not tools,
                router_id=_router_id_label(router),
                transcript=transcript,
                session_id=session,
            )
        )

    if "spawn_subagent" in wanted_names:
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
    return registry


@dataclass
class ClassicChatSession:
    """Per-session state for the classic REPL. Consolidates the
    closures that used to live inside cli.chat()."""

    character: Character
    adapter: ModelAdapter
    console: Console
    transcript: Transcript
    workspace_path: Path
    retrieval_state: _RetrievalState
    thinking: _ThinkingSpinner
    stream_renderer: _StreamRenderer
    speaker: str
    session: str
    channel: str
    persona: bool
    rewrite_on_tools: bool
    top_k: int
    memories: int
    memories_threshold: float
    facts: int
    facts_threshold: float
    retriever: VoiceRetriever | None
    memory_store: EpisodicStore | None
    semantic_store: SemanticStore | None
    registry: ToolRegistry | None
    router: Router | None
    ab_adapter: BeadsAdapter | None
    ctx_meter: ContextMeter
    # Optional HookPipeline override. None means use the orchestrator's
    # module-default pipeline; the CLI injects a custom pipeline when
    # --summarize-tool-results is set (adds a post_tool summarizer).
    hooks: HookPipeline | None = None
    approved_tools: set[str] = field(default_factory=set)

    def tool_label(self, name: str) -> str:
        if self.registry is not None and name in self.registry:
            return self.registry.get(name).spec.label
        return name

    def warn_once(self, msg: str) -> None:
        self.console.print(f"[yellow]⚠ {msg}[/yellow]")

    def confirm_write_tool(self, call: ToolCall) -> bool:
        from harness.cli import _describe_call, _pre_validate_write_call

        refusal = _pre_validate_write_call(call, self.workspace_path)
        if refusal is not None:
            self.console.print(f"[red]🚫 refusing {call.name}: {refusal}[/red]")
            return False
        if call.name in self.approved_tools:
            return True
        label = self.tool_label(call.name)
        summary = _describe_call(call, self.workspace_path)
        self.console.print(f"[yellow]🔧 Airton wants to [bold]{label}[/bold] — {summary}[/yellow]")
        answer = self.console.input("   approve? [y/N/always]: ").strip().lower()
        if answer == "always":
            self.approved_tools.add(call.name)
            return True
        return answer.startswith("y")

    def render_tool_event(self, event: ToolLoopEvent) -> None:
        from harness.cli import _render_tool_event

        _render_tool_event(
            event,
            console=self.console,
            thinking=self.thinking,
            stream_renderer=self.stream_renderer,
            tool_label=self.tool_label,
        )

    def run_turn(self, user_input: str) -> None:
        """Handle one user turn: retrieval → system prompt → model
        (+ tool loop when active) → optional persona rewrite →
        persist. All UI output goes through `self.console` / the
        thinking spinner / stream renderer."""
        from harness.cli import (
            _build_tool_grounding_block,
            _persist_tool_exchange,
            _render_ab_memories_block,
            _render_fact_block,
            _render_memory_block,
            _retrieve_turn_context,
            _stream_or_complete,
        )

        self.transcript.append(
            session=self.session,
            channel=self.channel,
            speaker=self.speaker,
            role="user",
            content=user_input,
        )
        # Start the spinner immediately so the user sees acknowledgement
        # of their submission while retrieval warms up.
        self.thinking.start()

        examples, recalled, known_facts = _retrieve_turn_context(
            user_input=user_input,
            speaker=self.speaker,
            retriever=self.retriever,
            memory_store=self.memory_store,
            semantic_store=self.semantic_store,
            top_k=self.top_k,
            memories=self.memories,
            memories_threshold=self.memories_threshold,
            facts=self.facts,
            facts_threshold=self.facts_threshold,
            state=self.retrieval_state,
            warn=self.warn_once,
        )

        if examples:
            system_content = self.character.system_prompt(
                include_samples=examples, now=date.today()
            )
        else:
            system_content = self.character.system_prompt(now=date.today())

        if recalled:
            system_content = f"{system_content}\n\n{_render_memory_block(recalled)}"

        if self.ab_adapter is not None:
            ab_mem_block = _render_ab_memories_block(self.ab_adapter)
            if ab_mem_block is not None:
                system_content = f"{system_content}\n\n{ab_mem_block}"

        if self.registry is not None:
            system_content = (
                f"{system_content}\n\n"
                f"{_build_tool_grounding_block(self.registry, self.workspace_path)}"
            )

        if known_facts:
            system_content = f"{system_content}\n\n{_render_fact_block(known_facts)}"

        system = ChatMessage(role="system", content=system_content)

        summary_msg, history = self.ctx_meter.load_history()
        history_messages: list[ChatMessage] = []
        if summary_msg is not None:
            history_messages.append(summary_msg)
        history_messages.extend(history)

        self.console.print(f"[bold green]{self.character.name} ›[/bold green]")
        streamed = False
        if self.registry is not None:
            initial_messages: list[ChatMessage] = [system, *history_messages]
            loop_result = run_tool_loop(
                self.adapter,  # type: ignore[arg-type]
                initial_messages,
                self.registry,
                confirm=self.confirm_write_tool,
                observe=self.render_tool_event,
                router=self.router,
                hooks=self.hooks,
            )
            streamed = True
            _persist_tool_exchange(
                self.transcript,
                session=self.session,
                channel=self.channel,
                character_name=self.character.name,
                initial_count=len(initial_messages),
                loop_messages=loop_result.messages,
            )
            draft = loop_result.content
            # Small models (gemma4 8B) sometimes bail after a tool result —
            # empty content AND no further tool calls. Nudge once before
            # falling back to the sentinel.
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
                retry = self.adapter.complete_with_tools(  # type: ignore[attr-defined]
                    nudge_msgs,
                    tools=self.registry.specs(),
                    max_tokens=2048,
                    temperature=0.3,
                )
                if retry.content.strip():
                    draft = retry.content
            # Skip the rewriter when (a) rewrite-on-tools is off (default) —
            # rewriter compresses prose that summarize / investigate tasks
            # need, or (b) the tool loop left no substantive draft.
            if self.persona and self.rewrite_on_tools and draft.strip():
                self.console.print("\n[dim]*— voice pass —*[/dim]")
                rewrite_msgs = build_rewriter_messages(self.character, draft)
                reply, _ = _stream_or_complete(
                    self.adapter,
                    rewrite_msgs,
                    stream_renderer=self.stream_renderer,
                    temperature=0.2,
                    max_tokens=2048,
                )
            else:
                reply = draft or "(no reply — model returned empty text after tool calls)"
        else:
            reply, streamed = _stream_or_complete(
                self.adapter,
                [system, *history_messages],
                stream_renderer=self.stream_renderer,
            )

        self.thinking.stop()
        self.stream_renderer.stop()
        self.transcript.append(
            session=self.session,
            channel=self.channel,
            speaker=self.character.name,
            role="assistant",
            content=reply,
        )
        if not streamed:
            self.console.print(Markdown(reply))
        self.console.print()


def run_classic_chat(
    *,
    console: Console,
    session: str,
    channel: str,
    speaker: str,
    model: str,
    model_repo: str | None,
    lora_path: str | None,
    draft_repo: str | None,
    summarize_tool_results: bool,
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
    rewrite_on_tools: bool,
    workspace: str | None,
    compact_at: float,
    compact_keep_recent: int,
    dev: bool,
    include_internal: bool,
    router_enabled: bool,
    router_repo: str,
    router_mode: str,
) -> None:
    """Classic-mode chat entry point. Handles setup, REPL loop, and
    teardown; dispatches per-turn work to `ClassicChatSession.run_turn`."""
    import typer

    from harness.cli import (
        _maybe_bd_adapter,
        _maybe_retriever,
        _open_episodic_store,
        _open_semantic_store,
        _print_session_end_retro,
        _render_chat_header,
        _resolve_adapter,
        _RetrievalState,
        _StreamRenderer,
        _ThinkingSpinner,
    )

    character = load_character(settings.character_path)
    workspace_path = Path(workspace).expanduser().resolve() if workspace else settings.root
    if tools and not workspace_path.is_dir():
        raise typer.BadParameter(f"workspace {workspace_path} is not a directory")

    adapter = _resolve_adapter(
        model,
        persona=persona and not tools,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
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

        router_adapter: Any = MLXAdapter(repo=router_repo)
        router = (
            GrammarRouter(adapter=router_adapter)
            if router_mode == "grammar"
            else ModelRouter(adapter=router_adapter)
        )

    retriever = _maybe_retriever(character, top_k)
    memory_store = _open_episodic_store(character) if memories > 0 else None
    semantic_store = _open_semantic_store() if facts > 0 else None
    transcript = Transcript(settings.character_db_path)
    compaction_store = CompactionStore(settings.character_db_path) if compact_at > 0 else None

    retrieval_state = _RetrievalState()
    ab_adapter = _maybe_bd_adapter(character, include_internal=include_internal or dev)

    registry = build_classic_registry(
        tools=tools,
        tool_set=tool_set,
        tools_add=tools_add,
        tools_drop=tools_drop,
        workspace_path=workspace_path,
        adapter=adapter,
        character=character,
        transcript=transcript,
        memory_store=memory_store,
        semantic_store=semantic_store,
        speaker=speaker,
        session=session,
        ab_adapter=ab_adapter,
        console=console,
        retrieval_state=retrieval_state,
        persona=persona,
        router=router,
    )

    thinking = _ThinkingSpinner(console)
    stream_renderer = _StreamRenderer(console, show_suppressions=dev)

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

    ctx_meter = ContextMeter(
        adapter=adapter,
        character=character,
        transcript=transcript,
        compaction_store=compaction_store,
        session=session,
        console=console,
    )

    hooks = _build_hook_pipeline(
        summarize_tool_results=summarize_tool_results,
        router=router,
        router_repo=router_repo,
        console=console,
    )

    chat_session = ClassicChatSession(
        character=character,
        adapter=adapter,
        console=console,
        transcript=transcript,
        workspace_path=workspace_path,
        retrieval_state=retrieval_state,
        thinking=thinking,
        stream_renderer=stream_renderer,
        speaker=speaker,
        session=session,
        channel=channel,
        persona=persona,
        rewrite_on_tools=rewrite_on_tools,
        top_k=top_k,
        memories=memories,
        memories_threshold=memories_threshold,
        facts=facts,
        facts_threshold=facts_threshold,
        retriever=retriever,
        memory_store=memory_store,
        semantic_store=semantic_store,
        registry=registry,
        router=router,
        ab_adapter=ab_adapter,
        ctx_meter=ctx_meter,
        hooks=hooks,
    )

    if ab_adapter is not None:
        # Session-resume protocol — show thought-graph state so ab + user
        # resume from the bead graph rather than cold.
        console.print(f"[dim]{build_resume_summary(ab_adapter)}[/dim]")

    try:
        while True:
            ctx_meter.maybe_compact(
                compact_at=compact_at,
                compact_keep_recent=compact_keep_recent,
                thinking=thinking,
                ab_adapter=ab_adapter,
            )
            ctx_meter.print_ctx()
            user_input = console.input("[bold cyan]you › [/bold cyan]").strip()
            if not user_input:
                continue
            if ab_adapter is not None:
                ab_adapter.reset_turn_counter()
            if user_input.lower() in _EXIT_COMMANDS:
                _print_session_end_retro(ab_adapter)
                break
            if user_input.lower() in _RETRO_COMMANDS:
                handle_retro_slash(ab_adapter, console)
                continue
            if user_input.lower() in _EDIT_COMMANDS:
                handle_edit_slash(transcript=transcript, session=session, console=console)
                continue
            if user_input.lower() in _CLEAR_COMMANDS:
                handle_clear_slash(ctx_meter, console)
                continue
            chat_session.run_turn(user_input)
    except (KeyboardInterrupt, EOFError):
        console.print()
        _print_session_end_retro(ab_adapter)
        console.print("[dim]bye.[/dim]")
    finally:
        # Order matters: stop anything that could still be painting the
        # terminal first (spinner, stream) so a later exception doesn't
        # leave a live region hanging. Stores close last — idempotent
        # and safe under exceptions.
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
