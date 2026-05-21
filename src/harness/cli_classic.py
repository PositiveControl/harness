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
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markdown import Markdown

from harness.character import Character, load_character
from harness.cli_repl import (
    ContextMeter,
    handle_clear_slash,
    handle_edit_slash,
    handle_help_slash,
    handle_retro_slash,
)
from harness.compaction import CompactionStore
from harness.config import settings
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.orchestrator import ToolLoopEvent, run_tool_loop
from harness.persona.banter import BanterStreakTracker, load_default_tracker
from harness.persona.rewriter import build_rewriter_messages
from harness.retrieval import VoiceRetriever
from harness.retrieval.contract import StoreBundle
from harness.router import GrammarRouter, ModelRouter, Router
from harness.store import EpisodicStore, SemanticStore
from harness.store.audit import AuditStore, record_turn_audit
from harness.store.bd_adapter import BeadsAdapter
from harness.store.transcript import Transcript
from harness.tools import (
    AssembleContextTool,
    CalcTool,
    ConsolidateMemoryTool,
    DateMathTool,
    EditFileTool,
    FetchUrlTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    LoadToolTool,
    NowTool,
    PythonEvalTool,
    ReadFileTool,
    RememberEventTool,
    RememberFactTool,
    ScribeSessionTool,
    SearchFactsTool,
    SearchMemoryTool,
    SearchScholarTool,
    SearchWebTool,
    ShellTool,
    StatsTool,
    SunTool,
    Tool,
    ToolCall,
    ToolCatalog,
    ToolRegistry,
    ToolResult,
    ToolSearchTool,
    TzConvertTool,
    WriteFileTool,
    resolve_tool_names,
    seed_builtins_into,
)
from harness.tools.ab_ops import build_resume_summary

if TYPE_CHECKING:
    from harness.cli import _RetrievalState, _StreamRenderer, _ThinkingSpinner
    from harness.orchestrator.hooks import HookPipeline


_EXIT_COMMANDS = frozenset({"/exit", "/quit", ":q"})
_RETRO_COMMANDS = frozenset({"/retro"})
_EDIT_COMMANDS = frozenset({"/edit", "/capture"})
# /clear and /reset are aliases — /reset is the more-discoverable name
# users guess; /clear is what's been wired since the original watermark
# work (harness-c1r). Mid-loop session-id swap (harness-hb8's stretch
# goal) isn't supported in the classic REPL — `/exit` and restart with
# `--session NEW_NAME` is the documented path.
_CLEAR_COMMANDS = frozenset({"/clear", "/reset"})
_HELP_COMMANDS = frozenset({"/help", "/?"})


def _classic_session_tool_catalog() -> ToolCatalog:
    """Build a fresh ToolCatalog seeded with the built-in metadata
    table for a classic-CLI session (harness-ozx1). Mirrors
    cli._session_tool_catalog. Persistence lands in rqg0.5."""
    cat = ToolCatalog()
    seed_builtins_into(cat, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))
    return cat


def _make_write_file_redirect_hook(
    *,
    registry: ToolRegistry | None,
    workspace_path: Path | None,
) -> Any:
    """Wire a WriteFileRedirectHook against the session's live
    registry + workspace, or return None if neither is available
    (e.g. tool-less chat).

    Note (harness-hf4r): the hook is constructed regardless of
    whether write_file is currently in the registry. Sessions with
    `--tool-set minimal` start without write_file, then the model
    discovers and load_tool's it mid-session. If the hook had been
    constructed eagerly with a 'write_file not in registry → None'
    short-circuit, the redirect would never have fired on those
    lazy-loaded turns — the exact case that prompted the redirect
    in the first place (Mark's 2026-05-20T21:47 GTA session). The
    hook's own check() already gates on `ctx.call.name == "write_file"`
    so the always-constructed hook is a no-op cost on every non-
    write_file call.

    Three closures bridge the hook (pure data) to the live session:

    - `read_existing(path)` reads `<workspace>/<path>` as UTF-8 text
      and returns the content. Non-existent files and decode failures
      yield None so the hook treats them as 'not redirectable' and
      Continues.
    - `ensure_edit_file_active()` adds edit_file to the active set
      when it's registered-but-inactive (the load_tool companion
      pairing usually means it's already active alongside
      write_file). Returns True iff edit_file is callable after the
      call. Returns False (causing the hook to fall through to
      Continue) when edit_file isn't even in the catalog — the
      existing multi-round write_file → unknown_tool → load_tool
      recovery path runs unchanged.
    - `invoke_edit_file(path, old, new)` dispatches the registry's
      edit_file tool and returns its ToolResult so the hook can feed
      it back as the Skip payload.
    """
    if registry is None or workspace_path is None:
        return None

    from harness.orchestrator.hooks import WriteFileRedirectHook

    root = workspace_path.resolve()

    def read_existing(path: str) -> str | None:
        try:
            target = (root / path).resolve()
            # Reject paths that escape the workspace root — the hook
            # treats them as 'not redirectable' so write_file's own
            # escape check fires the same error the model is used to.
            target.relative_to(root)
        except (OSError, ValueError):
            return None
        if not target.exists() or not target.is_file():
            return None
        try:
            return target.read_text()
        except (OSError, UnicodeDecodeError):
            return None

    def ensure_edit_file_active() -> bool:
        if "edit_file" not in registry:
            return False
        if "edit_file" in registry.active_names():
            return True
        try:
            registry.set_active(set(registry.active_names()) | {"edit_file"})
        except Exception:
            return False
        return "edit_file" in registry.active_names()

    def invoke_edit_file(path: str, old: str, new: str) -> ToolResult:
        return registry.call(
            "edit_file",
            {"path": path, "old_string": old, "new_string": new},
        )

    return WriteFileRedirectHook(
        read_existing=read_existing,
        ensure_edit_file_active=ensure_edit_file_active,
        invoke_edit_file=invoke_edit_file,
    )


def _make_auto_load_on_unknown_hook(
    registry: ToolRegistry | None,
) -> Any:
    """Wire an AutoLoadOnUnknownHook against the session's live
    registry. Returns None when there's no registry (tool-less chat)
    or when `load_tool` isn't itself in the registry — without
    load_tool the closure has nothing to dispatch.

    Reuses `LoadToolTool.call` as the activation primitive so the
    auto-loader inherits all the existing edge cases: builder
    failures, synthesized-tool restart hints, companion auto-load,
    no-op confirmation when already active. The closure converts the
    string-shaped LoadToolTool result into a bool by re-checking the
    registry's active set after the call (harness-2uso)."""
    if registry is None:
        return None
    try:
        load_tool_obj = registry.get("load_tool")
    except KeyError:
        return None
    if not isinstance(load_tool_obj, LoadToolTool):
        # Pessimistic guard: some other tool got registered under the
        # name 'load_tool'. Skip wiring rather than dispatch into an
        # unknown shape.
        return None

    from harness.orchestrator.hooks import AutoLoadOnUnknownHook

    def try_activate(name: str) -> bool:
        # Fast path — already active, no work to do.
        if name in registry.active_names():
            return True
        try:
            load_tool_obj.call(name=name)
        except Exception:
            # Any error inside load_tool is contained: the unknown_tool
            # error path will fire downstream as before.
            return False
        return name in registry.active_names()

    return AutoLoadOnUnknownHook(try_activate=try_activate)


def _build_hook_pipeline(
    *,
    summarize_tool_results: bool,
    router: Router | None,
    router_repo: str,
    console: Console,
    character_path: Path,
    character: Character,
    registry: ToolRegistry | None = None,
    workspace_path: Path | None = None,
) -> HookPipeline | None:
    """Build a HookPipeline override when the character has a corpus
    chunks dir (so FabricatedSectionHook gets its anchor index) OR
    the summarizer is requested. Returns None when neither — the
    orchestrator falls back to its module default.

    `valid_section_anchors` (harness-aise) is loaded once at startup
    from two sources, unioned:

      * `<character>/corpus/chunks/*.jsonl` — chunked corpora used by
        FAA personas (airton_c{,1,_tfr}). Rows carry a `section`
        field stamped by the chunker.
      * `<character>/document_trees:` markdown sources — used by
        markdown-corpus personas (airton_f). Numeric section
        headings (`## 1. Scope`, `### 2.1 Greeter`) become §N / §N.N
        anchors.

    Empty for characters with neither, which keeps FabricatedSectionHook
    silent. The cost is one file walk per source; no embedder.

    The summarizer reuses the router's adapter when available — it's
    a small MLX model already loaded into the process. Otherwise we
    build a fresh MLX adapter from `router_repo`. Either way the
    extra model cost is bounded (~1 GB for a 3B-4bit router)."""
    from harness.orchestrator.section_index import (
        collect_valid_anchors,
        collect_valid_anchors_from_markdown,
    )

    valid_anchors = collect_valid_anchors(character_path / "corpus" / "chunks")
    if character.document_trees:
        valid_anchors = valid_anchors | collect_valid_anchors_from_markdown(
            character.document_trees
        )
    grammar = character.citation_grammar
    catchers = character.catchers
    # Build the write_file → edit_file redirect hook (harness-hnt7)
    # when the session has both a registry containing write_file and a
    # workspace path. None otherwise — the orchestrator's module
    # default pipeline doesn't carry the hook either, so behavior is
    # unchanged for those sessions.
    redirect_hook = _make_write_file_redirect_hook(
        registry=registry,
        workspace_path=workspace_path,
    )
    # Build the auto-load-on-unknown hook (harness-2uso) — only fires
    # when the session has a registry that already contains load_tool.
    # Saves 2 rounds per first-use of catalog-known tools.
    auto_load_hook = _make_auto_load_on_unknown_hook(registry=registry)
    if (
        not summarize_tool_results
        and not valid_anchors
        and grammar is None
        and not catchers
        and redirect_hook is None
        and auto_load_hook is None
    ):
        return None

    from harness.orchestrator.hooks import (
        ToolResultSummarizerHook,
        default_hook_pipeline,
    )

    pipeline = default_hook_pipeline(
        valid_section_anchors=valid_anchors,
        citation_grammar=grammar,
        catchers=catchers,
        scope_redirect_template=character.scope_redirect_template,
        character_name=character.name,
        write_file_redirect_hook=redirect_hook,
        auto_load_on_unknown_hook=auto_load_hook,
    )

    if not summarize_tool_results:
        return pipeline

    summarizer_adapter: Any
    if router is not None and hasattr(router, "adapter"):
        summarizer_adapter = router.adapter  # duck-typed; ModelRouter + GrammarRouter both carry it
        console.print(f"[dim]tool-result summarizer: reusing router adapter ({router_repo})[/dim]")
    else:
        from harness.model.mlx import MLXAdapter

        summarizer_adapter = MLXAdapter(repo=router_repo)
        console.print(
            f"[dim]tool-result summarizer: loading {router_repo} (first turn is slower)[/dim]"
        )

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
        _open_fetch_denylist,
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

    # Character-local synonym expansion for search_memory (harness-ajn).
    # Loads the shared corpus/synonyms.yaml (also ingest-consumed) plus
    # corpus/query_synonyms.yaml (expander-only, to prevent lay-paraphrase
    # entries from diluting ingest-side precision on jargon queries).
    # NullQueryExpander when both are absent.
    from harness.retrieval.query_expander import (
        default_query_only_synonyms_path,
        default_synonyms_path,
        load_query_expander,
    )

    query_expander = load_query_expander(
        default_synonyms_path(settings.character_path),
        query_only_path=default_query_only_synonyms_path(settings.character_path),
    )

    # Per-character tabular store (harness-kgpi). Same pattern as the
    # TUI builder — None for characters that don't ship tabular_tables;
    # the AssembleContextTool then surfaces "tabular slot needs a
    # tabular store" for any contract slot that wants one.
    tabular_store = None
    if character is not None and character.tabular_tables and memory_store is not None:
        from harness.store.tabular import build_tabular_store_for_character

        tabular_store = build_tabular_store_for_character(
            character_path=settings.character_path,
            embedder=memory_store.embedder,
            tabular_tables=character.tabular_tables,
        )

    # Per-character document-tree store (harness-px7k). Same pattern.
    tree_store = None
    if character is not None and character.document_trees and memory_store is not None:
        from harness.store.document_tree import build_document_tree_store_for_character

        tree_store = build_document_tree_store_for_character(
            character_path=settings.character_path,
            embedder=memory_store.embedder,
            document_trees=character.document_trees,
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
        # Reckon-profile primitives. Stateless, workspace-independent
        # — registered alongside fs/git so any profile can pick them up
        # via --tools-add (harness-1u2h).
        "now": lambda: NowTool(),
        "date_math": lambda: DateMathTool(),
        "calc": lambda: CalcTool(),
        "python_eval": lambda: PythonEvalTool(),
        "tz_convert": lambda: TzConvertTool(),
        "stats": lambda: StatsTool(),
        "sun": lambda: SunTool(),
        # tool_search — agent discovery primitive (harness-ozx1).
        # Build a fresh session-scoped catalog seeded from the
        # builtin metadata table; registry plumbing surfaces live
        # spec descriptions over the catalog's empty defaults.
        "tool_search": lambda: ToolSearchTool(
            catalog=_classic_session_tool_catalog(),
            registry=registry,
        ),
        # load_tool — agent-driven working-set expansion (harness-atsz),
        # extended in harness-cm4v to build catalog-only builtins on
        # demand. Closure captures the local `builders` dict by name,
        # so the lazy-build path sees the full builder map at call time.
        "load_tool": lambda: LoadToolTool(
            catalog=_classic_session_tool_catalog(),
            registry=registry,
            builders=builders,
        ),
        "search_memory": (
            lambda: (
                SearchMemoryTool(store=memory_store, user_id=speaker, expander=query_expander)
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
        # search_scholar: structured academic-paper search across
        # Semantic Scholar + OpenAlex. Hardcoded API endpoints; the
        # character profile decides whether to include it.
        "search_scholar": lambda: SearchScholarTool(),
        # search_web reuses the same allowlist as fetch_url to rerank
        # results: hosts in the allowlist surface first with an
        # `[allowlisted]` marker; non-allowlisted hits stay visible
        # with `[external]`. Pairs with fetch_url so the agent sees
        # what's out there but is biased toward sources it can act on.
        # default_site_filter scopes the character's *default* search
        # to one host (airton_f → scholar.google.com); the model can
        # still override by including its own `site:` operator.
        "search_web": lambda: SearchWebTool(
            allowed_hosts=(
                frozenset(character.fetch_url_allowed_hosts)
                if character.fetch_url_allowed_hosts
                else None
            ),
            default_site_filter=character.search_web_default_site_filter,
            # Share the FetchUrlTool denylist so a 403 logged this
            # session deprioritizes the host on the next search
            # (harness-xncq).
            denylist=_open_fetch_denylist(),
        ),
        # Mirror cli.py (TUI path): honor character.fetch_url_allowed_hosts
        # so per-character allowlists (airton_f's scholar.google.com /
        # arxiv.org / doi.org, atc's aviation sources, etc.) actually
        # apply in the classic REPL. Empty tuple → None → no allowlist
        # (open web).
        "fetch_url": lambda: FetchUrlTool(
            allowed_hosts=(
                frozenset(character.fetch_url_allowed_hosts)
                if character.fetch_url_allowed_hosts
                else None
            ),
            denylist=_open_fetch_denylist(),
        ),
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
        # Phase 3 contract orchestrator wiring (harness-xysp +
        # harness-kgpi). Contracts under character/<name>/contracts/*.yaml;
        # tabular store wired when the character declares
        # `tabular_tables:` in core.yaml.
        "assemble_context": (
            lambda: (
                AssembleContextTool(
                    stores=StoreBundle(
                        episodic=memory_store, tabular=tabular_store, tree=tree_store
                    ),
                    contracts_dir=settings.character_path / "contracts",
                    user_id=speaker,
                    error_log_path=settings.data_path / "logs" / "assemble_context_errors.log",
                )
                if character is not None
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

    # Wire the builtin catalog so unknown-tool errors include the
    # load_tool-recovery hint when the model calls a catalog-known
    # tool that wasn't activated (harness-yczi).
    registry = ToolRegistry(catalog=_classic_session_tool_catalog())
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
                hooks=default_hook_pipeline(
                    citation_grammar=character.citation_grammar,
                    catchers=character.catchers,
                    scope_redirect_template=character.scope_redirect_template,
                    character_name=character.name,
                ),
                router=router,
            )
        )

    if not registry.names():
        return None

    from harness.tools.profiles import apply_profile_descriptions

    apply_profile_descriptions(registry, tool_set, character=character)
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
    chain_rewrites: bool
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
    # Per-turn audit log (harness-ywp.2). None disables auditing (e.g.
    # when the store can't open); otherwise one row lands per turn
    # after the assistant reply is rendered and before return.
    audit_store: AuditStore | None = None
    approved_tools: set[str] = field(default_factory=set)
    # Resolved `--memory-scope` filter for episodic + semantic search.
    # None = no filter (default 'all'); tuple = NULL OR session_id IN
    # (...). Used by `_retrieve_turn_context` per turn (harness-w3mo).
    allowed_sessions: tuple[str, ...] | None = None
    # Recency-RRF gate (harness-w3mo step 5). When weight > 0, the
    # store fuses a third RRF tier ordered by session recency. ranks
    # is None when weight is 0 — keeps the gate fully off rather than
    # paying the dict-lookup cost.
    recency_ranks: dict[str, int] | None = None
    recency_weight: float = 0.0
    # Banter intercept tracker (epic harness-jjm9). Loaded by the chat
    # bootstrap when the active character ships a jokes.yaml; None
    # otherwise. Holds the per-session 1-joke-then-3-redirects cycle
    # state so smartass / empty-signal prompts deflect with a joke
    # instead of fabricating a rule chunk from thin retrieval.
    banter_tracker: BanterStreakTracker | None = None

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
            allowed_sessions=self.allowed_sessions,
            recency_ranks=self.recency_ranks,
            recency_weight=self.recency_weight,
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

        # Topic-boundary signal (harness-eftf + harness-w3mo). Cheap
        # (~25 tokens), fires when retrieval is muted (post-/clear)
        # OR when --memory-scope is bounding retrieval to a session
        # subset. Different wording per cause; the helper picks.
        from harness.cli import _topic_boundary_suffix

        system_content = (
            f"{system_content}{_topic_boundary_suffix(self.retrieval_state, self.allowed_sessions)}"
        )

        system = ChatMessage(role="system", content=system_content)

        summary_msg, history = self.ctx_meter.load_history()
        history_messages: list[ChatMessage] = []
        if summary_msg is not None:
            history_messages.append(summary_msg)
        history_messages.extend(history)

        self.console.print(f"[bold green]{self.character.name} ›[/bold green]")
        streamed = False
        # Track the tool-loop result across both branches so the
        # per-turn audit record (harness-ywp.2) can summarise the
        # turn uniformly, whether tools ran or not.
        loop_result: Any = None
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
                memory_block_attached=bool(recalled),
                force_search_memory=self.character.require_search_memory,
                force_assemble_context=(
                    self.character.default_contract_role
                    if self.character.require_assemble_context
                    else None
                ),
                banter_tracker=self.banter_tracker,
                scope_redirect_template=self.character.scope_redirect_template,
                scope_lexicon=self.character.scope_lexicon,
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
                rewrite_msgs = build_rewriter_messages(self.character, draft, focus="style")
                reply, _ = _stream_or_complete(
                    self.adapter,
                    rewrite_msgs,
                    stream_renderer=self.stream_renderer,
                    temperature=0.2,
                    max_tokens=2048,
                )
                if self.chain_rewrites and reply.strip():
                    self.console.print("\n[dim]*— concrete pass —*[/dim]")
                    concrete_msgs = build_rewriter_messages(self.character, reply, focus="concrete")
                    reply, _ = _stream_or_complete(
                        self.adapter,
                        concrete_msgs,
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
        record_turn_audit(
            self.audit_store,
            session=self.session,
            character=self.character.name,
            user_id=self.speaker,
            user_message=user_input,
            model_reply=reply,
            loop_result=loop_result,
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
    harvest_skills: bool,
    harvest_memories: bool,
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
    chain_rewrites: bool = False,
    workspace: str | None,
    compact_at: float,
    compact_keep_recent: int,
    auto_scribe: bool,
    dev: bool,
    include_internal: bool,
    router_enabled: bool,
    router_repo: str,
    router_mode: str,
    allowed_sessions: tuple[str, ...] | None = None,
    recency_ranks: dict[str, int] | None = None,
    recency_weight: float = 0.0,
) -> None:
    """Classic-mode chat entry point. Handles setup, REPL loop, and
    teardown; dispatches per-turn work to `ClassicChatSession.run_turn`."""
    import typer

    from harness.cli import (
        _maybe_bd_adapter,
        _maybe_retriever,
        _open_audit_store,
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
        chain_rewrites=chain_rewrites,
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
            GrammarRouter(adapter=router_adapter, persona_scope_hint=character.scope_hint)
            if router_mode == "grammar"
            else ModelRouter(adapter=router_adapter, persona_scope_hint=character.scope_hint)
        )

    retriever = _maybe_retriever(character, top_k)
    memory_store = _open_episodic_store(character) if memories > 0 else None
    semantic_store = _open_semantic_store() if facts > 0 else None
    transcript = Transcript(settings.character_db_path)
    audit_store = _open_audit_store()
    compaction_store = CompactionStore(settings.character_db_path) if compact_at > 0 else None

    retrieval_state = _RetrievalState()
    ab_adapter = _maybe_bd_adapter(character, include_internal=include_internal or dev)

    # Harvest newly-closed thought:* beads into episodic memory so
    # decisions/observations made since last session are searchable
    # this session. Idempotent; no-op when either substrate is
    # missing. See harness-j5b (sota punch #7 follow-up).
    from harness.cli import _maybe_harvest_bd_memories, _maybe_harvest_skills

    _maybe_harvest_skills(ab_adapter, memory_store, enabled=harvest_skills)
    # Mirror bd memories onto the same episodic substrate so identity
    # and biographical questions don't hallucinate (harness-9yd).
    _maybe_harvest_bd_memories(ab_adapter, memory_store, enabled=harvest_memories)

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
        auto_scribe=auto_scribe,
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
        memory_store=memory_store,
        semantic_store=semantic_store,
        scribe_user_id=speaker,
        auto_scribe=auto_scribe,
        retrieval_state=retrieval_state,
    )

    hooks = _build_hook_pipeline(
        summarize_tool_results=summarize_tool_results,
        router=router,
        router_repo=router_repo,
        console=console,
        character_path=settings.character_path,
        character=character,
        registry=registry,
        workspace_path=workspace_path,
    )

    banter_tracker = load_default_tracker(settings.character_path)

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
        chain_rewrites=chain_rewrites,
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
        audit_store=audit_store,
        allowed_sessions=allowed_sessions,
        recency_ranks=recency_ranks,
        recency_weight=recency_weight,
        banter_tracker=banter_tracker,
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
            if user_input.lower() in _HELP_COMMANDS:
                handle_help_slash(console)
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
        audit_store.close()
        if compaction_store is not None:
            compaction_store.close()
        if memory_store is not None:
            memory_store.close()
        if semantic_store is not None:
            semantic_store.close()
