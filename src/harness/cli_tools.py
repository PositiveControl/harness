"""Tool-registry construction for a chat session.

Step 4 of docs/cli-extraction-plan.md. Owns the builder map, the
per-session stores the builders need (tabular / document-tree / fetch
denylist), the ops-tool injection, the deferred `introspect` +
`spawn_subagent` pass, the grounding block the system prompt grows when
tools are active, and the eval-time spec resolver the router uses.

The builder map is the single place that answers "which tools does the
model get this turn". Every drop is reported through `warnings_out`
rather than silently shrinking the registry (harness-akq).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from harness.character import Character
from harness.cli_apps import app
from harness.cli_bd import _maybe_bd_adapter
from harness.cli_introspect import list_cli_commands
from harness.config import settings
from harness.model.adapter import ModelAdapter
from harness.router import GrammarRouter, Router
from harness.store.bd_adapter import BeadsAdapter
from harness.store.episodic import EpisodicStore
from harness.store.fetch_denylist import FetchDenylistStore
from harness.store.semantic import SemanticStore
from harness.store.transcript import Transcript
from harness.tools import (
    AssembleContextTool,
    CalcTool,
    DateMathTool,
    EditFileTool,
    FetchUrlTool,
    GeographyTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    GlobTool,
    GrepTool,
    IntrospectContext,
    IntrospectTool,
    ListDirTool,
    LoadToolTool,
    NowTool,
    OutlineTool,
    PythonEvalTool,
    PythonStreamTool,
    ReadFileTool,
    RememberEventTool,
    RememberFactTool,
    SearchFactsTool,
    SearchMemoryTool,
    SearchScholarTool,
    SearchWebTool,
    ShellTool,
    StatsTool,
    StreamEditTool,
    SunTool,
    Tool,
    ToolCatalog,
    ToolRegistry,
    ToolSearchTool,
    ToolSpec,
    TranscriptIngestTool,
    TzConvertTool,
    WriteFileTool,
    resolve_tool_names,
    seed_builtins_into,
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
from harness.tools.phraseology_lint import PhraseologyLintTool

console = Console()


# Source identifier the phraseology lint pipeline filters its
# retrieval slate against. Matches the principle prefix the
# `scripts/atc_ingest.py` script writes for JO 7110.65 rows
# ("JO_7110.65 §<section>"). Applied to keep the JO-only lint
# contract intact when the active character has a multi-source
# corpus (airton_c carries JO + AIM + 14 CFR + PCG + PHAK; airton_c1
# is JO-only and the filter is a no-op there). Every lint call site
# applies it conditional on `"phraseology" in character.tool_descriptions`
# — which is the data-driven equivalent of the old "atc-family"
# membership check (harness-a2sa).
_ATC_LINT_SOURCE_FILTER: str = "JO_7110.65"


def _open_fetch_denylist() -> FetchDenylistStore:
    """Open the per-character fetch_url denylist on the shared SQLite
    (harness-4dgm). Always live — denylist behavior is on whenever
    fetch_url is in the registry; absence means the table is just
    empty. Callers close() at session teardown."""
    return FetchDenylistStore(settings.character_db_path)


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


def _build_tabular_store_for_session(
    character: Character | None,
    memory_store: EpisodicStore | None,
) -> object | None:
    """Build a per-character TabularStore from `character.tabular_tables`,
    or return None if the character doesn't ship table data
    (harness-kgpi).

    Reuses the episodic store's embedder so dense vectors are
    consistent across stores. Idempotent: re-runs at session start
    drop+recreate the data tables from the CSVs, with a fresh schema
    embedding under the current embedder. `object` return type dodges
    a top-of-file import of the TabularStore class; the only callers
    treat it as a duck-typed bundle member.
    """
    if character is None or not character.tabular_tables or memory_store is None:
        return None
    from harness.store.tabular import build_tabular_store_for_character

    return build_tabular_store_for_character(
        character_path=settings.character_path,
        embedder=memory_store.embedder,
        tabular_tables=character.tabular_tables,
    )


def _build_document_tree_store_for_session(
    character: Character | None,
    memory_store: EpisodicStore | None,
) -> object | None:
    """Build a per-character DocumentTreeStore from
    `character.document_trees`, or return None when the character
    doesn't ship hierarchical-doc data (harness-px7k).

    Mirrors `_build_tabular_store_for_session`: same embedder as
    episodic (so dense vectors are consistent across stores),
    idempotent on re-launch (the store dedups on `(document_id, path)`),
    and `object` return type to dodge a top-of-file import of the
    DocumentTreeStore class.
    """
    if character is None or not character.document_trees or memory_store is None:
        return None
    from harness.store.document_tree import build_document_tree_store_for_character

    return build_document_tree_store_for_character(
        character_path=settings.character_path,
        embedder=memory_store.embedder,
        document_trees=character.document_trees,
    )


def _build_assemble_context_tool(
    *,
    character: Character | None,
    memory_store: EpisodicStore | None,
    speaker: str,
    tabular_store: object | None,
    tree_store: object | None,
) -> AssembleContextTool | None:
    """Construct the AssembleContextTool from character + store handles.

    Character convention: contracts live under
    `character/<name>/contracts/*.yaml`. When the directory doesn't
    exist (most characters), the tool still registers but its
    description says 'no contracts registered' and any call returns a
    polite error. Cheap to ship the tool unconditionally — the cost
    of an unused builder is one extra dict entry.

    `tabular_store` (harness-kgpi) — when the character ships
    `tabular_tables:` declarations, the per-character TabularStore
    flows into StoreBundle.tabular so tabular slots in contracts
    actually resolve. None for characters without table data.

    `tree_store` (harness-px7k) — analogous wiring for `document_trees:`.
    A per-character DocumentTreeStore flows into StoreBundle.tree so
    tree-slot contracts resolve end-to-end. None for characters
    without hierarchical-doc data.
    """
    if character is None:
        return None
    contracts_dir = settings.character_path / "contracts"
    from harness.retrieval.contract import StoreBundle
    from harness.store.document_tree import DocumentTreeStore
    from harness.store.tabular import TabularStore

    bundle = StoreBundle(
        episodic=memory_store,
        tabular=tabular_store if isinstance(tabular_store, TabularStore) else None,
        tree=tree_store if isinstance(tree_store, DocumentTreeStore) else None,
    )
    return AssembleContextTool(
        stores=bundle,
        contracts_dir=contracts_dir,
        user_id=speaker,
        error_log_path=settings.data_path / "logs" / "assemble_context_errors.log",
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


def _session_tool_catalog() -> ToolCatalog:
    """Build a fresh ToolCatalog seeded with the built-in metadata
    table for this session. Used by tool_search (harness-ozx1).

    Persistence is deferred to the hot-reload slice (rqg0.5) — for
    now every session starts from the built-in seed, which matches
    the catalog's defensive load policy (missing file = empty)."""
    cat = ToolCatalog()
    seed_builtins_into(cat, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))
    return cat


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

    # Verb-anchor map for the phraseology lint tool (harness-ptya).
    # Empty mapping when the file is absent so non-atc characters get
    # the unmodified pipeline.
    from harness.tools.phraseology_lint import (
        default_verb_anchors_path,
        load_verb_anchors,
    )

    phraseology_verb_anchors = load_verb_anchors(default_verb_anchors_path(settings.character_path))

    # Per-character tabular store (harness-kgpi). None for characters
    # that don't ship `tabular_tables:` in core.yaml — most characters
    # today. The TUI keeps a single instance so multiple builders that
    # need it (assemble_context, future tabular-flavored tools) share
    # one connection.
    tabular_store = _build_tabular_store_for_session(character, memory_store)
    # Per-character document-tree store (harness-px7k). Same single-
    # instance pattern as tabular_store — assemble_context (and any
    # future tree-flavored tool) gets the one populated handle.
    tree_store = _build_document_tree_store_for_session(character, memory_store)

    builders: dict[str, Callable[[], Tool | None]] = {
        "read_file": lambda: ReadFileTool(root=workspace_path),
        "outline": lambda: OutlineTool(root=workspace_path),
        "edit_file": lambda: EditFileTool(root=workspace_path),
        "write_file": lambda: WriteFileTool(root=workspace_path),
        "stream_edit": lambda: StreamEditTool(root=workspace_path),
        "python_stream": lambda: PythonStreamTool(root=workspace_path),
        "shell": lambda: ShellTool(cwd=workspace_path),
        "list_dir": lambda: ListDirTool(root=workspace_path),
        "grep": lambda: GrepTool(root=workspace_path),
        "glob": lambda: GlobTool(root=workspace_path),
        "git_status": lambda: GitStatusTool(root=workspace_path),
        "git_diff": lambda: GitDiffTool(root=workspace_path),
        "git_log": lambda: GitLogTool(root=workspace_path),
        # Reckon-profile primitives. All four are stateless,
        # workspace-independent, and need no stores — they construct
        # cheaply per session (harness-1u2h).
        "now": lambda: NowTool(),
        "date_math": lambda: DateMathTool(),
        "calc": lambda: CalcTool(),
        "python_eval": lambda: PythonEvalTool(),
        "tz_convert": lambda: TzConvertTool(),
        "stats": lambda: StatsTool(),
        "sun": lambda: SunTool(),
        # tool_search — discovery primitive (harness-ozx1). Constructs a
        # session-scoped catalog seeded with built-in metadata; passes
        # the registry along so live spec descriptions surface rather
        # than the catalog's empty seed default.
        "tool_search": lambda: ToolSearchTool(
            catalog=_session_tool_catalog(),
            registry=registry,
        ),
        # load_tool — agent-driven working-set expansion (harness-atsz),
        # extended in harness-cm4v to build catalog-only builtins on
        # demand. The lambda captures `builders` by reference (late
        # binding), so the dict-literal-after-load_tool tools are still
        # reachable when load_tool actually calls a builder.
        "load_tool": lambda: LoadToolTool(
            catalog=_session_tool_catalog(),
            registry=registry,
            builders=builders,
        ),
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
        # search_scholar: structured academic-paper search across
        # Semantic Scholar + OpenAlex. No allowlist plumbing — both
        # APIs are hardcoded endpoints, not user-configurable hosts.
        # The character profile decides whether to include it.
        "search_scholar": lambda: SearchScholarTool(),
        # geography: deterministic country/region lookup over the
        # built-in gazetteer (harness-e83u). No constructor args — the
        # data is static and offline. Pairs with the ScopeViolationHook
        # so the model can verify scope proactively.
        "geography": lambda: GeographyTool(),
        # search_web reuses fetch_url's allowlist to rerank results so
        # allowlisted hosts surface first with `[allowlisted]` markers;
        # external hits stay visible. Keeps the agent's view of the
        # web honest about what's fetchable without hiding the rest.
        # default_site_filter scopes the character's *default* search
        # to one host (airton_f → scholar.google.com); the model can
        # still override by including its own `site:` operator.
        "search_web": lambda: SearchWebTool(
            allowed_hosts=(
                frozenset(character.fetch_url_allowed_hosts)
                if character is not None and character.fetch_url_allowed_hosts
                else None
            ),
            default_site_filter=(
                character.search_web_default_site_filter if character is not None else None
            ),
            # Share the same denylist FetchUrlTool writes to so a 403
            # logged this session deprioritizes the host on the next
            # search (harness-xncq).
            denylist=_open_fetch_denylist(),
        ),
        "fetch_url": lambda: FetchUrlTool(
            allowed_hosts=(
                frozenset(character.fetch_url_allowed_hosts)
                if character is not None and character.fetch_url_allowed_hosts
                else None
            ),
            denylist=_open_fetch_denylist(),
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
        "assemble_context": lambda: _build_assemble_context_tool(
            character=character,
            memory_store=memory_store,
            speaker=speaker,
            tabular_store=tabular_store,
            tree_store=tree_store,
        ),
        "phraseology_lint": lambda: (
            PhraseologyLintTool(
                adapter=adapter,
                store=memory_store,
                grammar=character.citation_grammar if character is not None else None,
                user_id=speaker,
                verb_anchors=phraseology_verb_anchors or None,
                source_filter=(
                    _ATC_LINT_SOURCE_FILTER
                    if character is not None and "phraseology" in character.tool_descriptions
                    else None
                ),
            )
            if (memory_store is not None and adapter is not None)
            else None
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

    # Wire the session's builtin catalog into the registry so the
    # unknown-tool error path can distinguish 'name exists but isn't
    # loaded' from 'genuine miss' (harness-yczi). The same catalog
    # backs tool_search / load_tool, so the recovery hint we hand
    # back stays consistent with what those discovery tools see.
    registry = ToolRegistry(catalog=_session_tool_catalog())
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
                    hooks=default_hook_pipeline(
                        citation_grammar=character.citation_grammar
                        if character is not None
                        else None,
                        catchers=character.catchers if character is not None else (),
                        scope_redirect_template=character.scope_redirect_template
                        if character is not None
                        else None,
                        character_name=character.name if character is not None else None,
                    ),
                    router=router,
                )
            )

    if not registry.names():
        return None

    from harness.tools.profiles import apply_profile_descriptions

    apply_profile_descriptions(registry, tool_set, character=character)

    # Hot-reload synthesized tools (harness-t5kx). Failures here don't
    # block boot — they surface via warnings_out and quarantine the
    # broken catalog entry so `harness tool list` shows the carryover.
    _hot_reload_synthesized(registry, character, warnings_out=warnings_out)

    return registry


def _hot_reload_synthesized(
    registry: ToolRegistry,
    character: Character | None,
    *,
    warnings_out: list[str] | None,
) -> None:
    """Load every non-quarantined synthesized tool from the catalog
    into the registry. Idempotent across sessions: quarantined entries
    skip without retry; loaded names collide-check against the live
    registry."""
    from harness.tools.catalog import load_catalog
    from harness.tools.loader import load_synthesized_tools

    character_name = character.name if character is not None else None
    catalog_path = _resolve_catalog_path(character_name)
    if not catalog_path.exists():
        return  # never-synthesised character; nothing to hot-reload
    catalog = load_catalog(catalog_path)
    report = load_synthesized_tools(catalog=catalog, catalog_path=catalog_path, registry=registry)
    if warnings_out is not None:
        for name, reason in report.quarantined:
            warnings_out.append(f"tool {name!r} quarantined during hot-reload: {reason}")
        if report.already_quarantined:
            joined = ", ".join(report.already_quarantined)
            warnings_out.append(f"skipping quarantined synthesised tools (carryover): {joined}")


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


def _resolve_catalog_path(character: str | None) -> Path:
    """Default catalog location for the character. Operator override
    via `--catalog-path` on each command."""
    char_path = settings.character_path
    if character:
        char_path = char_path.parent / character
    return char_path / "data" / "tool_catalog.json"


def _resolve_router_tool_specs(tool_names: Sequence[str], workspace: Path) -> list[ToolSpec]:
    """Instantiate the minimum-viable set of tools needed to read their
    ToolSpecs for the router eval. Memory-dependent tools (search_memory,
    search_facts, remember_*) are skipped with a warning — the eval
    fixture can still cover them via the same tool name, but the router
    will see the spec from a dummy no-op tool below."""
    builders: dict[str, Callable[[], Tool]] = {
        "read_file": lambda: ReadFileTool(root=workspace),
        "outline": lambda: OutlineTool(root=workspace),
        "list_dir": lambda: ListDirTool(root=workspace),
        "grep": lambda: GrepTool(root=workspace),
        "glob": lambda: GlobTool(root=workspace),
        "search_web": lambda: SearchWebTool(),
        "geography": lambda: GeographyTool(),
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
