"""`harness memory` subcommands.

Step 5b of docs/cli-extraction-plan.md. Everything that reads or writes
the episodic + semantic stores from the shell: list / search, the fact
surface, scribe, consolidate, rebuild-embeddings, wipe, ingest, and the
two bd harvests.

Note that `list` and `search` open the episodic store with ingest=True,
so a first run against an empty db seeds the character's memories as a
side effect. Pinned in tests/test_cli_memory_commands.py rather than
quietly changed here — a move is not the place to alter behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from harness.character import Character, load_character
from harness.cli_adapter import _resolve_adapter
from harness.cli_apps import memory_app
from harness.cli_bd import _maybe_bd_adapter
from harness.cli_store import _open_episodic_store, _open_semantic_store
from harness.cli_tools import (
    _build_document_tree_store_for_session,
    _build_tabular_store_for_session,
)
from harness.config import settings
from harness.consolidate import run_consolidation
from harness.scribe import run_scribe
from harness.store.episodic import ensure_seeds_ingested
from harness.store.transcript import Transcript

console = Console()


def _resolve_memory_target(override: str | None) -> tuple[Character, Path]:
    """Resolve the (Character, db_path) pair a memory subcommand should
    operate on. `override` is the `--character` flag value; None falls
    back to `settings.character_name` (default 'airton' unless
    HARNESS_CHARACTER_NAME is set). Prints a one-line header showing
    the resolved target so the implicit default is never silent
    (harness-5t53). Raises typer.BadParameter when the character
    directory doesn't exist."""
    name = override or settings.character_name
    char_path = settings.root / "character" / name
    if not char_path.exists():
        raise typer.BadParameter(f"character {name!r} not found at {char_path}")
    character = load_character(char_path)
    db_path = settings.db_path_for(name)
    rel = db_path.relative_to(settings.root) if db_path.is_relative_to(settings.root) else db_path
    console.print(f"[dim](memory: {name} store at {rel})[/dim]")
    return character, db_path


@memory_app.command("list")
def memory_list(
    tier: str | None = typer.Option(None, help="Filter by tier: seed | consolidated | working"),
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to list. Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """List every record in the episodic store."""
    char, db_path = _resolve_memory_target(character)
    store = _open_episodic_store(char, db_path=db_path)
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
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to search. Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Search episodic memory by semantic similarity."""
    char, db_path = _resolve_memory_target(character)
    store = _open_episodic_store(char, db_path=db_path)
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
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to list. Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """List facts in the semantic store."""
    _, db_path = _resolve_memory_target(character)
    store = _open_semantic_store(db_path=db_path)
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
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to search. Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Semantic-search the fact store."""
    _, db_path = _resolve_memory_target(character)
    store = _open_semantic_store(db_path=db_path)
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
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to write to. Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Add a single fact to the semantic store."""
    _, db_path = _resolve_memory_target(character)
    store = _open_semantic_store(db_path=db_path)
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
    model: str = typer.Option("mlx", help="Adapter for extraction: echo | mlx | ollama | vllm"),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help="Override the model id. MLX: HF repo. Ollama: model tag. vLLM: base URL.",
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="LoRA adapter directory (from `mlx_lm.lora` training). Requires --model mlx.",
    ),
    window_size: int = typer.Option(20, help="Turns per extraction window"),
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to scribe into. "
        "Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Walk unprocessed transcript turns and extract candidate memories."""
    char, db_path = _resolve_memory_target(character)
    adapter = _resolve_adapter(model, model_repo=model_repo, lora_path=lora_path)
    episodic = _open_episodic_store(char, db_path=db_path)
    semantic = _open_semantic_store(db_path=db_path)
    if episodic is None or semantic is None:
        raise typer.Exit(code=1)
    transcript = Transcript(db_path)
    try:
        user_id = None if shared else user
        with Status(f"scribe running on session={session}…", console=console):
            summary = run_scribe(
                adapter,
                char,
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
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to consolidate. "
        "Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Promote working-tier memories to consolidated. Clusters similar
    episodes; groups semantic facts by (subject, predicate); marks
    superseded rows so they drop out of retrieval. Idempotent."""
    char, db_path = _resolve_memory_target(character)
    episodic = _open_episodic_store(char, ingest=False, db_path=db_path)
    semantic = _open_semantic_store(db_path=db_path)
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
def memory_rebuild_embeddings(
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to rebuild. Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Re-embed every active episodic and semantic record with the
    current embedder. Use after switching embedder models so existing
    data participates in search again."""
    char, db_path = _resolve_memory_target(character)
    episodic = _open_episodic_store(char, ingest=False, db_path=db_path)
    semantic = _open_semantic_store(db_path=db_path)
    if episodic is None or semantic is None:
        raise typer.Exit(code=1)
    # harness-pzpy: also rebuild the character's tabular schema
    # embeddings when one is wired. The store ID is the same — same
    # embedder feeds all three stores, and an embedder swap drifts
    # the tabular dim the same way it drifts episodic / semantic.
    tabular_store = _build_tabular_store_for_session(char, episodic)
    # harness-px7k: same treatment for the document-tree store.
    tree_store_for_rebuild = _build_document_tree_store_for_session(char, episodic)
    from harness.store.document_tree import DocumentTreeStore
    from harness.store.tabular import TabularStore  # local for type narrowing

    tabular_typed = tabular_store if isinstance(tabular_store, TabularStore) else None
    tree_typed = (
        tree_store_for_rebuild if isinstance(tree_store_for_rebuild, DocumentTreeStore) else None
    )
    try:
        ep_mismatched = episodic.count_mismatched_embeddings()
        sem_mismatched = semantic.count_mismatched_embeddings()
        tab_mismatched = (
            tabular_typed.count_mismatched_embeddings() if tabular_typed is not None else 0
        )
        tree_mismatched = tree_typed.count_mismatched_embeddings() if tree_typed is not None else 0
        tabular_blurb = (
            f"; tabular: {tab_mismatched} mismatched" if tabular_typed is not None else ""
        )
        tree_blurb = f"; tree: {tree_mismatched} mismatched" if tree_typed is not None else ""
        console.print(
            f"episodic: {ep_mismatched} mismatched; "
            f"semantic: {sem_mismatched} mismatched{tabular_blurb}{tree_blurb}."
        )
        with Status("re-embedding episodic…", console=console):
            ep_updated, _ = episodic.rebuild_embeddings()
        with Status("re-embedding semantic…", console=console):
            sem_updated, _ = semantic.rebuild_embeddings()
        tab_updated = 0
        if tabular_typed is not None:
            with Status("re-embedding tabular…", console=console):
                tab_updated, _ = tabular_typed.rebuild_embeddings()
        tree_updated = 0
        if tree_typed is not None:
            with Status("re-embedding tree…", console=console):
                tree_updated, _ = tree_typed.rebuild_embeddings()
        tab_msg = f" and {tab_updated} tabular" if tabular_typed is not None else ""
        tree_msg = f" and {tree_updated} tree" if tree_typed is not None else ""
        console.print(
            f"[green]rebuilt {ep_updated} episodic and {sem_updated} semantic"
            f"{tab_msg}{tree_msg} embeddings with {episodic.embedder.id}.[/green]"
        )
    finally:
        episodic.close()
        semantic.close()


@memory_app.command("wipe")
def memory_wipe(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to wipe. Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Clear episodic, semantic, and scribe-watermark data. Transcripts
    and character data are preserved. Use after switching embedder
    dimensions, or when you want to re-ingest from scratch."""
    _, db_path = _resolve_memory_target(character)
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
    conn = sqlite3.connect(db_path)
    try:
        for table in tables:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"DELETE FROM {table}")  # noqa: S608
        conn.commit()
    finally:
        conn.close()
    console.print("[yellow]wiped episodic, semantic, and scribe_watermark.[/yellow]")


@memory_app.command("ingest")
def memory_ingest(
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to ingest into. "
        "Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Force an ingestion pass of the character's seed memories.
    Idempotent — existing records with matching external_id are
    preserved."""
    char, db_path = _resolve_memory_target(character)
    store = _open_episodic_store(char, ingest=False, db_path=db_path)
    if store is None:
        raise typer.Exit(code=1)
    try:
        inserted = ensure_seeds_ingested(char, store)
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
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to harvest into. "
        "Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Harvest ab's bd thought-graph into the episodic store as
    tier='procedural' records so relevant decisions / observations
    surface on future user turns. Idempotent — re-running after new
    beads close picks up only the new ones."""
    from harness.skills import DEFAULT_HARVEST_LABELS, harvest_bd_skills

    char, db_path = _resolve_memory_target(character)
    ab_adapter = _maybe_bd_adapter(char, include_internal=True)
    if ab_adapter is None:
        console.print(
            "[red]no bd adapter available — set HARNESS_AB_BD_DIR to ab's "
            "bd working directory first[/red]"
        )
        raise typer.Exit(code=1)
    store = _open_episodic_store(char, db_path=db_path)
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
def memory_harvest_bd_memories(
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose store to harvest into. "
        "Default: HARNESS_CHARACTER_NAME env or 'airton'.",
    ),
) -> None:
    """Mirror bd's persistent memories (bd remember / retro record)
    into the episodic store as tier='procedural' records so the main
    retrieval path surfaces them on identity / biographical questions.
    Idempotent on external_id='bd-mem:<key>' — re-running after new
    `bd remember` calls picks up only the new keys (harness-9yd)."""
    from harness.skills import harvest_bd_memories

    char, db_path = _resolve_memory_target(character)
    ab_adapter = _maybe_bd_adapter(char, include_internal=True)
    if ab_adapter is None:
        console.print(
            "[red]no bd adapter available — set HARNESS_AB_BD_DIR to ab's "
            "bd working directory first[/red]"
        )
        raise typer.Exit(code=1)
    store = _open_episodic_store(char, db_path=db_path)
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
