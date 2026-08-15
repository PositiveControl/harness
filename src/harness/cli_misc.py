"""The remaining `harness` subcommand groups: tool, plan, denylist,
web, phraseology.

Step 5d of docs/cli-extraction-plan.md — the last extraction the plan
calls for. Grouped into one module because none of the five is big
enough to earn its own file, and they share the same shape: a thin
Typer surface over a store or a factory that lives elsewhere.

  tool         inspect + manage the tool catalog (harness-rqg0)
  plan         bootstrap runtime-typed plans (harness-ptdw)
  denylist     manage the fetch_url denylist (harness-4dgm)
  web          serve a character over HTTP (harness-3jz1.9)
  phraseology  one-shot ATC utterance lint (airton_c1 only)

Covered by tests/test_cli_misc_commands.py, plus the pre-existing
subprocess suites tests/test_cli_tool.py and tests/test_cli_plan.py,
which drive the real binary and so survive a move on their own.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from harness.character import load_character
from harness.cli_adapter import _resolve_adapter
from harness.cli_apps import denylist_app, phraseology_app, plan_app, tool_app, web_app
from harness.cli_store import _open_episodic_store
from harness.cli_tools import (
    _ATC_LINT_SOURCE_FILTER,
    _open_fetch_denylist,
    _resolve_catalog_path,
)
from harness.config import settings
from harness.model import AdapterName, ModelAdapter, make_adapter
from harness.store.bd_adapter import BeadsAdapter, BeadsAdapterError
from harness.tools import ToolCatalog, seed_builtins_into

console = Console()


@phraseology_app.command("lint")
def phraseology_lint_cmd(
    utterance: str = typer.Argument(
        ...,
        help='Controller utterance to lint, e.g. "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF."',
    ),
    scenario_hint: str | None = typer.Option(
        None,
        "--scenario",
        help="Optional operational class: departure | arrival | handoff | emergency.",
    ),
    model: str = typer.Option("mlx", help="Adapter: echo | mlx | ollama | vllm"),
    model_repo: str | None = typer.Option(None, "--model-repo"),
    lora_path: str | None = typer.Option(None, "--lora-path"),
    draft_repo: str | None = typer.Option(None, "--draft-repo"),
    k: int = typer.Option(
        8,
        help=(
            "Top-K hybrid retrieval. The candidate-anchor set used to "
            "gate the model's citation is built from this slate."
        ),
    ),
    temperature: float = typer.Option(
        0.0,
        help="Sampling temperature. Default 0 for stable verdicts.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Lint one ATC controller utterance against JO 7110.65.

    Reuses the airton_c1 hardened retrieval stack (BGE-small + FTS5 +
    RRF) plus a cite-grounding gate so the verdict either anchors to a
    real candidate section or refuses with `out_of_scope`. No persona
    rewriting — the lint output is structured JSON, not voice-shaped
    prose.
    """
    character = load_character(settings.character_path)
    # Gate on character ownership of the phraseology profile rather
    # than hard-coding the family list (harness-a2sa). If a character
    # declares phraseology overrides in tool_descriptions.yaml, it
    # owns the lint pipeline; otherwise the prompt + cite-grounding
    # rules don't fit and we refuse rather than silently retrieve
    # against unrelated seeds.
    if "phraseology" not in character.tool_descriptions:
        raise typer.BadParameter(
            f"phraseology lint requires a character that declares the "
            f"`phraseology` profile in tool_descriptions.yaml "
            f"(got {character.name!r}). Set HARNESS_CHARACTER_NAME=airton_c1."
        )

    adapter = _resolve_adapter(
        model,
        persona=False,
        character=character,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
    )
    memory_store = _open_episodic_store(character, ingest=False)
    if memory_store is None:
        raise typer.BadParameter(
            "phraseology lint needs the episodic store — install with "
            "`uv sync --extra all` and run `harness memory ingest`."
        )

    try:
        from harness.tools.phraseology_lint import (
            default_verb_anchors_path,
            lint_utterance,
            load_verb_anchors,
        )

        verb_anchors = load_verb_anchors(default_verb_anchors_path(settings.character_path))
        verdict = lint_utterance(
            utterance,
            adapter=adapter,
            episodic_store=memory_store,
            grammar=character.citation_grammar,
            scenario_hint=scenario_hint,
            user_id=None,  # rulebook seeds are shared (user_id IS NULL)
            k=k,
            temperature=temperature,
            verb_anchors=verb_anchors or None,
            source_filter=_ATC_LINT_SOURCE_FILTER,
        )
    finally:
        memory_store.close()

    payload = {
        "utterance": utterance,
        "scenario_hint": scenario_hint,
        "verdict": verdict.verdict,
        "expected_section": verdict.expected_section,
        "expected_phraseology": verdict.expected_phraseology,
        "mismatch": verdict.mismatch,
        "citation_quote": verdict.citation_quote,
    }
    if as_json:
        console.print_json(json.dumps(payload))
        return

    verdict_color = {
        "ok": "green",
        "wrong": "red",
        "incomplete": "yellow",
        "out_of_scope": "dim",
    }[verdict.verdict]
    console.print(f"[bold]utterance[/bold]  {utterance}")
    if scenario_hint:
        console.print(f"[bold]scenario [/bold]  {scenario_hint}")
    console.print(f"[bold]verdict  [/bold]  [{verdict_color}]{verdict.verdict}[/{verdict_color}]")
    if verdict.expected_section:
        console.print(f"[bold]section  [/bold]  §{verdict.expected_section}")
    if verdict.expected_phraseology:
        console.print(f"[bold]canonical[/bold]  {verdict.expected_phraseology}")
    if verdict.mismatch:
        console.print(f"[bold]mismatch [/bold]  [yellow]{verdict.mismatch}[/yellow]")
    if verdict.citation_quote:
        console.print(f"[bold]quote    [/bold]  [dim]{verdict.citation_quote}[/dim]")


@plan_app.command("bootstrap")
def plan_bootstrap(
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose bd state to read. Default: HARNESS_CHARACTER_NAME.",
    ),
    plan_id: str | None = typer.Option(
        None,
        "--plan-id",
        help="Explicit Plan id. Defaults to 'bd:<assignee>' so re-running on the "
        "same assignee produces the same Plan id.",
    ),
    assignee: str = typer.Option(
        "mark",
        "--assignee",
        help="Bd assignee whose beads enter the Plan. Use 'airton_b' for the ab data-plane.",
    ),
    plans_dir: Path | None = typer.Option(
        None,
        "--plans-dir",
        help="Where the JsonPlanStore writes plan files. Default: <character>/data/plans/.",
    ),
    include_closed: bool = typer.Option(
        False,
        "--include-closed",
        help="Include closed beads as `achieved` Subgoals (default excludes them).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the would-be Plan and exit; don't write anything.",
    ),
) -> None:
    """Read the character's bd state and write the resulting Plan to a
    JsonPlanStore. Idempotent — re-running on the same bd state produces
    structurally-identical Plans (timestamps update on each write).
    Harness-snn2."""
    from harness.plan import (
        JsonPlanStore,
        build_plan_from_bd,
    )

    char_path = settings.character_path
    if character:
        char_path = char_path.parent / character
    char = load_character(char_path)
    bd_dir = settings.bd_dir_for(char.name)

    adapter = BeadsAdapter(
        bd_dir,
        default_exclude_assignee=char.bd_exclude_assignee,
        ab_assignee=char.bd_assignee,
    )
    try:
        adapter.verify()
    except BeadsAdapterError as exc:
        console.print(f"[red]bd not available at {bd_dir}: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    plan = build_plan_from_bd(
        adapter,
        assignee=assignee,
        plan_id=plan_id,
        include_closed=include_closed,
    )

    subgoal_count = sum(1 for sid in plan.subgoals if sid != plan.root_subgoal_id)
    console.print(
        f"[bold]plan bootstrap[/bold] {char.name} → {plan.id} "
        f"({subgoal_count} subgoals from bd assignee={assignee!r})"
    )

    if dry_run:
        console.print("[yellow](dry-run; nothing written)[/yellow]")
        from rich.json import JSON

        console.print(JSON.from_data(plan.to_dict()))
        return

    resolved_dir = plans_dir or (char_path / "data" / "plans")
    store = JsonPlanStore(resolved_dir)
    store.save(plan)
    console.print(f"[green]wrote {resolved_dir / (plan.id + '.json')}[/green]")


def _load_or_seed_catalog(path: Path) -> ToolCatalog:
    """Load from disk if present; else seed from the builtin metadata
    table and return (without writing). The CLI list / show paths use
    this so a never-saved catalog still surfaces builtins."""
    from harness.tools import load_catalog as _load

    if path.exists():
        return _load(path)
    cat = ToolCatalog()
    seed_builtins_into(cat, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))
    return cat


@tool_app.command("list")
def tool_list(
    tag: str | None = typer.Option(None, "--tag", help="Filter to entries with this tag."),
    family: str | None = typer.Option(None, "--family", help="Filter to entries in this family."),
    origin: str | None = typer.Option(
        None,
        "--origin",
        help="Filter by origin: builtin | synthesized | external.",
    ),
    character: str | None = typer.Option(
        None,
        "--character",
        help="Character whose tool catalog to read. Default: HARNESS_CHARACTER_NAME.",
    ),
    catalog_path: Path | None = typer.Option(
        None,
        "--catalog-path",
        help="Override the catalog file path. Default: <character>/data/tool_catalog.json.",
    ),
) -> None:
    """List tools in the catalog (harness-fx24).

    Reads from <character>/data/tool_catalog.json; if missing,
    surfaces the builtin metadata table directly. Filters compose:
    --tag X --family Y --origin Z all intersect."""
    cat = _load_or_seed_catalog(catalog_path or _resolve_catalog_path(character))
    entries = cat.all()
    if tag:
        entries = [e for e in entries if tag in e.tags]
    if family:
        entries = [e for e in entries if e.family == family]
    if origin:
        entries = [e for e in entries if e.origin == origin]

    if not entries:
        console.print("[yellow]no catalog entries match the filter[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("name")
    table.add_column("family")
    table.add_column("origin")
    table.add_column("tags")
    table.add_column("registered_at")
    for entry in entries:
        tags = ", ".join(entry.tags) if entry.tags else "—"
        table.add_row(
            entry.name,
            entry.family or "—",
            entry.origin,
            tags,
            entry.registered_at or "—",
        )
    console.print(table)
    console.print(f"\n[dim]{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}[/dim]")


@tool_app.command("show")
def tool_show(
    name: str = typer.Argument(..., help="Tool name to inspect."),
    character: str | None = typer.Option(None, "--character", help="Character to read from."),
    catalog_path: Path | None = typer.Option(None, "--catalog-path"),
    source_preview_lines: int = typer.Option(
        20,
        "--source-preview-lines",
        help="How many lines of source to preview for synthesized tools.",
    ),
) -> None:
    """Show the full catalog entry for `name` — family, tags, tier,
    origin, registered_at, plus the first ~20 lines of source for
    synthesized tools."""
    cat = _load_or_seed_catalog(catalog_path or _resolve_catalog_path(character))
    entry = cat.get(name)
    if entry is None:
        console.print(f"[red]no tool named {name!r} in catalog[/red]")
        raise typer.Exit(code=1)
    console.print(f"[bold]{entry.name}[/bold]")
    console.print(f"  family       : {entry.family or '—'}")
    console.print(f"  tags         : {', '.join(entry.tags) if entry.tags else '—'}")
    console.print(f"  tier         : {entry.tier}")
    console.print(f"  origin       : {entry.origin}")
    console.print(f"  source_path  : {entry.source_path or '—'}")
    console.print(f"  registered_at: {entry.registered_at or '—'}")
    if entry.quarantined:
        reason = entry.quarantine_reason or "no reason"
        console.print(f"[yellow]  quarantined  : yes ({reason})[/yellow]")
    if entry.description:
        console.print("\n[bold]description[/bold]")
        console.print(f"  {entry.description}")
    if entry.source_path:
        src = Path(entry.source_path)
        if src.exists():
            console.print(f"\n[bold]source preview[/bold] ({src})")
            text = src.read_text()
            lines = text.splitlines()
            visible = lines[:source_preview_lines]
            for line in visible:
                console.print(f"  {line}")
            if len(lines) > source_preview_lines:
                remaining = len(lines) - source_preview_lines
                console.print(f"  [dim]… (+{remaining} more lines)[/dim]")
        else:
            console.print(f"\n[yellow]source_path {src} does not exist on disk[/yellow]")


@tool_app.command("drop")
def tool_drop(
    name: str = typer.Argument(..., help="Tool name to remove from the catalog."),
    character: str | None = typer.Option(None, "--character"),
    catalog_path: Path | None = typer.Option(None, "--catalog-path"),
    keep_source: bool = typer.Option(
        False,
        "--keep-source",
        help="Skip deleting the synthesized tool's source file; only remove from catalog.",
    ),
) -> None:
    """Remove a tool from the catalog. Refuses on builtin entries
    (they're declared in code, not the catalog file). For synthesized
    tools, deletes the source file unless --keep-source is set.
    Idempotent — dropping an absent name is a no-op + warning."""
    from harness.tools import load_catalog as _load
    from harness.tools import save_catalog as _save

    path = catalog_path or _resolve_catalog_path(character)
    if not path.exists():
        console.print(f"[yellow]no catalog file at {path}; nothing to drop[/yellow]")
        return
    cat = _load(path)
    entry = cat.get(name)
    if entry is None:
        console.print(f"[yellow]tool {name!r} not in catalog; nothing to drop[/yellow]")
        return
    if entry.origin == "builtin":
        console.print(
            f"[red]refusing to drop builtin tool {name!r} — "
            f"builtins are declared in code, not the catalog. "
            f"Add to TOOL_PROFILES drop set or remove from BUILTIN_TOOL_METADATA instead.[/red]"
        )
        raise typer.Exit(code=1)
    cat.drop(name)
    _save(cat, path)
    console.print(f"[green]dropped {name!r} from catalog[/green]")
    if entry.source_path and not keep_source:
        src = Path(entry.source_path)
        if src.exists():
            src.unlink()
            console.print(f"[green]deleted source {src}[/green]")
        else:
            console.print(f"[dim]source {src} already gone[/dim]")


@tool_app.command("synth-rebuild")
def tool_synth_rebuild(
    character: str | None = typer.Option(None, "--character"),
    catalog_path: Path | None = typer.Option(None, "--catalog-path"),
) -> None:
    """Re-validate every synthesized catalog entry via the sandbox
    validator (harness-l2ak). Surfaces drift — a synthesized tool
    whose source file is missing, whose import allowlist no longer
    accepts a builtin it uses, etc.

    Until rqg0.5 + l2ak ship the validator + hot-reload, this command
    reports synthesized-entry counts but doesn't yet revalidate. Use
    `harness tool list --origin synthesized` to see current entries.
    """
    path = catalog_path or _resolve_catalog_path(character)
    cat = _load_or_seed_catalog(path)
    synth = cat.by_origin("synthesized")
    if not synth:
        console.print("[dim]no synthesized tools in catalog[/dim]")
        return
    console.print(f"[bold]{len(synth)} synthesized tool(s):[/bold]")
    for entry in synth:
        marker = "[yellow]quarantined[/yellow]" if entry.quarantined else "[green]ok[/green]"
        console.print(f"  - {entry.name} ({entry.family}) — {marker}")
    console.print(
        "\n[dim]revalidation pending harness-l2ak (sandbox) + harness-t5kx (hot-reload).[/dim]"
    )


@denylist_app.command("list")
def denylist_list(
    include_expired: bool = typer.Option(
        False,
        "--include-expired",
        help="Show entries past their 30-day TTL alongside active ones.",
    ),
) -> None:
    """List hosts currently blocked by the fetch_url denylist.

    Active = last 401/403 within the TTL window (30 days). Expired
    entries stay on disk for audit but no longer gate fetches; pass
    --include-expired to see them too."""
    store = _open_fetch_denylist()
    try:
        entries = store.list_all(include_expired=include_expired)
    finally:
        store.close()
    if not entries:
        scope = "any" if include_expired else "active"
        console.print(f"[dim]no {scope} denylist entries[/dim]")
        return
    now = datetime.now(UTC)
    for entry in entries:
        active = entry.is_active(now=now, ttl_days=store.ttl_days)
        status_tag = "[green]active[/green]" if active else "[dim]expired[/dim]"
        age_days = (now - entry.last_seen_at).days
        console.print(
            f"  {entry.host}  HTTP {entry.last_status} {entry.last_reason}  "
            f"({entry.count} hit{'s' if entry.count != 1 else ''}, "
            f"last seen {age_days}d ago) {status_tag}"
        )


@denylist_app.command("clear")
def denylist_clear(
    host: str | None = typer.Option(
        None,
        "--host",
        help="Remove this host only. Without --host, all entries are dropped.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirm prompt when clearing everything.",
    ),
) -> None:
    """Remove entries from the fetch_url denylist.

    Use this after a host's 401/403 was transient and you want the
    agent to retry sooner than the 30-day TTL."""
    if host is None and not yes:
        confirm = typer.confirm("Clear ALL denylist entries?", default=False)
        if not confirm:
            console.print("[dim]cancelled[/dim]")
            return
    store = _open_fetch_denylist()
    try:
        removed = store.clear(host)
    finally:
        store.close()
    target = f"host {host!r}" if host else "all entries"
    console.print(f"[dim]removed {removed} row(s) for {target}[/dim]")


@denylist_app.command("add")
def denylist_add(
    host: str = typer.Argument(..., help="Hostname to block (e.g. example.com)."),
    status: int = typer.Option(
        403,
        "--status",
        help="HTTP status to record. Defaults to 403.",
    ),
    reason: str = typer.Option(
        "manual",
        "--reason",
        help="Free-text reason recorded with the entry.",
    ),
) -> None:
    """Manually add a host to the denylist.

    Useful when you already know a domain will refuse fetches and
    want to skip the first round-trip. Mirrors a fresh 401/403 record."""
    store = _open_fetch_denylist()
    try:
        entry = store.record(
            host=host,
            status=status,
            reason=reason,
            url=f"https://{host}/",
        )
    finally:
        store.close()
    console.print(
        f"[dim]added {entry.host} (HTTP {entry.last_status} {entry.last_reason}, "
        f"count={entry.count})[/dim]"
    )


@web_app.command("serve")
def web_serve(
    host: str = typer.Option(
        "0.0.0.0",  # noqa: S104 — Tailscale-only ingress relies on host firewall + ACLs
        "--host",
        help=(
            "Bind interface. Default 0.0.0.0; Tailscale-only enforcement "
            "is provided by the Mac firewall + Tailscale ACLs (standard "
            "tailnet pattern). Override to '127.0.0.1' for local-only "
            "development."
        ),
    ),
    port: int = typer.Option(8080, "--port", help="TCP port to bind."),
    model: str = typer.Option(
        "mlx",
        "--model",
        help="Adapter: echo | mlx | ollama | vllm. Echo is the fastest smoke test.",
    ),
    model_repo: str | None = typer.Option(
        None, "--model-repo", help="HF repo override for the chosen adapter."
    ),
    rate_limit: str = typer.Option(
        "120/minute",
        "--rate-limit",
        help=(
            "Per-IP rate-limit budget (slowapi syntax). Generous default "
            "for invite-list tailnet audiences."
        ),
    ),
    reload: bool = typer.Option(False, "--reload", help="Run uvicorn with auto-reload (dev only)."),
) -> None:
    """Serve the resolved character over HTTP via the harness.web
    factory (harness-3jz1.9). Mounts the base endpoints (/healthz,
    /character, /chat, /capabilities) plus any character-specific
    extension at harness.web.characters.<name>."""
    character = load_character(settings.character_path)
    if model == "mlx" and model_repo:
        from harness.model.mlx import MLXAdapter

        adapter: ModelAdapter = MLXAdapter(repo=model_repo)
    elif model == "ollama" and model_repo:
        from harness.model.ollama import OllamaAdapter

        adapter = OllamaAdapter(model=model_repo)
    else:
        try:
            adapter = make_adapter(cast(AdapterName, model))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    loader = getattr(adapter, "load", None)
    if callable(loader):
        with Status(f"loading {adapter.id}…", console=console):
            loader()

    from harness.web import build_character_app

    web_app_instance = build_character_app(character, adapter, rate_limit=rate_limit)
    console.print(
        f"[bold green]◈ {character.name}[/bold green] → http://{host}:{port}  (model={adapter.id})"
    )

    import uvicorn

    uvicorn.run(web_app_instance, host=host, port=port, reload=reload)
