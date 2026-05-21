"""CLI surface for the multi-turn driver — harness-s83d.

Exposes two subcommands under `harness drive`:

  - `harness drive plan` runs the planner phase. Two modes:
      * default — drive the LLM through the spec doc, emit a YAML draft.
      * `--commit DRAFT.yaml` — read a (reviewed) YAML draft and
        materialize it in bd as epic + child issues + dep edges.
  - `harness drive loop` runs the executor phase. Two modes:
      * default — start a fresh run against an `--epic` id.
      * `--resume LOOP_RUN_ID` — rehydrate an in-progress run.
      * `--list-runs` — enumerate `.harness/loop_runs/*.json` snapshots.

Name choice: `harness drive` (not `harness plan`/`harness loop`) so the
new subcommands don't collide with the existing `harness plan` runtime-
typed plan tools (harness-ptdw). Grouping under `drive` also makes the
relationship between planner and executor explicit.

`harness drive loop` refuses to run on a dirty git tree by default
(uncommitted changes confuse the "files touched this loop_run" diff in
the handoff builder). `--allow-dirty` overrides for operators who know
what they're doing.

The driver CLI does NOT import `_resolve_adapter` from `cli.py` — that
helper depends on `settings`, character loading, persona wrapping, and
all the rest of the chat machinery. The driver path needs none of that:
the executor turn skips persona-rewrite and voice-retrieval by design,
so `make_adapter(name)` is enough. For v0 the model is `--model echo`
unless overridden — the executor's actual work happens via real MLX/
Ollama runs but the CLI surface tests are model-agnostic.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from harness.character import load_character
from harness.driver.bd import DriverBd
from harness.driver.loop import LoopConfig, LoopResult, run_loop
from harness.driver.planner import (
    PlannerConfig,
    PlannerError,
    commit_plan,
    run_planner,
    write_draft,
)
from harness.driver.state import LoopRunState
from harness.model.factory import AdapterName, make_adapter

drive_app = typer.Typer(
    help="Multi-turn driver — planner + executor (harness-e9oq).",
    no_args_is_help=True,
)


# --- harness drive plan ------------------------------------------------


@drive_app.command("plan")
def plan_command(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to the spec doc the planner reads.",
    ),
    epic_title: str = typer.Option(
        "",
        "--epic-title",
        help="High-level title for the parent epic (required unless --commit).",
    ),
    draft_path: Path = typer.Option(
        Path("./harness-plan-draft.yaml"),
        "--draft",
        help="Where to write the YAML draft (or read it from when --commit).",
    ),
    max_plan_turns: int = typer.Option(
        5,
        "--max-plan-turns",
        help="Outer-loop cap on planner iterations.",
    ),
    workspace: Path = typer.Option(
        Path.cwd(),  # noqa: B008 — typer evaluates at call time, not import time
        "--workspace",
        help="Workspace root (planner read_file/grep/glob sandbox + bd cwd).",
    ),
    model: str = typer.Option(
        "echo",
        "--model",
        help="Model adapter for the planner: echo | mlx | ollama.",
    ),
    commit: bool = typer.Option(
        False,
        "--commit",
        help="Skip the planner; read --draft and materialize it in bd.",
    ),
) -> None:
    """Run the planner (default) or commit a reviewed YAML draft (--commit).

    Planner mode emits `--draft` (default ./harness-plan-draft.yaml).
    The operator reviews + edits, then re-runs with --commit to
    materialize.
    """
    if commit:
        bd = DriverBd(bd_dir=workspace)
        try:
            epic_id = commit_plan(draft_path, bd, spec)
        except PlannerError as exc:
            typer.echo(f"plan commit failed:\n{exc}", err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(f"committed plan: epic id = {epic_id}")
        return

    if not epic_title:
        raise typer.BadParameter("--epic-title is required when running the planner")
    if not spec.exists():
        raise typer.BadParameter(f"spec {spec} does not exist")

    adapter = make_adapter(_validate_model(model))
    config = PlannerConfig(
        spec_path=spec,
        epic_title=epic_title,
        workspace=workspace,
        max_plan_turns=max_plan_turns,
        draft_path=draft_path,
    )
    draft = run_planner(adapter, config)
    write_draft(draft, draft_path)
    typer.echo(
        f"wrote plan draft to {draft_path}: {len(draft.items)} item(s). "
        f"Review, then run `harness drive plan --commit {draft_path} --spec {spec}`."
    )


# --- harness drive loop ------------------------------------------------


@drive_app.command("loop")
def loop_command(
    epic: str = typer.Option(
        "",
        "--epic",
        help="bd id of the epic to drain (omitted when --resume or --list-runs).",
    ),
    resume_from: str = typer.Option(
        "",
        "--resume",
        help="Loop-run id to rehydrate from state. Mutually exclusive with --epic.",
    ),
    list_runs: bool = typer.Option(
        False,
        "--list-runs",
        help="Enumerate .harness/loop_runs/*.json snapshots and exit.",
    ),
    max_turns: int = typer.Option(
        20,
        "--max-turns",
        help="Hard cap on iterations.",
    ),
    workspace: Path = typer.Option(
        Path.cwd(),  # noqa: B008 — typer evaluates at call time, not import time
        "--workspace",
        help="Workspace root (bd_dir, git_root, sandbox for filesystem tools).",
    ),
    log_path: Path | None = typer.Option(
        None,
        "--log",
        help="Override progress log path. Default: <workspace>/.harness/loop_runs/<id>.log.",
    ),
    model: str = typer.Option(
        "echo",
        "--model",
        help="Model adapter for executor turns: echo | mlx | ollama.",
    ),
    character_path: Path = typer.Option(
        Path("./character/airton"),
        "--character",
        help="Character directory; system_prompt(include_samples=()) is used.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Assemble + print one handoff and exit without running the model.",
    ),
    allow_dirty: bool = typer.Option(
        False,
        "--allow-dirty",
        help="Allow running on a dirty git tree (uncommitted changes).",
    ),
) -> None:
    """Drive a bd epic to closure across multiple turns."""
    if list_runs:
        _list_runs(workspace)
        return

    if resume_from and epic:
        raise typer.BadParameter("--epic and --resume are mutually exclusive")
    if not (resume_from or epic):
        raise typer.BadParameter("one of --epic or --resume is required")

    if not allow_dirty and _git_tree_is_dirty(workspace):
        typer.echo(
            "git tree has uncommitted changes; the handoff's "
            "files-touched diff would be misleading. Pass --allow-dirty "
            "to override.",
            err=True,
        )
        raise typer.Exit(code=2)

    adapter = make_adapter(_validate_model(model))
    character = load_character(character_path)
    bd = DriverBd(bd_dir=workspace)
    config = LoopConfig(
        epic_id=epic or "",  # state file overrides when --resume is set
        workspace=workspace,
        character=character,
        max_turns=max_turns,
        resume_from=resume_from or None,
        dry_run=dry_run,
        log_path=log_path,
    )
    result = run_loop(adapter, bd, config)
    _print_result(result)
    raise typer.Exit(code=_exit_code(result))


# --- helpers ----------------------------------------------------------


def _validate_model(name: str) -> AdapterName:
    if name not in ("echo", "mlx", "ollama"):
        raise typer.BadParameter(f"--model must be one of: echo | mlx | ollama (got {name!r})")
    return name  # type: ignore[return-value]  # Literal narrowed by the check above


def _git_tree_is_dirty(workspace: Path) -> bool:
    """True iff `git status --porcelain` produces any output. On any
    git error (no repo, no git) returns False so the loop can still
    run in a non-git workspace — operators who actually want the
    dirty-check enabled will be inside a real repo."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607 — git on PATH is expected
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def _list_runs(workspace: Path) -> None:
    """Enumerate state files in `.harness/loop_runs/`. Each line:
    `<id>  epic=<epic>  turns=<n>  closed=<n>  started=<iso>`."""
    state_dir = LoopRunState.state_dir(workspace)
    if not state_dir.exists():
        typer.echo("(no loop runs yet)")
        return
    files = sorted(state_dir.glob("*.json"))
    if not files:
        typer.echo("(no loop runs yet)")
        return
    for path in files:
        try:
            state = LoopRunState.load(path)
        except Exception as exc:
            typer.echo(f"{path.stem}  (load error: {exc})")
            continue
        typer.echo(
            f"{state.loop_run_id}  epic={state.epic_id}  "
            f"turns={state.turns_used}  closed={len(state.closed_this_run)}  "
            f"started={state.started_at.isoformat(timespec='seconds')}"
        )


def _print_result(result: LoopResult) -> None:
    """Print a one-line summary + any halt / interrupt details that
    the operator needs to act on."""
    typer.echo(
        f"loop_run={result.loop_run_id} exit={result.exit_reason} "
        f"turns_used={result.turns_used} closed={len(result.closed)}"
    )
    if result.exit_reason == "halted":
        typer.echo(
            f"HALTED on {result.halted_on} — flagged via `bd human`. "
            f"Inspect with `bd show {result.halted_on}` and `bd human-list`.",
            err=True,
        )
    elif result.exit_reason == "interrupted":
        typer.echo(
            f"INTERRUPTED — resume with `harness drive loop --resume {result.loop_run_id}`.",
            err=True,
        )
    elif result.exit_reason == "exhausted":
        typer.echo(
            f"EXHAUSTED at max_turns. Resume with "
            f"`harness drive loop --resume {result.loop_run_id} --max-turns <N>`.",
            err=True,
        )


def _exit_code(result: LoopResult) -> int:
    """Map exit_reason → process exit code. Success / dry_run = 0;
    halted = 2 (operator action needed); exhausted / interrupted = 1."""
    if result.exit_reason in ("success", "dry_run"):
        return 0
    if result.exit_reason == "halted":
        return 2
    return 1


# Module-level smoke test entry — `python -m harness.driver.cli plan ...`
# delegates to the typer app. Keeps the driver CLI exercisable in
# isolation without the full `harness` entry point.
def main() -> None:
    drive_app()


if __name__ == "__main__":
    main()


__all__ = ["drive_app", "main"]
