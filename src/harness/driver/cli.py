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

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import typer

if TYPE_CHECKING:
    from harness.driver.auto_iterate import AutoIterateResult

from harness.character import load_character
from harness.driver.bd import DriverBd, DriverBdError
from harness.driver.loop import LoopConfig, LoopResult, run_loop
from harness.driver.plan_linter import score_bead
from harness.driver.planner import (
    PlannerConfig,
    PlannerError,
    commit_plan,
    run_planner,
    write_draft,
)
from harness.driver.state import LoopRunState
from harness.driver.workspace_verify import missing_entry_html_reason
from harness.model.adapter import ModelAdapter
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
        help="Model adapter for the planner: echo | mlx | ollama | vllm.",
    ),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help=(
            "Override default HF repo (mlx), Ollama model tag (ollama), "
            "or vLLM OpenAI-compatible base URL (vllm)."
        ),
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="LoRA adapter path (MLX only).",
    ),
    draft_repo: str | None = typer.Option(
        None,
        "--draft-repo",
        help="Speculative-decoding draft model HF repo (MLX only).",
    ),
    commit: bool = typer.Option(
        False,
        "--commit",
        help="Skip the planner; read --draft and materialize it in bd.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Mirror the planner's tool-loop event stream to stderr.",
    ),
) -> None:
    """Run the planner (default) or commit a reviewed YAML draft (--commit).

    Planner mode emits `--draft` (default ./harness-plan-draft.yaml).
    The operator reviews + edits, then re-runs with --commit to
    materialize. The planner ALWAYS writes a per-run event log under
    `<workspace>/.harness/planner_<ts>.log` regardless of --verbose;
    the flag adds live stderr mirroring for debugging an empty / wrong
    draft.
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

    adapter = _resolve_driver_adapter(
        model,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
    )
    config = PlannerConfig(
        spec_path=spec,
        epic_title=epic_title,
        workspace=workspace,
        max_plan_turns=max_plan_turns,
        draft_path=draft_path,
    )
    observer = _stderr_observer if verbose else None
    draft = run_planner(adapter, config, observe=observer)
    write_draft(draft, draft_path)
    typer.echo(
        f"wrote plan draft to {draft_path}: {len(draft.items)} item(s). "
        f"Review, then run `harness drive plan --commit {draft_path} --spec {spec}`."
    )
    if not draft.items:
        typer.echo(
            "WARNING: draft has zero items. Check the planner event log at "
            f"{workspace / '.harness'} (planner_<ts>.log) — the model likely "
            "emitted prose instead of tool calls, or every plan_add call "
            "failed validation. Re-run with --verbose to mirror events to "
            "stderr.",
            err=True,
        )


# --- harness drive lint-epic (harness-bpix) ----------------------------


@drive_app.command("lint-epic")
def lint_epic_command(
    epic: str = typer.Option("", "--epic", help="bd id of the epic to lint."),
    workspace: Path = typer.Option(
        Path.cwd(),  # noqa: B008 — typer evaluates at call time
        "--workspace",
        help="Workspace root (resolves the bd dir).",
    ),
    include_closed: bool = typer.Option(
        False,
        "--include-closed",
        help="Also score already-closed children (default: open only).",
    ),
) -> None:
    """Score each open child of an epic for decomposition risk and flag
    the over-scoped ones BEFORE a drive burns turns discovering them
    (harness-bpix). Pure text-shape heuristic — no model call. Exits 1
    when any bead is flagged so it can gate a pre-drive check."""
    if not epic:
        raise typer.BadParameter("--epic is required")
    bd = DriverBd(bd_dir=workspace)
    try:
        epic_issue = bd.show(epic)
    except DriverBdError as exc:
        typer.echo(f"lint-epic: cannot load epic {epic}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    child_ids = [
        str(d.get("id")) for d in (epic_issue.raw.get("dependencies") or []) if d.get("id")
    ]
    scored = []
    for cid in child_ids:
        try:
            child = bd.show(cid)
        except DriverBdError:
            continue
        if child.status == "closed" and not include_closed:
            continue
        scored.append(
            score_bead(
                cid,
                child.title,
                str(child.raw.get("description", "")),
                str(child.raw.get("acceptance_criteria", "") or ""),
                workspace=workspace,
            )
        )

    if not scored:
        typer.echo(f"lint-epic: no open children under {epic} to score.")
        return

    # Flagged first, then by sub-section spread — worst offenders on top.
    scored.sort(key=lambda c: (c.flagged, c.subsection_refs, c.description_chars), reverse=True)
    flagged = [c for c in scored if c.flagged]

    typer.echo(f"lint-epic {epic}: {len(scored)} open bead(s), {len(flagged)} flagged\n")
    typer.echo(f"  {'FLAG':<5}{'§subs':>6}{'chars':>7}{'bullets':>8}  id / title")
    for c in scored:
        mark = "⚠" if c.flagged else " "
        title = c.title if len(c.title) <= 52 else c.title[:49] + "..."
        cols = f"{mark:<5}{c.subsection_refs:>6}{c.description_chars:>7}{c.bullets:>8}"
        typer.echo(f"  {cols}  {c.bead_id}  {title}")
    if flagged:
        typer.echo("\nDecomposition candidates:")
        for c in flagged:
            typer.echo(f"  {c.bead_id} {c.title}")
            for r in c.reasons:
                typer.echo(f"    - {r}")
        typer.echo(
            "\nConsider splitting these into per-sub-section beads before driving "
            "(each one is a wall that costs ~5+ turns to discover mid-drive)."
        )
        raise typer.Exit(code=1)


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
        help="Model adapter for executor turns: echo | mlx | ollama | vllm.",
    ),
    model_repo: str | None = typer.Option(
        None,
        "--model-repo",
        help=(
            "Override default HF repo (mlx), Ollama model tag (ollama), "
            "or vLLM OpenAI-compatible base URL (vllm)."
        ),
    ),
    lora_path: str | None = typer.Option(
        None,
        "--lora-path",
        help="LoRA adapter path (MLX only).",
    ),
    draft_repo: str | None = typer.Option(
        None,
        "--draft-repo",
        help="Speculative-decoding draft model HF repo (MLX only).",
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
    allow_missing_smoke: bool = typer.Option(
        False,
        "--allow-missing-smoke",
        help=(
            "Allow running a browser-JS workspace that has no index.html "
            "(runtime smoke gate OFF → syntax-only closes). Refused by "
            "default (harness-9ugc false-success guard)."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Mirror the executor's per-turn tool-loop event stream to stderr.",
    ),
    no_verify: bool = typer.Option(
        False,
        "--no-verify",
        help=(
            "Disable post-turn forbidden-pattern verification "
            "(default checks for TODO/FIXME/XXX/HACK in modified files)."
        ),
    ),
    executor_max_rounds: int = typer.Option(
        12,
        "--executor-max-rounds",
        help=(
            "Per-turn round budget for the inner run_tool_loop. Default 12 "
            "(orchestrator's default 8 is too tight for multi-edit work)."
        ),
    ),
    max_attempts: int = typer.Option(
        3,
        "--max-attempts",
        help=(
            "Per-issue retry budget before the issue is parked (skip-on) "
            "or the run halts. Default 3. Raise it to give hard issues "
            "more tries on a run aiming for full closure."
        ),
    ),
    regression_guard: bool = typer.Option(
        True,
        "--regression-guard/--no-regression-guard",
        help=(
            "Guard the baseline against regressions (harness-16w6): fail "
            "verify when an edit deletes previously-defined symbols or "
            "shrinks the deliverable vs the last-green snapshot, and on "
            "park restore last-green so a break can't cascade. Default on."
        ),
    ),
    scratch_sweep: bool = typer.Option(
        True,
        "--scratch-sweep/--no-scratch-sweep",
        help=(
            "Archive agent-created scratch (plans, temp_*, validate_*, "
            "backups) into .harness/loop_runs/<id>_scratch/ when the issue "
            "that made it completes (harness-ul5z). Default on."
        ),
    ),
    snapshot: bool = typer.Option(
        True,
        "--snapshot/--no-snapshot",
        help=(
            "Tar+gzip the workspace to .harness/loop_runs/<id>_workspace.tar.gz "
            "before the first turn (default on; harness-9ijr). Disables for "
            "workspaces over the 100MB cap or when the operator wants to "
            "skip the safety net. Resume runs always skip the snapshot — the "
            "original run's tarball is the recovery point."
        ),
    ),
    plan_draft: Path | None = typer.Option(
        None,
        "--plan-draft",
        help=(
            "YAML plan draft committed for this epic (harness-xfh2). When "
            "set, the loop runs each item's `verify` shell commands after "
            "the model closes its bd issue; a non-zero exit reopens the "
            "issue and stashes the failure in the next handoff so the "
            "model self-corrects. Items without `verify` entries (and runs "
            "without --plan-draft) behave as before."
        ),
    ),
    use_fsm: bool = typer.Option(
        False,
        "--fsm/--no-fsm",
        help=(
            "Route each turn through the TurnFSM (ASSESS → WRITE_TEST → "
            "IMPLEMENT → VERIFY → CLOSE) instead of the legacy "
            "single-shot executor (harness-kbnl). The FSM constrains "
            "the tool roster per phase + requires explicit meta-tool "
            "transitions (submit_assessment, submit_failing_test, etc.) "
            "to advance. Off by default while the FSM bakes in."
        ),
    ),
    tdd: bool = typer.Option(
        True,
        "--tdd/--no-tdd",
        help=(
            "When --fsm is set, require the WRITE_TEST phase (default). "
            "--no-tdd routes ASSESS directly to IMPLEMENT, skipping the "
            "failing-test gate. Use for runs where TDD genuinely doesn't "
            "apply (UI/visual changes, docs). The model can also opt "
            "out per-issue via submit_assessment(tdd_applicable=False) — "
            "this flag is the operator's blanket override."
        ),
    ),
    summarize_tool_results: bool = typer.Option(
        True,
        "--summarize-tool-results/--no-summarize-tool-results",
        help=(
            "Compress high-noise tool outputs (grep / list_dir / "
            "search_web / fetch_url …) before they reach the model's "
            "context (harness-tu4o). Default ON for drive runs because "
            "they're unattended and a single long executor turn can "
            "otherwise pile up tool results until the model rejects "
            "the prompt. Uses --model as the summarizer adapter."
        ),
    ),
    auto_close_on_claim: bool = typer.Option(
        True,
        "--auto-close-on-claim/--no-auto-close-on-claim",
        help=(
            "When the claim-without-close gate fires AND verify ran at "
            "least one step that passed, close the bd issue on the "
            "model's behalf (harness-b7m1). Default ON because small "
            "models routinely spend their entire retry budget "
            "re-verifying instead of running `bd close` even with "
            "explicit feedback. Falls back to the soft-hint retry when "
            "verify failed, no steps ran, or the bd close subprocess "
            "itself errored. Set --no-auto-close-on-claim to require "
            "the model to drive the close itself."
        ),
    ),
    skip_on_max_attempts: bool = typer.Option(
        True,
        "--skip-on-max-attempts/--no-skip-on-max-attempts",
        help=(
            "On max-attempts exhaustion for a bd issue, park it via "
            "`bd flag_human` and continue the drive against the next "
            "ready issue (harness-zcrd). Default ON because one hard "
            "issue shouldn't kill the whole drive — operators pick up "
            "parked issues via `bd human list` after the run. Set "
            "--no-skip-on-max-attempts for the pre-zcrd behavior where "
            "max-attempts halts the entire run."
        ),
    ),
    render_milestone: str | None = typer.Option(
        None,
        "--render-milestone",
        help=(
            "bd issue id that marks 'the workspace should now render "
            "something' (e.g. the §2 world-map issue). The blank-canvas "
            "smoke check is suppressed until this issue closes, so an "
            "incremental from-scratch build whose early sections (§1 "
            "skeleton) legitimately render nothing isn't failed on every "
            "turn (harness-6dsn). Console/page-error checks always run. "
            "Omit to enforce blank-canvas always (right for an "
            "already-built game)."
        ),
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

    # harness-9ugc: refuse a browser-JS workspace with no entry index.html
    # — the runtime smoke gate would be silently OFF and the drive would
    # false-close on syntax only (the run 3c7c9da2 failure).
    if not allow_missing_smoke:
        smoke_gap = missing_entry_html_reason(workspace)
        if smoke_gap is not None:
            typer.echo(f"refusing to drive: {smoke_gap}", err=True)
            raise typer.Exit(code=2)

    adapter = _resolve_driver_adapter(
        model,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
    )
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
        extra_observer=_stderr_observer if verbose else None,
        forbidden_patterns=() if no_verify else ("TODO", "FIXME", "XXX", "HACK"),
        executor_max_rounds=executor_max_rounds,
        max_attempts_per_issue=max_attempts,
        regression_guard=regression_guard,
        scratch_sweep=scratch_sweep,
        snapshot=snapshot,
        plan_draft_path=plan_draft,
        use_fsm=use_fsm,
        tdd_required=tdd,
        summarize_tool_results=summarize_tool_results,
        auto_close_on_claim=auto_close_on_claim,
        skip_on_max_attempts=skip_on_max_attempts,
        render_milestone_id=render_milestone,
    )
    result = run_loop(adapter, bd, config)
    _print_result(result)
    raise typer.Exit(code=_exit_code(result))


# --- harness drive auto-iterate (harness-3f8e) ------------------------


@drive_app.command("auto-iterate")
def auto_iterate_command(
    epic: str = typer.Option("", "--epic", help="bd id of the epic to drain."),
    spec: Path | None = typer.Option(
        None,
        "--spec",
        help=(
            "Path to the source spec used by the critic. Default: resolve "
            "from the epic's `plan-source:<file>` label."
        ),
    ),
    max_passes: int = typer.Option(
        8,
        "--max-passes",
        help="Hard cap on drive+critic passes.",
    ),
    convergence_streak: int = typer.Option(
        2,
        "--convergence-streak",
        help="Empty critic passes in a row before declaring converged.",
    ),
    critic_max_findings: int = typer.Option(
        10,
        "--critic-max-findings",
        help="Cap on beads filed per critic pass.",
    ),
    critic_slice: bool = typer.Option(
        True,
        "--critic-slice/--no-critic-slice",
        help="Symbol-scoped critique: critique one symbol-aligned slice at a time "
        "(anchors the model to real code, harness-a0yj). --no-critic-slice uses the "
        "legacy whole-file prompt with a context budget (harness-zk3c).",
    ),
    # Forwarded `drive loop` options (same names + semantics).
    max_turns: int = typer.Option(20, "--max-turns"),
    workspace: Path = typer.Option(
        Path.cwd(),  # noqa: B008 — typer evaluates at call time
        "--workspace",
    ),
    model: str = typer.Option("echo", "--model"),
    model_repo: str | None = typer.Option(None, "--model-repo"),
    lora_path: str | None = typer.Option(None, "--lora-path"),
    draft_repo: str | None = typer.Option(None, "--draft-repo"),
    context_window: int | None = typer.Option(
        None,
        "--context-window",
        help="Override the adapter context window (vLLM only). Set this to match "
        "a server relaunched with a larger --max-model-len so the critic budgets "
        "against the real window (harness-zk3c). Default: adapter's own (vLLM 32768).",
    ),
    character_path: Path = typer.Option(Path("./character/airton"), "--character"),
    allow_dirty: bool = typer.Option(False, "--allow-dirty"),
    allow_missing_smoke: bool = typer.Option(False, "--allow-missing-smoke"),
    executor_max_rounds: int = typer.Option(12, "--executor-max-rounds"),
    max_attempts: int = typer.Option(3, "--max-attempts"),
    regression_guard: bool = typer.Option(True, "--regression-guard/--no-regression-guard"),
    use_fsm: bool = typer.Option(False, "--fsm/--no-fsm"),
    tdd: bool = typer.Option(True, "--tdd/--no-tdd"),
    summarize_tool_results: bool = typer.Option(
        True, "--summarize-tool-results/--no-summarize-tool-results"
    ),
    auto_close_on_claim: bool = typer.Option(
        True, "--auto-close-on-claim/--no-auto-close-on-claim"
    ),
    skip_on_max_attempts: bool = typer.Option(
        True, "--skip-on-max-attempts/--no-skip-on-max-attempts"
    ),
    render_milestone: str | None = typer.Option(None, "--render-milestone"),
) -> None:
    """Drive + critic until convergence: drains the epic, asks the model
    to propose follow-up bugs grounded in the spec, files them as
    auto-blocked beads, and drives again. Stops when N consecutive
    critic passes return zero findings or `--max-passes` hits.
    """
    if not epic:
        raise typer.BadParameter("--epic is required")
    if not allow_dirty and _git_tree_is_dirty(workspace):
        typer.echo(
            "git tree has uncommitted changes; pass --allow-dirty to override.",
            err=True,
        )
        raise typer.Exit(code=2)
    if not allow_missing_smoke:
        smoke_gap = missing_entry_html_reason(workspace)
        if smoke_gap is not None:
            typer.echo(f"refusing to drive: {smoke_gap}", err=True)
            raise typer.Exit(code=2)

    from harness.driver.auto_iterate import AutoIterateConfig, run_auto_iterate

    adapter = _resolve_driver_adapter(
        model,
        model_repo=model_repo,
        lora_path=lora_path,
        draft_repo=draft_repo,
        context_window=context_window,
    )
    character = load_character(character_path)
    bd = DriverBd(bd_dir=workspace)
    loop_config = LoopConfig(
        epic_id=epic,
        workspace=workspace,
        character=character,
        max_turns=max_turns,
        executor_max_rounds=executor_max_rounds,
        max_attempts_per_issue=max_attempts,
        regression_guard=regression_guard,
        use_fsm=use_fsm,
        tdd_required=tdd,
        summarize_tool_results=summarize_tool_results,
        auto_close_on_claim=auto_close_on_claim,
        skip_on_max_attempts=skip_on_max_attempts,
        render_milestone_id=render_milestone,
    )
    config = AutoIterateConfig(
        loop_config=loop_config,
        spec_path=spec,
        max_passes=max_passes,
        convergence_streak=convergence_streak,
        critic_max_findings=critic_max_findings,
        critic_slice_mode=critic_slice,
    )
    result: AutoIterateResult = run_auto_iterate(adapter, bd, config)
    _print_auto_iterate_result(result)
    raise typer.Exit(code=_auto_iterate_exit_code(result))


def _print_auto_iterate_result(result: AutoIterateResult) -> None:
    spec_mode = "resolved" if result.spec_resolved else "artifact-only"
    typer.echo(
        f"auto-iterate: passes={result.passes_run} "
        f"critic_findings={result.critic_findings_total} "
        f"spec={spec_mode} "
        f"exit={result.exit_reason}"
    )
    if result.exit_reason == "no_work":
        typer.echo(
            "  no ready work under the epic — drive ran zero turns. "
            "The epic may be complete, or its remaining work isn't wired "
            "as a direct dependency (ready_under_epic is one level deep)."
        )
    if result.filed_beads:
        typer.echo(f"filed beads ({len(result.filed_beads)}):")
        for bid in result.filed_beads:
            typer.echo(f"  - {bid}")
    for i, dr in enumerate(result.drive_results, start=1):
        typer.echo(
            f"  pass {i}: drive={dr.exit_reason} closed={len(dr.closed)} turns={dr.turns_used}"
        )


def _auto_iterate_exit_code(result: AutoIterateResult) -> int:
    if result.exit_reason == "converged":
        return 0
    if result.exit_reason == "drive_halted":
        return 1
    if result.exit_reason == "critic_failed":
        # The critic's model call failed (outage/timeout) — distinct from a
        # clean finish. Non-zero so a wrapper retries rather than trusting
        # a false "converged" (harness-fote).
        return 3
    if result.exit_reason == "stuck":
        # Drive made zero progress and stalled (exhausted / all-parked) with
        # no new critic findings — the work is stuck, not done. Distinct code
        # so a wrapper can branch on it (harness-dqoy).
        return 4
    if result.exit_reason == "no_work":
        # The epic's ready queue was empty from the start (e.g. all children
        # already closed) — the drive ran zero turns and the critic had an
        # unchanged artifact. Non-zero so a no-op run isn't read as a real
        # finish; distinct from "stuck" so a wrapper can tell "nothing to do"
        # from "jammed" (harness-eh07).
        return 5
    # passes_exhausted: ran out of budget with critic still finding bugs.
    # Surface as non-zero so a CI / cron wrapper sees the unfinished state.
    return 2


# --- helpers ----------------------------------------------------------


def _stderr_observer(event: object) -> None:
    """Planner --verbose mirror — formats a ToolLoopEvent for stderr.
    Imports inside the function so the type isn't required at module
    load (keeps the CLI's import graph shallow)."""
    from harness.driver.planner import _format_event
    from harness.orchestrator import ToolLoopEvent

    if isinstance(event, ToolLoopEvent):
        typer.echo(_format_event(event), err=True)


_ADAPTER_NAMES: tuple[str, ...] = get_args(AdapterName)


def _validate_model(name: str) -> AdapterName:
    if name not in _ADAPTER_NAMES:
        choices = " | ".join(_ADAPTER_NAMES)
        raise typer.BadParameter(f"--model must be one of: {choices} (got {name!r})")
    return name  # type: ignore[return-value]  # Literal narrowed by the check above


def _resolve_driver_adapter(
    name: str,
    *,
    model_repo: str | None,
    lora_path: str | None,
    draft_repo: str | None,
    context_window: int | None = None,
) -> ModelAdapter:
    """Driver-scoped adapter resolver (harness-gu6k). Mirrors the
    relevant subset of cli._resolve_adapter:
      - default path: make_adapter(name)
      - --model-repo: instantiate MLXAdapter or OllamaAdapter directly
      - --lora-path / --draft-repo: mlx-only, raise BadParameter otherwise

    Persona wrapping / chain_rewrites are NOT supported — drive
    deliberately runs without voice retrieval or persona rewrite (see
    harness-ml66)."""
    validated = _validate_model(name)
    if lora_path and validated != "mlx":
        raise typer.BadParameter("--lora-path requires --model mlx.")
    if draft_repo and validated != "mlx":
        raise typer.BadParameter("--draft-repo requires --model mlx.")
    if context_window is not None and validated != "vllm":
        raise typer.BadParameter("--context-window requires --model vllm.")
    if not (model_repo or lora_path or draft_repo or context_window):
        return make_adapter(validated)
    if validated == "mlx":
        from harness.model.mlx import MLXAdapter

        mlx_kwargs: dict[str, object] = {}
        if model_repo:
            mlx_kwargs["repo"] = model_repo
        if lora_path:
            mlx_kwargs["adapter_path"] = lora_path
        if draft_repo:
            mlx_kwargs["draft_repo"] = draft_repo
        return MLXAdapter(**mlx_kwargs)  # type: ignore[arg-type]
    if validated == "ollama":
        from harness.model.ollama import OllamaAdapter

        return OllamaAdapter(model=model_repo) if model_repo else OllamaAdapter()
    if validated == "vllm":
        from harness.model.vllm import VllmAdapter

        vllm_kwargs: dict[str, object] = {}
        if model_repo:
            vllm_kwargs["base_url"] = model_repo
        if context_window is not None:
            vllm_kwargs["context_window"] = context_window
        return VllmAdapter(**vllm_kwargs)  # type: ignore[arg-type]
    # echo: model_repo not meaningful; refuse rather than silently ignore.
    raise typer.BadParameter(
        f"--model-repo not supported for --model {validated}; use mlx, ollama, or vllm."
    )


# Paths that are routinely "dirty" during normal harness usage and
# don't represent meaningful uncommitted work (harness-m0g4):
#   - `.beads/`: bd's own state files (export-state.json, interactions.jsonl,
#     issues.jsonl). Every `bd close` / `bd update` an executor turn runs
#     via shell modifies these — the loop is GOING to dirty them
#     legitimately. Excluding them up front is the only way the dirty
#     gate stays useful.
_DIRTY_CHECK_IGNORE_PREFIXES: tuple[str, ...] = (".beads/",)


def _git_tree_is_dirty(workspace: Path) -> bool:
    """True iff any TRACKED, non-`.beads/` file is modified relative to
    HEAD. Untracked files are ignored — they're invisible to the
    handoff's `git diff --name-status` and can't make the diff
    misleading. On any git error (no repo, no git) returns False so the
    loop can still run in a non-git workspace.

    Uses `git diff --name-only HEAD` (staged + unstaged tracked
    changes) instead of `git status --porcelain` so untracked files
    don't gate (harness-m0g4)."""
    # Strip inherited ``GIT_*`` env vars so the explicit ``cwd=`` actually
    # binds to ``workspace``'s repo. Without this, a parent process that
    # set ``GIT_DIR`` / ``GIT_WORK_TREE`` (e.g. a wrapping pre-commit
    # hook or a driver invoking the loop from inside another git op)
    # would silently make this check inspect the WRONG repo.
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],  # noqa: S607 — git on PATH is expected
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            env=clean_env,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    for raw_line in (result.stdout or "").splitlines():
        path = raw_line.strip()
        if not path:
            continue
        if any(path.startswith(prefix) for prefix in _DIRTY_CHECK_IGNORE_PREFIXES):
            continue
        return True
    return False


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
    the operator needs to act on. For dry-run exits, also render the
    assembled handoff so the operator can verify the executor's view
    before burning real turns (harness-2pj3)."""
    typer.echo(
        f"loop_run={result.loop_run_id} exit={result.exit_reason} "
        f"turns_used={result.turns_used} closed={len(result.closed)}"
    )
    if result.exit_reason == "dry_run" and result.handoffs:
        # One handoff per dry-run by contract (the loop assembles + exits
        # after the first iteration). Render to stdout so it's pipeable.
        typer.echo("")
        typer.echo(result.handoffs[0].render())
    elif result.exit_reason == "halted":
        typer.echo(
            f"HALTED on {result.halted_on} — flagged via `bd human`. "
            f"Inspect with `bd show {result.halted_on}` and `bd human list`.",
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
    elif result.exit_reason == "partial":
        parked = ", ".join(result.parked_issues)
        typer.echo(
            f"PARTIAL — ready queue drained but {len(result.parked_issues)} "
            f"issue(s) parked for operator pickup: {parked}. These (and "
            f"anything depending on them) are NOT done. Inspect with "
            f"`bd human list`; clear a park to resume.",
            err=True,
        )


# --- harness drive logs <list|prune> (harness-830a) -----------------


logs_app = typer.Typer(
    help="Inspect + prune the .harness/loop_runs/ directory.",
    no_args_is_help=True,
)


@logs_app.command("list")
def logs_list_command(
    workspace: Path = typer.Option(
        Path.cwd(),  # noqa: B008 — typer evaluates at call time, not import time
        "--workspace",
        help="Workspace whose .harness/loop_runs/ to inspect.",
    ),
) -> None:
    """List loop runs newest-first with size + status. Same data as
    `harness drive loop --list-runs` (legacy flag) but as a proper
    subcommand and ordered by recency."""
    state_dir = LoopRunState.state_dir(workspace)
    if not state_dir.exists():
        typer.echo("(no loop runs yet)")
        return
    rows = _logs_inventory(state_dir)
    if not rows:
        typer.echo("(no loop runs yet)")
        return
    for row in rows:
        typer.echo(row)


@logs_app.command("prune")
def logs_prune_command(
    workspace: Path = typer.Option(
        Path.cwd(),  # noqa: B008 — typer evaluates at call time, not import time
        "--workspace",
        help="Workspace whose .harness/loop_runs/ to prune.",
    ),
    keep: int = typer.Option(
        10,
        "--keep",
        min=1,
        help="Keep the N most-recent runs in the active dir; archive the rest.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be archived without touching the filesystem.",
    ),
) -> None:
    """Archive older loop runs into .harness/loop_runs/archive/.

    Each archived run becomes: <id>.log.gz + <id>.json (kept readable
    for inspect tools) + <id>_workspace.tar.gz (preserved as-is, it's
    already compressed). Active dir keeps the N newest runs by
    started_at."""
    state_dir = LoopRunState.state_dir(workspace)
    if not state_dir.exists():
        typer.echo("(no loop runs yet)")
        return
    archived = _prune_loop_runs(state_dir, keep=keep, dry_run=dry_run)
    if not archived:
        typer.echo(f"(nothing to prune — {keep} or fewer runs in {state_dir})")
        return
    action = "would archive" if dry_run else "archived"
    for run_id in archived:
        typer.echo(f"{action}: {run_id}")
    if not dry_run:
        typer.echo(f"\nArchive directory: {state_dir / 'archive'}")


def _logs_inventory(state_dir: Path) -> list[str]:
    """Build the operator-facing inventory of loop runs (newest-first).
    Each row: `<id>  started=<iso>  turns=<n>  closed=<n>  log=<size>`."""
    files = sorted(state_dir.glob("*.json"))
    entries: list[tuple[LoopRunState, Path]] = []
    for path in files:
        try:
            state = LoopRunState.load(path)
        except (OSError, ValueError, KeyError):
            continue
        entries.append((state, path))
    # Newest first by started_at.
    entries.sort(key=lambda item: item[0].started_at, reverse=True)
    rows: list[str] = []
    for state, _path in entries:
        log_path = state_dir / f"{state.loop_run_id}.log"
        log_size_kb = 0
        if log_path.exists():
            log_size_kb = log_path.stat().st_size // 1024
        rows.append(
            f"{state.loop_run_id}  "
            f"started={state.started_at.isoformat(timespec='seconds')}  "
            f"turns={state.turns_used}  "
            f"closed={len(state.closed_this_run)}  "
            f"log={log_size_kb}KB"
        )
    return rows


def _prune_loop_runs(state_dir: Path, *, keep: int, dry_run: bool) -> list[str]:
    """Move runs older than the most-recent `keep` into an archive/
    subdir. Returns the list of run_ids that were (or would be) moved.

    Archive shape: <id>.log → <id>.log.gz; .json + _workspace.tar.gz
    move as-is. The .log is the only un-compressed artifact and the
    biggest by volume, so gzipping it is the highest-leverage saving.

    Concurrency: not safe if another loop is writing to the same dir.
    Operator's job to run this when no loops are active. We don't
    fight for it."""
    import gzip
    import shutil

    files = sorted(state_dir.glob("*.json"))
    entries: list[tuple[LoopRunState, Path]] = []
    for path in files:
        try:
            state = LoopRunState.load(path)
        except (OSError, ValueError, KeyError):
            continue
        entries.append((state, path))
    if len(entries) <= keep:
        return []
    # Newest first; the trailing slice past `keep` is what gets archived.
    entries.sort(key=lambda item: item[0].started_at, reverse=True)
    to_archive = entries[keep:]
    archive_dir = state_dir / "archive"
    archived_ids: list[str] = []
    for state, json_path in to_archive:
        run_id = state.loop_run_id
        archived_ids.append(run_id)
        if dry_run:
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        # 1. Move + gzip the .log → archive/<id>.log.gz
        log_src = state_dir / f"{run_id}.log"
        if log_src.exists():
            log_dst = archive_dir / f"{run_id}.log.gz"
            with log_src.open("rb") as src, gzip.open(log_dst, "wb") as dst:
                shutil.copyfileobj(src, dst)
            log_src.unlink()
        # 2. Move the .json verbatim — small, readable in inspect tools.
        json_dst = archive_dir / json_path.name
        shutil.move(str(json_path), str(json_dst))
        # 3. Move the workspace tarball (if present) — already compressed.
        tar_src = state_dir / f"{run_id}_workspace.tar.gz"
        if tar_src.exists():
            tar_dst = archive_dir / tar_src.name
            shutil.move(str(tar_src), str(tar_dst))
    return archived_ids


drive_app.add_typer(logs_app, name="logs")


# --- exit-code helpers ------------------------------------------------


def _exit_code(result: LoopResult) -> int:
    """Map exit_reason → process exit code. Success / dry_run = 0;
    halted = 2 (operator action needed); partial / exhausted /
    interrupted = 1 (ran within constraints but the epic isn't done)."""
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
