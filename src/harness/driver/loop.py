"""Executor loop — harness-ml66.

Drives a bd epic to closure by iterating `bd ready --under-epic`, one
issue per turn, until the epic empties or a halt condition fires.
Bypasses `harness chat` and calls `orchestrator.tool_loop.run_tool_loop`
directly so the executor turn skips voice retrieval, persona rewrite,
and chat-history tracking — none of which apply to agent-to-agent
work.

Per-iteration flow:

  1. SIGINT check — clean exit with a session-state bead + state save.
  2. Turn-budget check — exit_exhausted if `turns_used >= max_turns`.
  3. `bd.ready_under_epic(epic_id)` — exit_success if empty.
  4. Take the top of ready; increment attempt counter for that id.
  5. Build the handoff. If `dry_run`, print and return without calling
     the model.
  6. Assemble messages: `[system=character.system_prompt(...) + handoff,
     user="Drive this bd issue to closure..."]`. No voice samples
     (stripped prompt, per harness-ml66).
  7. Call `run_tool_loop` with the coding-profile registry (filesystem
     + git + reckon + discovery + fetch_url; store-dependent tools
     skipped for v0).
  8. Post-turn outcome check: `bd.show(current_id).status == "closed"`
     AND the reply isn't the fabrication-fallback sentinel.
  9. On SUCCESS — record close, write a session-state bead, log,
     continue.
  10. On FIRST FAILURE — stash the reason in `state.last_failure`,
      save, log, retry the same issue.
  11. On SECOND FAILURE — `bd.flag_human`, write a session-state bead
      with `status=halted`, save, log, exit_halted.

Outcome detection is intentionally NOT scan-based on tool calls —
the executor's coding profile gives the model `shell`, so it'll close
via `bd close <id>` via shell rather than a structured `bd_close`
tool. Post-turn `bd.show` is mechanical and doesn't care which path
the model took.

State persists between turns to `<workspace>/.harness/loop_runs/<id>.json`
(crash-safe atomic write via `LoopRunState.save`). Progress log mirrors
the bead writes to a plain-text file so `tail -f` works during a long
run.

Resume contract (`--resume <loop_run_id>`): load the state file, pick
the next ready issue per the same `ready_under_epic` query, continue.
Any in-progress attempt count is preserved — a SIGINT mid-attempt
resumes on attempt #2 (operator-visible via the progress log).
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import signal
import subprocess
import tarfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from harness.character import Character
from harness.driver.bd import DriverBd, DriverBdError
from harness.driver.claim_detector import (
    detect_claim_in_shell_call,
    detect_claim_signal,
    last_shell_cmd_in_messages,
)
from harness.driver.handoff import Handoff, build_handoff
from harness.driver.planner import PlanDraft, PlannerError, VerifyStep
from harness.driver.precommit_verify_hook import (
    PreCloseVerifyHook,
    make_pre_close_verify_hook,
)
from harness.driver.state import LoopRunState
from harness.driver.workspace_guard import (
    DEFAULT_SCRATCH_PATTERNS,
    WorkspaceTooBigError,
    archive_workspace,
    detect_regression,
    list_workspace_files,
    restore_workspace,
    sweep_scratch,
)
from harness.driver.workspace_verify import (
    browser_smoke_skip_reason,
    default_workspace_verify_steps,
)
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.orchestrator import ToolLoopEvent, ToolLoopResult, run_tool_loop
from harness.orchestrator.hook_wiring import make_write_file_redirect_hook
from harness.orchestrator.hooks import (
    EXHAUSTED_FABRICATION_FALLBACK,
    HookPipeline,
    default_hook_pipeline,
)
from harness.tools import (
    CalcTool,
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
    PythonStreamTool,
    ReadFileTool,
    ShellTool,
    StreamEditTool,
    Tool,
    ToolCatalog,
    ToolRegistry,
    ToolSearchTool,
    WriteFileTool,
    seed_builtins_into,
)

# The user message stays fixed across the run — the handoff carries the
# noun, this is the verb. Mentions `bd close` explicitly so the model
# knows shelling to bd is how it signals completion (v0 has no
# structured bd_close tool — that's a future revision).
EXECUTOR_USER_MESSAGE = (
    "Drive the bd issue described in the session handoff to closure. "
    "Run `bd close <issue-id>` via shell when the acceptance criteria are met. "
    "Do not invent acceptance criteria the issue doesn't list. "
    "Edit the existing deliverable in place — do NOT rewrite a working file "
    "from scratch (deleting existing functions fails the regression gate) and "
    "do NOT create planning, temp, validate, or backup files; they are swept "
    "after the issue and only waste your round budget. "
    "Execute commands directly — do NOT echo intent strings or dry-runs. A "
    'shell `echo "Would run: bd close …"` or `echo "…closed…"` does nothing '
    "and is invisible to the loop; run `bd close <issue-id>` for real."
)


@dataclass
class LoopConfig:
    """Configuration for one `run_loop` invocation.

    - `epic_id`: bd epic this run drains. Ignored when `resume_from` is
      set (state file carries it).
    - `workspace`: workspace root. Used as `bd_dir` for `DriverBd`,
      `cwd` for git diff, sandbox root for filesystem tools, and
      anchor for `.harness/loop_runs/*.json`.
    - `character`: loaded `Character` used for the (stripped) system
      prompt. `character.system_prompt(include_samples=())` strips
      voice few-shot — identity + values + style rules survive.
    - `max_turns`: hard cap on iterations. Default 20.
    - `profile`: pinned to "coding" today; field present so future
      revisions can swap without changing the run_loop signature.
    - `resume_from`: loop_run_id of an existing state file. When set,
      `epic_id` and `max_turns` are ignored — the state file owns
      those values.
    - `dry_run`: assemble the handoff and print it, then return
      without running the model. Useful for verifying handoff shape
      before burning real turns.
    - `log_path`: progress log file. Defaults to
      `<workspace>/.harness/loop_runs/<loop_run_id>.log`.
    """

    epic_id: str
    workspace: Path
    character: Character
    max_turns: int = 20
    profile: str = "coding"
    resume_from: str | None = None
    dry_run: bool = False
    log_path: Path | None = None
    # When set, mirror per-turn tool-loop events to this callable (in
    # addition to the existing file log). Used by the CLI's --verbose
    # flag to tee events to stderr. harness-9bpt.
    extra_observer: Callable[[ToolLoopEvent], None] | None = None
    # Forbidden substrings checked in workspace files modified during
    # this run (harness-k52f). After a turn closes its bd issue, if any
    # of these strings appear in any file modified since loop start,
    # the turn fails with the violation list as the reason — fed back
    # to the model via prior_attempt_failure on retry. Default catches
    # the spec-opening "no TODO / no future improvement" rule that
    # cross-cuts most workplan specs. Set to `()` to disable.
    forbidden_patterns: tuple[str, ...] = ("TODO", "FIXME", "XXX", "HACK")
    # Per-turn round budget for the inner run_tool_loop call
    # (harness-pipb). The orchestrator's default (8) is too tight for
    # executor turns that do multi-edit work — read spec, read
    # existing file, edit + verify + edit again — leading to
    # `wrap_up_forced` firing before the model finishes. 12 gives
    # ~50% more headroom while keeping each turn bounded (12 rounds *
    # ~10s/round ≈ 2 min upper bound on M4 Pro).
    executor_max_rounds: int = 12
    # harness-d8e3: max attempts per bd issue before the driver halts +
    # flags for human review. Was hard-coded at 2 (halt on the second
    # consecutive failure); bumped to 3 because the forbidden-pattern
    # audit fires AFTER a successful close, so the model only learns
    # about the failure in the handoff for the NEXT attempt. With cap=2,
    # a turn-1 failure that didn't close + a turn-2 close that fails
    # the audit halted the run with the model never having seen the
    # audit-failure feedback. cap=3 gives one more retry slot where
    # the model sees [PRIOR ATTEMPT FAILED: ... forbidden-pattern ...]
    # and can fix it.
    max_attempts_per_issue: int = 3
    # Tar+gzip the workspace into
    # .harness/loop_runs/<id>_workspace.tar.gz before the first turn
    # fires (harness-9ijr). Default on — gitignored workspaces are
    # otherwise unrecoverable when a runaway model rewrites a file
    # (today's GTA2 §1 regression that prompted this guard). Resume
    # runs (`resume_from` set) skip the snapshot — the original run's
    # snapshot is the operator's restore point and overwriting it
    # would defeat the purpose. Set False via --no-snapshot for
    # callers who know what they're doing (large workspaces over the
    # cap, fully-tracked git trees, throwaway scratch sessions).
    snapshot: bool = True
    # YAML plan draft this run is draining (harness-xfh2). When set, the
    # loop loads the draft once at startup and runs each item's `verify`
    # steps after the executor closes the corresponding bd issue. Any
    # step exiting non-zero reopens the bd issue, stashes the failure
    # in `last_failure`, and lets the next iteration retry — the model
    # sees the verify failure via `prior_attempt_failure` in the
    # handoff. None disables the gate entirely; drafts without `verify`
    # entries behave identically to None for the items they describe.
    plan_draft_path: Path | None = None
    # harness-kbnl: route turns through the TurnFSM (ASSESS → WRITE_TEST →
    # IMPLEMENT → VERIFY → CLOSE) instead of the legacy single-shot
    # _run_executor_turn. Off by default while the FSM bakes in; once
    # stable the default flips and the legacy path moves to a
    # --legacy-turn escape hatch. CLI wires this via --fsm / --no-fsm.
    use_fsm: bool = False
    # harness-kbnl: when use_fsm=True, require the WRITE_TEST phase by
    # default. Operator can flip to False with --no-tdd for runs where
    # TDD genuinely doesn't apply (UI/visual changes, docs). Model can
    # also opt out per-issue via submit_assessment(tdd_applicable=False)
    # — that's the model's judgment call, this is the operator's.
    tdd_required: bool = True
    # harness-tu4o: drive-loop analog of chat's compaction guard. The
    # drive loop rebuilds messages fresh each iteration (no cross-turn
    # history), so the budget blow-up that bites long-running runs
    # happens INSIDE a single run_tool_loop call — tool results pile
    # up across rounds until vLLM / MLX rejects the prompt. The right
    # analog for "compaction" here is the ToolResultSummarizerHook,
    # which compresses high-noise tool outputs (grep / list_dir /
    # search_web / fetch_url etc.) before they reach the model's
    # context. Default ON for drive runs because they're unattended;
    # the chat CLI keeps it opt-in.
    summarize_tool_results: bool = True
    # harness-b7m1: when the claim-without-close gate fires AND verify
    # ran at least one step that passed, close the bd issue on the
    # model's behalf instead of returning a soft hint and burning another
    # retry. Loop f36cf2e4 confirmed that small models can spend 3 turns
    # x 12 rounds re-verifying instead of running `bd close` even with
    # explicit feedback in the handoff. When verify is the contract and
    # verify passed, the harness should be willing to drive the close.
    # Falls back to the soft hint when verify failed, no steps ran, or
    # the bd close subprocess itself failed. Set False via
    # `--no-auto-close-on-claim` to restore the pre-b7m1 behavior.
    auto_close_on_claim: bool = True
    # harness-zcrd: when an issue exhausts max_attempts_per_issue, park
    # it (`bd flag_human` + add to state.parked_issues) and continue
    # the drive against the next ready issue instead of halting the
    # whole run. Loop 1a6e4437 closed 6 issues then stopped on §9a
    # (genuinely complex pickups task) — pre-zcrd, ONE hard issue
    # killed the whole drive. With skip-on, the drive completes
    # everything it can and the operator picks up parked issues via
    # `bd human list`. Set False via `--no-skip-on-max-attempts` to
    # restore the halt-the-whole-run behavior.
    skip_on_max_attempts: bool = True
    # harness-6dsn: id of the bd issue that marks "the workspace should
    # now render something" (e.g. §2 world map). The blank-canvas smoke
    # check (harness-7bxm) is suppressed until this issue is closed —
    # an incremental from-scratch build has legitimately-blank skeletons
    # (§1 project shape) that the check would otherwise fail on every
    # turn, blocking the whole build. None = enforce blank-canvas always
    # (the right default for verifying an already-built game). The
    # console / page-error smoke checks always run regardless. CLI wires
    # this via --render-milestone.
    render_milestone_id: str | None = None
    # harness-16w6: regression guard + no-punting. Maintains a
    # last-green workspace snapshot (refreshed on every close) and (a)
    # fails verify when the deliverable lost previously-defined symbols
    # or shrank dramatically vs last-green — the stub-rewrite smoke
    # can't see — feeding the regression back so the model fixes it
    # before advancing; (b) on park (max attempts), if the issue left
    # the baseline broken, restores last-green so the break can't
    # cascade to later issues. Off via --no-regression-guard.
    regression_guard: bool = True
    # harness-ul5z: archive agent-created scratch (planning / temp /
    # validate / backup files) into .harness/loop_runs/<id>_scratch/
    # when the issue that created it completes, so it stops
    # accumulating + getting re-read on later turns. Off via
    # --no-scratch-sweep. Patterns matched against the file basename.
    scratch_sweep: bool = True
    scratch_patterns: tuple[str, ...] = DEFAULT_SCRATCH_PATTERNS


@dataclass
class LoopResult:
    """What `run_loop` returns to the caller.

    `exit_reason` matches the lifecycle event names in the progress log:
      - "success":     ready_under_epic emptied with nothing parked —
                       the epic is genuinely complete.
      - "partial":     ready_under_epic emptied, but only because
                       parked issues were filtered out of it
                       (harness-iljv). The epic is NOT complete: the
                       parked issues — and anything depending on them —
                       are stranded pending operator pickup. Kept
                       distinct from "success" so callers don't read a
                       stalled epic as finished.
      - "exhausted":   turns_used reached max_turns.
      - "halted":      catastrophic failure (e.g. startup exception) OR
                       max-attempts halt with `skip_on_max_attempts=False`.
      - "interrupted": SIGINT mid-loop.
      - "dry_run":     --dry-run; one handoff printed, no turn ran.

    `parked_issues` (harness-zcrd): bd ids the drive parked via
    `bd flag_human` after max-attempts exhaustion. Populated when
    `LoopConfig.skip_on_max_attempts` is True (default); on the
    opt-out path the drive halts instead and parked_issues stays
    empty.
    """

    loop_run_id: str
    epic_id: str
    closed: list[str]
    halted_on: str | None
    turns_used: int
    exit_reason: Literal["success", "partial", "halted", "exhausted", "interrupted", "dry_run"]
    handoffs: list[Handoff] = field(default_factory=list)
    parked_issues: list[str] = field(default_factory=list)


def _blank_canvas_enforced(bd: DriverBd, milestone_id: str | None) -> bool:
    """Whether the smoke gate's blank-canvas check should be enforced
    this turn (harness-6dsn).

    - No milestone configured → always enforce (the right default for
      verifying an already-built game; preserves pre-6dsn behavior).
    - Milestone configured → enforce only once that issue is CLOSED.
      Before then the build is still wiring up its first render, so a
      blank canvas is expected, not a bug.
    - Milestone lookup fails (bad id, bd error) → enforce + let the
      caller log it. Failing toward the stricter gate keeps a typo'd
      milestone loud rather than silently disabling a safety check."""
    if milestone_id is None:
        return True
    try:
        issue = bd.show(milestone_id)
    except DriverBdError:
        return True
    return issue.status == "closed"


# --- top-level entry --------------------------------------------------


def run_loop(adapter: ModelAdapter, bd: DriverBd, config: LoopConfig) -> LoopResult:
    """Run the executor loop until success / halt / exhaustion / interrupt."""
    state = _load_or_init_state(config)
    log_path = _resolve_log_path(config, state)
    log = _open_log(log_path)
    log(
        f"loop_run={state.loop_run_id} epic={state.epic_id} max_turns={state.max_turns} "
        f"resumed={'yes' if config.resume_from else 'no'}"
    )
    # harness-xfh2: load the per-item verify map once per run. Empty
    # when no draft path is supplied or the draft has no `verify`
    # entries — `_run_issue_verify` short-circuits to None in that case
    # and the original trust-the-close path is unchanged.
    verify_map = _load_verify_map(config.plan_draft_path)
    if verify_map:
        log(f"verify gate active: {len(verify_map)} item(s) with verify steps")

    # harness-oxj7: workspace-typed default verify steps. Scanned once
    # at run start and appended to every issue's per-item list so
    # closes are gated on the artifact still parsing — defense in
    # depth alongside the per-write parse-gate (harness-h6wa). Empty
    # tuple when the workspace has no recognized file types or the
    # required parsers aren't installed; the loop's existing behavior
    # is unchanged in that case.
    # harness-e74o: this initial snapshot reflects the workspace state
    # BEFORE the first turn fires. It's the binding used for the
    # FSM-turn path's prebuilt handoff and as a non-empty starting point
    # for non-FSM turns; the legacy path recomputes per-turn (post-write)
    # so empty workspaces still pick up the model's freshly-created files
    # when the gate runs.
    enforce_blank = _blank_canvas_enforced(bd, config.render_milestone_id)
    default_verify_steps = default_workspace_verify_steps(
        config.workspace, enforce_blank_canvas=enforce_blank
    )
    if default_verify_steps:
        log(f"workspace-typed verify defaults active: {len(default_verify_steps)} step(s)")
    else:
        log("workspace-typed verify defaults: none at startup (will recompute per turn)")
    if config.render_milestone_id is not None and not enforce_blank:
        log(
            "blank-canvas smoke check SUPPRESSED until render milestone "
            f"{config.render_milestone_id} closes (harness-6dsn); console/"
            "page-error checks still active"
        )
    # harness-7bxm: never let a degraded runtime-verify gate stay silent.
    # If the workspace is a browser app but the smoke-execute gate can't
    # run, say so loudly — this is the b85f4008 false-close root cause.
    smoke_skip = browser_smoke_skip_reason(config.workspace)
    if smoke_skip is not None:
        log(f"WARNING: {smoke_skip}")

    # harness-9ijr: snapshot the workspace once per fresh run BEFORE
    # the first turn fires. Resume runs inherit the original
    # snapshot — overwriting it would lose the operator's recovery
    # point. SnapshotTooBigError aborts the run; the operator chooses
    # between narrowing --workspace and passing --no-snapshot.
    if config.snapshot and config.resume_from is None:
        snapshot_path = _snapshot_workspace(config.workspace, state.loop_run_id)
        log(f"workspace snapshot: {snapshot_path}")

    # harness-16w6: last-green rollback target + harness-ul5z scratch
    # census. last_green is refreshed on every close (so it's the
    # current issue's clean starting point) and used both as the
    # regression-comparison baseline and the park rollback target. Seed
    # it from the start workspace only if it's already green — never
    # enshrine a broken baseline as the restore point.
    last_green: Path | None = None
    if config.regression_guard:
        if _baseline_is_green(default_verify_steps, config.workspace):
            last_green = _refresh_last_green(config, state, log)
            if last_green is not None:
                log(f"regression guard: last-green baseline set ({last_green.name})")
        else:
            log(
                "regression guard: workspace not green at start; last-green deferred to first close"
            )
    issue_start_files: dict[str, set[str]] = {}

    with _sigint_guard() as interrupted:
        while True:
            if interrupted.is_set():
                return _exit_interrupted(bd, state, config.workspace, log)
            if state.turns_used >= state.max_turns:
                return _exit_exhausted(state, config.workspace, log)

            try:
                ready = bd.ready_under_epic(state.epic_id)
            except DriverBdError as exc:
                log(f"bd ready_under_epic failed: {exc}; halting")
                return _exit_halted(
                    bd,
                    state,
                    current_id=state.epic_id,
                    reason=str(exc),
                    workspace=config.workspace,
                    log=log,
                )
            # harness-zcrd: skip-and-flag mode parks max-attempts
            # exhausted issues. Filter them out of ready so the next
            # iteration grabs the next genuinely-ready issue instead of
            # cycling back to the parked one. The bd-side flag (set in
            # _park_issue) doesn't necessarily exclude the issue from
            # `bd ready` — the filter is the load-bearing mechanism here.
            if state.parked_issues:
                parked = set(state.parked_issues)
                ready = [issue for issue in ready if issue.id not in parked]
            if not ready:
                # harness-iljv: distinguish a genuinely-complete epic
                # from one whose ready queue only emptied because we
                # filtered out parked issues above. The latter leaves
                # the parked issues (and their dependents) stranded, so
                # it's a "partial", not a "success" — callers and the
                # exit code must be able to tell the difference.
                if state.parked_issues:
                    return _exit_partial(state, config.workspace, log)
                return _exit_success(state, config.workspace, log)

            current = ready[0]
            attempt = state.attempt_counts.get(current.id, 0) + 1
            state.attempt_counts[current.id] = attempt
            # harness-ul5z: census the workspace the first time we touch
            # this issue, so the scratch sweep on completion can tell
            # which files the issue itself created vs pre-existing ones.
            if current.id not in issue_start_files:
                issue_start_files[current.id] = list_workspace_files(config.workspace)
            prior_failure = state.last_failure.get(current.id) if attempt > 1 else None
            # harness-lefw: signal targeted-fix mode when the loop has
            # already touched this issue OR the operator left a
            # "REGRESSION" marker in notes (their convention when
            # reopening a previously-closed issue). _issue_has_regression
            # tolerates a missing bd lookup — bd.show is called again
            # inside build_handoff and a transient miss there raises.
            targeted_fix = attempt > 1 or _issue_has_regression(bd, current.id)

            handoff = build_handoff(
                state,
                current.id,
                bd,
                git_root=config.workspace,
                prior_attempt_failure=prior_failure,
                workspace=config.workspace,
                targeted_fix=targeted_fix,
                forbidden_patterns=config.forbidden_patterns,
            )

            if config.dry_run:
                log(f"dry_run: handoff for {current.id} (attempt {attempt}) — exiting")
                # Stash the handoff so callers (eg. tests) can inspect.
                return LoopResult(
                    loop_run_id=state.loop_run_id,
                    epic_id=state.epic_id,
                    closed=list(state.closed_this_run),
                    halted_on=None,
                    turns_used=state.turns_used,
                    exit_reason="dry_run",
                    handoffs=[handoff],
                )

            turn_observer = _make_turn_observer(
                log_path, state.turns_used + 1, config.extra_observer
            )
            # harness-nlj7: per-turn pre-close verify hook. current_id and
            # regression_snapshot change between turns, so the factory
            # runs HERE (not at run startup) — the closure captures
            # this turn's verify config.
            pre_close_verify = make_pre_close_verify_hook(
                verify_map=verify_map,
                bd=bd,
                current_issue_id=current.id,
                workspace=config.workspace,
                default_steps=default_verify_steps,
                regression_snapshot=last_green if config.regression_guard else None,
            )
            if config.use_fsm:
                (
                    turn_success,
                    turn_reason,
                    turn_reply,
                    turn_last_shell,
                ) = _run_fsm_turn_via_driver(
                    adapter=adapter,
                    character=config.character,
                    bd=bd,
                    state=state,
                    config=config,
                    current_issue=current,
                    prior_failure=prior_failure,
                    targeted_fix=targeted_fix,
                    verify_map=verify_map,
                    default_verify_steps=default_verify_steps,
                    observe=turn_observer,
                    pre_close_verify=pre_close_verify,
                )
            else:
                (
                    turn_success,
                    turn_reason,
                    turn_reply,
                    turn_last_shell,
                ) = _run_executor_turn(
                    adapter=adapter,
                    character=config.character,
                    handoff=handoff,
                    workspace=config.workspace,
                    observe=turn_observer,
                    max_rounds=config.executor_max_rounds,
                    summarize_tool_results=config.summarize_tool_results,
                    pre_close_verify=pre_close_verify,
                )
            state.turns_used += 1

            # harness-e74o: recompute workspace-typed verify defaults
            # AFTER the model's turn lands. Workspaces that start empty
            # (the §1 case) have no .js / .py files at startup, so the
            # run-startup snapshot of `default_verify_steps` is empty
            # and the b7m1 auto-close gate's `steps_ran > 0` precondition
            # never trips. By recomputing here we pick up the files the
            # model just wrote and the per-turn verify reflects current
            # reality. Recompute is cheap (a few directory walks) and
            # idempotent — empty workspaces still yield ().
            # harness-6dsn: recompute blank-canvas enforcement too — the
            # render milestone may have closed on a prior turn, flipping
            # the smoke gate from skeleton-tolerant to render-strict.
            default_verify_steps = default_workspace_verify_steps(
                config.workspace,
                enforce_blank_canvas=_blank_canvas_enforced(bd, config.render_milestone_id),
            )

            success, reason, forbidden_warnings = _classify_post_turn(
                bd,
                current.id,
                turn_success,
                turn_reason,
                workspace=config.workspace,
                started_at=state.started_at,
                forbidden_patterns=config.forbidden_patterns,
            )

            # harness-oh8e: forbidden_patterns is warn-only. Surface the
            # violations to the progress log so the operator sees them
            # at audit time, but DON'T fail the close — drives that
            # actually finish substantive work shouldn't halt over a
            # leftover TODO comment. The previous behavior (fail +
            # reopen + retry, harness-2u0t) burned the retry budget on
            # over-verification rounds that never reached `bd close`.
            if forbidden_warnings:
                warning_tail = "; ".join(forbidden_warnings[:5])
                extra = (
                    f" (+{len(forbidden_warnings) - 5} more)" if len(forbidden_warnings) > 5 else ""
                )
                log(
                    f"[WARN] {current.id} closed with forbidden-pattern hits: {warning_tail}{extra}"
                )

            # harness-pfvj + harness-24pn + harness-b7m1:
            # claim-without-close detection. Two signals compose into a
            # single gate — either is enough to route the turn through
            # the verify gate as a pseudo-close:
            #
            #   (a) detect_claim_signal(turn_reply) — completion phrases
            #       in the model's prose ("issue resolved", "meets
            #       acceptance criteria", "tests pass", etc.). Catches
            #       the d4e01d68 + 94534703 turn 1+2 reply shapes.
            #
            #   (b) detect_claim_in_shell_call(turn_last_shell) —
            #       celebratory `echo` as the model's terminal shell
            #       action ("echo \"Fix applied successfully\""). Catches
            #       the 94534703 finalization-gesture pattern that the
            #       prose detector alone missed (the strongest signal
            #       was in the shell cmd, not the reply text).
            #
            # Outcomes (post-verify):
            #   - verify failed → reason gets the verify failure tail.
            #     Next handoff tells the model exactly what's broken.
            #   - verify passed AND auto-close enabled AND at least one
            #     step ran → harness-b7m1 auto-close: bd close on the
            #     model's behalf, _on_success, continue. The verify gate
            #     IS the contract; the model just forgot the final step.
            #   - verify passed but no steps ran (no defaults applicable
            #     + no per-issue steps) → soft hint pointing the model
            #     at the missing bd close. Auto-close is unsafe here
            #     because verify had nothing to corroborate the claim.
            #   - auto-close subprocess itself failed → fall back to
            #     soft hint so the model can try again.
            if (
                not success
                and _is_still_open_reason(reason)
                and (detect_claim_signal(turn_reply) or detect_claim_in_shell_call(turn_last_shell))
            ):
                verify_failure, steps_ran = _run_issue_verify(
                    verify_map,
                    bd,
                    current.id,
                    config.workspace,
                    default_steps=default_verify_steps,
                    regression_snapshot=last_green if config.regression_guard else None,
                )
                if verify_failure is not None:
                    reason = f"{_CLAIM_WITHOUT_CLOSE_PREFIX} {verify_failure}"
                elif (
                    config.auto_close_on_claim
                    and steps_ran > 0
                    and _try_auto_close(bd, current.id, log)
                ):
                    # Drive the close on the model's behalf; bypass the
                    # success branch's redundant verify by handling the
                    # close-and-continue here.
                    log(
                        f"turn {state.turns_used}: {current.id} AUTO_CLOSED "
                        f"(claim + verify passed, {steps_ran} step{'s' if steps_ran != 1 else ''})"
                    )
                    _on_success(bd, state, current.id, log)
                    last_green = _post_close_housekeeping(
                        config, state, issue_start_files.pop(current.id, set()), last_green, log
                    )
                    _save_state(state, config.workspace)
                    continue
                else:
                    reason = f"{_CLAIM_WITHOUT_CLOSE_PREFIX} {_CLAIM_WITHOUT_CLOSE_NO_VERIFY_HINT}"

            if success:
                # harness-xfh2: the verify gate runs only when the bd
                # close looked clean. A non-zero verify exit reopens the
                # issue, stashes the failure, and falls through to next
                # turn — same retry budget as a "close failed" path. The
                # reopen + stash combo is the contract: ready_under_epic
                # picks up the now-open issue next iteration, and the
                # next handoff carries `verify_failed: ...` so the model
                # self-corrects.
                verify_failure, _ = _run_issue_verify(
                    verify_map,
                    bd,
                    current.id,
                    config.workspace,
                    default_steps=default_verify_steps,
                    regression_snapshot=last_green if config.regression_guard else None,
                )
                if verify_failure is None:
                    _on_success(bd, state, current.id, log)
                    last_green = _post_close_housekeeping(
                        config, state, issue_start_files.pop(current.id, set()), last_green, log
                    )
                    _save_state(state, config.workspace)
                    continue
                reopened = _try_reopen(bd, current.id)
                state.last_failure[current.id] = f"verify_failed: {verify_failure}"
                _save_state(state, config.workspace)
                reopen_note = "reopened" if reopened else "REOPEN_FAILED"
                log(
                    f"turn {state.turns_used}: {current.id} VERIFY_FAIL "
                    f"({reopen_note}; {verify_failure})"
                )
                # Verify failure counts as the iteration's failure for
                # attempt accounting — halt OR park at
                # config.max_attempts_per_issue. harness-zcrd: skip-on
                # parks the issue and moves to the next ready one;
                # skip-off preserves the legacy halt-the-whole-run
                # behavior so operators who depend on that signal can
                # opt out.
                if attempt >= config.max_attempts_per_issue:
                    fail_reason = f"verify_failed: {verify_failure}"
                    if config.skip_on_max_attempts:
                        _park_issue(bd, state, current_id=current.id, reason=fail_reason, log=log)
                        _post_park_housekeeping(
                            config,
                            state,
                            issue_start_files.pop(current.id, set()),
                            last_green,
                            default_verify_steps,
                            log,
                        )
                        _save_state(state, config.workspace)
                        continue
                    return _exit_halted(
                        bd,
                        state,
                        current_id=current.id,
                        reason=fail_reason,
                        workspace=config.workspace,
                        log=log,
                    )
                continue

            log(f"turn {state.turns_used}: {current.id} attempt={attempt} FAIL ({reason})")

            if attempt < config.max_attempts_per_issue:
                state.last_failure[current.id] = reason
                _save_state(state, config.workspace)
                continue

            # Final consecutive failure (harness-d8e3 + zcrd): park
            # under skip-on (the new default), halt under skip-off.
            if config.skip_on_max_attempts:
                _park_issue(bd, state, current_id=current.id, reason=reason, log=log)
                _post_park_housekeeping(
                    config,
                    state,
                    issue_start_files.pop(current.id, set()),
                    last_green,
                    default_verify_steps,
                    log,
                )
                _save_state(state, config.workspace)
                continue
            return _exit_halted(
                bd, state, current_id=current.id, reason=reason, workspace=config.workspace, log=log
            )


# --- per-turn -----------------------------------------------------


# Observer type for the executor's per-turn tool-loop event stream
# (harness-9bpt). Same shape as planner's PlannerObserver.
ExecutorObserver = Callable[[ToolLoopEvent], None]


def _make_turn_observer(
    log_path: Path,
    turn_index: int,
    extra: ExecutorObserver | None,
) -> ExecutorObserver:
    """Build an observer that:
      1. Appends one line per ToolLoopEvent to `log_path` (prefixed
         with `turn N |` so the operator can grep per-turn slices).
      2. Optionally forwards to `extra` — used by the CLI's --verbose
         flag to mirror to stderr.

    Best-effort IO: log write failures don't crash the turn. Observer
    failures in `extra` are suppressed for the same reason (matches
    the planner's _compose_observers contract)."""

    def emit(event: ToolLoopEvent) -> None:
        line = f"turn {turn_index} | {_format_executor_event(event)}"
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        if extra is not None:
            with contextlib.suppress(Exception):
                extra(event)

    return emit


def _format_executor_event(event: ToolLoopEvent) -> str:
    """Same shape as planner's _format_event — timestamp + kind + most
    relevant payload. Kept separate so the driver module doesn't import
    from planner."""
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    parts: list[str] = [ts, event.kind]
    if event.call is not None:
        args_preview = repr(event.call.arguments)
        if len(args_preview) > 200:
            args_preview = args_preview[:200] + "...]"
        parts.append(f"call={event.call.name} args={args_preview}")
    if event.result is not None:
        success = "ok" if event.result.success else f"FAIL[{event.result.error}]"
        output_preview = event.result.output.replace("\n", " ")[:160]
        parts.append(f"result={success} output={output_preview!r}")
    if event.delta:
        parts.append(f"delta={event.delta}")
    if event.catcher:
        parts.append(f"catcher={event.catcher}")
    return " | ".join(parts)


def _run_fsm_turn_via_driver(
    *,
    adapter: ModelAdapter,
    character: Character,
    bd: DriverBd,
    state: LoopRunState,
    config: LoopConfig,
    current_issue: Any,
    prior_failure: str | None,
    targeted_fix: bool,
    verify_map: Mapping[str, Sequence[VerifyStep]],
    default_verify_steps: Sequence[VerifyStep],
    observe: ExecutorObserver | None,
    pre_close_verify: PreCloseVerifyHook | None = None,
) -> tuple[bool, str, str, str | None]:
    """Adapter that wraps `run_fsm_turn` to match the legacy
    `_run_executor_turn` return shape (succeeded, reason, reply,
    last_shell_cmd).

    Builds a phase-aware handoff_builder closure: each phase asks
    for a fresh Handoff that reflects the FSM's current state
    (phase, prior assessment, prior test_cmd). The verify steps for
    the current bd issue are pulled from `verify_map` and passed
    into `run_fsm_turn` so the VERIFY phase can execute them.

    Side effects:
      - Persists `state.last_turn_phase[issue_id]` so a halt mid-FSM
        carries forward (next attempt resumes in IMPLEMENT if the
        prior attempt halted there with an assessment).
      - Persists `state.last_assessment[issue_id]` + `state.last_test_cmd[issue_id]`
        for the same reason."""
    from harness.driver.fsm_turn import (
        FsmTurnResult,
        phase_instructions,
        run_fsm_turn,
    )
    from harness.driver.handoff import build_handoff
    from harness.driver.turn_fsm import TurnPhase

    issue_id = current_issue.id

    def builder(
        phase: TurnPhase,
        prior_assessment: dict[str, Any] | None,
        prior_test_cmd: str | None,
    ) -> Handoff:
        return build_handoff(
            state,
            issue_id,
            bd,
            git_root=config.workspace,
            prior_attempt_failure=prior_failure,
            workspace=config.workspace,
            targeted_fix=targeted_fix,
            phase=phase.value,
            phase_instructions=phase_instructions(phase),
            prior_assessment=prior_assessment if prior_assessment else None,
            prior_test_cmd=prior_test_cmd,
        )

    initial_phase = TurnPhase.ASSESS
    saved_phase_value = state.last_turn_phase.get(issue_id)
    if saved_phase_value:
        with contextlib.suppress(ValueError):
            initial_phase = TurnPhase(saved_phase_value)
            # Don't resume into a terminal phase — start fresh.
            if initial_phase in {TurnPhase.DONE, TurnPhase.HALTED}:
                initial_phase = TurnPhase.ASSESS

    prior_assessment = state.last_assessment.get(issue_id)
    prior_test_cmd = state.last_test_cmd.get(issue_id)
    # harness-oxj7: workspace-typed defaults run BEFORE per-item steps
    # so cheap parse-checks fail fast ahead of slower operator-authored
    # gates. The FSM's `_resolve_verify_outcome` walks the sequence in
    # order and short-circuits on first non-zero exit.
    per_item_verify = verify_map.get(current_issue.title, ()) if verify_map else ()
    verify_steps = tuple(default_verify_steps) + tuple(per_item_verify)

    try:
        result: FsmTurnResult = run_fsm_turn(
            adapter=adapter,
            character=character,
            bd=bd,
            handoff_builder=builder,
            workspace=config.workspace,
            current_issue_id=issue_id,
            initial_phase=initial_phase,
            prior_assessment=prior_assessment,
            prior_test_cmd=prior_test_cmd,
            verify_steps=verify_steps,
            tdd_required=config.tdd_required,
            observe=observe,
            summarize_tool_results=config.summarize_tool_results,
            pre_close_verify=pre_close_verify,
        )
    except Exception as exc:
        # harness-tu4o: see _run_executor_turn for the same rationale.
        # A context-overflow rejection inside the FSM's inner tool
        # loop becomes a turn failure so the loop's retry budget runs;
        # next iteration rebuilds the prompt fresh.
        if _is_context_overflow(exc):
            return False, f"context_exhausted: {exc}", "", None
        raise

    # Persist FSM state for resume. Plain string values keep the
    # .json dump operator-readable.
    state.last_turn_phase[issue_id] = result.final_phase.value
    if result.last_assessment is not None:
        state.last_assessment[issue_id] = result.last_assessment
    if result.last_test_cmd is not None:
        state.last_test_cmd[issue_id] = result.last_test_cmd

    return result.succeeded, result.reason, result.reply, result.last_shell_cmd


def _build_driver_hook_pipeline(
    *,
    adapter: ModelAdapter,
    registry: Any,
    workspace: Path,
    summarize_tool_results: bool,
    pre_close_verify: PreCloseVerifyHook | None = None,
) -> HookPipeline:
    """Drive-loop hook pipeline (harness-tu4o). Mirrors
    `cli_classic._build_hook_pipeline` but trimmed to what the executor
    actually needs: WriteFileRedirectHook (the safety-shrink guard) and
    the optional ToolResultSummarizerHook (the high-noise output
    compressor — the drive-loop analog of chat's compaction).

    `summarize_tool_results=True` reuses `adapter` as the summarizer.
    That's the same compromise the chat CLI makes when no router model
    is loaded: cheap (one extra adapter call per high-noise tool
    result), and correct (the summarizer prompt is short and
    deterministic). Failure to summarize is non-fatal — the hook
    falls through with `Continue` and the raw output reaches the
    model untouched.

    `pre_close_verify` (harness-nlj7) gates ``bd close <current_issue>``
    on workspace verify. When supplied, it runs ahead of
    ShellEchoNoopHook so a verify-fail Skips the close before the echo
    detector ever sees it."""
    from harness.orchestrator.hooks import ShellEchoNoopHook, ToolResultSummarizerHook

    write_file_redirect_hook = make_write_file_redirect_hook(
        registry=registry,
        workspace_path=workspace,
    )
    pipeline = default_hook_pipeline(write_file_redirect_hook=write_file_redirect_hook)
    # harness-nlj7: pre-close verify gate. Runs FIRST in pre_tool so a
    # verify failure Skips the bd close before any downstream hook sees
    # it. The model gets a verify_blocked failure result instead of a
    # close-success ack, killing the wrap_up_forced narration spiral.
    if pre_close_verify is not None:
        pipeline.pre_tool.append(pre_close_verify)
    # harness-jmkc: drive-only. Catch the model echoing "Would run: bd
    # close X" / "Issue closed…" instead of executing the close — a
    # narration no-op that left issues open and burned attempts.
    pipeline.pre_tool.append(ShellEchoNoopHook())
    if summarize_tool_results:
        pipeline.post_tool.append(ToolResultSummarizerHook(summarizer=adapter))
    return pipeline


def _is_context_overflow(exc: BaseException) -> bool:
    """Heuristic: does this exception look like the model server
    rejected the prompt for exceeding context window?

    Covers the shapes we've actually seen in the wild:
      - vLLM: RuntimeError('vLLM returned HTTP 400 ... maximum context
        length is 32768 tokens')
      - MLX / Ollama: tokenizer / adapter errors that mention 'context'
        or 'maximum'.

    Used by `_run_executor_turn` so a single turn's context blow-up
    becomes a turn failure (which the loop's retry budget handles)
    instead of an unhandled exception that tears down the run."""
    msg = str(exc).lower()
    return "maximum context length" in msg or (
        "context" in msg and ("exceed" in msg or "too long" in msg)
    )


def _run_executor_turn(
    *,
    adapter: ModelAdapter,
    character: Character,
    handoff: Handoff,
    workspace: Path,
    observe: ExecutorObserver | None = None,
    max_rounds: int = 12,
    summarize_tool_results: bool = True,
    pre_close_verify: PreCloseVerifyHook | None = None,
) -> tuple[bool, str, str, str | None]:
    """Run one executor turn. Returns (succeeded, reason, reply, last_shell_cmd).

    `succeeded` is computed against the post-turn ToolLoopResult only —
    the caller is responsible for the post-turn `bd.show` outcome check
    (success requires BOTH a clean reply AND the bd issue actually
    being closed). This split keeps the test surface small: the turn
    runner is a pure function of its inputs, and the caller composes
    the bd check on top.

    `reply` is the model's final text content for the turn — the same
    string the operator sees in the progress log. Returned so the caller
    can run claim-detection (harness-pfvj) when the bd issue stays open
    after the turn: a confident-but-not-closed reply gets routed through
    the verify gate as a pseudo-close.

    `last_shell_cmd` is the `cmd` argument from the LAST shell tool call
    this turn (or None if no shell call happened). Used by the
    claim-without-close gate (harness-24pn) so a turn whose final shell
    action is `echo "Fix applied successfully"` — the model's
    celebratory finalization gesture instead of `bd close` — gets
    classified as a claim and routed through the verify path.

    `observe`, when set, receives every `ToolLoopEvent` from the inner
    `run_tool_loop` — same shape as the planner's observer
    (harness-9bpt). The caller is responsible for writing to a log
    file / stderr; this function is just the seam."""
    registry = _build_executor_registry(workspace)
    # harness-d6ak (smaller surgery): include_style_rules=False strips
    # the chat-shaped "How you speak" block AND the voice few-shot.
    # Identity + values + directives + thought-graph survive — the
    # model still has a generative anchor and knows its taboos, but
    # the prose-by-default / 1-4-sentences / 'Don't know plainly'
    # rules that bias toward narration are gone. Full d6ak revert
    # (cdd2fd9) kept these; turned out the persona-prompt removal
    # was too aggressive — model went silent. This carve-out keeps
    # the anchor and drops only the conflicting style guidance.
    base_prompt = character.system_prompt(include_samples=(), include_style_rules=False)
    system_prompt = f"{base_prompt}\n\n{handoff.render()}"
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=EXECUTOR_USER_MESSAGE),
    ]
    # harness-lefw: WriteFileRedirectHook safety-shrink guard.
    # harness-tu4o: optional ToolResultSummarizerHook so high-noise tool
    # outputs (grep / list_dir / fetch_url …) get compressed before they
    # land in the model's context.
    hooks = _build_driver_hook_pipeline(
        adapter=adapter,
        registry=registry,
        workspace=workspace,
        summarize_tool_results=summarize_tool_results,
        pre_close_verify=pre_close_verify,
    )
    try:
        result: ToolLoopResult = run_tool_loop(
            adapter,  # type: ignore[arg-type]  # narrower _ToolCapableAdapter, checked at runtime
            messages,
            registry,
            hooks=hooks,
            observe=observe,
            max_rounds=max_rounds,
        )
    except Exception as exc:
        # harness-tu4o: a context-window rejection inside the inner
        # tool loop is a turn failure, not a run-killing crash. The
        # loop's retry budget (2 attempts before halt + bd_human) will
        # take over — and because each iteration rebuilds the prompt
        # from scratch, the next attempt starts with a clean slate.
        if _is_context_overflow(exc):
            return False, f"context_exhausted: {exc}", "", None
        raise
    last_shell_cmd = last_shell_cmd_in_messages(result.messages)
    if result.content.strip() == EXHAUSTED_FABRICATION_FALLBACK.strip():
        return False, "fabrication_fallback fired", result.content, last_shell_cmd
    return True, "", result.content, last_shell_cmd


def _classify_post_turn(
    bd: DriverBd,
    current_id: str,
    turn_success: bool,
    turn_reason: str,
    *,
    workspace: Path | None = None,
    started_at: datetime | None = None,
    forbidden_patterns: tuple[str, ...] = (),
) -> tuple[bool, str, tuple[str, ...]]:
    """Combine the turn outcome with the post-turn bd state.

    Issue must actually be closed for the turn to count as a real win —
    a clean reply with the issue still open means the model didn't
    finish the work, regardless of how confidently it claimed to.

    Forbidden-pattern verification (harness-k52f, harness-oh8e): when
    `forbidden_patterns` is non-empty and `workspace` + `started_at`
    are supplied, the function scans workspace files modified since
    `started_at` for any of the patterns. Hits no longer fail the turn
    (warn-only); they're returned in the third tuple element so the
    caller can log them and the operator can audit post-hoc. The bd
    issue stays closed because the model did the substantive work even
    if it left a TODO comment behind. The previous fail-the-turn
    behavior consumed retry budget on over-verification rounds that
    never reached `bd close` (drive halt 715f3edb)."""
    if not turn_success:
        return False, turn_reason, ()
    try:
        issue = bd.show(current_id)
    except DriverBdError as exc:
        return False, f"post-turn bd.show failed: {exc}", ()
    if issue.status != "closed":
        return False, f"issue still {issue.status} after turn", ()
    # bd issue closed; warn-only forbidden-pattern check.
    warnings: tuple[str, ...] = ()
    if forbidden_patterns and workspace is not None and started_at is not None:
        violations = _find_violations(workspace, started_at, forbidden_patterns)
        if violations:
            warnings = tuple(violations)
    return True, "", warnings


def _find_violations(
    workspace: Path,
    started_at: datetime,
    forbidden_patterns: tuple[str, ...],
) -> list[str]:
    """Walk `workspace` for files modified after `started_at` and check
    each for any of `forbidden_patterns`. Returns one entry per
    (path, pattern) hit. Empty = no violations (harness-k52f).

    Skips:
      - hidden dirs (any path component starting with `.`)
      - files > 1MB (binary-shaped, not worth scanning)
      - files that aren't decodable as UTF-8 text
    Best-effort: IO errors are silently skipped — the verification is
    a safety net, not a contract gate."""
    if not forbidden_patterns:
        return []
    started_ts = started_at.timestamp()
    violations: list[str] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(workspace).parts
        if any(p.startswith(".") for p in rel_parts):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime < started_ts:
            continue
        if stat.st_size > 1024 * 1024:
            continue
        try:
            content = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(workspace)
        for pattern in forbidden_patterns:
            if pattern in content:
                violations.append(f"{rel}: contains {pattern!r}")
    return violations


# --- verify gate (harness-xfh2) -----------------------------------


# Tail of stderr (or stdout, when stderr is empty) captured into the
# `last_failure` slot when a verify step exits non-zero. 200 chars per
# the xfh2 spec — enough to identify the failure mode without bloating
# the next turn's handoff.
VERIFY_STDERR_TAIL_CHARS: int = 200

# Per-step subprocess timeout in seconds. A verify step that hangs
# would otherwise block the whole loop; 60s is the same upper bound
# the orchestrator uses for shell tool calls.
VERIFY_TIMEOUT_SECONDS: int = 60


def _load_verify_map(path: Path | None) -> dict[str, tuple[VerifyStep, ...]]:
    """Parse the YAML plan draft into `{item.title: (verify steps,)}`.

    Missing path / unreadable YAML / malformed schema all degrade to an
    empty map — the verify gate is a safety net, not a contract gate.
    The model still sees `prior_attempt_failure` on any failure mode the
    rest of the loop catches; a broken draft just means verify itself
    is skipped this run."""
    if path is None or not path.exists():
        return {}
    try:
        text = path.read_text()
    except OSError:
        return {}
    try:
        draft = PlanDraft.from_yaml(text)
    except PlannerError:
        return {}
    return {item.title: tuple(item.verify) for item in draft.items if item.verify}


def _run_issue_verify(
    verify_map: Mapping[str, Sequence[VerifyStep]],
    bd: DriverBd,
    issue_id: str,
    workspace: Path,
    *,
    default_steps: Sequence[VerifyStep] = (),
    regression_snapshot: Path | None = None,
) -> tuple[str | None, int]:
    """Run the verify steps registered for `issue_id` (looked up by bd
    title). Returns ``(failure_msg_or_None, steps_run_count)``:

    - ``(None, 0)`` — no steps registered; nothing to gate on.
    - ``(None, N>0)`` — all N steps ran and passed; verified clean.
    - ``("…", N)`` — N-th step failed; message is the truncated tail.

    The count is the post-condition signal the auto-close gate
    (harness-b7m1) needs: "claim + verify passed" is only a trustworthy
    close signal when at least one step actually verified something.
    Without that distinction, an issue with zero registered steps would
    auto-close purely on the model's word — too aggressive.

    Title lookup, not bd-id lookup: the draft YAML carries the operator's
    titles and `commit_plan` materialized those verbatim into bd. A
    title mismatch (model renamed the issue post-commit, or the draft
    drifted) silently passes — the loop's existing classification
    handles those edges; we don't want to fail-closed on a clerical
    drift.

    `default_steps` (harness-oxj7) is the workspace-typed baseline
    list synthesized once per run by `default_workspace_verify_steps`.
    Runs BEFORE per-item steps so cheap parse-checks fail fast before
    any slower operator-authored gate. Empty list (the default) yields
    the pre-harness-oxj7 behavior."""
    try:
        issue = bd.show(issue_id)
    except DriverBdError as exc:
        # A bd.show failure here is rare (we just classified the turn,
        # which also called bd.show successfully). Treat as a soft pass
        # — the next iteration's bd.ready_under_epic will surface the
        # same issue if it's still open. Count 0 keeps auto-close from
        # firing on a lookup miss.
        return f"verify lookup failed (bd.show): {exc}", 0
    # harness-16w6: regression gate runs FIRST — before any shell step —
    # so a stub-rewrite that still loads (passes smoke) is caught here.
    # Counts as a verify failure so the existing reopen/retry/park path
    # feeds the "you deleted X/Y/Z" reason back and blocks the close.
    if regression_snapshot is not None and regression_snapshot.is_file():
        regression = detect_regression(workspace, regression_snapshot)
        if regression is not None:
            return f"regression: {regression}", 1
    per_item_steps = verify_map.get(issue.title, ()) if verify_map else ()
    steps: tuple[VerifyStep, ...] = tuple(default_steps) + tuple(per_item_steps)
    if not steps:
        return None, 0
    for i, step in enumerate(steps, start=1):
        exit_code, tail = _exec_verify_cmd(step, workspace)
        if exit_code != 0:
            preview = step.cmd if len(step.cmd) <= 80 else step.cmd[:77] + "..."
            msg = f"{preview} exit={exit_code}: {tail}" if tail else f"{preview} exit={exit_code}"
            return msg, i
    return None, len(steps)


def _exec_verify_cmd(step: VerifyStep, workspace: Path) -> tuple[int, str]:
    """Run a single verify command. Returns (exit_code, stderr/stdout
    tail truncated to VERIFY_STDERR_TAIL_CHARS).

    Separate from `_run_issue_verify` so tests can monkeypatch the
    subprocess seam without intercepting the lookup logic. shell=True is
    the default for operator convenience (pipes, $VAR); shell=False
    routes through `shlex.split` so `cmd: "node smoke.js arg"` Just
    Works. OSError + TimeoutExpired both surface as non-zero with the
    exception message in the tail — a hung verify is a failure."""
    args: str | list[str]
    use_shell = step.shell
    if use_shell:
        args = step.cmd
    else:
        try:
            args = shlex.split(step.cmd)
        except ValueError as exc:
            return 1, f"unparseable cmd: {exc}"
    try:
        result = subprocess.run(  # noqa: S603 — cmd from trusted operator-authored draft YAML
            args,
            shell=use_shell,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=VERIFY_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        return 1, f"executable not found: {exc}"
    except subprocess.TimeoutExpired:
        return 1, f"timeout after {VERIFY_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return 1, f"exec failed: {exc}"
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    tail_src = stderr or stdout
    tail = tail_src[-VERIFY_STDERR_TAIL_CHARS:]
    return result.returncode, tail


# --- claim-without-close (harness-pfvj) ---------------------------


# Prefix on `last_failure` entries that originated from the
# claim-detection path. Matches the `verify_failed:` convention from
# harness-xfh2 — operators and the next-turn handoff renderer can grep
# for the prefix to distinguish failure modes.
_CLAIM_WITHOUT_CLOSE_PREFIX: str = "claim_without_close:"

# Fallback message used when a claim is detected but no verify steps are
# registered for the issue. Tells the model the concrete next action —
# "you said you were done, but you didn't run bd close, run it." — so
# the next handoff has something more actionable than "issue still
# open after turn". Kept short; the executor's prompt budget is tight.
_CLAIM_WITHOUT_CLOSE_NO_VERIFY_HINT: str = (
    "model claimed success in its reply but did not invoke "
    "`bd close <id>` via shell — run it explicitly when the work is done"
)


def _is_still_open_reason(reason: str) -> bool:
    """True when `_classify_post_turn` returned the 'issue still
    {status} after turn' failure shape — the exact path claim-detection
    cares about. Other failure modes (forbidden-pattern hits, bd.show
    errors, fabrication-fallback) keep their original reason; the
    claim-detection rewrite only applies when the bd issue stayed
    open."""
    return reason.startswith("issue still ") and reason.endswith(" after turn")


def _try_auto_close(bd: DriverBd, issue_id: str, log: _LogWriter) -> bool:
    """Best-effort `bd close <id>` on the model's behalf (harness-b7m1).

    Used by the claim-without-close gate when the model claimed success
    AND verify ran at least one step that passed — the harness drives
    the close so the drive doesn't burn another retry on a verification
    loop the model can't exit (loop f36cf2e4 pattern).

    Returns True on success; False (with a log line) on bd failure so
    the caller can fall back to the soft-hint retry path."""
    close_reason = "auto-closed by drive: model claimed success + verify gate passed (harness-b7m1)"
    try:
        bd.close(issue_id, reason=close_reason)
    except DriverBdError as exc:
        log(f"auto-close failed for {issue_id}: {exc}")
        return False
    return True


def _try_reopen(bd: DriverBd, issue_id: str) -> bool:
    """Best-effort `bd update --status=open`. Returns True on success.
    A transient bd hiccup here is logged via the caller's `log()` but
    doesn't crash the loop — `last_failure` still gets stashed, and the
    operator sees a reopened-FAILED note in the progress log."""
    try:
        bd.reopen(issue_id)
    except DriverBdError:
        return False
    return True


# --- targeted-fix detection ---------------------------------------


# Operator convention when reopening a previously-closed bd issue: lead
# the notes with "REGRESSION YYYY-MM-DD:" so the driver can recognize
# the retry as a targeted fix. Case-sensitive substring match — the
# convention is established (e.g. harness-90j0 / harness-vjb6 reopened
# on 2026-05-21 both carry the literal "REGRESSION 2026-05-21:"). False
# positives (notes that happen to mention regressions in passing) are
# mostly harmless: the worst case is an extra MODE banner the model
# reads and complies with.
_REGRESSION_MARKER: str = "REGRESSION"


def _issue_has_regression(bd: DriverBd, issue_id: str) -> bool:
    """Best-effort check: does the bd issue's notes contain the
    operator's regression marker? Swallows DriverBdError so a transient
    bd hiccup degrades to 'no marker found' — build_handoff is about to
    call bd.show again anyway and will raise loudly on a real lookup
    failure. Pure read; no mutation."""
    try:
        issue = bd.show(issue_id)
    except DriverBdError:
        return False
    notes = issue.raw.get("notes") or ""
    return _REGRESSION_MARKER in notes


# --- tool registry ------------------------------------------------


def _build_executor_registry(workspace: Path) -> ToolRegistry:
    """Build the executor's tool registry. v0: the coding-profile
    subset that doesn't depend on stores / character. Store-dependent
    tools (search_memory, search_facts) are skipped — they need
    retrieval setup the driver doesn't provision. introspect /
    spawn_subagent are also skipped (need adapter+character wiring;
    can be added in a future revision).

    All filesystem tools sandboxed to `workspace`. The shell tool runs
    with `cwd=workspace`, so `bd close <id>` works as long as the
    workspace is a bd project root (the same constraint the rest of
    the harness already enforces)."""
    catalog = ToolCatalog()
    seed_builtins_into(catalog, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))
    registry = ToolRegistry(catalog=catalog)

    builders: dict[str, Tool] = {
        "read_file": ReadFileTool(root=workspace),
        "list_dir": ListDirTool(root=workspace),
        "grep": GrepTool(root=workspace),
        "glob": GlobTool(root=workspace),
        "edit_file": EditFileTool(root=workspace),
        "write_file": WriteFileTool(root=workspace),
        "stream_edit": StreamEditTool(root=workspace),
        "python_stream": PythonStreamTool(root=workspace),
        "shell": ShellTool(cwd=workspace),
        "git_status": GitStatusTool(root=workspace),
        "git_diff": GitDiffTool(root=workspace),
        "git_log": GitLogTool(root=workspace),
        "fetch_url": FetchUrlTool(),
        "now": NowTool(),
        "date_math": DateMathTool(),
        "calc": CalcTool(),
    }
    for tool in builders.values():
        registry.register(tool)
    # tool_search / load_tool need a reference to the registry they
    # operate on, so they construct AFTER the static tools are in
    # place. `load_tool`'s builder map is empty — the executor exposes
    # its full set upfront, so lazy-loading at runtime isn't needed.
    registry.register(ToolSearchTool(catalog=catalog, registry=registry))
    registry.register(LoadToolTool(catalog=catalog, registry=registry, builders={}))
    return registry


# --- state / log -------------------------------------------------


def _load_or_init_state(config: LoopConfig) -> LoopRunState:
    if config.resume_from is not None:
        path = LoopRunState.state_path(config.workspace, config.resume_from)
        state = LoopRunState.load(path)
        # Override the loaded state's max_turns with the config value
        # (harness-iai4). The CLI's --max-turns is user intent — saved
        # state's stale value would silently make a "resume with a
        # higher cap" run instantly exhaust. Persist immediately so a
        # subsequent reload reflects the new cap.
        if state.max_turns != config.max_turns:
            state.max_turns = config.max_turns
            _save_state(state, config.workspace)
        return state
    sha = _git_head_sha(config.workspace)
    state = LoopRunState.fresh(
        epic_id=config.epic_id,
        max_turns=config.max_turns,
        started_at_sha=sha,
    )
    _save_state(state, config.workspace)
    return state


def _save_state(state: LoopRunState, workspace: Path) -> None:
    state.save(LoopRunState.state_path(workspace, state.loop_run_id))


def _git_head_sha(workspace: Path) -> str:
    """`git rev-parse HEAD` from the workspace. Returns "unknown" on
    any error — the handoff still functions with a degenerate sha
    (git diff just returns nothing), so this is non-fatal."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 — git on PATH is expected
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _resolve_log_path(config: LoopConfig, state: LoopRunState) -> Path:
    if config.log_path is not None:
        return config.log_path
    return LoopRunState.state_dir(config.workspace) / f"{state.loop_run_id}.log"


def _open_log(log_path: Path) -> _LogWriter:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return _LogWriter(log_path)


class _LogWriter:
    """Tiny append-only log writer. One line per call. Timestamps each
    line so `tail -f` is informative. Best-effort — IO errors don't
    crash the loop (a crashed log isn't worse than no log)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def __call__(self, message: str) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}\n"
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            # Log a warning to stderr but don't crash the loop — a
            # missing log is annoying, not corrupting.
            logging.getLogger(__name__).warning("loop log write failed: %s", exc)


# --- exit paths --------------------------------------------------


def _exit_success(state: LoopRunState, workspace: Path, log: _LogWriter) -> LoopResult:
    # harness-iljv: reached only when the ready queue emptied with
    # nothing parked, so the epic is genuinely complete. A run that
    # parked anything exits via _exit_partial instead.
    log(f"loop_run={state.loop_run_id} SUCCESS (epic complete)")
    _save_state(state, workspace)  # harness-3zu3: persist on every exit path
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=None,
        turns_used=state.turns_used,
        exit_reason="success",
        parked_issues=list(state.parked_issues),
    )


def _exit_partial(state: LoopRunState, workspace: Path, log: _LogWriter) -> LoopResult:
    """The ready queue emptied only because parked issues were filtered
    out (harness-iljv). Some work may have closed, but the parked issues
    — and anything depending on them — are stranded pending operator
    pickup, so this is reported distinctly from "success"."""
    parked_tail = ", ".join(state.parked_issues)
    log(
        f"loop_run={state.loop_run_id} PARTIAL "
        f"(ready queue drained; {len(state.parked_issues)} parked: {parked_tail})"
    )
    _save_state(state, workspace)  # harness-3zu3: persist on every exit path
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=None,
        turns_used=state.turns_used,
        exit_reason="partial",
        parked_issues=list(state.parked_issues),
    )


def _exit_exhausted(state: LoopRunState, workspace: Path, log: _LogWriter) -> LoopResult:
    log(
        f"loop_run={state.loop_run_id} EXHAUSTED "
        f"(turns_used={state.turns_used} max={state.max_turns})"
    )
    _save_state(state, workspace)  # harness-3zu3: persist on every exit path
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=None,
        turns_used=state.turns_used,
        exit_reason="exhausted",
        parked_issues=list(state.parked_issues),
    )


def _exit_halted(
    bd: DriverBd,
    state: LoopRunState,
    *,
    current_id: str,
    reason: str,
    workspace: Path,
    log: _LogWriter,
) -> LoopResult:
    log(f"loop_run={state.loop_run_id} HALTED on {current_id}: {reason}")
    # harness-3zu3: persist before returning so a resume sees the
    # post-halt attempt_counts + turns_used (the increments that led to
    # the halt), not the stale pre-halt values — otherwise the operator
    # gets one retry instead of the expected budget after reopen.
    _save_state(state, workspace)
    with contextlib.suppress(DriverBdError):
        bd.flag_human(current_id, reason=f"loop halted: {reason}")
    with contextlib.suppress(DriverBdError):
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=current_id,
            status="halted",
            body=reason,
        )
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=current_id,
        turns_used=state.turns_used,
        exit_reason="halted",
        parked_issues=list(state.parked_issues),
    )


def _exit_interrupted(
    bd: DriverBd, state: LoopRunState, workspace: Path, log: _LogWriter
) -> LoopResult:
    log(f"loop_run={state.loop_run_id} INTERRUPTED")
    _save_state(state, workspace)  # harness-3zu3: persist on every exit path
    with contextlib.suppress(DriverBdError):
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=state.epic_id,
            status="interrupted",
            body=f"SIGINT received at turns_used={state.turns_used}",
        )
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=None,
        turns_used=state.turns_used,
        exit_reason="interrupted",
        parked_issues=list(state.parked_issues),
    )


def _park_issue(
    bd: DriverBd,
    state: LoopRunState,
    *,
    current_id: str,
    reason: str,
    log: _LogWriter,
) -> None:
    """harness-zcrd: park an issue after max-attempts exhaustion.

    Records the bd-id in `state.parked_issues` (filtered out of
    subsequent ready_under_epic results), flags via `bd flag_human`
    so the operator sees it in `bd human list`, writes a session-state
    bead, and logs a PARKED line. Does NOT return — the caller
    `continue`s the loop to pick up the next ready issue.

    `bd flag_human` and session-state writes are best-effort: a bd
    hiccup here doesn't break the drive's forward motion. The
    in-memory `state.parked_issues` is the load-bearing filter; the
    bd flag is operator-facing signal."""
    log(f"loop_run={state.loop_run_id} PARKED {current_id}: {reason}")
    if current_id not in state.parked_issues:
        state.parked_issues.append(current_id)
    with contextlib.suppress(DriverBdError):
        bd.flag_human(current_id, reason=f"drive parked after max attempts: {reason}")
    with contextlib.suppress(DriverBdError):
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=current_id,
            status="parked",
            body=reason,
        )


def _on_success(bd: DriverBd, state: LoopRunState, current_id: str, log: _LogWriter) -> None:
    state.closed_this_run.append(current_id)
    state.last_failure.pop(current_id, None)
    log(f"turn {state.turns_used}: {current_id} CLOSED")
    with contextlib.suppress(DriverBdError):
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=current_id,
            status="success",
            body=f"closed at turn {state.turns_used}",
        )


# --- signal handling ----------------------------------------------


class _InterruptFlag:
    """Set on SIGINT; checked at the top of every loop iteration."""

    def __init__(self) -> None:
        self._flag = False

    def set(self) -> None:
        self._flag = True

    def is_set(self) -> bool:
        return self._flag


@contextlib.contextmanager
def _sigint_guard() -> Iterator[_InterruptFlag]:
    """Install a SIGINT handler that flips the interrupt flag, restore
    the previous handler on exit. Safe to nest under outer handlers —
    the previous handler is restored on context exit."""
    flag = _InterruptFlag()

    def handler(_signum: int, _frame: object) -> None:
        flag.set()

    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGINT, previous)


# --- workspace snapshot (harness-9ijr) ----------------------------


# Names that are pruned from the snapshot walk (entire subtree
# skipped). Targets the high-cost / not-our-state directories
# operators routinely have in workspaces. `.harness` is special:
# the snapshot tar lives INSIDE this dir, so excluding it prevents
# the tar from trying to archive itself.
DEFAULT_SNAPSHOT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".harness", ".git", "node_modules", ".venv", "__pycache__"}
)

# File-suffix exclusions. `.pyc` byte-compiled artifacts and OS
# scratch files don't help recovery and bloat the tar; skipping is
# pure win. Tuple matches `str.endswith`'s signature.
DEFAULT_SNAPSHOT_EXCLUDE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")

# Pre-tar uncompressed size cap. 100MB matches the spec — large
# enough to capture realistic scratch workspaces (the GTA2 case is
# <1MB), small enough to force operators to think before they
# snapshot a node_modules-laden tree.
SNAPSHOT_SIZE_CAP_BYTES: int = 100 * 1024 * 1024


class SnapshotTooBigError(RuntimeError):
    """Raised when the pre-tar workspace size exceeds the cap. The
    message names the cap so operators know the threshold to clear
    or the flag to bypass it."""


def _snapshot_path(workspace: Path, loop_run_id: str) -> Path:
    """`.harness/loop_runs/<id>_workspace.tar.gz` under workspace."""
    return workspace / ".harness" / "loop_runs" / f"{loop_run_id}_workspace.tar.gz"


def _iter_snapshot_files(workspace: Path) -> Iterator[Path]:
    """Walk workspace yielding files to include in the snapshot.

    Uses os.walk with followlinks=False to avoid infinite loops on
    self-referential symlinks. Prunes DEFAULT_SNAPSHOT_EXCLUDE_DIRS
    in-place so we never recurse into them (cheap — saves the cost
    of statting every node_modules file). Skips files whose basename
    ends with DEFAULT_SNAPSHOT_EXCLUDE_SUFFIXES."""
    for root, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = [d for d in dirs if d not in DEFAULT_SNAPSHOT_EXCLUDE_DIRS]
        for fname in files:
            if fname.endswith(DEFAULT_SNAPSHOT_EXCLUDE_SUFFIXES):
                continue
            yield Path(root) / fname


def _snapshot_workspace(
    workspace: Path,
    loop_run_id: str,
    *,
    size_cap_bytes: int = SNAPSHOT_SIZE_CAP_BYTES,
) -> Path:
    """tar+gzip `workspace` to `.harness/loop_runs/<id>_workspace.tar.gz`.

    Returns the snapshot path on success. Raises SnapshotTooBigError when
    pre-tar uncompressed total exceeds size_cap_bytes — caller MUST
    abort the run rather than continue without a recoverable
    starting point.

    Pre-walk size accounting runs first; only after the cap check
    passes does the tar open. This means an over-cap workspace
    leaves no partial tar behind. Files that vanish between the
    size walk and the tar pass (unlikely in practice; the loop's
    workspace shouldn't be churning) are silently skipped from the
    tar — `tarfile.add` raises but we don't catch; let it surface
    so the operator sees the race condition."""
    total_bytes = 0
    files_to_include: list[Path] = []
    for path in _iter_snapshot_files(workspace):
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
        if total_bytes > size_cap_bytes:
            mb = size_cap_bytes // (1024 * 1024)
            raise SnapshotTooBigError(
                f"workspace size > {mb}MB cap (rooted at {workspace}). "
                f"Pass --no-snapshot to skip the snapshot guard, or "
                f"narrow --workspace to a smaller subtree."
            )
        files_to_include.append(path)

    snapshot_path = _snapshot_path(workspace, loop_run_id)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snapshot_path, "w:gz") as tf:
        for path in files_to_include:
            arcname = path.relative_to(workspace)
            tf.add(path, arcname=str(arcname), recursive=False)
    return snapshot_path


# --- harness-16w6 / harness-ul5z: regression guard + scratch hygiene ---


def _last_green_path(workspace: Path, loop_run_id: str) -> Path:
    """`.harness/loop_runs/<id>_lastgreen.tar.gz` — the rollback target,
    overwritten on every close (so it's always the current issue's clean
    starting point)."""
    return workspace / ".harness" / "loop_runs" / f"{loop_run_id}_lastgreen.tar.gz"


def _scratch_archive_dir(workspace: Path, loop_run_id: str) -> Path:
    return workspace / ".harness" / "loop_runs" / f"{loop_run_id}_scratch"


def _baseline_is_green(default_steps: Sequence[VerifyStep], workspace: Path) -> bool:
    """True iff every workspace-typed default verify step passes — i.e.
    the deliverable loads/parses. No bd / no per-item steps; this is the
    standalone baseline check the regression guard uses at start and on
    park. Empty steps (nothing to verify) counts as green."""
    for step in default_steps:
        exit_code, _ = _exec_verify_cmd(step, workspace)
        if exit_code != 0:
            return False
    return True


def _refresh_last_green(config: LoopConfig, state: LoopRunState, log: _LogWriter) -> Path | None:
    """Snapshot the (now-green) workspace as the rollback target. Returns
    the snapshot path, or None if the snapshot couldn't be taken (over
    cap / IO error) — a missing last-green just disables rollback for the
    next issue, it never fails the run."""
    dest = _last_green_path(config.workspace, state.loop_run_id)
    try:
        return archive_workspace(config.workspace, dest, size_cap_bytes=SNAPSHOT_SIZE_CAP_BYTES)
    except (WorkspaceTooBigError, OSError) as exc:
        log(f"regression guard: could not refresh last-green snapshot ({exc}); rollback disabled")
        return None


def _sweep_issue_scratch(
    config: LoopConfig,
    state: LoopRunState,
    baseline_files: set[str],
    log: _LogWriter,
) -> None:
    """Archive scratch files the issue created (vs `baseline_files`) into
    `.harness/loop_runs/<id>_scratch/`. No-op when scratch_sweep is off."""
    if not config.scratch_sweep:
        return
    archive_dir = _scratch_archive_dir(config.workspace, state.loop_run_id)
    moved = sweep_scratch(
        config.workspace,
        baseline_files,
        patterns=config.scratch_patterns,
        archive_dir=archive_dir,
    )
    if moved:
        log(
            f"scratch sweep: archived {len(moved)} file(s) -> "
            f"{archive_dir.name}: {', '.join(moved)}"
        )


def _post_close_housekeeping(
    config: LoopConfig,
    state: LoopRunState,
    baseline_files: set[str],
    last_green: Path | None,
    log: _LogWriter,
) -> Path | None:
    """After a close: archive the issue's scratch, then refresh the
    last-green snapshot (the workspace just passed verify, so it's the
    new clean baseline). Returns the (possibly updated) last-green path."""
    _sweep_issue_scratch(config, state, baseline_files, log)
    if config.regression_guard:
        return _refresh_last_green(config, state, log)
    return last_green


def _post_park_housekeeping(
    config: LoopConfig,
    state: LoopRunState,
    baseline_files: set[str],
    last_green: Path | None,
    default_steps: Sequence[VerifyStep],
    log: _LogWriter,
) -> None:
    """After a park (harness-16w6 no-punting): if the parked issue left
    the baseline degraded — either it no longer loads (red verify) OR it
    regressed vs last-green (lost symbols / shrank, which still "loads"
    but is gutted) — restore last-green so the break can't cascade. The
    restore also removes the issue's scratch (files added since the
    snapshot). If the baseline is still healthy (incomplete-but-intact
    work), keep it and just sweep scratch."""
    has_green = config.regression_guard and last_green is not None and last_green.is_file()
    if has_green:
        assert last_green is not None  # narrowed by has_green
        red = not _baseline_is_green(default_steps, config.workspace)
        regressed = detect_regression(config.workspace, last_green) is not None
        if red or regressed:
            restored, removed = restore_workspace(config.workspace, last_green)
            why = "broken" if red else "regressed (lost code vs last-green)"
            log(
                f"loop_run={state.loop_run_id} ROLLED_BACK after park: baseline {why}; "
                f"restored last-green ({restored} files, removed {len(removed)} issue-added)"
            )
            return
    _sweep_issue_scratch(config, state, baseline_files, log)


__all__ = [
    "DEFAULT_SNAPSHOT_EXCLUDE_DIRS",
    "DEFAULT_SNAPSHOT_EXCLUDE_SUFFIXES",
    "EXECUTOR_USER_MESSAGE",
    "SNAPSHOT_SIZE_CAP_BYTES",
    "VERIFY_STDERR_TAIL_CHARS",
    "VERIFY_TIMEOUT_SECONDS",
    "LoopConfig",
    "LoopResult",
    "SnapshotTooBigError",
    "run_loop",
]
