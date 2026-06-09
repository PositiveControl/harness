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
from harness.driver.handoff import Handoff, build_handoff, substantial_artifacts
from harness.driver.planner import PlanDraft, PlannerError, VerifyStep
from harness.driver.precommit_verify_hook import (
    PreCloseVerifyHook,
    make_pre_close_verify_hook,
)
from harness.driver.state import LoopRunState
from harness.driver.turn_fsm import PREMISE_UNMET_REASON_PREFIX
from harness.driver.vision_qa import build_rubric, run_advisory_qa
from harness.driver.workspace_guard import (
    DEFAULT_SCRATCH_PATTERNS,
    WorkspaceTooBigError,
    archive_workspace,
    detect_regression,
    list_workspace_files,
    restore_workspace,
    sweep_scratch,
    workspace_changed,
)
from harness.driver.workspace_verify import (
    browser_smoke_skip_reason,
    default_workspace_verify_steps,
)
from harness.model import make_vision_adapter
from harness.model.adapter import ChatMessage, ModelAdapter, PromptBudgetError
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
    OutlineTool,
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
    "and is invisible to the loop; run `bd close <issue-id>` for real. "
    "To read code, prefer `outline <path>` to map a file's functions/classes "
    "cheaply, then `read_file <path> symbol=<name>` (e.g. symbol='Foo.bar') to "
    "pull a whole function or class. Reach for `read_file` offset/limit line "
    "ranges only for non-code files or when you already know the exact lines — "
    "blind line ranges tend to return half a function."
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
    # harness-s0el9: sampling temperature for the executor's inner
    # run_tool_loop. Lower than `harness chat`'s 0.5 default because the
    # drive is autonomous code editing, where determinism curbs the wild
    # destructive edits a higher temperature invites (drive loop_run=
    # 9a1e7970 harness-rxtpz: a high-variance sed deleted the run-over
    # block instead of adding it). 0.2 matches the critic's generate temp.
    # Override per run with `--executor-temperature`.
    executor_temperature: float = 0.2
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
    # harness-6y2dc: bd ids to filter out of `ready` up-front, in
    # addition to this run's own `state.parked_issues`. auto_iterate
    # carries the union of prior passes' parked issues here so a
    # fresh-state pass doesn't re-drive an issue that already parked
    # (hit max_attempts) last pass. Without it, each pass starts with an
    # empty LoopRunState, re-queries `bd ready` — which still lists
    # parked-but-open issues — and re-drives them from cold, burning the
    # whole pass budget re-discovering they can't close. Nothing mutates
    # the workspace between passes (the critic only reads + files beads),
    # so a parked issue faces the identical workspace next pass and would
    # park again; skipping it is correct. Empty for standalone run_loop.
    skip_issue_ids: frozenset[str] = frozenset()
    # harness-ke4hx.4: advisory VLM browser-QA endpoint. When set, the
    # smoke step captures a post-settle screenshot and, after each turn's
    # verify resolves, a VLM judges it against the bead's acceptance
    # rubric — logged + appended to .harness/vision_qa.jsonl, NEVER gating
    # (Phase 1 is signal-collection only). None = off; the smoke gate runs
    # exactly as before. CLI wires this from settings.vision_base_url.
    vision_base_url: str | None = None


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
    # harness-s0el9: per-parked-issue gate test command (from
    # `state.last_test_cmd`). Lets an outer auto-iterate pass fingerprint
    # the gate file a parked issue was stuck on, so a between-pass gate
    # repair can un-strand it instead of carrying it skipped forever.
    parked_test_cmds: dict[str, str] = field(default_factory=dict)


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
    # harness-ke4hx.4: advisory VLM browser-QA. Resolved once per run —
    # None when vision_base_url is unset (feature dormant; smoke gate
    # unchanged). The screenshot the smoke step captures and the JSONL
    # audit both live under .harness so they ride the run's lifecycle.
    vision_adapter = make_vision_adapter(config.vision_base_url)
    vision_shot = config.workspace / ".harness" / "vision_shot.png" if vision_adapter else None
    vision_qa_log = config.workspace / ".harness" / "vision_qa.jsonl"
    if vision_adapter is not None:
        log(f"advisory vision-QA active: {vision_adapter.id} (non-gating)")

    enforce_blank = _blank_canvas_enforced(bd, config.render_milestone_id)
    default_verify_steps = default_workspace_verify_steps(
        config.workspace, enforce_blank_canvas=enforce_blank, capture_shot=vision_shot
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

    # harness-xxdr: ambient HARNESS_VLLM_TRACE default. If unset, point
    # at .harness/loop_runs/<run_id>.vllm_trace.jsonl so the trace
    # colocates with run state and gets cleaned by the same lifecycle.
    # User-set value wins; restored on exit so chat / non-loop callers
    # in the same process aren't surprised by tracing. ExitStack so the
    # rest of the function keeps its indentation level — the env-var
    # manager is a sibling concern to the sigint guard, not nested.
    with contextlib.ExitStack() as stack:
        ambient_trace = stack.enter_context(ambient_vllm_trace(config.workspace, state.loop_run_id))
        if ambient_trace is not None:
            log(f"vllm trace: {ambient_trace}")
        interrupted = stack.enter_context(_sigint_guard())
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
            # harness-6y2dc: ALSO filter config.skip_issue_ids — issues
            # parked by a PRIOR auto_iterate pass. A fresh-state pass
            # otherwise re-drives them from cold (the wheel-spin), since
            # parked-but-open issues still come back from `bd ready`.
            skip_set = set(state.parked_issues) | config.skip_issue_ids
            stranded_now = {issue.id for issue in ready if issue.id in skip_set}
            if skip_set:
                ready = [issue for issue in ready if issue.id not in skip_set]
            if not ready:
                # harness-iljv: distinguish a genuinely-complete epic
                # from one whose ready queue only emptied because we
                # filtered out parked issues above. The latter leaves
                # the parked issues (and their dependents) stranded, so
                # it's a "partial", not a "success" — callers and the
                # exit code must be able to tell the difference.
                # `stranded_now` is precise: "partial" only when a
                # skipped id was actually present in this pass's ready
                # (carried-parked work still open), not merely configured.
                if state.parked_issues or stranded_now:
                    return _exit_partial(
                        state, config.workspace, log, stranded=sorted(stranded_now)
                    )
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
            # harness-psnz1: an investigation-loop retry — the
            # prior attempt left the workspace byte-identical to the issue
            # baseline (zero net edits vs last-green) despite burning its
            # round budget on reads. Run b74bef10 cw1m parked after 3 such
            # attempts (0 edits, 65 read/grep/outline calls): the per-turn
            # repeat-counter resets each attempt, so the loop is invisible
            # to it. Surface it in the handoff as a "stop reading, edit
            # now" directive. Reuses Fix A's workspace_changed signal.
            last_green = _last_green_if_present(last_green, log)
            prior_made_no_edits = (
                attempt > 1
                and config.regression_guard
                and last_green is not None
                and not workspace_changed(config.workspace, last_green)
            )
            if prior_made_no_edits:
                # Log the nudge injection so the investigation-loop break
                # is attributable in the run log (it was previously a
                # silent prompt-only signal — loop_run=498a4d79 fired it on
                # every retry with no trace, so its effect couldn't be
                # measured).
                log(
                    f"loop_run={state.loop_run_id} no-edit-nudge injected for "
                    f"{current.id} (attempt {attempt}: prior attempt left the "
                    f"workspace byte-identical to last-green)"
                )
            # harness-lefw: signal targeted-fix mode when the loop has
            # already touched this issue OR the operator left a
            # "REGRESSION" marker in notes (their convention when
            # reopening a previously-closed issue). _issue_has_regression
            # tolerates a missing bd lookup — bd.show is called again
            # inside build_handoff and a transient miss there raises.
            #
            # harness-8tjnv: ALSO target-fix when the workspace already
            # holds a built source artifact (>=50 lines). A foundational
            # bead (e.g. "§1 game.js skeleton") re-served after later
            # sections built the file would otherwise get a fresh-build
            # handoff and the model recreates the skeleton over the built
            # game (run b085854e: 2000→51→2000). The write_file hard-block
            # self-gates on file existence, so this only adds protection.
            targeted_fix = (
                attempt > 1
                or _issue_has_regression(bd, current.id)
                or bool(substantial_artifacts(config.workspace))
            )

            handoff = build_handoff(
                state,
                current.id,
                bd,
                git_root=config.workspace,
                prior_attempt_failure=prior_failure,
                workspace=config.workspace,
                targeted_fix=targeted_fix,
                forbidden_patterns=config.forbidden_patterns,
                prior_made_no_edits=prior_made_no_edits,
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
                    prior_made_no_edits=prior_made_no_edits,
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
                    executor_temperature=config.executor_temperature,
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
                capture_shot=vision_shot,
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

            # harness-ke4hx.4: advisory vision-QA. Judges the smoke
            # screenshot against this bead's rubric AFTER the gate outcome
            # (`success`) is already decided — it logs + records, and NEVER
            # changes pass/fail. Degrades to a logged skip when the
            # adapter, screenshot, or endpoint is unavailable.
            if vision_adapter is not None and vision_shot is not None:
                run_advisory_qa(
                    vision_adapter,
                    vision_shot,
                    build_rubric(current.title, str(current.raw.get("description", ""))),
                    gate_passed=success,
                    issue_id=current.id,
                    jsonl_path=vision_qa_log,
                    log=log,
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
            #   (c) harness-82r1v — made_edits: the model edited
            #       issue-relevant source files this attempt (workspace
            #       differs from the last-green baseline) but produced no
            #       claim phrase. The claim signal is only a hint; the
            #       verify gate is the contract. In run 29f4a974 / gta
            #       6182c539 a duplicate-read tripped wrap_up_forced,
            #       truncating the reply to a non-claim stub AFTER the
            #       work (score state + render, verify-green) was done —
            #       (a)/(b) missed it, the turn scored FAIL, and the
            #       correct edits were discarded on the inter-attempt
            #       restore. Treating real edits as a trigger routes the
            #       turn through the same verify gate. Gated on actual
            #       edits so a no-op turn over an already-green workspace
            #       can't false-close (steps_ran > 0 below is the second
            #       guard: verify must have something to corroborate).
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
            # The snapshot can vanish mid-run (operator cleanup of
            # .harness/loop_runs, tmpreaper, disk eviction); drop a stale
            # path so the guard degrades to rollback-disabled instead of
            # crashing on the gone tar.
            last_green = _last_green_if_present(last_green, log)
            made_edits = (
                last_green is not None
                and config.regression_guard
                and workspace_changed(config.workspace, last_green)
            )
            if (
                not success
                and _is_still_open_reason(reason)
                and (
                    detect_claim_signal(turn_reply)
                    or detect_claim_in_shell_call(turn_last_shell)
                    or made_edits
                )
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
                elif smoke_skip is not None:
                    # harness-tro2c: the strongest gate (runtime smoke) is
                    # degraded for this browser app — the steps that ran are
                    # syntax-only, so they can't corroborate a runtime claim.
                    # Refuse to auto-close on the model's behalf (the exact
                    # vector that false-closed eznk/8aav/ray4 in run b085854e
                    # and 8 beads in run 3c7c9da2). Fall through to a soft hint;
                    # the issue retries then parks rather than closing blind.
                    reason = (
                        f"{_CLAIM_WITHOUT_CLOSE_PREFIX} runtime smoke gate is OFF "
                        "(syntax-only verify) — not auto-closing a browser app on a "
                        "claim the gate can't corroborate. Close manually after a "
                        "real runtime check, or install the browser extra "
                        "(uv sync --extra browser && playwright install chromium)."
                    )
                elif (
                    config.auto_close_on_claim
                    and steps_ran > 0
                    and _try_auto_close(bd, current.id, log)
                ):
                    # Drive the close on the model's behalf; bypass the
                    # success branch's redundant verify by handling the
                    # close-and-continue here.
                    trigger = (
                        "claim"
                        if (
                            detect_claim_signal(turn_reply)
                            or detect_claim_in_shell_call(turn_last_shell)
                        )
                        else "edits"
                    )
                    log(
                        f"turn {state.turns_used}: {current.id} AUTO_CLOSED "
                        f"({trigger} + verify passed, "
                        f"{steps_ran} step{'s' if steps_ran != 1 else ''})"
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
                # harness-2qth: between attempts, restore workspace to
                # last-green so the next attempt starts from the clean
                # baseline instead of inheriting this attempt's broken
                # edits. See loop_run=3e295564 turns 16-19: attempts 2-5
                # each compounded onto an already-broken file.
                # harness-iteip: restore only fires now when the attempt
                # left the baseline broken/regressed — a green workspace
                # keeps its edits.
                _inter_attempt_restore(
                    config,
                    state,
                    last_green,
                    issue_id=current.id,
                    default_steps=default_verify_steps,
                    log=log,
                )
                continue

            # harness-r0s61: the turn failed (fabrication-fallback, claim
            # without verifiable work, …). If the model already shell-closed
            # the bead before failing, reopen it so a failed turn can never
            # leave a closed bead behind (run b085854e turn 9 / harness-zbnq).
            if _reconcile_failed_close(bd, current.id, log=log):
                reason = f"{reason}; reopened unverified model-close"

            log(f"turn {state.turns_used}: {current.id} attempt={attempt} FAIL ({reason})")

            # harness-u1il5 follow-on: a premise-unmet halt (ASSESS called
            # flag_blocked — the thing the bead asks to verify/fix doesn't
            # exist because an upstream dependency never landed it) is not
            # retryable: every attempt faces the identical false premise.
            # Park-and-flag immediately instead of burning the retry budget
            # (loop_run=498a4d79: §15a-iii gating parked after 3 futile
            # attempts trying to gate inputs that §9b-i/§10 closed blind).
            if reason.startswith(PREMISE_UNMET_REASON_PREFIX):
                log(
                    f"turn {state.turns_used}: {current.id} PREMISE_UNMET — "
                    f"parking without retry ({reason})"
                )
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
                    bd,
                    state,
                    current_id=current.id,
                    reason=reason,
                    workspace=config.workspace,
                    log=log,
                )

            if attempt < config.max_attempts_per_issue:
                state.last_failure[current.id] = reason
                _save_state(state, config.workspace)
                # harness-2qth: same per-attempt rollback as the
                # verify-fail branch above. harness-iteip: green,
                # non-regressed workspaces keep their edits.
                _inter_attempt_restore(
                    config,
                    state,
                    last_green,
                    issue_id=current.id,
                    default_steps=default_verify_steps,
                    log=log,
                )
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
    the planner's _compose_observers contract).

    Streaming `token_delta` events are coalesced (harness): each streamed
    chunk was a separate log line — hundreds per model call, each a few
    characters — which buried the readable events and reopened the file
    per fragment. They're buffered instead and flushed as a single
    `model_output | chars=N preview=…` line at `model_call_end`. Raw
    deltas are still forwarded to `extra` so live streaming (CLI --verbose
    / TUI) is unaffected."""

    delta_buffer: list[str] = []

    def write_line(line: str) -> None:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def emit(event: ToolLoopEvent) -> None:
        if event.kind == "token_delta":
            # Buffer the chunk; don't write a per-fragment line. Still
            # forward so live consumers see the stream in real time.
            if event.delta:
                delta_buffer.append(event.delta)
            if extra is not None:
                with contextlib.suppress(Exception):
                    extra(event)
            return
        # Flush the coalesced stream as one summary line at the call
        # boundary, before the model_call_end line itself.
        if event.kind == "model_call_end" and delta_buffer:
            streamed = "".join(delta_buffer)
            delta_buffer.clear()
            ts = datetime.now(UTC).isoformat(timespec="seconds")
            preview = streamed.replace("\n", " ")[:200]
            write_line(
                f"turn {turn_index} | {ts} | model_output | "
                f"chars={len(streamed)} preview={preview!r}"
            )
        write_line(f"turn {turn_index} | {_format_executor_event(event)}")
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
    prior_made_no_edits: bool = False,
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
            prior_made_no_edits=prior_made_no_edits,
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
            executor_temperature=config.executor_temperature,
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


# harness-xxdr: name + default-path resolution for the vLLM trace.
# The adapter (`harness.model.vllm._vllm_trace`) reads this env var;
# the driver SETS it (when unset) to colocate traces alongside the
# .json/.log/.tar.gz loop-run artifacts. User-supplied values win.
_VLLM_TRACE_ENV = "HARNESS_VLLM_TRACE"


def _default_vllm_trace_path(workspace: Path, loop_run_id: str) -> Path:
    """`.harness/loop_runs/<id>.vllm_trace.jsonl` — sibling of the
    run state .json / .log / .tar.gz so a cleanup pass on the
    loop_runs directory sweeps the trace too."""
    return workspace / ".harness" / "loop_runs" / f"{loop_run_id}.vllm_trace.jsonl"


@contextlib.contextmanager
def ambient_vllm_trace(workspace: Path, loop_run_id: str) -> Iterator[Path | None]:
    """If HARNESS_VLLM_TRACE is unset, install a default path scoped to
    this run for the duration of the context. Restore the original
    environment on exit so chat / non-loop callers running in the same
    process aren't surprised by tracing they didn't ask for.

    Public so auto_iterate can reuse it to wrap the post-drive critic
    pass under the same per-run trace file (harness-5t0a)."""
    prior = os.environ.get(_VLLM_TRACE_ENV)
    if prior is not None:
        yield None
        return
    path = _default_vllm_trace_path(workspace, loop_run_id)
    os.environ[_VLLM_TRACE_ENV] = str(path)
    try:
        yield path
    finally:
        os.environ.pop(_VLLM_TRACE_ENV, None)


def _build_driver_hook_pipeline(
    *,
    adapter: ModelAdapter,
    registry: Any,
    workspace: Path,
    summarize_tool_results: bool,
    pre_close_verify: PreCloseVerifyHook | None = None,
    targeted_fix: bool = False,
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
    from harness.driver.bd_warning_filter import make_bd_auto_export_warning_filter
    from harness.orchestrator.hooks import ShellEchoNoopHook, ToolResultSummarizerHook
    from harness.orchestrator.parse_gate_escalation import (
        make_parse_gate_escalation_pair,
    )

    write_file_redirect_hook = make_write_file_redirect_hook(
        registry=registry,
        workspace_path=workspace,
        targeted_fix=targeted_fix,
    )
    pipeline = default_hook_pipeline(write_file_redirect_hook=write_file_redirect_hook)
    # harness-nlj7: pre-close verify gate. Runs FIRST in pre_tool so a
    # verify failure Skips the bd close before any downstream hook sees
    # it. The model gets a verify_blocked failure result instead of a
    # close-success ack, killing the wrap_up_forced narration spiral.
    if pre_close_verify is not None:
        pipeline.pre_tool.append(pre_close_verify)
    # harness-0v6d: parse-gate escalation pair. The observer counts
    # parse-gate failures per file in post_tool; once a file has
    # tripped twice this turn, the escalation hook intercepts the
    # NEXT edit_file / stream_edit / python_stream call against that
    # path and Skips it with a whole-file-rewrite nudge. Both share a
    # per-pipeline ParseGateState (resets between turns).
    gate_observer, gate_escalation = make_parse_gate_escalation_pair()
    pipeline.pre_tool.append(gate_escalation)
    # harness-jmkc: drive-only. Catch the model echoing "Would run: bd
    # close X" / "Issue closed…" instead of executing the close — a
    # narration no-op that left issues open and burned attempts.
    pipeline.pre_tool.append(ShellEchoNoopHook())
    pipeline.post_tool.append(gate_observer)
    # harness-rtwm: strip bd's "auto-export: git add failed" warning
    # from shell-tool output when the workspace is gitignored. Enabled
    # automatically based on a one-shot `git check-ignore`; tracked
    # workspaces get a hook that no-ops on every call.
    pipeline.post_tool.append(make_bd_auto_export_warning_filter(workspace))
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
    # PromptBudgetError is the harness's OWN pre-flight / 422-backstop
    # budget rejection (harness.model.adapter). It's an unambiguous typed
    # signal — match it by type, not by message. Its wording ("…window
    # remain for generation…") contains none of the string heuristics
    # below, so a grep that slurped a .harness trace into the prompt was
    # tearing the whole run down instead of failing one turn.
    if isinstance(exc, PromptBudgetError):
        return True
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
    executor_temperature: float = 0.5,
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
        # harness-8tjnv: the handoff already carries targeted_fix (set by
        # run_loop on retry / REGRESSION / built-artifact); reuse it so
        # the write_file hard-block lines up with the TARGETED-FIX banner.
        targeted_fix=handoff.targeted_fix,
    )
    try:
        result: ToolLoopResult = run_tool_loop(
            adapter,  # type: ignore[arg-type]  # narrower _ToolCapableAdapter, checked at runtime
            messages,
            registry,
            hooks=hooks,
            observe=observe,
            max_rounds=max_rounds,
            temperature=executor_temperature,
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


def _reconcile_failed_close(bd: DriverBd, issue_id: str, *, log: Callable[[str], None]) -> bool:
    """harness-r0s61: a failed turn must never leave a closed bead.

    The model closes issues by running ``bd close <id>`` via shell (the
    intended mechanism — DRIVE_CLOSE_INSTRUCTION; there is no structured
    ``bd_close`` tool yet). If that close ran but the turn then failed —
    e.g. the tool loop exhausted and emitted the fabrication-fallback
    sentinel *after* the close, or a claim-without-verifiable-work gate
    tripped — the driver records the turn as FAIL but the bead stays
    CLOSED in bd. The fabrication/convergence gate only governs the
    driver's accounting and the workspace (last-green restore); it never
    reverses a model-initiated close, so an unverified close leaks in
    (run b085854e, turn 9, harness-zbnq).

    Reconcile bd state with the driver's verdict: if the bead is closed
    on a failure path, reopen it so it returns to the ready set (or is
    parked while open) instead of silently counting as done. Returns
    True iff a reopen was performed. Pure best-effort — a bd lookup or
    reopen hiccup degrades to "leave as-is" and is logged."""
    try:
        issue = bd.show(issue_id)
    except DriverBdError:
        return False
    if issue.status != "closed":
        return False
    reopened = _try_reopen(bd, issue_id)
    note = "reopened" if reopened else "REOPEN_FAILED"
    log(f"{issue_id}: turn failed but bead was closed — {note} (unverified close)")
    return reopened


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
        "outline": OutlineTool(root=workspace),
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


def _parked_test_cmds(state: LoopRunState) -> dict[str, str]:
    """Gate test command for each parked issue (harness-s0el9). Only ids
    that reached a WRITE_TEST phase have a `last_test_cmd` entry; the rest
    map to nothing and the outer pass treats them as "no resolvable gate"."""
    return {
        pid: state.last_test_cmd[pid] for pid in state.parked_issues if pid in state.last_test_cmd
    }


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
        parked_test_cmds=_parked_test_cmds(state),
    )


def _exit_partial(
    state: LoopRunState,
    workspace: Path,
    log: _LogWriter,
    stranded: Sequence[str] = (),
) -> LoopResult:
    """The ready queue emptied only because parked issues were filtered
    out (harness-iljv). Some work may have closed, but the parked issues
    — and anything depending on them — are stranded pending operator
    pickup, so this is reported distinctly from "success".

    `stranded` (harness-q1uci) is the set of ready issues filtered out of
    THIS pass — `state.parked_issues` (parked this run) PLUS any carried
    `config.skip_issue_ids` parked by a PRIOR auto-iterate pass. The latter
    don't appear in `state.parked_issues` (fresh per pass), so logging only
    the parked count read "0 parked" while three issues were actually
    stranded — making an `exit=stuck` opaque. Log both so the culprits are
    named."""
    parked_tail = ", ".join(state.parked_issues) or "(none)"
    stranded_tail = ", ".join(stranded) or "(none)"
    log(
        f"loop_run={state.loop_run_id} PARTIAL "
        f"(ready queue drained; {len(state.parked_issues)} parked: {parked_tail}; "
        f"{len(stranded)} stranded: {stranded_tail})"
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
        parked_test_cmds=_parked_test_cmds(state),
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
        parked_test_cmds=_parked_test_cmds(state),
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
    try:
        bd.flag_human(current_id, reason=f"loop halted: {reason}")
    except DriverBdError as exc:
        log(f"loop_run={state.loop_run_id} WARN flag_human({current_id}) failed: {exc}")
    try:
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=current_id,
            status="halted",
            body=reason,
        )
    except DriverBdError as exc:
        log(f"loop_run={state.loop_run_id} WARN session_state({current_id}) failed: {exc}")
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=current_id,
        turns_used=state.turns_used,
        exit_reason="halted",
        parked_issues=list(state.parked_issues),
        parked_test_cmds=_parked_test_cmds(state),
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
        parked_test_cmds=_parked_test_cmds(state),
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
    try:
        bd.flag_human(current_id, reason=f"drive parked after max attempts: {reason}")
    except DriverBdError as exc:
        # Best-effort, but log the failure — a silent suppress here hid a
        # bd-CLI drift that left `bd human list` empty after every park.
        log(f"loop_run={state.loop_run_id} WARN flag_human({current_id}) failed: {exc}")
    try:
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=current_id,
            status="parked",
            body=reason,
        )
    except DriverBdError as exc:
        log(f"loop_run={state.loop_run_id} WARN session_state({current_id}) failed: {exc}")


def _on_success(bd: DriverBd, state: LoopRunState, current_id: str, log: _LogWriter) -> None:
    # harness-4k2p: dedupe at append time. An issue reopened mid-run
    # (operator note + status reset, or a resume after reopen) and then
    # re-closed would otherwise land twice in closed_this_run, inflating
    # the audit count. Membership check keeps it a set-like ordered list.
    if current_id not in state.closed_this_run:
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


def _last_green_if_present(last_green: Path | None, log: _LogWriter) -> Path | None:
    """Return ``last_green`` if its snapshot file is still on disk; else
    log once and return None.

    The snapshot lives under ``.harness/loop_runs/`` inside the workspace,
    so it can disappear out from under a live run — an operator cleaning
    the dir mid-run (the observed case), a tmpreaper, disk eviction. The
    regression-guard contract is that a missing last-green disables
    rollback for the remainder of the run; it never fails the run. Nulling
    it here keeps every downstream check (`workspace_changed`,
    `detect_regression`, restore, park) consistently in the
    rollback-disabled state instead of some guarding `.is_file()` and the
    two raw `workspace_changed` calls crashing on the gone tar."""
    if last_green is not None and not last_green.is_file():
        log(
            f"regression guard: last-green snapshot {last_green.name} vanished; "
            "rollback disabled for the remainder of this run"
        )
        return None
    return last_green


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


def _inter_attempt_restore(
    config: LoopConfig,
    state: LoopRunState,
    last_green: Path | None,
    *,
    issue_id: str,
    default_steps: Sequence[VerifyStep],
    log: _LogWriter,
) -> None:
    """harness-2qth: between attempts on the same issue, roll the
    workspace back to last-green so the next attempt starts clean
    instead of compounding edits onto a broken baseline.

    harness-iteip: restore ONLY when the failed attempt left the
    workspace broken (red verify) or regressed vs last-green (lost
    symbols / shrank). A failed attempt that left a healthy, green
    workspace — correct work the model never `bd close`d, or a
    non-claim wrap-up stub — keeps its edits so the next attempt builds
    on them. Wiping a green workspace forces a cold restart that invites
    the fabricate-from-scratch spiral (run 29f4a974 / gta 6182c539,
    where the score work was done on attempt 2, restored away, then
    fabricated on attempt 3 → park). The next handoff still carries the
    failure reason, so the model fixes forward instead of redoing.

    Mirrors `_post_park_housekeeping`'s restore gate. No-op when
    regression_guard is off or no green baseline exists yet."""
    if not config.regression_guard or last_green is None or not last_green.is_file():
        return
    red = not _baseline_is_green(default_steps, config.workspace)
    regressed = detect_regression(config.workspace, last_green) is not None
    if not (red or regressed):
        log(
            f"loop_run={state.loop_run_id} retry of {issue_id}: workspace green, "
            f"not regressed — keeping edits (no restore)"
        )
        return
    restored, removed = restore_workspace(config.workspace, last_green)
    why = "broken" if red else "regressed (lost code vs last-green)"
    log(
        f"loop_run={state.loop_run_id} RESTORED before retry of {issue_id}: "
        f"baseline {why}; last-green ({restored} files, removed {len(removed)} issue-added)"
    )
    # harness-75tto: the restore reverts to a last-green snapshot taken
    # BEFORE this issue's WRITE_TEST file existed, so it deletes that file
    # (one of the "removed issue-added" above). The carried test_cmd still
    # points at it — the next attempt's VERIFY would run a deleted test
    # forever (exit 2 "can't open file" -> verify-retry ceiling -> park;
    # loop_run=9a1e7970, harness-rxtpz). Drop the carried test_cmd when its
    # script was removed, so the next attempt re-enters WRITE_TEST and
    # re-authors a real gate against the restored baseline instead of
    # reusing a phantom.
    carried = state.last_test_cmd.get(issue_id)
    if carried is not None:
        from harness.driver.fsm_turn import _test_cmd_file_missing

        if _test_cmd_file_missing(carried, config.workspace):
            state.last_test_cmd.pop(issue_id, None)
            log(
                f"loop_run={state.loop_run_id} cleared carried test_cmd for {issue_id}: "
                f"gate script removed by restore — WRITE_TEST will re-author next attempt"
            )
    # harness-s0el9: a REGRESSED restore (not merely red) means the failed
    # attempt's net effect was to DELETE working code — the destructive-edit
    # pattern (sed range-delete on game.js) the 30B falls into instead of
    # adding. The restore undoes the damage, but the next attempt will repeat
    # it unless told. Append an explicit additive directive to the carried
    # failure reason so the next handoff's [PRIOR ATTEMPT FAILED] block steers
    # the model to build forward (loop_run=9a1e7970, harness-rxtpz turn 3).
    if regressed and not red:
        nudge = (
            "your previous edit REMOVED working code (regressed vs last-green — "
            "lost symbols), so it was rolled back. This bead asks you to ADD "
            "behavior: make additive edits, do NOT delete or range-delete "
            "existing functions/blocks."
        )
        prior = state.last_failure.get(issue_id, "")
        state.last_failure[issue_id] = f"{prior} | NOTE: {nudge}" if prior else nudge


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
