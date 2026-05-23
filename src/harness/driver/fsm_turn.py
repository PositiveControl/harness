"""FSM-driven executor turn — harness-kbnl.

Drives one bd issue through the TurnPhase FSM. Each phase is a
bounded `run_tool_loop` call with its own tool roster, round budget,
user message, and exit condition. The FSM (built from
`turn_fsm.build_turn_fsm`) decides which phase comes next based on
PhaseOutcome events the driver constructs from each phase's
meta-tool captures.

Replaces the linear `_run_executor_turn` for runs with
`LoopConfig.use_fsm=True`. The legacy path stays in `loop.py` for
back-compat + as a fallback when the operator runs `--no-fsm`.

Per-phase recipe (same across phases):

  1. Build a phase-scoped tool registry (`_phase_registry`).
  2. Compose the per-phase user message (`_phase_user_message`).
  3. Run `run_tool_loop` with the phase's round budget.
  4. Inspect meta-tool captures + tool-loop outcome to construct
     a `PhaseOutcome`.
  5. Feed outcome to the FSM; loop until terminal.

The driver's verify gate (harness-xfh2) runs inside the VERIFY phase:
the FSM's `verify_passed` / `verify_failed` outcomes are derived from
running the registered verify steps + (when present) the
`prior_test_cmd` captured during WRITE_TEST.
"""

from __future__ import annotations

import contextlib
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.character import Character
from harness.driver.bd import DriverBd, DriverBdError
from harness.driver.handoff import Handoff
from harness.driver.planner import VerifyStep
from harness.driver.turn_fsm import (
    DEFAULT_PHASE_BUDGETS,
    PhaseOutcome,
    TurnPhase,
    assessment_skipped_tdd,
    assessment_submitted,
    build_turn_fsm,
    close_failed,
    close_succeeded,
    failing_test_submitted,
    green_test_outcome,
    implement_complete,
    phase_no_progress,
    verify_failed,
    verify_passed,
)
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.orchestrator import ToolLoopEvent, ToolLoopResult, run_tool_loop
from harness.orchestrator.hooks import EXHAUSTED_FABRICATION_FALLBACK
from harness.orchestrator.no_write_streak import NoWriteStreakDetector
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
    ReadFileTool,
    ShellTool,
    Tool,
    ToolCatalog,
    ToolRegistry,
    ToolSearchTool,
    WriteFileTool,
    seed_builtins_into,
)
from harness.tools.turn_phase_meta import (
    SkipTestPhaseTool,
    SubmitAssessmentTool,
    SubmitFailingTestTool,
    SubmitImplementationCompleteTool,
)

ExecutorObserver = Callable[[ToolLoopEvent], None]


@dataclass
class FsmTurnResult:
    """Outcome of one FSM-driven executor turn.

    - `final_phase`: terminal TurnPhase (DONE or HALTED).
    - `succeeded`: True iff `final_phase == DONE` AND the bd issue
      actually closed during CLOSE.
    - `reason`: short failure reason when not succeeded; empty
      string on success. Format mirrors the legacy turn's reason
      string so callers / handoffs / state files can carry it
      through unchanged.
    - `reply`: text of the last model reply across all phases.
      Used by audit + the legacy claim-detector path (which is
      now redundant under FSM but kept addressable for debugging).
    - `last_assessment`: the assessment payload from ASSESS phase,
      or None if ASSESS didn't complete. Threaded into LoopRunState
      so a halted-mid-FSM turn carries the work forward.
    - `last_test_cmd`: the test command from WRITE_TEST phase, or
      None when not on the TDD path. Carried forward similarly.
    - `phase_trace`: list of (from, to, name) tuples mirroring the
      FSM's own trace — used by audit + progress log."""

    final_phase: TurnPhase
    succeeded: bool
    reason: str
    reply: str
    last_assessment: dict[str, Any] | None
    last_test_cmd: str | None
    phase_trace: list[tuple[TurnPhase, TurnPhase, str]]


# --- per-phase tool rosters ---------------------------------------


def _read_only_tools(workspace: Path) -> dict[str, Tool]:
    """Tools available in ASSESS phase + as the read-tier baseline
    for every other phase. No file mutations; no shell access
    (shell can have side effects via redirects)."""
    return {
        "read_file": ReadFileTool(root=workspace),
        "list_dir": ListDirTool(root=workspace),
        "grep": GrepTool(root=workspace),
        "glob": GlobTool(root=workspace),
        "git_status": GitStatusTool(root=workspace),
        "git_diff": GitDiffTool(root=workspace),
        "git_log": GitLogTool(root=workspace),
        "fetch_url": FetchUrlTool(),
        "now": NowTool(),
        "date_math": DateMathTool(),
        "calc": CalcTool(),
    }


def _write_tier_tools(workspace: Path) -> dict[str, Tool]:
    """edit_file, write_file, shell — the file-mutating set
    available in IMPLEMENT. The IMPLEMENT registry layers these on
    top of `_read_only_tools`."""
    return {
        "edit_file": EditFileTool(root=workspace),
        "write_file": WriteFileTool(root=workspace),
        "shell": ShellTool(cwd=workspace),
    }


def _build_phase_registry(
    phase: TurnPhase,
    workspace: Path,
    *,
    submit_assessment: SubmitAssessmentTool,
    skip_test_phase: SkipTestPhaseTool,
    submit_failing_test: SubmitFailingTestTool,
    submit_implementation_complete: SubmitImplementationCompleteTool,
) -> ToolRegistry:
    """Construct the phase-scoped tool registry. Each phase has its
    own narrow roster — the model literally cannot call `edit_file`
    during ASSESS because the schema doesn't include it.

    Meta-tools are wired per-phase: `submit_assessment` lives in
    ASSESS, `submit_failing_test` / `skip_test_phase` in WRITE_TEST,
    `submit_implementation_complete` in IMPLEMENT. Each is the
    transition trigger for its phase.

    Meta-tools are passed in (not constructed here) so the caller
    can inspect `captured` after run_tool_loop returns — the meta
    tools are stateful by design (harness-kbnl)."""
    catalog = ToolCatalog()
    seed_builtins_into(catalog, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))
    registry = ToolRegistry(catalog=catalog)

    read_tools = _read_only_tools(workspace)
    phase_tools: dict[str, Tool] = dict(read_tools)

    if phase == TurnPhase.ASSESS:
        # Read-tier only + submit_assessment. No writes, no shell —
        # ASSESS is a pure orient/decide phase.
        phase_tools["submit_assessment"] = submit_assessment
    elif phase == TurnPhase.WRITE_TEST:
        # Allow write_file (for the test file) + shell (to run the
        # test and capture failure output). No edit_file — writing
        # a new test file is the expected shape, not editing an
        # existing one.
        phase_tools["write_file"] = _write_tier_tools(workspace)["write_file"]
        phase_tools["shell"] = _write_tier_tools(workspace)["shell"]
        phase_tools["submit_failing_test"] = submit_failing_test
        phase_tools["skip_test_phase"] = skip_test_phase
    elif phase == TurnPhase.IMPLEMENT:
        # Full coding profile — edit + write + shell.
        phase_tools.update(_write_tier_tools(workspace))
        phase_tools["submit_implementation_complete"] = submit_implementation_complete
    elif phase == TurnPhase.VERIFY:
        # Read-only + shell (to re-run the test command). No file
        # mutations — VERIFY validates, it doesn't modify.
        phase_tools["shell"] = _write_tier_tools(workspace)["shell"]
    elif phase == TurnPhase.CLOSE:
        # Shell only (so the model can `bd close <id>`). Read-only
        # tools dropped — CLOSE is a single-action phase.
        phase_tools = {"shell": _write_tier_tools(workspace)["shell"]}

    for tool in phase_tools.values():
        registry.register(tool)

    # tool_search / load_tool are bookkeeping; available everywhere
    # so the model can introspect the schema if needed. Cheap; no
    # meaningful security surface here.
    registry.register(ToolSearchTool(catalog=catalog, registry=registry))
    registry.register(LoadToolTool(catalog=catalog, registry=registry, builders={}))
    return registry


# --- per-phase user messages --------------------------------------


_PHASE_INSTRUCTIONS: Mapping[TurnPhase, str] = {
    TurnPhase.ASSESS: (
        "You are in the ASSESS phase.\n\n"
        "AUTHORITY: The [Current issue:] block in the handoff IS the source "
        "of truth for this turn. It already contains the spec_quote, "
        "acceptance criteria, design notes, and any REGRESSION markers — "
        "everything you need to plan this turn's work. The bd issue is "
        "the curated, per-turn slice; you do NOT need the upstream "
        "artifacts that produced it.\n\n"
        "DO NOT re-read upstream artifacts (plan-draft YAML, .artifacts/, "
        "spec source files). Their relevant content is already rendered "
        "in the handoff above. Re-reading them wastes turn budget without "
        "adding new information.\n\n"
        "DO read the workspace files the bd issue references — these are "
        "the actual artifacts you will modify or examine for this turn.\n\n"
        "Then call `submit_assessment` with three non-empty fields:\n"
        "  - current_state: what the artifact looks like NOW (concrete)\n"
        "  - gap: how that differs from the acceptance criteria\n"
        "  - approach: how you plan to close the gap\n"
        "Set tdd_applicable=false ONLY when the issue genuinely admits "
        "no unit test (UI tweak, docs); justify in the approach field. "
        "Do NOT modify files in this phase — you don't have edit tools."
    ),
    TurnPhase.WRITE_TEST: (
        "You are in the WRITE_TEST phase. Write a failing test that "
        "proves the gap exists, run it, capture its red output, then "
        "call `submit_failing_test` with the test_path, the test_cmd "
        "that runs it, and the captured failure_output. The same "
        "test_cmd will be run again in VERIFY to prove the implementation "
        "makes the test pass.\n"
        "If you genuinely cannot write a test for this issue (no test "
        "runner available, opaque side effect, etc.) call "
        "`skip_test_phase` with a concrete reason. Use sparingly."
    ),
    TurnPhase.IMPLEMENT: (
        "You are in the IMPLEMENT phase. Make the changes required by "
        "the assessment's approach. When done, call "
        "`submit_implementation_complete` with a short summary of "
        "the changes. The VERIFY phase that follows will re-run the "
        "test you wrote earlier (if any) plus any verify steps "
        "registered for this issue.\n"
        "If you wrote a test in WRITE_TEST, your implementation must "
        "make that exact test pass. Do NOT modify the test to make it "
        "pass — that defeats the purpose of TDD."
    ),
    TurnPhase.VERIFY: (
        "You are in the VERIFY phase. The driver will execute the "
        "registered verify steps + the test command from WRITE_TEST "
        "automatically. You don't need to do anything — just produce "
        "a brief 'verify run' reply and the driver will pick up the "
        "outcome."
    ),
    TurnPhase.CLOSE: (
        "You are in the CLOSE phase. Run `bd close <issue-id>` via "
        "shell to close the bd issue. The verify steps already passed "
        "in the previous phase — this is the final action of the turn."
    ),
}


def phase_instructions(phase: TurnPhase) -> str:
    """Operator-facing per-phase user message. Public so the driver
    can embed it in the handoff render + the per-phase user-role
    ChatMessage."""
    return _PHASE_INSTRUCTIONS.get(phase, "")


_PHASE_USER_PROMPTS: Mapping[TurnPhase, str] = {
    TurnPhase.ASSESS: (
        "Read the file(s) referenced in the session handoff, then "
        "call submit_assessment with current_state, gap, and approach."
    ),
    TurnPhase.WRITE_TEST: (
        "Write a test file that proves the gap exists. Run the test "
        "and confirm it fails. Then call submit_failing_test with "
        "the path, command, and failure output. If TDD doesn't apply "
        "here, call skip_test_phase with a concrete reason."
    ),
    TurnPhase.IMPLEMENT: (
        "Make the changes described in your earlier assessment's "
        "approach. When the work is done, call "
        "submit_implementation_complete with a short change summary."
    ),
    TurnPhase.VERIFY: (
        "Acknowledge the verify run. The driver will execute the "
        "verify steps + WRITE_TEST command. One short reply is enough."
    ),
    TurnPhase.CLOSE: ("Close the bd issue: `bd close <issue-id>` via shell."),
}


# --- one phase ----------------------------------------------------


@dataclass(frozen=True)
class _PhaseExecutionResult:
    """Internal: what one phase's run_tool_loop produced."""

    tool_loop_result: ToolLoopResult
    succeeded_tools: set[str]


def _run_one_phase(
    *,
    adapter: ModelAdapter,
    character: Character,
    handoff: Handoff,
    workspace: Path,
    phase: TurnPhase,
    registry: ToolRegistry,
    user_prompt: str,
    max_rounds: int,
    observe: ExecutorObserver | None,
    summarize_tool_results: bool = True,
) -> _PhaseExecutionResult:
    """Drive one phase: assemble the system+user messages, install
    the default hook pipeline + WriteFileRedirectHook (+ the optional
    ToolResultSummarizerHook from harness-tu4o), call run_tool_loop.
    Returns the bare result; caller maps it to a PhaseOutcome."""
    from harness.driver.loop import _build_driver_hook_pipeline

    base_prompt = character.system_prompt(include_samples=())
    system_prompt = f"{base_prompt}\n\n{handoff.render()}\n\n{phase_instructions(phase)}"
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    hooks = _build_driver_hook_pipeline(
        adapter=adapter,
        registry=registry,
        workspace=workspace,
        summarize_tool_results=summarize_tool_results,
    )
    succeeded_tools: set[str] = set()
    # Wrap observe to also sniff succeeded tool names — needed for
    # IMPLEMENT's "did any write succeed" outcome.
    original_observe = observe

    def relay(event: ToolLoopEvent) -> None:
        if (
            event.kind == "tool_call_end"
            and event.call is not None
            and event.result is not None
            and event.result.success
        ):
            succeeded_tools.add(event.call.name)
        if original_observe is not None:
            original_observe(event)

    # harness-41b3: arm the no-write-streak detector only for IMPLEMENT.
    # Other phases (ASSESS / WRITE_TEST / VERIFY / CLOSE) legitimately
    # do read-only work; nudging them toward edit_file would be wrong.
    # Constructed fresh per phase invocation — the detector holds
    # per-turn state and must not leak across phases.
    no_write_streak = NoWriteStreakDetector() if phase is TurnPhase.IMPLEMENT else None
    result: ToolLoopResult = run_tool_loop(
        adapter,  # type: ignore[arg-type]  # ModelAdapter satisfies _ToolCapableAdapter at runtime
        messages,
        registry,
        hooks=hooks,
        observe=relay,
        max_rounds=max_rounds,
        no_write_streak=no_write_streak,
    )
    return _PhaseExecutionResult(tool_loop_result=result, succeeded_tools=succeeded_tools)


# --- per-phase outcome resolvers ----------------------------------


def _resolve_assess_outcome(
    submit_assessment: SubmitAssessmentTool,
) -> PhaseOutcome:
    latest = submit_assessment.latest()
    if latest is None:
        return phase_no_progress(TurnPhase.ASSESS, reason="no submit_assessment call")
    if latest.get("tdd_applicable", True):
        return assessment_submitted(
            current_state=str(latest["current_state"]),
            gap=str(latest["gap"]),
            approach=str(latest["approach"]),
        )
    return assessment_skipped_tdd(
        current_state=str(latest["current_state"]),
        gap=str(latest["gap"]),
        approach=str(latest["approach"]),
        reason=str(latest.get("approach", "tdd not applicable")),
    )


def _resolve_write_test_outcome(
    submit_failing_test: SubmitFailingTestTool,
    skip_test_phase: SkipTestPhaseTool,
    *,
    workspace: Path,
) -> PhaseOutcome:
    skip = skip_test_phase.latest()
    if skip is not None:
        return PhaseOutcome(
            kind="test_phase_skipped",
            detail=f"skipped: {skip['reason'][:120]}",
            payload={"reason": skip["reason"]},
        )
    latest = submit_failing_test.latest()
    if latest is None:
        return phase_no_progress(
            TurnPhase.WRITE_TEST,
            reason="no submit_failing_test or skip_test_phase call",
        )
    # Sanity-check the test by re-executing — if it's already green,
    # the model claimed a failing test that isn't actually failing.
    # Re-route through green_test_outcome so the FSM short-circuits
    # to CLOSE rather than continuing into IMPLEMENT on a fake red.
    exit_code, _tail = _exec_test_cmd(latest["test_cmd"], workspace)
    if exit_code == 0:
        return green_test_outcome(
            test_path=latest["test_path"],
            test_cmd=latest["test_cmd"],
        )
    return failing_test_submitted(
        test_path=latest["test_path"],
        test_cmd=latest["test_cmd"],
        failure_output=latest["failure_output"],
    )


def _resolve_implement_outcome(
    submit_implementation_complete: SubmitImplementationCompleteTool,
    succeeded_tools: set[str],
) -> PhaseOutcome:
    latest = submit_implementation_complete.latest()
    if latest is not None:
        return implement_complete(summary=str(latest["summary"]))
    # No explicit completion call. If ANY write tool succeeded this
    # phase, treat it as implicit progress and transition to VERIFY
    # (rather than halting outright) — verify will pick up the slack.
    if {"edit_file", "write_file"} & succeeded_tools:
        return PhaseOutcome(
            kind="implement_some_writes",
            detail="writes landed without explicit complete signal",
            payload={"succeeded_tools": sorted(succeeded_tools)},
        )
    return phase_no_progress(
        TurnPhase.IMPLEMENT,
        reason="no submit_implementation_complete + no successful writes",
    )


def _resolve_verify_outcome(
    *,
    verify_steps: Sequence[VerifyStep],
    test_cmd: str | None,
    workspace: Path,
) -> PhaseOutcome:
    """Run registered verify steps + (when set) the test command from
    WRITE_TEST. First non-zero exit short-circuits to verify_failed
    with the captured tail. All-pass returns verify_passed."""
    # Run the captured failing test first — it's the most specific
    # signal for the TDD path.
    if test_cmd:
        exit_code, tail = _exec_test_cmd(test_cmd, workspace)
        if exit_code != 0:
            return verify_failed(failure_tail=f"test {test_cmd!r} exit={exit_code}: {tail}")
    for step in verify_steps:
        exit_code, tail = _exec_test_cmd(step.cmd, workspace, shell_mode=step.shell)
        if exit_code != 0:
            preview = step.cmd if len(step.cmd) <= 80 else step.cmd[:77] + "..."
            return verify_failed(failure_tail=f"{preview} exit={exit_code}: {tail}")
    return verify_passed()


def _resolve_close_outcome(
    bd: DriverBd,
    issue_id: str,
) -> PhaseOutcome:
    """Inspect bd to confirm the issue actually closed during the
    CLOSE phase. Doesn't run `bd close` itself — that's the model's
    job via shell. We just check post-condition: status == closed."""
    try:
        issue = bd.show(issue_id)
    except DriverBdError as exc:
        return close_failed(reason=f"bd.show failed: {exc}")
    if issue.status == "closed":
        return close_succeeded()
    return close_failed(reason=f"bd issue still {issue.status} after CLOSE phase")


# --- per-step subprocess seam --------------------------------------


_VERIFY_TIMEOUT_SECONDS: int = 60
_VERIFY_TAIL_CHARS: int = 200


def _exec_test_cmd(cmd: str, workspace: Path, *, shell_mode: bool = True) -> tuple[int, str]:
    """Re-execute a single test command. Mirrors `_exec_verify_cmd`
    in `loop.py` (harness-xfh2) — same timeout, same tail length,
    same failure shape. Kept separate so the FSM module doesn't
    import private helpers from the legacy executor module."""
    args: str | list[str]
    if shell_mode:
        args = cmd
    else:
        try:
            args = shlex.split(cmd)
        except ValueError as exc:
            return 1, f"unparseable cmd: {exc}"
    try:
        result = subprocess.run(  # noqa: S603 — cmd from trusted local source
            args,
            shell=shell_mode,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_VERIFY_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        return 1, f"executable not found: {exc}"
    except subprocess.TimeoutExpired:
        return 1, f"timeout after {_VERIFY_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return 1, f"exec failed: {exc}"
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    tail_src = stderr or stdout
    return result.returncode, tail_src[-_VERIFY_TAIL_CHARS:]


# --- public entry point -------------------------------------------


def run_fsm_turn(
    *,
    adapter: ModelAdapter,
    character: Character,
    bd: DriverBd,
    handoff_builder: Callable[[TurnPhase, dict[str, Any] | None, str | None], Handoff],
    workspace: Path,
    current_issue_id: str,
    initial_phase: TurnPhase = TurnPhase.ASSESS,
    prior_assessment: dict[str, Any] | None = None,
    prior_test_cmd: str | None = None,
    verify_steps: Sequence[VerifyStep] = (),
    phase_budgets: Mapping[TurnPhase, int] | None = None,
    tdd_required: bool = True,
    observe: ExecutorObserver | None = None,
    summarize_tool_results: bool = True,
) -> FsmTurnResult:
    """Drive `current_issue_id` through the TurnPhase FSM.

    `handoff_builder(phase, assessment, test_cmd)` produces a fresh
    Handoff for each phase — caller-owned because the handoff
    composition (bd queries, git diff, thoughts) is the driver's
    contract, not this module's.

    `tdd_required=False` (e.g. CLI --no-tdd) is handled by
    inspecting the resolved ASSESS outcome and rerouting it through
    `assessment_skipped_tdd` BEFORE feeding to the FSM. The
    underlying transition table is unchanged.

    Returns FsmTurnResult with the terminal phase + success boolean +
    threaded assessment/test_cmd payload for resume.
    """
    budgets = dict(DEFAULT_PHASE_BUDGETS) if phase_budgets is None else dict(phase_budgets)

    submit_assessment = SubmitAssessmentTool()
    skip_test_phase_tool = SkipTestPhaseTool()
    submit_failing_test = SubmitFailingTestTool()
    submit_implementation_complete = SubmitImplementationCompleteTool()

    fsm = build_turn_fsm(initial=initial_phase)
    last_reply = ""
    captured_assessment: dict[str, Any] | None = prior_assessment
    captured_test_cmd: str | None = prior_test_cmd

    while not fsm.is_terminal():
        phase = fsm.state
        # Build phase-scoped registry. Meta-tool dataclasses persist
        # across phases (they accumulate captures) but only the
        # phase-relevant ones are registered — the model can't call
        # submit_assessment in IMPLEMENT, etc.
        registry = _build_phase_registry(
            phase,
            workspace,
            submit_assessment=submit_assessment,
            skip_test_phase=skip_test_phase_tool,
            submit_failing_test=submit_failing_test,
            submit_implementation_complete=submit_implementation_complete,
        )
        handoff = handoff_builder(phase, captured_assessment, captured_test_cmd)
        user_prompt = _PHASE_USER_PROMPTS.get(phase, "Proceed with this phase.")
        max_rounds = budgets.get(phase, 4)

        execution = _run_one_phase(
            adapter=adapter,
            character=character,
            handoff=handoff,
            workspace=workspace,
            phase=phase,
            registry=registry,
            user_prompt=user_prompt,
            max_rounds=max_rounds,
            observe=observe,
            summarize_tool_results=summarize_tool_results,
        )
        if execution.tool_loop_result.content:
            last_reply = execution.tool_loop_result.content

        # Map per-phase results to PhaseOutcome.
        outcome = _resolve_phase_outcome(
            phase,
            submit_assessment=submit_assessment,
            skip_test_phase=skip_test_phase_tool,
            submit_failing_test=submit_failing_test,
            submit_implementation_complete=submit_implementation_complete,
            succeeded_tools=execution.succeeded_tools,
            bd=bd,
            issue_id=current_issue_id,
            verify_steps=verify_steps,
            test_cmd=captured_test_cmd,
            workspace=workspace,
        )

        # Apply tdd_required override: if operator said --no-tdd and
        # the assessment came back tdd_applicable=True, reroute to the
        # skip path. Keeps the FSM's transition table simple.
        if (
            phase == TurnPhase.ASSESS
            and not tdd_required
            and outcome.kind == "assessment_submitted"
        ):
            outcome = assessment_skipped_tdd(
                current_state=str(outcome.payload.get("current_state", "")),
                gap=str(outcome.payload.get("gap", "")),
                approach=str(outcome.payload.get("approach", "")),
                reason="--no-tdd flag set on this loop run",
            )

        # Capture the assessment + test_cmd for the next phase's
        # handoff render + the run's state persistence.
        if outcome.kind in {"assessment_submitted", "assessment_skipped"}:
            captured_assessment = dict(outcome.payload)
        if outcome.kind == "failing_test_submitted":
            captured_test_cmd = str(outcome.payload.get("test_cmd", ""))

        try:
            _new_state, _name = fsm.handle(outcome)
        except Exception as exc:  # NoMatchingTransitionError + any guard error
            # The FSM rejected the outcome — halt with a descriptive reason.
            # In practice this happens when an outcome.kind isn't in the
            # transition table for the current phase (e.g. a mid-CLOSE
            # verify_failed event). The reason carries the offending kind.
            fsm.force(TurnPhase.HALTED, reason=f"unhandled outcome: {outcome.kind}")
            with contextlib.suppress(Exception):
                # Surface the exception detail too if available.
                last_reply = last_reply or f"FSM halt: {exc}"
            break

    # Build the final result. DONE means success; HALTED means failure.
    if fsm.state == TurnPhase.DONE:
        return FsmTurnResult(
            final_phase=TurnPhase.DONE,
            succeeded=True,
            reason="",
            reply=last_reply,
            last_assessment=captured_assessment,
            last_test_cmd=captured_test_cmd,
            phase_trace=list(fsm.trace),
        )

    # HALTED — derive reason from the last trace step or the last
    # outcome kind. The trace always has at least one entry once we've
    # entered the loop; force() also writes to it.
    reason = _halt_reason_from_trace(fsm.trace) or "halted (no transitions taken)"
    if last_reply.strip() == EXHAUSTED_FABRICATION_FALLBACK.strip():
        reason = f"fabrication_fallback fired during {fsm.state.value}"
    return FsmTurnResult(
        final_phase=TurnPhase.HALTED,
        succeeded=False,
        reason=reason,
        reply=last_reply,
        last_assessment=captured_assessment,
        last_test_cmd=captured_test_cmd,
        phase_trace=list(fsm.trace),
    )


def _resolve_phase_outcome(
    phase: TurnPhase,
    *,
    submit_assessment: SubmitAssessmentTool,
    skip_test_phase: SkipTestPhaseTool,
    submit_failing_test: SubmitFailingTestTool,
    submit_implementation_complete: SubmitImplementationCompleteTool,
    succeeded_tools: set[str],
    bd: DriverBd,
    issue_id: str,
    verify_steps: Sequence[VerifyStep],
    test_cmd: str | None,
    workspace: Path,
) -> PhaseOutcome:
    if phase == TurnPhase.ASSESS:
        return _resolve_assess_outcome(submit_assessment)
    if phase == TurnPhase.WRITE_TEST:
        return _resolve_write_test_outcome(
            submit_failing_test, skip_test_phase, workspace=workspace
        )
    if phase == TurnPhase.IMPLEMENT:
        return _resolve_implement_outcome(submit_implementation_complete, succeeded_tools)
    if phase == TurnPhase.VERIFY:
        return _resolve_verify_outcome(
            verify_steps=verify_steps, test_cmd=test_cmd, workspace=workspace
        )
    if phase == TurnPhase.CLOSE:
        return _resolve_close_outcome(bd, issue_id)
    # Terminal phases shouldn't reach here.
    return phase_no_progress(phase, reason="reached resolver in terminal phase")


def _halt_reason_from_trace(trace: list[tuple[TurnPhase, TurnPhase, str]]) -> str:
    """Last transition's name, or the forced halt reason if the
    final entry was a force()."""
    if not trace:
        return ""
    _, _, name = trace[-1]
    return name


__all__ = [
    "FsmTurnResult",
    "phase_instructions",
    "run_fsm_turn",
]
