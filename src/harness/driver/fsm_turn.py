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
import re
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.character import Character
from harness.driver.bd import DriverBd, DriverBdError
from harness.driver.claim_detector import last_shell_cmd_in_messages
from harness.driver.handoff import Handoff
from harness.driver.planner import VerifyStep
from harness.driver.precommit_verify_hook import PreCloseVerifyHook
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
    premise_unmet,
    verify_failed,
    verify_passed,
)
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.orchestrator import ToolLoopEvent, ToolLoopResult, run_tool_loop
from harness.orchestrator.hooks import EXHAUSTED_FABRICATION_FALLBACK
from harness.orchestrator.no_write_streak import (
    WRITE_TOOL_NAMES,
    NoSubmitStreakDetector,
    NoTestSubmitStreakDetector,
    NoWriteStreakDetector,
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
from harness.tools.turn_phase_meta import (
    FlagBlockedTool,
    SkipTestPhaseTool,
    SubmitAssessmentTool,
    SubmitFailingTestTool,
    SubmitImplementationCompleteTool,
)

ExecutorObserver = Callable[[ToolLoopEvent], None]

# harness-hs50i: turn-level ceilings the per-phase budgets can't bypass.
# Run d45fd2f7 turn 3 ran 2h50m without terminating: every IMPLEMENT
# pass landed ≥1 successful write, so `implement_some_writes` routed to
# VERIFY, verify failed, and `verify->implement (retry)` looped — 114
# wrap_up_forced events, 1,179 rounds, one turn. The per-phase round
# budgets bound each tool loop but nothing bounded the IMPLEMENT↔VERIFY
# cycle itself.
#
# _MAX_VERIFY_RETRIES caps the verify_failed → IMPLEMENT transitions per
# turn. Three retries is the same per-issue patience as the loop's
# default --max-attempts; a model that hasn't gone green after three
# full IMPLEMENT passes inside one turn isn't converging — halt the turn
# and let the loop's attempt/park machinery decide what's next.
_MAX_VERIFY_RETRIES = 3
# _MAX_PHASE_EXECUTIONS bounds total phase runs per turn regardless of
# transition shape, so any future cycle in the table (or a guard bug)
# degrades to a halted turn instead of an unbounded one. The longest
# legitimate walk is 5 linear phases + _MAX_VERIFY_RETRIES extra
# IMPLEMENT+VERIFY pairs = 11; 16 leaves headroom without permitting a
# runaway.
_MAX_PHASE_EXECUTIONS = 16


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
      FSM's own trace — used by audit + progress log.
    - `last_shell_cmd`: the `cmd` argument from the most recent
      shell tool call across all FSM phases, or None. Used by the
      legacy claim-without-close gate to detect celebratory `echo`
      finalization gestures (harness-24pn)."""

    final_phase: TurnPhase
    succeeded: bool
    reason: str
    reply: str
    last_assessment: dict[str, Any] | None
    last_test_cmd: str | None
    phase_trace: list[tuple[TurnPhase, TurnPhase, str]]
    last_shell_cmd: str | None = None


# --- per-phase tool rosters ---------------------------------------


def _read_only_tools(workspace: Path) -> dict[str, Tool]:
    """Tools available in ASSESS phase + as the read-tier baseline
    for every other phase. No file mutations; no shell access
    (shell can have side effects via redirects)."""
    return {
        "read_file": ReadFileTool(root=workspace),
        "outline": OutlineTool(root=workspace),
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
    """edit_file, write_file, shell (+ stream_edit when available) — the
    file-mutating set available in IMPLEMENT. The IMPLEMENT registry
    layers these on top of `_read_only_tools`.

    stream_edit (harness-bw27 file-ops bench) is the robust multi-edit /
    in-place sed/awk path — it doesn't depend on the model reproducing an
    exact `old_string`, which is where edit_file fails on large files
    (loop_run=498a4d79 turn 5: old_string mismatch → edit_file_dedup_loop
    spun out the attempt). StreamEditTool resolves awk/sed/cut/tr at
    construction and raises if a verb is missing; guard it so a host
    without one of those binaries degrades to the edit_file path instead
    of breaking the whole IMPLEMENT registry."""
    tools: dict[str, Tool] = {
        "edit_file": EditFileTool(root=workspace),
        "write_file": WriteFileTool(root=workspace),
        "shell": ShellTool(cwd=workspace),
    }
    # A coreutils verb (awk/sed/cut/tr) missing from PATH makes
    # StreamEditTool raise at construction — skip the tool rather than
    # fail the whole IMPLEMENT registry; edit_file/write_file still work.
    with contextlib.suppress(ValueError):
        tools["stream_edit"] = StreamEditTool(root=workspace)
    return tools


def _build_phase_registry(
    phase: TurnPhase,
    workspace: Path,
    *,
    submit_assessment: SubmitAssessmentTool,
    skip_test_phase: SkipTestPhaseTool,
    submit_failing_test: SubmitFailingTestTool,
    submit_implementation_complete: SubmitImplementationCompleteTool,
    flag_blocked: FlagBlockedTool,
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
        # ASSESS is a pure orient/decide phase. flag_blocked is the
        # escape hatch when the bead's premise is unmet (the thing to
        # verify/fix doesn't exist) — checked before submit_assessment.
        phase_tools["submit_assessment"] = submit_assessment
        phase_tools["flag_blocked"] = flag_blocked
    elif phase == TurnPhase.WRITE_TEST:
        # Allow write_file (a new test file when none fits) + edit_file
        # (APPEND a case to an existing test file whose subject matches —
        # the drive otherwise litters the workspace with one
        # test_<issue>.py per bead) + shell (run the test, capture red
        # output). stream_edit stays withheld: bulk multi-file edits are
        # an IMPLEMENT-phase shape, not test authoring.
        phase_tools["write_file"] = _write_tier_tools(workspace)["write_file"]
        phase_tools["edit_file"] = _write_tier_tools(workspace)["edit_file"]
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

    # Phase-aware unavailable-tool messaging (loop_run=3a0f6368). The
    # catalog is seeded with every builtin so tool_search stays useful,
    # but that means a withheld tool gets the generic "call load_tool,
    # then retry" error while load_tool (builders={}) answers "restart
    # the session" — two contradictory dead ends the model ping-pongs
    # between until the round budget dies. Register a per-phase hint on
    # both surfaces so calls to a deliberately-withheld tool explain the
    # real recovery path: finish this phase via its meta-tool.
    hints = _phase_unavailable_hints(phase)
    for tool_name, hint in hints.items():
        registry.set_unavailable_hint(tool_name, hint)

    # tool_search / load_tool are bookkeeping; available everywhere
    # so the model can introspect the schema if needed. Cheap; no
    # meaningful security surface here.
    registry.register(ToolSearchTool(catalog=catalog, registry=registry))
    registry.register(
        LoadToolTool(catalog=catalog, registry=registry, builders={}, unavailable_hints=hints)
    )
    return registry


# Write-tier tool names a phase may withhold. shell is listed separately
# because WRITE_TEST / VERIFY / CLOSE keep it while dropping the editors.
_EDITOR_TOOL_NAMES: tuple[str, ...] = ("edit_file", "write_file", "stream_edit")


def _phase_unavailable_hints(phase: TurnPhase) -> dict[str, str]:
    """Per-phase messages for tools the phase deliberately withholds.

    Keyed by tool name; applied to both the registry's unknown-tool
    error and load_tool's catalog-hit-no-builder branch so the two
    surfaces agree. Only covers the write tier — the observed failure
    mode (loop_run=3a0f6368: six turns burned chasing edit_file in
    ASSESS/WRITE_TEST) is models reaching for editors before the FSM
    grants them."""
    no_load = "Do NOT call load_tool — tools cannot be activated mid-phase."
    if phase == TurnPhase.ASSESS:
        hint = (
            "File edits and shell are not available in ASSESS — it is a "
            "read-only phase. Editing happens later, in the IMPLEMENT "
            "phase. Finish ASSESS by calling `submit_assessment` (or "
            f"`flag_blocked` if the premise is unmet). {no_load}"
        )
        return dict.fromkeys((*_EDITOR_TOOL_NAMES, "shell"), hint)
    if phase == TurnPhase.WRITE_TEST:
        hint = (
            "`stream_edit` is not available in WRITE_TEST — it is for "
            "bulk source edits in IMPLEMENT. This phase authors tests: "
            "`write_file` for a new test file, or `edit_file` to APPEND a "
            "case to an existing test file whose subject matches. Editing "
            "source unlocks in IMPLEMENT, after `submit_failing_test` "
            f"(or `skip_test_phase`). {no_load}"
        )
        return dict.fromkeys(("stream_edit",), hint)
    if phase == TurnPhase.VERIFY:
        hint = (
            "File mutations are not available in VERIFY — the driver "
            "re-runs the verify steps automatically. Produce a brief "
            f"'verify run' reply; fixes happen on the next turn. {no_load}"
        )
        return dict.fromkeys(_EDITOR_TOOL_NAMES, hint)
    if phase == TurnPhase.CLOSE:
        hint = (
            "Only `shell` is available in CLOSE. Run `bd close "
            f"<issue-id>` to finish the turn. {no_load}"
        )
        return dict.fromkeys(_EDITOR_TOOL_NAMES, hint)
    # IMPLEMENT has the full write tier; nothing to hint.
    return {}


# --- per-phase user messages --------------------------------------


# Read-strategy steering (harness-nfzw). The symbol-aware read tools
# (outline + read_file symbol=) are registered in every read-capable
# phase, but the model defaults to offset/limit out of habit unless told
# otherwise. Appended to the phases that actually read code.
_READ_STRATEGY_HINT = (
    "\n\nWhen reading code, prefer `outline <path>` to map a file's "
    "functions/classes, then `read_file <path> symbol=<name>` "
    "(e.g. symbol='Foo.bar') to read a whole function or class. Avoid "
    "blind `offset`/`limit` line ranges on source files — they tend to "
    "return half a function."
)


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
        "no unit test (UI tweak, docs); justify in the approach field.\n\n"
        "WRONG-TARGET ESCAPE: if the spec is grounded against a DIFFERENT "
        "artifact / project / language than this workspace — e.g. it names "
        "a Python 'snake_game.py' with food/score but the workspace is a "
        "JavaScript driving game — set premise_mismatch=true on "
        "submit_assessment and explain the mismatch in `gap`. This parks "
        "the bead for an operator. Do NOT 'adapt' a mis-grounded spec to "
        "the workspace and close it — flag it. (Distinct from BLOCKED below, "
        "which is an absent upstream precondition in the RIGHT project.)\n\n"
        "BLOCKED ESCAPE: if this bead asks you to verify or fix something "
        "that DOES NOT EXIST in the workspace yet — e.g. 'verify the fire-"
        "key gating' when there is no fire handler at all because an "
        "upstream bead never built it — call `flag_blocked(missing, reason)` "
        "instead of submit_assessment. Name the concrete absent artifact "
        "(function / file / symbol / keycode). This parks the bead for an "
        "operator. Use it ONLY for a genuinely-absent precondition — if the "
        "thing exists and is merely hard to change, do the work.\n\n"
        "Do NOT modify files in this phase — you don't have edit tools." + _READ_STRATEGY_HINT
    ),
    TurnPhase.WRITE_TEST: (
        "You are in the WRITE_TEST phase. Write a failing test that "
        "proves the gap exists, run it, capture its red output, then "
        "call `submit_failing_test` with the test_path, the test_cmd "
        "that runs it, and the captured failure_output. The same "
        "test_cmd will be run again in VERIFY to prove the implementation "
        "makes the test pass.\n\n"
        "PREFER EXTENDING AN EXISTING TEST FILE. Before creating a new "
        "file, `glob` for existing tests (e.g. 'test_*.py', '*_test.py', "
        "'**/*.test.js') and check whether one already covers this "
        "subject area (same module/feature). If so, `edit_file` to APPEND "
        "your new case to it. Only `write_file` a NEW test file when no "
        "existing file is a sensible home — one test_<feature>.py per "
        "feature, NOT one per bead. Keep the new test minimal: a single "
        "focused case that proves THIS gap, not a broad suite.\n"
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
        "pass — that defeats the purpose of TDD." + _READ_STRATEGY_HINT
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


# Phase-exit capture tools allowed in the forced wrap-up round
# (loop_run=dfc38c5f). When a phase's round budget exhausts one call
# short of the exit signal — the dfc38c5f shape: red test written and
# run, budget gone before submit_failing_test — the wrap-up round may
# still emit the exit call instead of having it stripped. These are
# zero-side-effect capture tools; executing them late is always safe.
_PHASE_EXIT_TOOLS: Mapping[TurnPhase, frozenset[str]] = {
    TurnPhase.ASSESS: frozenset({"submit_assessment", "flag_blocked"}),
    TurnPhase.WRITE_TEST: frozenset({"submit_failing_test", "skip_test_phase"}),
    TurnPhase.IMPLEMENT: frozenset({"submit_implementation_complete"}),
}


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
    # harness-gmu9f: write-tier tools CALLED this phase regardless of
    # success — lets IMPLEMENT distinguish "tried to write, every edit was
    # a no-op/identical/dedup (change likely already present)" from "never
    # attempted a write" (pure read-thrash). The former routes to VERIFY;
    # only the latter halts no-progress.
    attempted_write_tools: set[str]


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
    pre_close_verify: PreCloseVerifyHook | None = None,
) -> _PhaseExecutionResult:
    """Drive one phase: assemble the system+user messages, install
    the default hook pipeline + WriteFileRedirectHook (+ the optional
    ToolResultSummarizerHook from harness-tu4o), call run_tool_loop.
    Returns the bare result; caller maps it to a PhaseOutcome.

    `pre_close_verify` (harness-nlj7), when supplied, gates ``bd close
    <current_issue>`` shell calls on workspace verify. A no-op for any
    phase that doesn't run shell — but kept in the pipeline regardless
    so the wiring stays uniform across phases."""
    from harness.driver.loop import _build_driver_hook_pipeline

    # harness-d6ak (smaller surgery): keep the character's identity +
    # values + directives so the model still has a generative anchor,
    # but drop the chat-shaped "How you speak" style rules ("prose by
    # default", "1-4 sentences", "say 'Don't know' plainly") and the
    # voice examples. Those bias the model toward narration; the
    # drive needs tool calls. Full d6ak revert (cdd2fd9) kept the
    # style rules because removing the entire prompt made the model
    # go silent — this narrower carve-out targets only the rules
    # that conflict with the agent role.
    base_prompt = character.system_prompt(include_samples=(), include_style_rules=False)
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
        pre_close_verify=pre_close_verify,
        # harness-8tjnv: carry targeted-fix through to the write_file
        # hard-block so the FSM path matches the legacy executor path.
        targeted_fix=handoff.targeted_fix,
    )
    succeeded_tools: set[str] = set()
    # harness-gmu9f: write-tier tools CALLED this phase regardless of
    # outcome — a no-op/identical/dedup-rejected edit still lands here.
    attempted_write_tools: set[str] = set()
    # Wrap observe to also sniff succeeded tool names — needed for
    # IMPLEMENT's "did any write succeed" outcome.
    original_observe = observe

    def relay(event: ToolLoopEvent) -> None:
        if event.kind == "tool_call_end" and event.call is not None and event.result is not None:
            if event.call.name in WRITE_TOOL_NAMES:
                attempted_write_tools.add(event.call.name)
            if event.result.success:
                succeeded_tools.add(event.call.name)
        if original_observe is not None:
            original_observe(event)

    # Arm a per-phase stall detector. IMPLEMENT (harness-41b3): no-write
    # streak — "looking instead of writing". ASSESS (loop_run=ad30d9ad
    # parked hewc/cw1m on read-without-submitting): no-submit streak —
    # "reading instead of deciding", nudging toward submit_assessment /
    # flag_blocked before the round budget halts the phase. WRITE_TEST
    # (loop_run=3a0f6368 turn 2: genuine red test written + run, then the
    # budget burned without submit_failing_test): no-test-submit streak —
    # "testing instead of submitting". Constructed fresh per phase
    # invocation; the detector holds per-turn state and must not leak
    # across phases. VERIFY / CLOSE get none.
    no_write_streak: (
        NoWriteStreakDetector | NoSubmitStreakDetector | NoTestSubmitStreakDetector | None
    ) = None
    if phase is TurnPhase.IMPLEMENT:
        no_write_streak = NoWriteStreakDetector()
    elif phase is TurnPhase.ASSESS:
        no_write_streak = NoSubmitStreakDetector()
    elif phase is TurnPhase.WRITE_TEST:
        no_write_streak = NoTestSubmitStreakDetector()
    result: ToolLoopResult = run_tool_loop(
        adapter,  # type: ignore[arg-type]  # ModelAdapter satisfies _ToolCapableAdapter at runtime
        messages,
        registry,
        hooks=hooks,
        observe=relay,
        max_rounds=max_rounds,
        no_write_streak=no_write_streak,
        # loop_run=dfc38c5f: let the forced wrap-up round still emit
        # this phase's exit-signal call instead of stripping it.
        wrap_up_tools=_PHASE_EXIT_TOOLS.get(phase, frozenset()),
    )
    return _PhaseExecutionResult(
        tool_loop_result=result,
        succeeded_tools=succeeded_tools,
        attempted_write_tools=attempted_write_tools,
    )


# --- per-phase outcome resolvers ----------------------------------


def _resolve_assess_outcome(
    submit_assessment: SubmitAssessmentTool,
    flag_blocked: FlagBlockedTool,
) -> PhaseOutcome:
    # flag_blocked wins over submit_assessment: a premise-unmet signal
    # short-circuits the normal ASSESS exit so the loop parks-and-flags
    # rather than driving IMPLEMENT against a precondition that's absent.
    blocked = flag_blocked.latest()
    if blocked is not None:
        return premise_unmet(missing=str(blocked["missing"]), reason=str(blocked["reason"]))
    latest = submit_assessment.latest()
    if latest is None:
        return phase_no_progress(TurnPhase.ASSESS, reason="no submit_assessment call")
    # harness-jbz4z: the spec is grounded against a DIFFERENT artifact /
    # project / language than this workspace (model self-reported via
    # premise_mismatch). Park-and-flag rather than driving — and eventually
    # CLOSING — an unrelated change against a mis-grounded bead (drive
    # gta_r2, harness-rorj: a snake-game score+food bead closed against a JS
    # driving game). Distinct from flag_blocked, which is an absent UPSTREAM
    # precondition; here the bead targets the wrong codebase entirely.
    if latest.get("premise_mismatch", False):
        return premise_unmet(
            missing="workspace does not match the issue spec target",
            reason=str(latest["gap"]),
        )
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


# harness-k6du2: shell words that run no test and assert nothing about the
# code under change. A verify gate built ENTIRELY from these is degenerate:
# its exit code is fixed (e.g. `echo ... && exit 1` is always red, `true` is
# always green) regardless of what the implementation does, so it can never
# go green by fixing code — every VERIFY retry burns against a tautology
# (drive gta_r2 loop_run=1c7e5c58, harness-491j5). A real gate has at least
# one segment that runs a test runner / interpreter / script / assertion,
# none of which appear here. `test` / `[` are deliberately EXCLUDED — they
# assert (file/string predicates) and are a legitimate, if weak, gate.
_NOOP_GATE_WORDS: frozenset[str] = frozenset(
    {"echo", "printf", "exit", "true", "false", ":", "cd", "pwd", "sleep", "export"}
)


def _is_degenerate_test_cmd(cmd: str) -> bool:
    """True when `cmd` invokes no test — every segment is a shell no-op
    (echo/exit/cd/...) so its exit code is decoupled from the code under
    change. Such a command can never transition red->green by editing
    code, so accepting it as the WRITE_TEST gate guarantees the
    verify-retry budget is spent on a tautology."""
    # Split on shell separators; a single real-test segment redeems the cmd.
    segments = re.split(r"&&|\|\||;|\|", cmd)
    saw_segment = False
    for segment in segments:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            # Unparseable segment — can't prove it's a no-op; treat as real.
            return False
        if not tokens:
            continue
        # Strip leading `VAR=val` env assignments to reach the command word.
        word = next((t for t in tokens if "=" not in t.split(" ", 1)[0]), tokens[0])
        saw_segment = True
        if word not in _NOOP_GATE_WORDS:
            return False
    # All segments were no-ops (and there was at least one) -> degenerate.
    return saw_segment


# harness-75tto: file extensions a runnable test script carries. Used to pull
# the script path out of a verify command so we can check it still exists.
_TEST_SCRIPT_EXTS: tuple[str, ...] = (".py", ".js", ".mjs", ".cjs", ".ts")
_TEST_RUNNER_WORDS: frozenset[str] = frozenset(
    {"python", "python3", "pytest", "node", "deno", "bun", "ruby", "py.test"}
)


def _test_cmd_script(cmd: str) -> str | None:
    """Best-effort extract the test-script path a verify command runs.

    Returns the first token that looks like a runnable test file (carries a
    known script extension), with any pytest `::node` selector stripped.
    Returns None when no script path is determinable — callers must treat
    None as "can't tell", never as "missing", so a parse miss can't drop a
    real gate. Handles the common shapes the driver sees: `cd <dir> && python
    test_x.py`, `pytest path/test_x.py::t`, `node game.test.js`."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    saw_runner = False
    for tok in tokens:
        base = tok.split("::", 1)[0]
        if tok in _TEST_RUNNER_WORDS or tok.endswith("/pytest"):
            saw_runner = True
        if base.endswith(_TEST_SCRIPT_EXTS):
            return base
    # A runner with no file arg (e.g. bare `pytest`) is determinable-but-no-path.
    _ = saw_runner
    return None


# harness-75tto: substrings in a test runner's output that mean it could not
# RUN the test at all (vs. ran it and saw a failing assertion). A non-zero
# exit for one of these reasons is decoupled from the code under change — the
# gate is red no matter what the implementation does — so it must not be
# accepted or reused as a WRITE_TEST gate, the same trap as the degenerate
# gate. Matched case-insensitively against the re-exec tail.
_UNRUNNABLE_TEST_SIGNATURES: tuple[str, ...] = (
    "can't open file",  # python <missing>.py -> exit 2
    "no such file or directory",
    "no module named",  # ModuleNotFoundError on the test module itself
    "modulenotfounderror",
    "importerror",
    "syntaxerror",  # the test file itself won't parse
    "file or directory not found",  # pytest collection
    "errors during collection",
    "command not found",  # wrong interpreter
)


def _is_unrunnable_test_output(exit_code: int, tail: str) -> bool:
    """True when a non-zero `exit_code` came from the runner failing to LOAD
    the test (missing file, bad import, syntax error, collection error) rather
    than from a genuine assertion failure (harness-75tto). Such a red can
    never go green by editing the code under test, so it's not a valid gate.
    Operates on the re-exec result so it shares the seam tests stub."""
    if exit_code == 0:
        return False
    low = tail.lower()
    return any(sig in low for sig in _UNRUNNABLE_TEST_SIGNATURES)


def _test_cmd_file_missing(cmd: str, workspace: Path) -> bool:
    """True iff `cmd` names a test script that does NOT exist in the
    workspace (harness-75tto). A command whose script is absent exits
    non-zero for a reason decoupled from the code (`python foo.py` on a
    missing foo.py -> exit 2, "can't open file"), so it is red forever and
    must not be accepted or reused as the gate — the sibling of the
    degenerate-gate trap. The drive failure: an inter-attempt regression
    restore deleted the WRITE_TEST file, but the carried test_cmd still
    pointed at it, so VERIFY ran a phantom test to the retry ceiling and
    parked (loop_run=9a1e7970, harness-rxtpz). Conservative: a script we
    can't parse out returns False (treated as present), never a false drop."""
    script = _test_cmd_script(cmd)
    if script is None:
        return False
    candidate = Path(script)
    resolved = candidate if candidate.is_absolute() else workspace / candidate
    return not resolved.exists()


def _resolve_write_test_outcome(
    submit_failing_test: SubmitFailingTestTool,
    skip_test_phase: SkipTestPhaseTool,
    *,
    workspace: Path,
    prior_test_cmd: str | None = None,
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
        # harness-axjt8: this attempt's WRITE_TEST captured neither a
        # submit nor a skip. Before halting "no test" (and burning the
        # attempt), reuse a test a PRIOR attempt already established —
        # carried forward as prior_test_cmd. A re-attempt shouldn't have
        # to re-discover a test it has; the model oscillates submit/skip/
        # nothing across attempts and any barren one would otherwise park
        # the bead (loop_run=783a4e30: b65f2 / y2gqc). Mirrors gmu9f's
        # IMPLEMENT re-attempt idempotence, opposite phase.
        if prior_test_cmd and not _is_degenerate_test_cmd(prior_test_cmd):
            exit_code, tail = _exec_test_cmd(prior_test_cmd, workspace)
            if exit_code == 0:
                # Carried test now passes — the implementation already
                # landed; short-circuit to CLOSE rather than re-driving.
                return green_test_outcome(
                    test_path="(carried from prior attempt)",
                    test_cmd=prior_test_cmd,
                )
            # harness-75tto: don't reuse a carried gate that can't even RUN
            # (its script was deleted by an inter-attempt restore, bad
            # import, etc.) — that red is a phantom, not a real failure.
            # Fall through to the 'no test' halt so WRITE_TEST re-authors.
            if not _is_unrunnable_test_output(exit_code, tail):
                return failing_test_submitted(
                    test_path="(carried from prior attempt)",
                    test_cmd=prior_test_cmd,
                    failure_output="reused failing test carried from a prior attempt (still red)",
                )
        return phase_no_progress(
            TurnPhase.WRITE_TEST,
            reason="no submit_failing_test or skip_test_phase call",
        )
    # harness-k6du2: reject a degenerate gate (echo/exit-only, runs no
    # test) BEFORE the red-check. It's red, but for a reason decoupled from
    # the code, so VERIFY would burn its whole retry budget against a
    # tautology. Halt WRITE_TEST instead — the next attempt can re-author a
    # real test.
    if _is_degenerate_test_cmd(str(latest["test_cmd"])):
        return phase_no_progress(
            TurnPhase.WRITE_TEST,
            reason=(
                f"degenerate verify gate: command runs no test (no-op/exit only): "
                f"{str(latest['test_cmd'])[:120]}"
            ),
        )
    # Sanity-check the test by re-executing — if it's already green,
    # the model claimed a failing test that isn't actually failing.
    # Re-route through green_test_outcome so the FSM short-circuits
    # to CLOSE rather than continuing into IMPLEMENT on a fake red.
    exit_code, tail = _exec_test_cmd(latest["test_cmd"], workspace)
    if exit_code == 0:
        return green_test_outcome(
            test_path=latest["test_path"],
            test_cmd=latest["test_cmd"],
        )
    # harness-75tto: a non-zero exit that means the runner couldn't LOAD the
    # test (missing script, bad import, syntax/collection error) is not a
    # real failing test — it's red regardless of the code, so VERIFY would
    # burn its retry budget on a phantom. Halt WRITE_TEST to re-author.
    if _is_unrunnable_test_output(exit_code, tail):
        return phase_no_progress(
            TurnPhase.WRITE_TEST,
            reason=(
                f"verify gate could not run the test (not a real failure): {tail.strip()[:120]}"
            ),
        )
    return failing_test_submitted(
        test_path=latest["test_path"],
        test_cmd=latest["test_cmd"],
        failure_output=latest["failure_output"],
    )


def _resolve_implement_outcome(
    submit_implementation_complete: SubmitImplementationCompleteTool,
    succeeded_tools: set[str],
    attempted_write_tools: set[str],
) -> PhaseOutcome:
    latest = submit_implementation_complete.latest()
    if latest is not None:
        return implement_complete(summary=str(latest["summary"]))
    # No explicit completion call. If ANY write tool succeeded this
    # phase, treat it as implicit progress and transition to VERIFY
    # (rather than halting outright) — verify will pick up the slack.
    # WRITE_TOOL_NAMES (incl. stream_edit) is shared with the no-write
    # streak detector so "what counts as a write" stays in lock-step.
    if WRITE_TOOL_NAMES & succeeded_tools:
        return PhaseOutcome(
            kind="implement_some_writes",
            detail="writes landed without explicit complete signal",
            payload={"succeeded_tools": sorted(succeeded_tools)},
        )
    # harness-gmu9f: no write SUCCEEDED, but writes were ATTEMPTED — every
    # edit was a no-op / identical / dedup-rejected. On a re-attempt of a
    # bead whose prior attempt already landed the change, that's exactly
    # what the model produces (it re-reads, finds its edit present, can
    # only emit idempotent no-ops). Halting "no writes" here is misleading
    # and burns the attempt without even running verify. Route to VERIFY
    # and let it arbitrate the artifact: if the change is present + the
    # bead's checks pass, it closes; if not, verify_failed retries. Only a
    # phase with ZERO write attempts (pure read-thrash) still halts
    # no-progress below.
    if WRITE_TOOL_NAMES & attempted_write_tools:
        return PhaseOutcome(
            kind="implement_writes_attempted",
            detail="writes attempted but none landed (no-op/identical/dedup); verify arbitrates",
            payload={"attempted_write_tools": sorted(attempted_write_tools)},
        )
    return phase_no_progress(
        TurnPhase.IMPLEMENT,
        reason="no submit_implementation_complete + no write attempts",
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
    pre_close_verify: PreCloseVerifyHook | None = None,
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
    flag_blocked = FlagBlockedTool()

    fsm = build_turn_fsm(initial=initial_phase)
    last_reply = ""
    # harness-24pn: track the most recent shell cmd across all phases so
    # FsmTurnResult can expose it to the legacy claim-without-close gate.
    # Updated after each phase by scanning that phase's tool_loop_result
    # messages in reverse.
    last_shell_cmd: str | None = None
    captured_assessment: dict[str, Any] | None = prior_assessment
    captured_test_cmd: str | None = prior_test_cmd
    # When ASSESS emits premise_unmet, stash its marked detail so the
    # HALTED reason carries the PREMISE_UNMET_REASON_PREFIX (the trace-
    # derived reason would only carry the transition name). The loop
    # keys off that prefix to park-and-flag instead of retrying.
    premise_unmet_reason: str | None = None
    # harness-hs50i counters — see _MAX_VERIFY_RETRIES /
    # _MAX_PHASE_EXECUTIONS above.
    verify_retries = 0
    phase_executions = 0

    while not fsm.is_terminal():
        phase = fsm.state
        phase_executions += 1
        if phase_executions > _MAX_PHASE_EXECUTIONS:
            fsm.force(
                TurnPhase.HALTED,
                reason=(
                    f"phase-execution ceiling: {_MAX_PHASE_EXECUTIONS} phase runs "
                    f"in one turn without reaching a terminal state (harness-hs50i)"
                ),
            )
            break
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
            flag_blocked=flag_blocked,
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
            pre_close_verify=pre_close_verify,
        )
        if execution.tool_loop_result.content:
            last_reply = execution.tool_loop_result.content
        # harness-24pn: scan this phase's messages for the most recent
        # shell call's cmd. Phases run in temporal order, so a later
        # phase's shell call overrides an earlier one. The driver's
        # claim-without-close gate inspects the final value to catch
        # celebratory `echo` finalization gestures.
        phase_shell = last_shell_cmd_in_messages(execution.tool_loop_result.messages)
        if phase_shell is not None:
            last_shell_cmd = phase_shell

        # Map per-phase results to PhaseOutcome.
        outcome = _resolve_phase_outcome(
            phase,
            submit_assessment=submit_assessment,
            skip_test_phase=skip_test_phase_tool,
            submit_failing_test=submit_failing_test,
            submit_implementation_complete=submit_implementation_complete,
            flag_blocked=flag_blocked,
            succeeded_tools=execution.succeeded_tools,
            attempted_write_tools=execution.attempted_write_tools,
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
        if outcome.kind == "premise_unmet":
            premise_unmet_reason = outcome.detail

        # harness-hs50i: bound the IMPLEMENT↔VERIFY cycle. Counted on
        # the outcome (pre-handle) so the cap halts BEFORE re-entering
        # IMPLEMENT for a pass the budget would just burn.
        if phase == TurnPhase.VERIFY and outcome.kind == "verify_failed":
            verify_retries += 1
            if verify_retries >= _MAX_VERIFY_RETRIES:
                fsm.force(
                    TurnPhase.HALTED,
                    reason=(
                        f"verify-retry ceiling: {_MAX_VERIFY_RETRIES} failed "
                        f"verify passes in one turn (harness-hs50i); "
                        f"last failure: {outcome.detail[:200]}"
                    ),
                )
                break

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
            last_shell_cmd=last_shell_cmd,
        )

    # HALTED — derive reason from the last trace step or the last
    # outcome kind. The trace always has at least one entry once we've
    # entered the loop; force() also writes to it.
    reason = _halt_reason_from_trace(fsm.trace) or "halted (no transitions taken)"
    if premise_unmet_reason is not None:
        # Premise-unmet halt: carry the marked reason verbatim so the loop
        # parks-and-flags rather than retrying (PREMISE_UNMET_REASON_PREFIX).
        reason = premise_unmet_reason
    elif last_reply.strip() == EXHAUSTED_FABRICATION_FALLBACK.strip():
        # Annotate, don't overwrite (loop_run=3a0f6368): the fallback
        # firing is a symptom of the phase stalling out, and replacing
        # the trace-derived reason ("write_test->halted (no test)") with
        # the catcher name erased the actual halt cause from
        # state.last_failure + the retry handoff for all six turns.
        reason = f"{reason}; fabrication_fallback fired"
    return FsmTurnResult(
        final_phase=TurnPhase.HALTED,
        succeeded=False,
        reason=reason,
        reply=last_reply,
        last_assessment=captured_assessment,
        last_test_cmd=captured_test_cmd,
        phase_trace=list(fsm.trace),
        last_shell_cmd=last_shell_cmd,
    )


def _resolve_phase_outcome(
    phase: TurnPhase,
    *,
    submit_assessment: SubmitAssessmentTool,
    skip_test_phase: SkipTestPhaseTool,
    submit_failing_test: SubmitFailingTestTool,
    submit_implementation_complete: SubmitImplementationCompleteTool,
    flag_blocked: FlagBlockedTool,
    succeeded_tools: set[str],
    attempted_write_tools: set[str],
    bd: DriverBd,
    issue_id: str,
    verify_steps: Sequence[VerifyStep],
    test_cmd: str | None,
    workspace: Path,
) -> PhaseOutcome:
    if phase == TurnPhase.ASSESS:
        return _resolve_assess_outcome(submit_assessment, flag_blocked)
    if phase == TurnPhase.WRITE_TEST:
        # harness-axjt8: test_cmd here is captured_test_cmd — a test
        # carried forward from a prior attempt (None on the first). The
        # resolver reuses it instead of halting "no test" on a barren
        # re-attempt.
        return _resolve_write_test_outcome(
            submit_failing_test, skip_test_phase, workspace=workspace, prior_test_cmd=test_cmd
        )
    if phase == TurnPhase.IMPLEMENT:
        return _resolve_implement_outcome(
            submit_implementation_complete, succeeded_tools, attempted_write_tools
        )
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
