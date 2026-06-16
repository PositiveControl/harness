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
from harness.driver.gate_blind import (
    GATE_BLIND_IDIOM_NOTE,
    eval_blind_reference,
    eval_blind_typeof_guard,
)
from harness.driver.handoff import Handoff
from harness.driver.planner import VerifyStep
from harness.driver.precommit_verify_hook import PreCloseVerifyHook
from harness.driver.premise_guard import (
    flag_blocked_names_own_deliverable,
    flag_blocked_names_withheld_tool,
)
from harness.driver.turn_fsm import (
    DEFAULT_PHASE_BUDGETS,
    PhaseOutcome,
    TurnPhase,
    assessment_already_satisfied,
    assessment_satisfied_pending_verify,
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
from harness.driver.workspace_verify import workspace_has_browser_js
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
      finalization gestures (harness-24pn).
    - `gate_suspect` (loop_run=dae002aa): True when the turn halted
      because the WRITE_TEST gate failed byte-identically across
      IMPLEMENT passes that touched the source — the test never
      observes the code under change. The loop must DROP the carried
      test_cmd for this issue (state.last_test_cmd) so the next
      attempt re-authors a real gate instead of reusing the broken
      one; `last_test_cmd` is already None on these results.
    - `last_test_fail_tail` (harness-smplj): the most recent VERIFY
      test-step failure tail this turn, or None. The loop persists it
      per issue (state.last_test_fail_tail) so the NEXT attempt seeds its
      detector with it — a byte-identical failure across attempts trips
      gate-suspect even when each turn halts at the hs50i ceiling before
      seeing two identical tails in one turn. None on success / suspect
      (the gate is being dropped anyway)."""

    final_phase: TurnPhase
    succeeded: bool
    reason: str
    reply: str
    last_assessment: dict[str, Any] | None
    last_test_cmd: str | None
    phase_trace: list[tuple[TurnPhase, TurnPhase, str]]
    last_shell_cmd: str | None = None
    gate_suspect: bool = False
    last_test_fail_tail: str | None = None


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
        "ALREADY-SATISFIED ESCAPE: if the workspace ALREADY fully meets this "
        "bead's acceptance and NO change is needed — e.g. the deliverable "
        "landed during earlier work — set already_satisfied=true on "
        "submit_assessment. Structural (scaffold/skeleton) beads route "
        "straight to CLOSE; behavioral beads route to VERIFY, where the "
        "carried test + verify steps arbitrate your claim (green closes, red "
        "sends you to IMPLEMENT). Do NOT write a new 'failing' test for a "
        "behavior that already works — say already_satisfied instead. Do NOT "
        "use the flag to dodge work that genuinely remains, and do NOT "
        "re-create an existing scaffold from scratch — that destroys code "
        "other beads added.\n\n"
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
        "pass — that defeats the purpose of TDD.\n\n"
        "GO STRAIGHT TO THE EDIT. Your assessment above (current_state / "
        "gap / approach) ALREADY located the change and named the plan — "
        "that orientation is done. Do NOT re-run outline / grep / read_file "
        "to re-survey files you already assessed this turn: it burns the "
        "IMPLEMENT round budget without new information, and the phase halts "
        "'no writes' with the work unstarted (loop_run=6308eb21: harness-vsv "
        "re-read + grepped game.js 8x in IMPLEMENT and parked having edited "
        "nothing). Read AT MOST one file once to confirm an exact insertion "
        "point, then call edit_file / stream_edit. The deliverable of this "
        "phase is an EDIT, not a finding." + _READ_STRATEGY_HINT
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


# loop_run=dae002aa: appended to the WRITE_TEST user prompt when the
# workspace is browser-authored JS. Every gate the model wrote that run
# was always-red for one of two reasons this hint preempts: (a) loading
# the source in bare Node crashes on `document` before any assertion
# runs; (b) avoiding (a) by never loading the source at all, asserting
# on mock copies the implementation can't reach. The stub recipe is the
# minimal one that lets sloppy-mode `eval` hoist the source's function
# declarations into the test scope.
_BROWSER_TEST_HINT = (
    "\n\nWORKSPACE NOTE — browser JS: the source uses browser globals "
    "(document / canvas / window) at top level, so a bare require()/eval in "
    "Node crashes with 'document is not defined' BEFORE any assertion runs. "
    "That red is a load failure, not a failing test, and will be rejected. "
    "Stub the globals FIRST, then load the REAL source file, then assert:\n"
    "  const fs = require('fs');\n"
    "  const stub = new Proxy(function () {}, "
    "{ get: () => stub, apply: () => stub });\n"
    "  global.document = { getElementById: () => stub, "
    "addEventListener: () => {} };\n"
    "  global.window = { addEventListener: () => {} };\n"
    "  global.requestAnimationFrame = () => {};\n"
    "  eval(fs.readFileSync('game.js', 'utf8')); "
    "// function declarations land in this scope\n"
    "SCOPING PITFALL (harness-815wm): only FUNCTION declarations (and "
    "`var`) escape a direct eval — top-level `let`/`const` bindings do "
    "NOT. `eval(src); traffic.length` throws 'traffic is not defined' "
    "even when the source declares `let traffic = []`. To assert on "
    "let/const state, append your assertions INTO the eval'd string so "
    "they run in the same scope:\n"
    "  const src = fs.readFileSync('game.js', 'utf8');\n"
    '  eval(src + "\\n;if (!Array.isArray(traffic) || traffic.length '
    "!== 6) { console.error('FAIL: expected 6 cars'); process.exit(1); "
    '}");\n'
    "or assert on the source text itself (readFileSync + regex) for "
    "presence-shaped checks.\n"
    "RUNTIME STATE IS OFTEN EMPTY HEADLESS (harness-vsv, loop_run=6308eb21): "
    "arrays like `traffic` / `peds` and objects like `foot` are populated by "
    "an init that runs on a canvas-resize / animation-frame / DOM event that "
    "does NOT fire under a bare Node eval — so `traffic.length` is 0 and "
    "per-element property checks find nothing even when the source is correct. "
    "For a behavioral / physics requirement (cruise accel, walking speed, "
    "collision rule), the RELIABLE gate asserts on the SOURCE TEXT — read the "
    "file and regex for the required code shape:\n"
    "  const src = fs.readFileSync('game.js', 'utf8');\n"
    "  if (!/speed\\s*\\+=\\s*80\\s*\\*\\s*dt/.test(src)) { "
    "console.error('FAIL: no 80 px/s^2 cruise accel'); process.exit(1); }\n"
    "Use the eval-runtime path ONLY for state the source initializes "
    "SYNCHRONOUSLY at load. When neither a runtime nor a source-text gate "
    "genuinely fits, call skip_test_phase with that reason — VERIFY's "
    "source-presence + runtime smoke gates still arbitrate the close.\n"
    "Do NOT re-define mock copies of the functions under test in the test "
    "file — the test must observe the real source, and must process.exit(1) "
    "when the gap is present."
)


# harness-vsv close-target fix (loop_run=adfc7bd6): the CLOSE phase prompt
# carries a generic `bd close <issue-id>` template — it never names the actual
# id. The model inferred the id from the handoff's current-issue line
# (`harness-vsv · harness-l3tgq — Cruise behavior`) and closed the PARENT epic
# `harness-l3tgq` instead of `harness-vsv`. That close was rejected ("blocked by
# open issues [harness-vsv]") and the turn halted close->halted with the work
# already implemented but the issue unclosed. Inject the exact id so the model
# closes the current issue, not a parent referenced for context.
def _close_target_directive(issue_id: str) -> str:
    return (
        f"\n\nCLOSE THIS EXACT ISSUE: run `bd close {issue_id}`. Close "
        f"`{issue_id}` and NOTHING ELSE — NOT the parent epic, NOT any other id "
        f"shown for context in the issue title (the `· <parent>` reference is "
        f"orientation, not your target). loop_run=adfc7bd6 closed the parent "
        f"epic by mistake; that close is rejected (blocked by this still-open "
        f"issue) and the turn halts with the work done but the issue unclosed."
    )


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
    executor_temperature: float = 0.5,
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
        temperature=executor_temperature,
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
    *,
    structural_bead: bool = False,
    verify_steps: Sequence[VerifyStep] = (),
    prior_test_cmd: str | None = None,
    workspace: Path | None = None,
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
    # harness-1kd9t: a structural bead whose scaffold already exists has no
    # gap to close — route ASSESS straight to CLOSE instead of IMPLEMENT,
    # where the model "builds the skeleton" by rewriting the now-populated
    # file and trips the regression guard (loop_run=7b371136). Gated to
    # structural beads: their acceptance IS code-presence/file-structure,
    # which the CLOSE phase's pre_close_verify hook re-checks, so a wrong
    # claim fails verify rather than closing unfinished behavioral work. For
    # non-structural beads the flag is ignored — they fall through to the
    # normal TDD path.
    if structural_bead and latest.get("already_satisfied", False):
        return assessment_already_satisfied(
            current_state=str(latest["current_state"]),
            reason=str(latest.get("gap") or "scaffold already present"),
        )
    # loop_run=dae002aa: a BEHAVIORAL bead claiming already_satisfied is
    # never trusted to CLOSE, but it shouldn't be forced through WRITE_TEST
    # either — turns 10-12 of harness-8i9 assessed "no gap, drawPedestrian
    # already implemented" (correctly), got the flag ignored, and the forced
    # TDD path manufactured an always-red scope-only test that parked a done
    # bead. Route to VERIFY and let the carried test + registered verify
    # steps arbitrate: green → CLOSE, red → IMPLEMENT. Honored only when
    # there IS a real gate to arbitrate — with no carried non-degenerate
    # test and no verify steps the flag is ignored (fall through to the
    # normal TDD path) so a behavioral bead can't close on zero evidence.
    if not structural_bead and latest.get("already_satisfied", False):
        carried_gate_is_real = (
            prior_test_cmd is not None
            and workspace is not None
            and not _is_degenerate_test_cmd(prior_test_cmd)
            and not _test_cmd_file_missing(prior_test_cmd, workspace)
        )
        if carried_gate_is_real or verify_steps:
            return assessment_satisfied_pending_verify(
                current_state=str(latest["current_state"]),
                gap=str(latest["gap"]),
                approach=str(latest["approach"]),
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
    # loop_run=dae002aa (harness-uy4): a Node test that loads browser-
    # authored source without stubbing the DOM crashes at LOAD —
    # `ReferenceError: document is not defined` at game.js line 5 — before
    # any assertion runs. That red is decoupled from the implementation
    # (4 attempts burned the verify-retry ceiling against it). Scoped to
    # the named browser globals; a bare "referenceerror" would also match
    # the legitimate red `ReferenceError: drawTile is not defined`, which
    # IS the gap.
    "document is not defined",
    "window is not defined",
    "navigator is not defined",
    # loop_run=069d6172 (harness-xdgtb): the same bare-Node load crash via a
    # different browser global — `ReferenceError: requestAnimationFrame is
    # not defined` thrown from the eval'd source's own top-level game loop,
    # before any assertion. Not caught by the eval-blind tell (the source
    # CALLS rAF, doesn't declare it) nor the scaffold-crash check (it's a
    # ReferenceError at `at eval`, not a TypeError in the test file). Same
    # browser-global class as document/window — name the common timing /
    # storage / dialog globals so the gate is rejected at submit instead of
    # burning attempts until the byte-identical detector catches it.
    "requestanimationframe is not defined",
    "cancelanimationframe is not defined",
    "requestidlecallback is not defined",
    "localstorage is not defined",
    "sessionstorage is not defined",
    "alert is not defined",
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


# loop_run=dae002aa: constructs whose presence in a test file proves it
# LOADS something outside itself — a module import, a file read, a
# subprocess. Lower-cased before matching. A test with NONE of these can
# only assert on its own scope and mocks, so its verdict is fixed at
# authoring time: `typeof drawPedestrian === 'undefined'` in an empty
# scope is red forever no matter what lands in game.js (harness-8i9
# burned 4 attempts x 3 verify retries against exactly that test while
# the implementation it "tested" sat finished in the workspace).
_SOURCE_LOAD_MARKERS: tuple[str, ...] = (
    "require(",  # JS CommonJS module load
    "import ",  # Python import / ESM static import
    "import(",  # ESM dynamic import
    "readfilesync",  # fs.readFileSync — the eval-the-source pattern
    "readfile(",  # fs.readFile / aiofiles
    "open(",  # Python file read
    "exec",  # child_process.exec / execSync / Python exec(open(...))
    "spawn",  # child_process.spawn
    "subprocess",  # Python subprocess
    "__import__",
)

# Cap on how much of a submitted test file the lint reads. A real test
# is a few KB; the cap just bounds a pathological submission.
_GATE_LINT_READ_CAP: int = 262_144

# Test-file suffixes the source-load lint understands. A .sh / .yaml /
# other gate shape is skipped — the lint can't reason about its load
# semantics and a false reject would block a legitimate gate.
_GATE_LINT_SUFFIXES: tuple[str, ...] = _TEST_SCRIPT_EXTS


def _test_never_loads_source(workspace: Path, test_path: str) -> str | None:
    """Rejection message when the submitted test file provably never
    loads any artifact outside itself — no import, no require, no file
    read, no subprocess (loop_run=dae002aa). Such a test asserts only on
    its own scope/mocks, so it is implementation-insensitive: red forever
    (or green forever), never a gate. Conservative: unreadable / missing
    / unrecognized-suffix files return None (other guards own those)."""
    candidate = Path(test_path)
    resolved = candidate if candidate.is_absolute() else workspace / candidate
    if resolved.suffix not in _GATE_LINT_SUFFIXES or not resolved.is_file():
        return None
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")[:_GATE_LINT_READ_CAP]
    except (OSError, PermissionError):
        return None
    low = content.lower()
    if any(marker in low for marker in _SOURCE_LOAD_MARKERS):
        return None
    return (
        f"the test file {test_path} never loads any source artifact — no "
        f"import / require / file read / subprocess anywhere in it. A test "
        f"that only asserts on its own scope and mock copies is red forever "
        f"regardless of the implementation: it cannot observe the code under "
        f"change, so it can never go green in VERIFY. Load the REAL source "
        f"file first (e.g. eval(fs.readFileSync('<source>.js', 'utf8')) after "
        f"stubbing any browser globals it touches, or a plain import), then "
        f"assert on what it defines, and resubmit."
    )


# loop_run=df358902 (harness-491j5 / harness-c6jqy): the eval-the-source
# idiom fails a SECOND way beyond the let/const ReferenceError gate_blind
# catches. `eval(readFileSync('game.js')); global.player.x = …` never
# attaches the source's bindings to `global`, so `global.player` is
# `undefined` and the test's OWN setup throws
# `TypeError: Cannot set properties of undefined` at its line N — a crash
# in the harness scaffolding, before any assertion runs. The exit is
# non-zero, so red_check accepts it and the byte-identical detector only
# catches it after the verify-retry ceiling. This is the first-failure
# tell: an uncaught property-access TypeError whose top frame is the test
# file is broken scaffolding, not a gap the implementation can close.
#
# Scoped to "cannot read/set propert{y,ies} of undefined/null" — the
# reference-is-undefined fingerprint. Deliberately NOT
# "X is not a function" / "X is not defined": those name the deliverable
# the bead must build (game.spawnPed is not a function) and ARE the gap.
_SCAFFOLD_CRASH_RE = re.compile(
    r"TypeError:\s*cannot (?:read|set) propert(?:y|ies) of (?:undefined|null)",
    re.IGNORECASE,
)


def _frame_references_test_file(tail: str, test_path: str) -> bool:
    """True when a stack frame in `tail` points into the test file — a
    Node frame (``… test_x.js:25:20``) or a Python traceback line
    (``File "test_x.py", line 25``). Matched on basename so absolute and
    relative frame paths both hit. A crash whose top frame is the test
    file originates in the harness, not in the source under test."""
    name = re.escape(Path(test_path).name)
    js_frame = re.search(rf"\b{name}:\d+", tail)
    py_frame = re.search(rf'"[^"\n]*{name}",\s*line\s+\d+', tail)
    return bool(js_frame or py_frame)


def _test_scaffold_crash_output(test_path: str, tail: str) -> str | None:
    """Rejection message when a failing-test submission is a scaffold
    crash (harness-c6jqy): an uncaught property-access TypeError thrown
    from the test file's OWN code (top frame inside the test file) before
    any assertion — the test's handle on the source under test is
    undefined because its setup failed to expose it. Red forever
    regardless of the implementation. Returns None for any other shape
    (genuine assertion failure, missing-deliverable crash, source-side
    error) so it never false-rejects a real gate."""
    if not _SCAFFOLD_CRASH_RE.search(tail):
        return None
    if not _frame_references_test_file(tail, test_path):
        return None
    return (
        f"test_cmd exits non-zero, but the failure is an uncaught TypeError in "
        f"the test's OWN scaffolding ({test_path}), not an assertion: the test "
        f"crashed reading/setting a property of `undefined` before it could "
        f"check any behavior. The usual cause is the eval-the-source idiom — "
        f"`eval(fs.readFileSync('<source>.js'))` does NOT attach the source's "
        f"top-level `let`/`const`/`var` bindings to `global`, so `global.<name>` "
        f"is undefined. That red is decoupled from the implementation and can "
        f"never go green. Fix the harness so the source's symbols are reachable "
        f"(append the assertions INTO the eval'd string so they share its scope, "
        f"or have the source export and require it), confirm the test then fails "
        f"on the MISSING BEHAVIOR, and resubmit."
    )


def _lint_submitted_gate(
    workspace: Path, test_path: str, test_cmd: str, exit_code: int, tail: str
) -> str | None:
    """Submit-time gate lint (loop_run=dae002aa), wired into
    SubmitFailingTestTool.gate_lint. Runs only after red_check proved the
    command exits non-zero; rejects the two always-red shapes that a red
    exit alone can't distinguish from a genuine failing assertion:

      1. The runner couldn't LOAD/run the test (missing module, syntax
         error, browser global in a bare Node run) — red for a reason
         decoupled from the implementation.
      2. The test never loads any source artifact — it asserts on its own
         mocks, so the implementation is invisible to it.

    Rejecting at submit time gives the model an in-phase retry with the
    reason in hand; letting either shape through spends the verify-retry
    ceiling + the issue's whole attempt budget on a gate that can never
    go green (harness-8i9 / harness-uy4 parked exactly this way).

    Third shape (harness-815wm): the test loads the source via direct
    eval but throws ReferenceError on a top-level let/const binding the
    source DOES declare — eval scoping makes those bindings invisible to
    the test's own code, so the red is structural, not the gap.

    Fourth shape (harness-c6jqy): the test crashes with an uncaught
    property-access TypeError in its OWN scaffolding (`Cannot set
    properties of undefined` at the test's line) before any assertion —
    the eval-the-source binding never attached to `global`, so the SUT
    handle is undefined. Red forever, like the others."""
    if _is_unrunnable_test_output(exit_code, tail):
        return (
            f"test_cmd exits non-zero, but because the runner could not LOAD "
            f"or run the test — not because an assertion failed: "
            f"{tail.strip()[-200:]} — that red is decoupled from the "
            f"implementation and can never go green by changing the source. "
            f"Fix the load error first (stub browser globals like document/"
            f"window before loading browser-authored source, correct the "
            f"path/import), confirm the test fails on the MISSING BEHAVIOR, "
            f"then resubmit."
        )
    blind = eval_blind_reference(tail, test_path, workspace)
    if blind is None:
        # The `typeof`-guarded sibling throws nothing, so the runtime tell
        # above can't see it — fall back to the static read of the test text.
        blind = eval_blind_typeof_guard(test_path, workspace)
    if blind is not None:
        return (
            f"test_cmd exits non-zero, but the red is structural, not the "
            f"gap: {blind}. Fix the test so its assertions can observe the "
            f"source's state, confirm it fails on the MISSING BEHAVIOR, "
            f"then resubmit."
        )
    crash = _test_scaffold_crash_output(test_path, tail)
    if crash is not None:
        return crash
    return _test_never_loads_source(workspace, test_path)


# harness-1ttpl (loop_run=d12941b6): WRITE_TEST kept halting "no test" because
# the model never called submit_failing_test — it looped read_file/glob over
# the test files ALREADY on disk and concluded it "cannot run a test" (it had
# shell the whole time). The prior_test_cmd reuse below only helps once a PRIOR
# attempt submitted a gate; here nothing ever submitted, yet a ready red gate
# sits in the workspace, unused. Auto-adopt it: glob test files, keep the ones
# whose name overlaps the bead subject, run them, and adopt the first that is
# red, runnable, and passes the same submit-time lint a real submission would.
_TEST_FILE_GLOBS: tuple[str, ...] = (
    "test_*.py",
    "*_test.py",
    "test_*.js",
    "*_test.js",
    "*.test.js",
    "*.test.mjs",
    "test_*.mjs",
)
# Generic verbs / scaffolding words a bead title or test filename carries that
# carry no subject signal — matching on them would adopt an unrelated gate.
_ADOPT_STOP_WORDS: frozenset[str] = frozenset(
    {
        "test",
        "tests",
        "implement",
        "add",
        "fix",
        "make",
        "ensure",
        "correct",
        "update",
        "support",
        "handle",
        "mode",
        "spec",
        "game",
        "should",
        "the",
        "and",
        "for",
        "with",
        "per",
        "behavior",
        "behaviour",
        "requirement",
        "requirements",
    }
)
# Cap how many candidate gates we actually execute, bounding the cost of the
# no-submit recovery path. Best name-overlap first, so the cap rarely bites.
_MAX_ADOPT_CANDIDATES = 3


def _subject_tokens(text: str) -> set[str]:
    """Domain tokens of a bead title / test filename for overlap matching.
    Lowercased, ≥3 chars, generic verbs/scaffolding words dropped."""
    return {
        tok
        for tok in re.split(r"[^a-z0-9]+", text.lower())
        if len(tok) >= 3 and tok not in _ADOPT_STOP_WORDS
    }


def _default_test_cmd_for(rel_path: Path) -> str | None:
    """A bare runner command for an on-disk test file, or None when the
    extension needs a toolchain we can't assume present (e.g. .ts)."""
    suffix = rel_path.suffix
    if suffix == ".py":
        return f"python3 {rel_path.as_posix()}"
    if suffix in (".js", ".mjs", ".cjs"):
        return f"node {rel_path.as_posix()}"
    return None


def _adopt_existing_red_gate(workspace: Path, issue_title: str) -> PhaseOutcome | None:
    """Find an existing test file that is already this bead's red gate and
    adopt it as `failing_test_submitted`, so a model that never called
    submit_failing_test doesn't park the bead when the gate is right there.

    Conservative by construction: requires name-overlap with the bead
    subject, only adopts a RED + runnable result, and routes it through the
    same `_lint_submitted_gate` a real submission passes (rejects
    blind/scaffold-crash/never-loads-source shapes). Green candidates are
    skipped — a filename-matched green test is too weak a signal to close
    the bead on. Returns None when nothing qualifies (caller halts)."""
    title_tokens = _subject_tokens(issue_title)
    if not title_tokens:
        return None
    candidates: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for pattern in _TEST_FILE_GLOBS:
        for path in workspace.rglob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            overlap = title_tokens & _subject_tokens(path.stem)
            if overlap:
                candidates.append((len(overlap), path))
    # Best overlap first; deterministic tiebreak on path so the choice is stable.
    candidates.sort(key=lambda c: (-c[0], str(c[1])))
    for _score, path in candidates[:_MAX_ADOPT_CANDIDATES]:
        rel = path.relative_to(workspace)
        cmd = _default_test_cmd_for(rel)
        if cmd is None:
            continue
        exit_code, tail = _exec_test_cmd(cmd, workspace)
        if exit_code == 0:
            continue  # green: too weak to adopt as the gate
        if _is_unrunnable_test_output(exit_code, tail):
            continue  # red for a load reason, not the gap
        if _lint_submitted_gate(workspace, str(rel), cmd, exit_code, tail) is not None:
            continue  # blind / scaffold-crash / never-loads — not a real gate
        return failing_test_submitted(
            test_path=str(rel),
            test_cmd=cmd,
            failure_output=(
                f"adopted existing red gate {rel} — the model never submitted one: "
                f"{tail.strip()[-300:]}"
            ),
        )
    return None


def _resolve_write_test_outcome(
    submit_failing_test: SubmitFailingTestTool,
    skip_test_phase: SkipTestPhaseTool,
    *,
    workspace: Path,
    prior_test_cmd: str | None = None,
    issue_title: str = "",
) -> PhaseOutcome:
    skip = skip_test_phase.latest()
    latest = submit_failing_test.latest()
    # harness-15eeq: a skip is a legitimate escape ONLY when the issue has no
    # established gate. Once a prior attempt submitted a real (non-degenerate,
    # runnable) failing test, a later skip_test_phase is the model disowning
    # its own gate to dodge the work — "the test is wrong, skip it" — which it
    # provably isn't, since it runs and fails on the gap (drive loop_run=
    # 9a1e7970, harness-rxtpz). A fresh submit this attempt likewise overrides
    # the skip. Honor the skip only when there is no gate at all; otherwise
    # fall through to reuse the carried gate / accept the fresh submit.
    carried_gate_is_real = (
        prior_test_cmd is not None
        and not _is_degenerate_test_cmd(prior_test_cmd)
        and not _test_cmd_file_missing(prior_test_cmd, workspace)
    )
    if skip is not None and latest is None and not carried_gate_is_real:
        return PhaseOutcome(
            kind="test_phase_skipped",
            detail=f"skipped: {skip['reason'][:120]}",
            payload={"reason": skip["reason"]},
        )
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
        # harness-1ttpl: no submission this attempt and no carried gate — before
        # halting, adopt an existing on-disk test that is already this bead's red
        # gate (loop_run=d12941b6: the model read the gate files repeatedly but
        # never ran or submitted one).
        adopted = _adopt_existing_red_gate(workspace, issue_title)
        if adopted is not None:
            return adopted
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
    # harness-pfr5a: a submission that came through a wired red_check
    # carries its execution proof (exit + tail captured at submit time,
    # guaranteed non-zero — exit 0 was rejected as a tool error before
    # it could land in `captured`). Reuse it rather than running the
    # same command a second time; the unrunnable classification below
    # still applies to the stored tail. Submissions without the proof
    # (callers that wired no red_check) keep the re-execution.
    if "red_check_exit" in latest:
        exit_code, tail = int(latest["red_check_exit"]), latest.get("red_check_tail", "")
    else:
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
        exit_code, tail, output = _exec_test_cmd_capture(test_cmd, workspace)
        if exit_code != 0:
            # step/tail feed the gate-suspect tracker in run_fsm_turn
            # (loop_run=dae002aa): identical test-step tails across
            # IMPLEMENT passes that touched the source mark the gate as
            # implementation-insensitive.
            # harness-815wm: the eval-blind tell runs on the FULL output —
            # a ReferenceError on a name the eval'd source declares with
            # top-level let/const means the gate can never observe the
            # implementation, so run_fsm_turn halts on the FIRST failure
            # instead of burning the verify-retry ceiling.
            # harness-1ttpl follow-on: the submit-time lint catches both the
            # ReferenceError trap AND its `typeof`-guarded sibling, but a
            # gate CARRIED from a prior attempt (carried_gate_is_real) is
            # reused without re-submitting — so it never sees that lint. The
            # ReferenceError detector below keys on runtime output, which the
            # `typeof` trap never produces; fall back to the static text read
            # so a carried `typeof`-trap gate is caught at the FIRST verify
            # too, not after the byte-identical detector burns the budget.
            test_script = _test_cmd_script(test_cmd)
            return verify_failed(
                failure_tail=f"test {test_cmd!r} exit={exit_code}: {tail}",
                step="test",
                tail=tail,
                gate_blind=eval_blind_reference(output, test_script, workspace)
                or eval_blind_typeof_guard(test_script, workspace)
                or "",
            )
    for step in verify_steps:
        exit_code, tail = _exec_test_cmd(step.cmd, workspace, shell_mode=step.shell)
        if exit_code != 0:
            preview = step.cmd if len(step.cmd) <= 80 else step.cmd[:77] + "..."
            return verify_failed(
                failure_tail=f"{preview} exit={exit_code}: {tail}",
                step="verify",
                tail=tail,
            )
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
# harness-815wm: cap on the fuller output the gate-blind tell reads. A
# Node stack trace pushes the ReferenceError line out of the 200-char
# tail, so diagnosis needs the head of the error too.
_VERIFY_OUTPUT_CAP: int = 8_000


def _exec_test_cmd_capture(
    cmd: str, workspace: Path, *, shell_mode: bool = True
) -> tuple[int, str, str]:
    """Re-execute a single test command. Mirrors `_exec_verify_cmd`
    in `loop.py` (harness-xfh2) — same timeout, same tail length,
    same failure shape. Kept separate so the FSM module doesn't
    import private helpers from the legacy executor module.

    Returns ``(exit_code, tail, output)`` — `tail` is the classic
    200-char tail (gate-suspect byte-identical comparison keys on it);
    `output` is the same stream capped at `_VERIFY_OUTPUT_CAP` for
    diagnostics that need the error's head (harness-815wm)."""
    args: str | list[str]
    if shell_mode:
        args = cmd
    else:
        try:
            args = shlex.split(cmd)
        except ValueError as exc:
            msg = f"unparseable cmd: {exc}"
            return 1, msg, msg
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
        msg = f"executable not found: {exc}"
        return 1, msg, msg
    except subprocess.TimeoutExpired:
        msg = f"timeout after {_VERIFY_TIMEOUT_SECONDS}s"
        return 1, msg, msg
    except OSError as exc:
        msg = f"exec failed: {exc}"
        return 1, msg, msg
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    tail_src = stderr or stdout
    return (
        result.returncode,
        tail_src[-_VERIFY_TAIL_CHARS:],
        tail_src[-_VERIFY_OUTPUT_CAP:],
    )


def _exec_test_cmd(cmd: str, workspace: Path, *, shell_mode: bool = True) -> tuple[int, str]:
    exit_code, tail, _ = _exec_test_cmd_capture(cmd, workspace, shell_mode=shell_mode)
    return exit_code, tail


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
    prior_test_fail_tail: str | None = None,
    verify_steps: Sequence[VerifyStep] = (),
    phase_budgets: Mapping[TurnPhase, int] | None = None,
    tdd_required: bool = True,
    tdd_skip_reason: str | None = None,
    structural_bead: bool = False,
    deliverable_text: str = "",
    observe: ExecutorObserver | None = None,
    executor_temperature: float = 0.5,
    summarize_tool_results: bool = True,
    pre_close_verify: PreCloseVerifyHook | None = None,
) -> FsmTurnResult:
    """Drive `current_issue_id` through the TurnPhase FSM.

    `handoff_builder(phase, assessment, test_cmd)` produces a fresh
    Handoff for each phase — caller-owned because the handoff
    composition (bd queries, git diff, thoughts) is the driver's
    contract, not this module's.

    `tdd_required=False` (e.g. CLI --no-tdd, or a per-bead structural
    classification — harness-1kd9t) is handled by inspecting the resolved
    ASSESS outcome and rerouting it through `assessment_skipped_tdd` BEFORE
    feeding to the FSM. `tdd_skip_reason` overrides the recorded reason so
    the trace says WHY TDD was skipped (defaults to the --no-tdd text).

    `structural_bead=True` (harness-1kd9t) marks the bead as a
    scaffold/declaration unit, which enables the ASSESS-phase
    already_satisfied → CLOSE escape: an already-built scaffold closes
    (gated by pre_close_verify) instead of being forced through IMPLEMENT,
    where the model rewrites the populated file and trips the regression
    guard. The underlying transition table is unchanged.

    Returns FsmTurnResult with the terminal phase + success boolean +
    threaded assessment/test_cmd payload for resume.
    """
    budgets = dict(DEFAULT_PHASE_BUDGETS) if phase_budgets is None else dict(phase_budgets)

    submit_assessment = SubmitAssessmentTool()
    skip_test_phase_tool = SkipTestPhaseTool()

    # harness-pfr5a: gate the submission on the test actually being red.
    # An always-green test (exits 0 while printing failure text) is
    # rejected at submit time with a tool error, so the model fixes it
    # in-phase instead of the phase-end green short-circuit reading it
    # as "implementation already landed" and false-closing the bead.
    # loop_run=dae002aa: gate_lint rejects the two always-red shapes a red
    # exit alone can't distinguish from a genuine failing assertion — a
    # test the runner can't load, and a test that never loads the source.
    # harness-815wm: the red_check hands gate_lint the FULL (capped)
    # output, not the 200-char verify tail — a Node stack trace pushes
    # the lint-relevant head of the error (ReferenceError name,
    # SyntaxError location) out of a short tail.
    def _red_check(cmd: str) -> tuple[int, str]:
        exit_code, _tail, output = _exec_test_cmd_capture(cmd, workspace)
        return exit_code, output

    submit_failing_test = SubmitFailingTestTool(
        red_check=_red_check,
        gate_lint=lambda test_path, test_cmd, exit_code, tail: _lint_submitted_gate(
            workspace, test_path, test_cmd, exit_code, tail
        ),
    )
    submit_implementation_complete = SubmitImplementationCompleteTool()

    # harness-0t2f9: reject a flag_blocked whose `missing` restates the
    # bead's own deliverable (the absent artifact IS the task, not an
    # upstream precondition) before it can route ASSESS to PREMISE_UNMET.
    # Returns a corrective tool error so the model proceeds with ASSESS;
    # a genuine upstream precondition shares few deliverable tokens and
    # falls through to the normal park. Disabled when no bead text is
    # supplied (deliverable_text="") — trust-the-model, for unit callers.
    def _deliverable_check(missing: str) -> str | None:
        # harness-vsv: reject a flag_blocked that blames a write-tier editor
        # ASSESS withholds by design (available in IMPLEMENT). loop_run=065ff3c1
        # parked harness-vsv on attempt 1 — "edit_file tool not available" — a
        # phase-confusion, not an unmet premise. No deliverable_text needed: the
        # tool roster is the same regardless of bead text, so this fires for
        # unit callers too.
        withheld = flag_blocked_names_withheld_tool(missing, _EDITOR_TOOL_NAMES)
        if withheld is not None:
            return (
                f"flag_blocked rejected: '{withheld}' is not an absent premise — "
                f"it's a write-tier tool the ASSESS phase withholds by design. "
                f"It IS available in the IMPLEMENT phase. Do NOT flag; call "
                f"submit_assessment to plan the change and the FSM routes you to "
                f"IMPLEMENT, where {withheld} is in your toolset."
            )
        if not deliverable_text.strip():
            return None
        if flag_blocked_names_own_deliverable(missing, deliverable_text):
            return (
                f"flag_blocked rejected: '{missing}' names THIS bead's own "
                f"deliverable, not an absent UPSTREAM precondition. That the "
                f"artifact doesn't exist yet is the task — it's what this bead "
                f"exists to build. flag_blocked is only for a concrete symbol / "
                f"file the bead presupposes and a DIFFERENT, earlier bead owns. "
                f"Do not flag; proceed with ASSESS and call submit_assessment to "
                f"plan the implementation."
            )
        return None

    flag_blocked = FlagBlockedTool(deliverable_check=_deliverable_check)

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
    # Gate-suspect tracking (loop_run=dae002aa): consecutive test-step
    # verify failures with byte-identical output tails, where the
    # IMPLEMENT pass between them at least ATTEMPTED a write, mean the
    # test never observes the code under change — re-authoring the gate
    # is the move, not more IMPLEMENT passes. Attempted (not just landed)
    # writes count: on an already-implemented bead every edit is a no-op,
    # which is exactly the harness-8i9 shape that burned 4 attempts.
    # harness-smplj widened the arming on two axes: (1) an IMPLEMENT pass
    # that declared the work complete with NO edits is as suspect as one
    # that edited — the model believes it's done and the verdict won't
    # move; (2) the fail tail is seeded from the prior attempt (state),
    # so a byte-identical failure across attempts trips even when each
    # turn hits the hs50i verify-retry ceiling before seeing two in one
    # turn (loop_run=df358902, harness-491j5: 4 turns x 3 verifies = 12
    # byte-identical failures, none detected).
    # harness-smplj: seed from the prior attempt's persisted fail tail so a
    # byte-identical failure ACROSS attempts trips the detector on this
    # turn's FIRST verify pass — the cross-turn hole the per-turn-local
    # variable left open when each turn halted at the hs50i ceiling first.
    last_test_fail_tail: str | None = prior_test_fail_tail
    implement_touched_source = False
    # harness-smplj: the model declared IMPLEMENT complete this pass WITHOUT
    # editing the source — it believes the bead is done. A subsequent
    # byte-identical verify fail is as suspect as one after an edit: the
    # gate verdict won't move because the model isn't changing the source.
    implement_complete_no_edits = False
    gate_suspect = False
    # Browser-JS census, computed once per turn: gates the WRITE_TEST
    # stub-the-DOM hint (loop_run=dae002aa).
    browser_workspace = workspace_has_browser_js(workspace)

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
        if phase is TurnPhase.WRITE_TEST and browser_workspace:
            user_prompt += _BROWSER_TEST_HINT
        if phase is TurnPhase.CLOSE:
            user_prompt += _close_target_directive(current_issue_id)
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
            executor_temperature=executor_temperature,
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
        # Gate-suspect input (loop_run=dae002aa): did the most recent
        # IMPLEMENT pass touch the source? Attempted counts — a no-op
        # edit on an already-implemented bead still proves the model
        # acted on the source while the gate's verdict didn't move.
        if phase is TurnPhase.IMPLEMENT:
            implement_touched_source = bool(
                WRITE_TOOL_NAMES & (execution.succeeded_tools | execution.attempted_write_tools)
            )

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
            structural_bead=structural_bead,
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
                reason=tdd_skip_reason or "--no-tdd flag set on this loop run",
            )

        # Capture the assessment + test_cmd for the next phase's
        # handoff render + the run's state persistence.
        if outcome.kind in {
            "assessment_submitted",
            "assessment_skipped",
            "assessment_satisfied_pending_verify",
        }:
            captured_assessment = dict(outcome.payload)
        if outcome.kind == "failing_test_submitted":
            captured_test_cmd = str(outcome.payload.get("test_cmd", ""))
        if outcome.kind == "premise_unmet":
            premise_unmet_reason = outcome.detail

        # harness-smplj: did the IMPLEMENT pass declare the work complete
        # with NO source edits? Keyed on the resolved outcome (implement_complete
        # fires when submit_implementation_complete was recorded) rather than a
        # fresh tool-call sniff, so it survives the auto-resolve path. Recomputed
        # every IMPLEMENT pass: a later pass that touches source clears it.
        if phase == TurnPhase.IMPLEMENT:
            implement_complete_no_edits = (
                outcome.kind == "implement_complete" and not implement_touched_source
            )

        # harness-hs50i: bound the IMPLEMENT↔VERIFY cycle. Counted on
        # the outcome (pre-handle) so the cap halts BEFORE re-entering
        # IMPLEMENT for a pass the budget would just burn.
        if phase == TurnPhase.VERIFY and outcome.kind == "verify_failed":
            # Gate-suspect check (loop_run=dae002aa): the test step failed
            # with the SAME output tail as the previous verify pass, and
            # the IMPLEMENT pass between them touched the source. The gate
            # never observes the code under change — more IMPLEMENT passes
            # can't move it. Halt, mark the result so the loop drops the
            # carried test_cmd, and let the next attempt re-author.
            if outcome.payload.get("step") == "test":
                # harness-815wm: the eval-blind tell fired — the test threw
                # ReferenceError on a binding the eval'd source declares at
                # top level with let/const, so the gate is structurally
                # unable to observe the implementation. Halt on the FIRST
                # failure (the byte-identical detector below would burn
                # another IMPLEMENT pass first, and the verify-retry
                # ceiling three) and drop the carried test so the next
                # attempt re-authors with the diagnosis in hand.
                blind_msg = str(outcome.payload.get("gate_blind", ""))
                if blind_msg:
                    gate_suspect = True
                    captured_test_cmd = None
                    fsm.force(
                        TurnPhase.HALTED,
                        reason=f"gate-blind verify gate (harness-815wm): {blind_msg}",
                    )
                    break
                this_tail = str(outcome.payload.get("tail", ""))
                # harness-smplj: arm when the gate verdict is byte-identical
                # AND the model has stopped meaningfully moving the source —
                # either it edited (touched_source) or it declared the work
                # complete with no edits (complete_no_edits). The compared
                # tail may be from a prior verify THIS turn or, when seeded
                # from state, the prior ATTEMPT's tail (cross-turn trip).
                implement_settled = implement_touched_source or implement_complete_no_edits
                if this_tail and this_tail == last_test_fail_tail and implement_settled:
                    gate_suspect = True
                    captured_test_cmd = None
                    # harness-815wm: the old text said the re-authored gate
                    # should "LOAD the source artifact" — which steered the
                    # model back into the direct-eval idiom that cannot see
                    # top-level let/const. Name the pitfall instead when
                    # the workspace is browser JS.
                    pitfall = (
                        f" (browser-JS workspace: {GATE_BLIND_IDIOM_NOTE})"
                        if browser_workspace
                        else ""
                    )
                    fsm.force(
                        TurnPhase.HALTED,
                        reason=(
                            "suspect verify gate: the test failed byte-identically "
                            "across IMPLEMENT passes that edited the source or "
                            "declared it complete without edits — it never observes "
                            "the code under change "
                            "(loop_run=dae002aa/harness-smplj); the carried test is dropped so "
                            "the next attempt re-authors a gate whose assertions "
                            "can actually observe the source's state"
                            f"{pitfall}. last tail: {this_tail[:140]}"
                        ),
                    )
                    break
                last_test_fail_tail = this_tail
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
            last_test_fail_tail=None,
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
        gate_suspect=gate_suspect,
        # harness-smplj: on a gate-suspect halt the carried gate is being
        # dropped, so there's nothing to compare next attempt against —
        # clear the tail too. Otherwise persist it for the cross-turn trip.
        last_test_fail_tail=None if gate_suspect else last_test_fail_tail,
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
    structural_bead: bool = False,
) -> PhaseOutcome:
    if phase == TurnPhase.ASSESS:
        # verify_steps / test_cmd / workspace feed the behavioral
        # already_satisfied → VERIFY route (loop_run=dae002aa): the claim is
        # honored only when there's a real gate for VERIFY to arbitrate.
        return _resolve_assess_outcome(
            submit_assessment,
            flag_blocked,
            structural_bead=structural_bead,
            verify_steps=verify_steps,
            prior_test_cmd=test_cmd,
            workspace=workspace,
        )
    if phase == TurnPhase.WRITE_TEST:
        # harness-axjt8: test_cmd here is captured_test_cmd — a test
        # carried forward from a prior attempt (None on the first). The
        # resolver reuses it instead of halting "no test" on a barren
        # re-attempt.
        # harness-1ttpl: the bead title feeds the existing-red-gate auto-adopt
        # on a no-submit halt. Soft on bd errors — a missing title just
        # disables adoption, never blocks the resolve.
        issue_title = ""
        try:
            # Best-effort: a bd failure just disables adoption, never blocks.
            issue_title = bd.show(issue_id).title
        except Exception:
            issue_title = ""
        return _resolve_write_test_outcome(
            submit_failing_test,
            skip_test_phase,
            workspace=workspace,
            prior_test_cmd=test_cmd,
            issue_title=issue_title,
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
