"""Tests for the turn-phase meta-tools — harness-kbnl.

The meta-tools are the model's vocabulary for signaling FSM
transitions. Each is a stateful dataclass — its `captured` list
holds what landed during a phase, which the driver inspects to
construct the PhaseOutcome event. These tests pin the validation
+ capture semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.driver.fsm_executor import _resolve_assess_outcome
from harness.driver.planner import VerifyStep
from harness.driver.turn_fsm import PREMISE_UNMET_REASON_PREFIX
from harness.tools.turn_phase_meta import (
    FlagBlockedTool,
    SkipTestPhaseTool,
    SubmitAssessmentTool,
    SubmitFailingTestTool,
    SubmitImplementationCompleteTool,
)

# --- submit_assessment --------------------------------------------


def test_submit_assessment_captures_full_payload() -> None:
    """harness-kbnl: a valid assessment lands in `captured` with all
    four fields. Default tdd_applicable is True."""
    tool = SubmitAssessmentTool()
    result = tool.call(
        current_state="game.js has tileGrid with 28 rows of varying widths",
        gap="acceptance criteria require 30 rows by 40 chars each",
        approach="regenerate the entire tileGrid as 30 fixed-width strings",
    )
    assert "recorded" in result
    assert tool.latest() == {
        "current_state": "game.js has tileGrid with 28 rows of varying widths",
        "gap": "acceptance criteria require 30 rows by 40 chars each",
        "approach": "regenerate the entire tileGrid as 30 fixed-width strings",
        "tdd_applicable": True,
        "premise_mismatch": False,
        "already_satisfied": False,
    }


def test_submit_assessment_already_satisfied_path() -> None:
    """harness-1kd9t: already_satisfied=True is captured and the result
    text notes the route to close — the driver reads it (gated on the bead
    being structural) to route ASSESS straight to CLOSE."""
    tool = SubmitAssessmentTool()
    result = tool.call(
        current_state="index.html + game.js skeleton already present, gameLoop wired",
        gap="none — the scaffold from §1 already exists; no change needed",
        approach="no edit; the bead is already satisfied by earlier work",
        already_satisfied=True,
    )
    assert "already satisfied" in result
    latest = tool.latest()
    assert latest is not None
    assert latest["already_satisfied"] is True


def test_submit_assessment_premise_mismatch_path() -> None:
    """harness-jbz4z: premise_mismatch=True is captured and the result
    text signals the bead will be parked — the driver reads
    `premise_mismatch` to route ASSESS to premise_unmet."""
    tool = SubmitAssessmentTool()
    result = tool.call(
        current_state="workspace is a JS canvas driving game (game.js); no snake",
        gap="spec references snake_game.py with food/score — a different project",
        approach="cannot ground this bead against the current workspace",
        premise_mismatch=True,
    )
    assert "premise mismatch" in result
    latest = tool.latest()
    assert latest is not None
    assert latest["premise_mismatch"] is True


def test_submit_assessment_tdd_skip_path() -> None:
    """tdd_applicable=False is captured and the result text notes
    the skip — the driver reads `tdd_applicable` to pick the next
    FSM transition."""
    tool = SubmitAssessmentTool()
    result = tool.call(
        current_state="cli.py has --workspace flag wired to ChatConfig",
        gap="--workspace is missing from the loop subcommand parsers",
        approach="add the flag to driver/cli.py; documentation-only change",
        tdd_applicable=False,
    )
    assert "TDD skipped" in result
    latest = tool.latest()
    assert latest is not None
    assert latest["tdd_applicable"] is False


def test_submit_assessment_rejects_empty_field() -> None:
    """Empty / whitespace-only field is a validation error — the
    orchestrator surfaces it back to the model. Tests guard against
    a regression where short replies get accepted."""
    tool = SubmitAssessmentTool()
    with pytest.raises(ValueError, match="current_state"):
        tool.call(current_state="", gap="x" * 40, approach="x" * 40)


def test_submit_assessment_rejects_too_short_field() -> None:
    """Below the 20-char minimum is rejected — vacuous assessments
    ('done', 'ok', 'implemented') don't pass."""
    tool = SubmitAssessmentTool()
    with pytest.raises(ValueError, match="too short"):
        tool.call(current_state="ok", gap="x" * 40, approach="x" * 40)


def test_submit_assessment_latest_returns_none_before_call() -> None:
    """latest() is None when nothing has been captured. The driver
    uses None to construct a `phase_no_progress` PhaseOutcome."""
    assert SubmitAssessmentTool().latest() is None


def test_submit_assessment_multiple_calls_keep_history() -> None:
    """Multiple captures preserve order; latest() returns the most
    recent. Tests guard against accidental overwrite."""
    tool = SubmitAssessmentTool()
    long = "x" * 40
    tool.call(current_state=long + "_a", gap=long + "_a", approach=long + "_a")
    tool.call(current_state=long + "_b", gap=long + "_b", approach=long + "_b")
    assert len(tool.captured) == 2
    latest = tool.latest()
    assert latest is not None
    assert latest["current_state"].endswith("_b")


# --- skip_test_phase ----------------------------------------------


def test_skip_test_phase_captures_reason() -> None:
    tool = SkipTestPhaseTool()
    result = tool.call(reason="pure CSS change with no observable behavior")
    assert "skipped" in result
    latest = tool.latest()
    assert latest is not None
    assert latest["reason"].startswith("pure CSS change")


def test_skip_test_phase_rejects_short_reason() -> None:
    tool = SkipTestPhaseTool()
    with pytest.raises(ValueError, match="too short"):
        tool.call(reason="ui")


# --- submit_failing_test ------------------------------------------


def test_submit_failing_test_captures_test_cmd() -> None:
    """The test_cmd field is the load-bearing artifact — it's what
    the VERIFY phase will re-run. Test pins that it round-trips."""
    tool = SubmitFailingTestTool()
    result = tool.call(
        test_path="tests/test_tile_grid.py",
        test_cmd="pytest tests/test_tile_grid.py::test_grid_shape -v",
        failure_output="AssertionError: expected 30 rows, got 28 rows in tileGrid",
    )
    assert "tests/test_tile_grid.py" in result
    latest = tool.latest()
    assert latest is not None
    assert latest["test_cmd"] == "pytest tests/test_tile_grid.py::test_grid_shape -v"


def test_submit_failing_test_rejects_empty_failure_output() -> None:
    """A claimed failing test with no captured failure_output is
    not credible. The minimum char check is the cheap version of
    'show me the red' enforcement."""
    tool = SubmitFailingTestTool()
    with pytest.raises(ValueError, match="failure_output"):
        tool.call(
            test_path="tests/test_x_well_named.py",
            test_cmd="pytest tests/test_x_well_named.py",
            failure_output="",
        )


def test_submit_failing_test_red_check_rejects_green_cmd() -> None:
    """harness-pfr5a: a test_cmd that exits 0 is an always-green test —
    it proves nothing about the gap, and the hand-pasted failure_output
    is contradicted by the actual run (loop_run=ca3c96b4: harness-8i9
    false-closed on exactly this). The submission must be rejected and
    nothing captured, so the model fixes the test in-phase."""
    ran: list[str] = []

    def fake_green(cmd: str) -> tuple[int, str]:
        ran.append(cmd)
        return 0, "=== Test Suite === FAIL: drawPedestrian is undefined"

    tool = SubmitFailingTestTool(red_check=fake_green)
    with pytest.raises(ValueError, match="exited 0"):
        tool.call(
            test_path="test_pedestrian_splat_render.js",
            test_cmd="node test_pedestrian_splat_render.js",
            failure_output="FAIL: drawPedestrian function is undefined in game.js",
        )
    assert ran == ["node test_pedestrian_splat_render.js"]
    assert tool.latest() is None


def test_submit_failing_test_red_check_accepts_red_cmd() -> None:
    """A test_cmd that exits non-zero is genuinely red — the submission
    lands in `captured` with the execution proof attached, so the
    phase-end resolver can reuse it instead of running the command a
    second time."""
    tool = SubmitFailingTestTool(red_check=lambda _cmd: (1, "AssertionError: no splat"))
    result = tool.call(
        test_path="test_pedestrian_splat_render.js",
        test_cmd="node test_pedestrian_splat_render.js",
        failure_output="AssertionError: dead ped not rendered as #5a0a0a circle",
    )
    assert "recorded" in result
    latest = tool.latest()
    assert latest is not None
    assert latest["test_cmd"] == "node test_pedestrian_splat_render.js"
    assert latest["red_check_exit"] == "1"
    assert latest["red_check_tail"] == "AssertionError: no splat"


def test_submit_failing_test_no_red_check_trusts_model() -> None:
    """Without a wired red_check (callers with no workspace), the
    pre-pfr5a trust-the-model behavior is preserved."""
    tool = SubmitFailingTestTool()
    tool.call(
        test_path="tests/test_y_well_named.py",
        test_cmd="true",
        failure_output="claimed failure output that is never re-executed here",
    )
    assert tool.latest() is not None


# --- submit_implementation_complete -------------------------------


def test_submit_implementation_complete_captures_summary() -> None:
    tool = SubmitImplementationCompleteTool()
    result = tool.call(summary="regenerated tileGrid in game.js with 30 rows of 40 chars each")
    assert "recorded" in result
    latest = tool.latest()
    assert latest is not None
    assert "tileGrid" in latest["summary"]


def test_submit_implementation_complete_rejects_vague_summary() -> None:
    """'done' / 'fixed' / 'ok' don't tell a future-self anything.
    The 20-char floor catches the worst of these."""
    tool = SubmitImplementationCompleteTool()
    with pytest.raises(ValueError, match="too short"):
        tool.call(summary="done")


# --- specs are well-formed ----------------------------------------


def test_all_meta_tools_advertise_read_tier() -> None:
    """Meta-tools have no write side effects (they record into local
    state). They MUST advertise read-tier so write-tier confirmation
    doesn't gate them."""
    for tool in (
        SubmitAssessmentTool(),
        SkipTestPhaseTool(),
        SubmitFailingTestTool(),
        SubmitImplementationCompleteTool(),
    ):
        assert tool.spec.tier == "read", f"{tool.spec.name} must be read-tier"


def test_all_meta_tool_specs_have_required_fields() -> None:
    """JSON schema's `required` list must list every parameter the
    call signature treats as mandatory. Catches drift between the
    schema and the implementation."""
    for tool, mandatory in (
        (SubmitAssessmentTool(), {"current_state", "gap", "approach"}),
        (SkipTestPhaseTool(), {"reason"}),
        (
            SubmitFailingTestTool(),
            {"test_path", "test_cmd", "failure_output"},
        ),
        (SubmitImplementationCompleteTool(), {"summary"}),
    ):
        required = set(tool.spec.parameters.get("required", []))
        assert required == mandatory, f"{tool.spec.name} required={required} expected={mandatory}"


# --- flag_blocked (premise-unmet escape, harness-u1il5 follow-on) ----


def test_flag_blocked_captures_missing_and_reason() -> None:
    """A valid flag_blocked lands missing + reason in `captured` and
    latest() returns them — the driver reads this to emit premise_unmet."""
    tool = FlagBlockedTool()
    result = tool.call(
        missing="fireWeapon() / KeyJ handler",
        reason="bead verifies fire-key gating but no fire handler exists to gate",
    )
    assert "flagged blocked" in result
    assert tool.latest() == {
        "missing": "fireWeapon() / KeyJ handler",
        "reason": "bead verifies fire-key gating but no fire handler exists to gate",
    }


def test_flag_blocked_latest_none_before_call() -> None:
    """latest() is None until flag_blocked is called — the driver treats
    None as 'no premise-unmet signal' and falls through to assessment."""
    assert FlagBlockedTool().latest() is None


def test_flag_blocked_rejects_empty_missing() -> None:
    """An empty `missing` is rejected — the escape requires a concrete,
    operator-checkable absent artifact, not a blank claim."""
    tool = FlagBlockedTool()
    with pytest.raises(ValueError, match="missing"):
        tool.call(missing="", reason="x" * 40)


def test_flag_blocked_rejects_vague_reason() -> None:
    """A too-short reason is rejected — 'hard' / 'no' don't justify a
    park."""
    tool = FlagBlockedTool()
    with pytest.raises(ValueError, match="too short"):
        tool.call(missing="fireWeapon()", reason="no")


def test_flag_blocked_deliverable_check_rejects_and_does_not_capture() -> None:
    """harness-0t2f9: a deliverable_check that returns a message rejects
    the call (raises) and leaves nothing captured, so latest() stays None
    and the ASSESS resolver sees no premise-unmet signal."""
    tool = FlagBlockedTool(
        deliverable_check=lambda missing: f"rejected: {missing} is the deliverable"
    )
    with pytest.raises(ValueError, match="is the deliverable"):
        tool.call(missing="pedestrian spawning logic", reason="x" * 40)
    assert tool.latest() is None
    assert tool.captured == []


def test_flag_blocked_deliverable_check_none_honors_flag() -> None:
    """A deliverable_check returning None (genuine upstream) lets the flag
    land — captured + latest() carry the claim through to premise_unmet."""
    tool = FlagBlockedTool(deliverable_check=lambda _missing: None)
    tool.call(missing="drawCop() upstream symbol", reason="x" * 40)
    assert tool.latest() == {
        "missing": "drawCop() upstream symbol",
        "reason": "x" * 40,
    }


def test_flag_blocked_advertises_read_tier() -> None:
    """flag_blocked is read-tier — it records intent, mutates nothing."""
    assert FlagBlockedTool().spec.tier == "read"


def test_flag_blocked_copy_distinguishes_upstream_from_deliverable() -> None:
    """harness-pmz3i: the description must steer the model away from
    flagging a build/create bead's own absent deliverable as a block.
    The old copy ('the symbol the bead asks you to fix does not exist')
    trapped create-tasks — the deliverable never exists pre-implementation.
    Guard the distinguishing guidance so it can't silently regress."""
    desc = FlagBlockedTool().spec.description.lower()
    # Names the legit case: an upstream-owned precondition.
    assert "upstream" in desc
    # Explicitly forbids the trap: flagging this bead's own deliverable.
    assert "this bead" in desc
    assert "build" in desc
    # The `missing` param echoes the same constraint.
    missing_desc = FlagBlockedTool().spec.parameters["properties"]["missing"]["description"]
    assert "deliverable" in missing_desc.lower()


# --- premise_mismatch routing (harness-jbz4z) ---------------------


def test_assess_premise_mismatch_routes_to_premise_unmet() -> None:
    """A submitted assessment with premise_mismatch=True routes ASSESS to
    premise_unmet (park-and-flag), NOT assessment_submitted — so a bead
    grounded against a different project can't be driven and closed (drive
    gta_r2, harness-rorj)."""
    submit = SubmitAssessmentTool()
    submit.call(
        current_state="workspace is a JS driving game; no snake_game.py exists",
        gap="spec targets a Python snake game with food/score — wrong project",
        approach="cannot ground this bead against the current workspace",
        premise_mismatch=True,
    )
    outcome = _resolve_assess_outcome(submit, FlagBlockedTool())
    assert outcome.kind == "premise_unmet"
    assert outcome.detail.startswith(PREMISE_UNMET_REASON_PREFIX)
    # The gap is carried as the human-readable reason.
    assert "wrong project" in outcome.payload["reason"]


def test_assess_premise_mismatch_false_is_normal_assessment() -> None:
    """Default (premise_mismatch absent/False) still yields a normal
    assessment_submitted — the gate is opt-in and doesn't disturb the
    happy path."""
    submit = SubmitAssessmentTool()
    submit.call(
        current_state="game.js draws 28 tile rows of varying widths",
        gap="acceptance criteria require 30 fixed-width rows",
        approach="regenerate tileGrid as 30 strings of 40 chars",
    )
    outcome = _resolve_assess_outcome(submit, FlagBlockedTool())
    assert outcome.kind == "assessment_submitted"


def test_assess_already_satisfied_structural_routes_to_close() -> None:
    """harness-1kd9t: already_satisfied=True on a STRUCTURAL bead routes
    ASSESS → assessment_already_satisfied (→ CLOSE), skipping the
    rewrite-the-scaffold IMPLEMENT path."""
    submit = SubmitAssessmentTool()
    submit.call(
        current_state="index.html + game.js skeleton already present",
        gap="none — scaffold exists from earlier work",
        approach="no change needed; bead already satisfied",
        already_satisfied=True,
    )
    outcome = _resolve_assess_outcome(submit, FlagBlockedTool(), structural_bead=True)
    assert outcome.kind == "assessment_already_satisfied"


def test_assess_already_satisfied_ignored_when_not_structural() -> None:
    """already_satisfied is honored ONLY for structural beads — on a
    non-structural bead the flag is ignored and ASSESS proceeds normally,
    so the close-verify-gap can't be used to skip behavioral work."""
    submit = SubmitAssessmentTool()
    submit.call(
        current_state="update() has no police spawn call",
        gap="need a police.length < wanted guard gating spawnPoliceCar",
        approach="add the spawn guard to update()",
        already_satisfied=True,
    )
    outcome = _resolve_assess_outcome(submit, FlagBlockedTool(), structural_bead=False)
    assert outcome.kind == "assessment_submitted"


def test_assess_flag_blocked_still_wins_over_mismatch() -> None:
    """flag_blocked is checked before submit_assessment, so an explicit
    block still short-circuits regardless of a later mismatch flag."""
    submit = SubmitAssessmentTool()
    submit.call(
        current_state="x" * 30,
        gap="y" * 30,
        approach="z" * 30,
        premise_mismatch=True,
    )
    flag = FlagBlockedTool()
    flag.call(missing="fire_handler", reason="no fire input handler exists upstream")
    outcome = _resolve_assess_outcome(submit, flag)
    assert outcome.kind == "premise_unmet"
    assert "fire_handler" in outcome.payload["missing"]


# --- always-red gate rejection (loop_run=dae002aa) ----------------


def test_submit_failing_test_green_rejection_includes_exit_wiring() -> None:
    """loop_run=dae002aa (harness-purtm): four straight turns resubmitted
    tests that print FAIL but exit 0 — the abstract rejection never
    landed. The error must include copyable exit-code wiring for both
    runtimes so the fix is mechanical, not inferable."""
    tool = SubmitFailingTestTool(red_check=lambda _cmd: (0, "FAIL printed, exit 0"))
    with pytest.raises(ValueError, match=r"process\.exit\(1\)") as excinfo:
        tool.call(
            test_path="test_render.js",
            test_cmd="node test_render.js",
            failure_output="FAIL: render does not draw the grid (printed only)",
        )
    assert "sys.exit" in str(excinfo.value)


def test_submit_failing_test_gate_lint_rejects_and_captures_nothing() -> None:
    """loop_run=dae002aa: gate_lint runs after red_check proves non-zero
    exit and can still reject — an always-red gate (load crash /
    never-loads-source) raises with the lint's message so the model
    fixes the test in-phase, and nothing lands in `captured`."""
    tool = SubmitFailingTestTool(
        red_check=lambda _cmd: (1, "ReferenceError: document is not defined"),
        gate_lint=lambda _tp, _tc, _ec, tail: f"always-red gate: {tail}",
    )
    with pytest.raises(ValueError, match="always-red gate"):
        tool.call(
            test_path="test_drawtile.js",
            test_cmd="node test_drawtile.js",
            failure_output="ReferenceError: document is not defined at game.js line 5",
        )
    assert tool.latest() is None


def test_submit_failing_test_gate_lint_none_accepts() -> None:
    """A gate_lint that returns None accepts the submission — the lint
    is a veto, not a rewrite."""
    tool = SubmitFailingTestTool(
        red_check=lambda _cmd: (1, "AssertionError: drawTile missing"),
        gate_lint=lambda _tp, _tc, _ec, _tail: None,
    )
    tool.call(
        test_path="test_drawtile.js",
        test_cmd="node test_drawtile.js",
        failure_output="AssertionError: drawTile missing from the loaded source",
    )
    assert tool.latest() is not None


def test_submit_failing_test_gate_lint_skipped_without_red_check() -> None:
    """gate_lint depends on the red_check's exit/tail; without a wired
    red_check there's nothing to lint and the trust-the-model path is
    preserved."""
    tool = SubmitFailingTestTool(
        gate_lint=lambda _tp, _tc, _ec, _tail: "should never fire",
    )
    tool.call(
        test_path="tests/test_y.py",
        test_cmd="pytest tests/test_y.py",
        failure_output="claimed failure output that is never re-executed here",
    )
    assert tool.latest() is not None


# --- behavioral already_satisfied → VERIFY arbitration (loop_run=dae002aa)


def _satisfied_assessment() -> SubmitAssessmentTool:
    submit = SubmitAssessmentTool()
    submit.call(
        current_state="drawPedestrian (game.js:274) already renders dead peds as splats",
        gap="none — the behavior the bead asks for is already implemented",
        approach="no change needed; verification should arbitrate the claim",
        already_satisfied=True,
    )
    return submit


def test_assess_behavioral_satisfied_routes_to_verify_with_steps() -> None:
    """loop_run=dae002aa (harness-8i9): a behavioral bead claiming
    already_satisfied routes to VERIFY arbitration when registered
    verify steps exist — not to a forced WRITE_TEST that manufactures
    an always-red test against a done bead."""
    outcome = _resolve_assess_outcome(
        _satisfied_assessment(),
        FlagBlockedTool(),
        structural_bead=False,
        verify_steps=(VerifyStep(cmd="node --check game.js"),),
    )
    assert outcome.kind == "assessment_satisfied_pending_verify"
    # The assessment payload carries through for the handoff render.
    assert "drawPedestrian" in outcome.payload["current_state"]


def test_assess_behavioral_satisfied_routes_to_verify_with_carried_test(
    tmp_path: Path,
) -> None:
    """A real carried test (non-degenerate, script present) also
    qualifies as an arbitration gate."""
    script = tmp_path / "test_splat.js"
    script.write_text("require('./game.js');\nprocess.exit(1);\n")
    outcome = _resolve_assess_outcome(
        _satisfied_assessment(),
        FlagBlockedTool(),
        structural_bead=False,
        prior_test_cmd="node test_splat.js",
        workspace=tmp_path,
    )
    assert outcome.kind == "assessment_satisfied_pending_verify"


def test_assess_behavioral_satisfied_ignored_without_any_gate() -> None:
    """With no verify steps and no carried test there is nothing to
    arbitrate the claim — the flag is ignored (normal TDD path) so a
    behavioral bead can't close on zero evidence."""
    outcome = _resolve_assess_outcome(
        _satisfied_assessment(),
        FlagBlockedTool(),
        structural_bead=False,
    )
    assert outcome.kind == "assessment_submitted"


def test_assess_behavioral_satisfied_degenerate_carried_gate_does_not_qualify(
    tmp_path: Path,
) -> None:
    """A degenerate carried command (echo/exit only) is not an
    arbitration gate — its verdict is fixed, so the claim falls through
    to the normal path instead of 'verifying' against a tautology."""
    outcome = _resolve_assess_outcome(
        _satisfied_assessment(),
        FlagBlockedTool(),
        structural_bead=False,
        prior_test_cmd="echo nope && exit 1",
        workspace=tmp_path,
    )
    assert outcome.kind == "assessment_submitted"


def test_assess_structural_satisfied_still_routes_to_close() -> None:
    """The harness-1kd9t structural escape is unchanged: structural beads
    route straight to CLOSE regardless of gates."""
    outcome = _resolve_assess_outcome(
        _satisfied_assessment(),
        FlagBlockedTool(),
        structural_bead=True,
    )
    assert outcome.kind == "assessment_already_satisfied"
