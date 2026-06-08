"""Tests for the turn-phase meta-tools — harness-kbnl.

The meta-tools are the model's vocabulary for signaling FSM
transitions. Each is a stateful dataclass — its `captured` list
holds what landed during a phase, which the driver inspects to
construct the PhaseOutcome event. These tests pin the validation
+ capture semantics.
"""

from __future__ import annotations

import pytest

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
    }


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
