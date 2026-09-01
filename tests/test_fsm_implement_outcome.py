"""Unit tests for the IMPLEMENT phase outcome resolver — harness-gmu9f.

`_resolve_implement_outcome` decides what the IMPLEMENT phase produced
once its tool loop ends. The subtle case (harness-gmu9f): a re-attempt of
a bead whose prior attempt already landed the edit produces only no-op /
identical / dedup-rejected edits — writes ATTEMPTED, none SUCCEEDED. That
must route to VERIFY (let it arbitrate the artifact), NOT halt
"no writes" and burn the attempt. Only a phase with zero write attempts
(pure read-thrash) halts no-progress.
"""

from __future__ import annotations

from harness.driver.fsm_executor import _resolve_implement_outcome
from harness.tools.turn_phase_meta import SubmitImplementationCompleteTool


def test_implement_complete_when_submit_called() -> None:
    """An explicit submit_implementation_complete wins regardless of
    write state."""
    tool = SubmitImplementationCompleteTool()
    tool.call(summary="reordered grid before player")
    outcome = _resolve_implement_outcome(tool, set(), set())
    assert outcome.kind == "implement_complete"


def test_some_writes_when_a_write_succeeded() -> None:
    """A successful write without an explicit complete signal → VERIFY
    via implement_some_writes (existing behavior)."""
    tool = SubmitImplementationCompleteTool()
    outcome = _resolve_implement_outcome(
        tool, succeeded_tools={"edit_file"}, attempted_write_tools={"edit_file"}
    )
    assert outcome.kind == "implement_some_writes"


def test_writes_attempted_but_none_landed_routes_to_verify() -> None:
    """harness-gmu9f core: writes were attempted (edit_file called) but
    none succeeded — every edit was a no-op / identical / dedup-rejected.
    This is the re-attempt-of-already-edited-bead shape. Route to VERIFY
    (implement_writes_attempted), NOT a 'no writes' halt."""
    tool = SubmitImplementationCompleteTool()
    outcome = _resolve_implement_outcome(
        tool, succeeded_tools=set(), attempted_write_tools={"edit_file"}
    )
    assert outcome.kind == "implement_writes_attempted"
    assert outcome.payload["attempted_write_tools"] == ["edit_file"]


def test_no_write_attempts_halts_no_progress() -> None:
    """Pure read-thrash — the model never even tried to write — still
    halts no-progress (the original 'looking instead of writing' case)."""
    tool = SubmitImplementationCompleteTool()
    outcome = _resolve_implement_outcome(
        tool, succeeded_tools={"read_file", "grep"}, attempted_write_tools=set()
    )
    assert outcome.kind == "phase_no_progress"
