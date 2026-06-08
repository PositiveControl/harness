"""Unit tests for the WRITE_TEST phase outcome resolver — harness-axjt8.

The subtle case: a re-attempt whose WRITE_TEST captured neither a
submit_failing_test nor a skip_test_phase must NOT immediately halt
"no test" if a prior attempt already established a test (carried forward
as prior_test_cmd). It reuses that test — green → CLOSE, still-red →
IMPLEMENT — instead of burning the attempt. Only a genuine first attempt
with no carried test halts no-progress.
"""

from __future__ import annotations

from pathlib import Path

from harness.driver.fsm_turn import _resolve_write_test_outcome
from harness.tools.turn_phase_meta import SkipTestPhaseTool, SubmitFailingTestTool


def test_no_submit_no_skip_no_prior_halts_no_progress(tmp_path: Path) -> None:
    """First attempt, nothing captured, no carried test → legit
    'no test' halt (unchanged behavior)."""
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=None,
    )
    assert outcome.kind == "phase_no_progress"


def test_barren_reattempt_reuses_carried_red_test(tmp_path: Path) -> None:
    """harness-axjt8 core: no submit/skip this attempt, but a prior
    attempt left a still-red test → reuse it (failing_test_submitted),
    route to IMPLEMENT, NOT a 'no test' halt."""
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd="false",  # exits non-zero → still red
    )
    assert outcome.kind == "failing_test_submitted"
    assert outcome.payload["test_cmd"] == "false"


def test_barren_reattempt_carried_test_now_green_closes(tmp_path: Path) -> None:
    """When the carried test now PASSES, the implementation already
    landed — short-circuit to CLOSE via test_already_passes rather than
    re-driving IMPLEMENT."""
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd="true",  # exits zero → green
    )
    assert outcome.kind == "test_already_passes"


def test_skip_still_wins_over_carried_test(tmp_path: Path) -> None:
    """An explicit skip_test_phase this attempt still takes precedence —
    the carry-forward only fires when nothing was captured."""
    skip = SkipTestPhaseTool()
    skip.call(reason="driving physics does not admit a clean unit test")
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        skip,
        workspace=tmp_path,
        prior_test_cmd="false",
    )
    assert outcome.kind == "test_phase_skipped"
