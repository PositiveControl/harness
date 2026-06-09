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

from harness.driver.fsm_turn import _is_degenerate_test_cmd, _resolve_write_test_outcome
from harness.tools.turn_phase_meta import SkipTestPhaseTool, SubmitFailingTestTool

# Non-degenerate test doubles: they invoke an interpreter (python3), so the
# harness-k6du2 guard treats them as real tests — exit code stands in for
# red/green without tripping the no-op-gate detector.
_RED_DOUBLE = 'python3 -c "import sys; sys.exit(1)"'
_GREEN_DOUBLE = 'python3 -c "import sys; sys.exit(0)"'


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
        prior_test_cmd=_RED_DOUBLE,  # exits non-zero → still red
    )
    assert outcome.kind == "failing_test_submitted"
    assert outcome.payload["test_cmd"] == _RED_DOUBLE


def test_barren_reattempt_carried_test_now_green_closes(tmp_path: Path) -> None:
    """When the carried test now PASSES, the implementation already
    landed — short-circuit to CLOSE via test_already_passes rather than
    re-driving IMPLEMENT."""
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=_GREEN_DOUBLE,  # exits zero → green
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
        prior_test_cmd=_RED_DOUBLE,
    )
    assert outcome.kind == "test_phase_skipped"


def test_degenerate_submitted_gate_halts_no_progress(tmp_path: Path) -> None:
    """harness-k6du2: a submitted gate that runs no test (echo + exit 1)
    is always red for a reason decoupled from the code. Reject it BEFORE
    the red-check so VERIFY never burns its retry budget on the tautology;
    halt WRITE_TEST so the next attempt can author a real test. This is the
    drive gta_r2 / harness-491j5 failure mode."""
    submit = SubmitFailingTestTool()
    submit.call(
        test_path="(none)",
        test_cmd='echo "GAP CONFIRMED: handbrake missing" && exit 1',
        failure_output="GAP CONFIRMED: handbrake missing",
    )
    outcome = _resolve_write_test_outcome(
        submit,
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=None,
    )
    assert outcome.kind == "phase_no_progress"
    assert "degenerate verify gate" in outcome.detail


def test_degenerate_carried_gate_not_reused(tmp_path: Path) -> None:
    """A degenerate test_cmd persisted by a pre-fix run must not be reused
    on a barren re-attempt — fall through to the legit 'no test' halt."""
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd="cd /tmp && echo nope && exit 1",
    )
    assert outcome.kind == "phase_no_progress"


def test_is_degenerate_test_cmd_classification() -> None:
    """Direct coverage of the no-op-gate detector."""
    # Degenerate: every segment is a shell no-op.
    assert _is_degenerate_test_cmd('echo "x" && exit 1')
    assert _is_degenerate_test_cmd("cd /tmp && echo hi && exit 1")
    assert _is_degenerate_test_cmd("exit 1")
    assert _is_degenerate_test_cmd("true")
    assert _is_degenerate_test_cmd("false")
    assert _is_degenerate_test_cmd("cd repo && printf done")
    # Real gates: at least one segment runs a test / interpreter / assertion.
    assert not _is_degenerate_test_cmd("pytest tests/test_x.py::test_y -v")
    assert not _is_degenerate_test_cmd("cd repo && python3 test_handbrake.py")
    assert not _is_degenerate_test_cmd('python3 -c "import sys; sys.exit(1)"')
    assert not _is_degenerate_test_cmd("node game.test.js")
    assert not _is_degenerate_test_cmd("cd repo && npm test && echo ok")
    # `test`/`[` assert (file/string predicates) — a legitimate weak gate.
    assert not _is_degenerate_test_cmd("test -f output.txt")
    assert not _is_degenerate_test_cmd("[ -f output.txt ]")
    # Empty / whitespace is not degenerate (no segment to run).
    assert not _is_degenerate_test_cmd("")
    assert not _is_degenerate_test_cmd("   ")
