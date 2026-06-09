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

from harness.driver.fsm_turn import (
    _is_degenerate_test_cmd,
    _is_unrunnable_test_output,
    _resolve_write_test_outcome,
    _test_cmd_file_missing,
    _test_cmd_script,
)
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


# --- missing test-script gate (harness-75tto) ---------------------


def test_submitted_gate_with_missing_script_halts(tmp_path: Path) -> None:
    """harness-75tto: a submitted gate whose test script is absent from the
    workspace is red only because the file can't be opened (exit 2), not a
    real failure. Reject -> phase_no_progress so VERIFY doesn't burn its
    retry budget on a phantom test (drive loop_run=9a1e7970, harness-rxtpz)."""
    submit = SubmitFailingTestTool()
    submit.call(
        test_path="test_runover_gap.py",
        test_cmd=f"cd {tmp_path} && python3 test_runover_gap.py",
        failure_output="FAIL: run-over detection missing",
    )
    outcome = _resolve_write_test_outcome(
        submit, SkipTestPhaseTool(), workspace=tmp_path, prior_test_cmd=None
    )
    assert outcome.kind == "phase_no_progress"
    assert "could not run the test" in outcome.detail


def test_submitted_gate_with_present_script_is_accepted(tmp_path: Path) -> None:
    """The guard must not reject a real gate: when the script exists, the
    normal red-check path runs (here the script exits 1 -> failing_test)."""
    script = tmp_path / "test_real_gap.py"
    script.write_text("import sys; print('FAIL'); sys.exit(1)\n")
    submit = SubmitFailingTestTool()
    submit.call(
        test_path="test_real_gap.py",
        test_cmd=f"cd {tmp_path} && python3 test_real_gap.py",
        failure_output="FAIL: real assertion failed, gap exists",
    )
    outcome = _resolve_write_test_outcome(
        submit, SkipTestPhaseTool(), workspace=tmp_path, prior_test_cmd=None
    )
    assert outcome.kind == "failing_test_submitted"


def test_carried_gate_with_missing_script_not_reused(tmp_path: Path) -> None:
    """A carried test_cmd whose script the inter-attempt restore deleted
    must NOT be reused — fall through to the 'no test' halt so WRITE_TEST
    re-authors instead of running a phantom."""
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=f"cd {tmp_path} && python3 test_gone.py",
    )
    assert outcome.kind == "phase_no_progress"


def test_test_cmd_script_extraction() -> None:
    """`_test_cmd_script` pulls the runnable script path; None when none."""
    assert _test_cmd_script("cd /ws && python test_x.py") == "test_x.py"
    assert _test_cmd_script("python3 path/to/test_y.py") == "path/to/test_y.py"
    assert _test_cmd_script("pytest tests/test_z.py::test_case -v") == "tests/test_z.py"
    assert _test_cmd_script("node game.test.js") == "game.test.js"
    # No script path -> None (don't false-positive into "missing").
    assert _test_cmd_script('python3 -c "import sys; sys.exit(1)"') is None
    assert _test_cmd_script("pytest") is None
    assert _test_cmd_script("echo hi && exit 1") is None


def test_test_cmd_file_missing_conservative_on_unparseable(tmp_path: Path) -> None:
    """A cmd with no determinable script is treated as present (False),
    never a false drop — only a parsed-but-absent script is missing."""
    # No script path -> not "missing".
    assert not _test_cmd_file_missing('python3 -c "import sys; sys.exit(1)"', tmp_path)
    # Parsed script, absent -> missing.
    assert _test_cmd_file_missing(f"cd {tmp_path} && python test_absent.py", tmp_path)
    # Parsed script, present -> not missing.
    (tmp_path / "test_here.py").write_text("pass\n")
    assert not _test_cmd_file_missing(f"cd {tmp_path} && python test_here.py", tmp_path)


def test_is_unrunnable_test_output_classification() -> None:
    """Distinguish 'runner couldn't load the test' from a genuine red."""
    # exit 0 is never unrunnable.
    assert not _is_unrunnable_test_output(0, "anything")
    # Load/collection failures -> unrunnable.
    assert _is_unrunnable_test_output(2, "python: can't open file '/x/test_y.py'")
    assert _is_unrunnable_test_output(2, "No such file or directory")
    assert _is_unrunnable_test_output(1, "ModuleNotFoundError: No module named 'game'")
    assert _is_unrunnable_test_output(1, "  File ...\n    SyntaxError: invalid syntax")
    assert _is_unrunnable_test_output(4, "ERROR: file or directory not found: test_z.py")
    assert _is_unrunnable_test_output(127, "python3: command not found")
    # A genuine assertion failure is a REAL red, not unrunnable.
    assert not _is_unrunnable_test_output(1, "AssertionError: expected 8 got 7")
    assert not _is_unrunnable_test_output(1, "FAIL: Run-over detection logic not found")
