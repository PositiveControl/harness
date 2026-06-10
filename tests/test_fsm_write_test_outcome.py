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
from typing import Any

import pytest

from harness.driver.fsm_turn import (
    _is_degenerate_test_cmd,
    _is_unrunnable_test_output,
    _resolve_write_test_outcome,
    _test_cmd_file_missing,
    _test_cmd_script,
    run_fsm_turn,
)
from harness.driver.handoff import Handoff
from harness.driver.turn_fsm import TurnPhase
from harness.model.adapter import ChatMessage
from harness.orchestrator import ToolLoopResult
from harness.tools.turn_phase_meta import SkipTestPhaseTool, SubmitFailingTestTool


class _FakeCharacter:
    def system_prompt(
        self,
        *,
        include_samples: Any = (),
        include_style_rules: bool = True,
    ) -> str:
        return "you are an executor."


class _FakeBd:
    """CLOSE-phase resolver surface; never reached — the scripted turn
    halts before CLOSE."""

    def show(self, issue_id: str) -> Any:  # pragma: no cover - not reached
        raise AssertionError("bd.show should not be called")


def _handoff(phase: TurnPhase, assessment: Any, test_cmd: Any) -> Handoff:
    return Handoff(
        loop_run_id="testrun1",
        epic_id="harness-e9oq",
        current_issue="harness-x (P2 task)\nTitle: do the thing",
        parent_epic_summary=None,
        files_touched=(),
        closed_this_run=(),
        decisions=(),
        observations=(),
        open_questions=(),
        prior_attempt_failure=None,
        phase=phase.value,
        prior_assessment=assessment,
        prior_test_cmd=test_cmd,
    )


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


def test_skip_honored_when_no_gate_exists(tmp_path: Path) -> None:
    """A skip with NO established gate (no fresh submit, no carried test) is
    a legitimate escape — honored."""
    skip = SkipTestPhaseTool()
    skip.call(reason="driving physics does not admit a clean unit test")
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        skip,
        workspace=tmp_path,
        prior_test_cmd=None,
    )
    assert outcome.kind == "test_phase_skipped"


def test_skip_ignored_when_real_carried_gate_exists(tmp_path: Path) -> None:
    """harness-15eeq: once a prior attempt established a real failing gate,
    skip_test_phase is the model trying to disown it to dodge the work. The
    skip is NOT honored — the carried gate is reused instead (the rxtpz
    dodge: 'the test is incorrectly written, skip it')."""
    skip = SkipTestPhaseTool()
    skip.call(reason="the test file is incorrectly written, skip it")
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        skip,
        workspace=tmp_path,
        prior_test_cmd=_RED_DOUBLE,  # a real, runnable, still-red gate
    )
    assert outcome.kind == "failing_test_submitted"


def test_skip_ignored_when_fresh_submit_present(tmp_path: Path) -> None:
    """A fresh submit_failing_test this attempt also overrides a skip — a
    real gate beats a dodge."""
    skip = SkipTestPhaseTool()
    skip.call(reason="actually let's skip this test phase entirely")
    submit = SubmitFailingTestTool()
    submit.call(
        test_path="(none)",
        test_cmd=_RED_DOUBLE,
        failure_output="real failure output proving the gap exists here",
    )
    outcome = _resolve_write_test_outcome(submit, skip, workspace=tmp_path, prior_test_cmd=None)
    assert outcome.kind == "failing_test_submitted"


def test_fresh_submit_with_red_check_proof_skips_reexecution(tmp_path: Path) -> None:
    """harness-pfr5a: a submission that carries its red_check proof is
    NOT re-executed at phase end. The test_cmd here is green if run —
    a re-execution would yield test_already_passes; trusting the stored
    proof yields failing_test_submitted."""
    submit = SubmitFailingTestTool(red_check=lambda _cmd: (1, "AssertionError: gap exists"))
    submit.call(
        test_path="tests/test_gap.py",
        test_cmd=_GREEN_DOUBLE,  # would be green if re-run
        failure_output="AssertionError: gap exists in the implementation",
    )
    outcome = _resolve_write_test_outcome(
        submit,
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=None,
    )
    assert outcome.kind == "failing_test_submitted"


def test_fresh_submit_unrunnable_classified_from_stored_proof(tmp_path: Path) -> None:
    """The unrunnable classification (harness-75tto) still applies to the
    stored red_check tail — a phantom red (test can't even load) halts
    WRITE_TEST rather than burning VERIFY's budget."""
    submit = SubmitFailingTestTool(
        red_check=lambda _cmd: (1, "ModuleNotFoundError: No module named 'game'")
    )
    submit.call(
        test_path="tests/test_gap.py",
        test_cmd=_RED_DOUBLE,
        failure_output="ModuleNotFoundError: No module named 'game'",
    )
    outcome = _resolve_write_test_outcome(
        submit,
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=None,
    )
    assert outcome.kind == "phase_no_progress"
    assert "could not run the test" in outcome.detail


def test_skip_honored_when_carried_gate_is_degenerate(tmp_path: Path) -> None:
    """A degenerate carried 'gate' isn't a real gate, so a skip is still a
    legitimate escape (the model isn't disowning anything of value)."""
    skip = SkipTestPhaseTool()
    skip.call(reason="no clean unit test for this UI tweak")
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        skip,
        workspace=tmp_path,
        prior_test_cmd="echo nope && exit 1",  # degenerate
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


# --- harness-pfr5a: submit-time red_check wiring -------------------


def test_run_fsm_turn_rejects_always_green_submission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-pfr5a wiring test: run_fsm_turn constructs the
    SubmitFailingTestTool with a workspace-scoped red_check, so a
    submission whose test_cmd exits 0 comes back as a tool ERROR (the
    always-green class — loop_run=ca3c96b4: harness-8i9 closed with no
    implementation because the phase-end green short-circuit read the
    always-green test as 'already landed'). Nothing is captured, so the
    phase resolves no-progress instead of green_test_outcome → CLOSE."""
    submit_results: list[Any] = []

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        names = registry.names()
        if "submit_assessment" in names:
            registry.call(
                "submit_assessment",
                {"current_state": "c" * 30, "gap": "g" * 30, "approach": "a" * 30},
            )
        elif "submit_failing_test" in names:
            submit_results.append(
                registry.call(
                    "submit_failing_test",
                    {
                        "test_path": "test_splat.js",
                        "test_cmd": _GREEN_DOUBLE,
                        "failure_output": "FAIL: drawPedestrian is undefined (hand-pasted)",
                    },
                )
            )
        content = "ok."
        return ToolLoopResult(
            content=content,
            messages=[ChatMessage(role="assistant", content=content)],
            rounds=1,
            events=[],
        )

    monkeypatch.setattr("harness.driver.fsm_turn.run_tool_loop", fake_run_tool_loop)

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]  # never reached; run_tool_loop is stubbed
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_FakeBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
    )

    # The submission was rejected with the red-proof error...
    assert submit_results, "scripted WRITE_TEST never reached submit_failing_test"
    assert all(not r.success for r in submit_results)
    assert "exited 0" in (submit_results[0].error or "")
    # ...so the turn cannot close on a fake red: it halts no-progress.
    assert not result.succeeded
    assert result.final_phase == TurnPhase.HALTED
