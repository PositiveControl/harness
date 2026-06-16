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
    _adopt_existing_red_gate,
    _is_degenerate_test_cmd,
    _is_unrunnable_test_output,
    _lint_submitted_gate,
    _resolve_write_test_outcome,
    _test_cmd_file_missing,
    _test_cmd_script,
    _test_never_loads_source,
    _test_scaffold_crash_output,
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


def _write_py_test(path: Path, *, red: bool, loads_source: bool) -> None:
    """Materialize a runnable python test file. `red` → exits non-zero on an
    AssertionError; `loads_source` → contains an `import` marker so the
    never-loads-source lint accepts it."""
    lines = []
    if loads_source:
        lines.append("import os  # source-load marker")
    lines.append("assert not " + ("True" if red else "False") + ', "gap"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_adopts_existing_red_gate_matching_subject(tmp_path: Path) -> None:
    """harness-1ttpl: no submit/skip, no carried gate, but a red on-disk test
    whose name overlaps the bead subject IS the gate — adopt it rather than
    halt 'no test'."""
    _write_py_test(tmp_path / "test_cruise_behavior.py", red=True, loads_source=True)
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=None,
        issue_title="NPC cruise acceleration toward maxSpeed",
    )
    assert outcome.kind == "failing_test_submitted"
    assert outcome.payload["test_path"] == "test_cruise_behavior.py"
    assert outcome.payload["test_cmd"] == "python3 test_cruise_behavior.py"


def test_no_adopt_when_no_name_overlap(tmp_path: Path) -> None:
    """A red test whose name shares no domain token with the bead subject is
    NOT this gate — halt rather than adopt an unrelated gate."""
    _write_py_test(tmp_path / "test_widget_layout.py", red=True, loads_source=True)
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=None,
        issue_title="NPC cruise acceleration toward maxSpeed",
    )
    assert outcome.kind == "phase_no_progress"


def test_no_adopt_green_existing_test(tmp_path: Path) -> None:
    """A name-matched but GREEN test is too weak a signal to adopt as the
    gate — closing the bead on it would skip the work. Halt instead."""
    _write_py_test(tmp_path / "test_cruise_behavior.py", red=False, loads_source=True)
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=None,
        issue_title="NPC cruise acceleration toward maxSpeed",
    )
    assert outcome.kind == "phase_no_progress"


def test_no_adopt_test_never_loads_source(tmp_path: Path) -> None:
    """A name-matched red test that never loads any source artifact is red
    forever (asserts on its own scope) — the submit-time lint rejects it, so
    adoption must too."""
    _write_py_test(tmp_path / "test_cruise_behavior.py", red=True, loads_source=False)
    outcome = _resolve_write_test_outcome(
        SubmitFailingTestTool(),
        SkipTestPhaseTool(),
        workspace=tmp_path,
        prior_test_cmd=None,
        issue_title="NPC cruise acceleration toward maxSpeed",
    )
    assert outcome.kind == "phase_no_progress"


def test_adopt_helper_picks_best_overlap(tmp_path: Path) -> None:
    """When several red tests match, the one with the most subject-token
    overlap wins (deterministic selection)."""
    _write_py_test(tmp_path / "test_cruise.py", red=True, loads_source=True)
    _write_py_test(tmp_path / "test_cruise_npc_acceleration.py", red=True, loads_source=True)
    outcome = _adopt_existing_red_gate(tmp_path, "NPC cruise acceleration toward maxSpeed")
    assert outcome is not None
    assert outcome.payload["test_path"] == "test_cruise_npc_acceleration.py"


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


# --- always-red gate lint (loop_run=dae002aa) ----------------------


def test_never_loads_source_rejects_scope_only_js_test(tmp_path: Path) -> None:
    """loop_run=dae002aa (harness-8i9): a test that checks
    `typeof drawPedestrian` in its own empty scope — no require, no
    readFileSync, no subprocess — is implementation-insensitive and must
    be rejected with a message that names the fix (load the source)."""
    test_file = tmp_path / "test_pedestrian_splat_missing.js"
    test_file.write_text(
        "// asserts only on its own scope\n"
        "if (typeof drawPedestrian === 'undefined') {\n"
        "  console.log('FAIL: drawPedestrian is undefined');\n"
        "  process.exitCode = 1;\n"
        "}\n"
    )
    msg = _test_never_loads_source(tmp_path, "test_pedestrian_splat_missing.js")
    assert msg is not None
    assert "never loads any source artifact" in msg


def test_never_loads_source_accepts_require(tmp_path: Path) -> None:
    test_file = tmp_path / "test_drawtile.js"
    test_file.write_text("const g = require('./game.js');\nprocess.exit(1);\n")
    assert _test_never_loads_source(tmp_path, "test_drawtile.js") is None


def test_never_loads_source_accepts_readfilesync_eval(tmp_path: Path) -> None:
    test_file = tmp_path / "test_drawtile.js"
    test_file.write_text("const fs = require('fs');\neval(fs.readFileSync('game.js', 'utf8'));\n")
    assert _test_never_loads_source(tmp_path, "test_drawtile.js") is None


def test_never_loads_source_accepts_python_import(tmp_path: Path) -> None:
    test_file = tmp_path / "test_game.py"
    test_file.write_text("import game\nassert game.draw_tile\n")
    assert _test_never_loads_source(tmp_path, "test_game.py") is None


def test_never_loads_source_conservative_on_missing_or_unknown(tmp_path: Path) -> None:
    """Missing file / unrecognized suffix → None (other guards own those);
    the lint must never false-drop what it can't read."""
    assert _test_never_loads_source(tmp_path, "test_absent.js") is None
    sh = tmp_path / "test_gate.sh"
    sh.write_text("exit 1\n")
    assert _test_never_loads_source(tmp_path, "test_gate.sh") is None


def test_unrunnable_signatures_cover_browser_global_load_crash() -> None:
    """loop_run=dae002aa (harness-uy4): `document is not defined` from a
    bare Node run of browser JS is a LOAD failure — red forever. The
    legitimate red `drawTile is not defined` (the gap itself) must NOT
    match: only the named browser globals classify as unrunnable."""
    assert _is_unrunnable_test_output(
        1, "ReferenceError: document is not defined\n    at eval (game.js:5)"
    )
    assert _is_unrunnable_test_output(1, "ReferenceError: window is not defined")
    assert not _is_unrunnable_test_output(1, "ReferenceError: drawTile is not defined")
    # loop_run=069d6172 (harness-xdgtb): the rAF game-loop load crash — same
    # browser-global class, a different name. Now rejected at submit.
    assert _is_unrunnable_test_output(
        1, "ReferenceError: requestAnimationFrame is not defined\n    at eval (<anonymous>:42)"
    )
    assert _is_unrunnable_test_output(1, "ReferenceError: localStorage is not defined")
    # Still conservative: a deliverable symbol absence is the gap, not a load crash.
    assert not _is_unrunnable_test_output(1, "ReferenceError: spawnPed is not defined")


def test_lint_submitted_gate_rejects_unrunnable_and_mockless(tmp_path: Path) -> None:
    """The combined submit-time lint: load-crash output → rejection;
    never-loads-source file → rejection; a real red gate → None."""
    crash = _lint_submitted_gate(
        tmp_path,
        "test_drawtile.js",
        "node test_drawtile.js",
        1,
        "ReferenceError: document is not defined",
    )
    assert crash is not None
    assert "could not LOAD" in crash

    scope_only = tmp_path / "test_scope.js"
    scope_only.write_text("if (typeof f === 'undefined') process.exitCode = 1;\n")
    mockless = _lint_submitted_gate(
        tmp_path, "test_scope.js", "node test_scope.js", 1, "FAIL: f undefined"
    )
    assert mockless is not None
    assert "never loads any source artifact" in mockless

    real = tmp_path / "test_real.js"
    real.write_text("require('./game.js');\nprocess.exit(1);\n")
    assert (
        _lint_submitted_gate(
            tmp_path, "test_real.js", "node test_real.js", 1, "AssertionError: no splat"
        )
        is None
    )


# --- crash-shaped red gate lint (harness-c6jqy) --------------------

# loop_run=df358902 (harness-491j5): eval-the-source idiom leaves
# global.player undefined, so the test's own setup throws before any
# assertion. Top frame is the test file.
_SCAFFOLD_CRASH_TAIL = (
    "=== Handbrake and Brake Test ===\n"
    "TypeError: Cannot set properties of undefined (setting 'speed')\n"
    "    at Object.<anonymous> (/ws/scratch/gta_r2/test_handbrake_and_collision.js:25:20)\n"
    "    at Module._compile (node:internal/modules/cjs/loader:1234:14)\n"
    "    at node:internal/main/run_main_module:28:49\n"
)


def test_scaffold_crash_rejects_property_of_undefined_in_test_file() -> None:
    """harness-491j5 regression: a TypeError reading/setting a property of
    undefined whose top frame is the test file is broken scaffolding, not
    a gap — rejected with an actionable harness-fix hint."""
    msg = _test_scaffold_crash_output("test_handbrake_and_collision.js", _SCAFFOLD_CRASH_TAIL)
    assert msg is not None
    assert "scaffolding" in msg
    assert "eval" in msg.lower()


def test_scaffold_crash_matches_read_and_python_frames() -> None:
    """The `read properties` variant and a Python traceback frame both
    classify — the fingerprint is reference-is-undefined from the test."""
    js_read = _test_scaffold_crash_output(
        "test_x.js",
        "TypeError: Cannot read properties of undefined (reading 'x')\n    at /ws/test_x.js:12:9\n",
    )
    assert js_read is not None
    py = _test_scaffold_crash_output(
        "test_x.py",
        "TypeError: Cannot read properties of undefined\n"
        '  File "/ws/test_x.py", line 12, in <module>\n',
    )
    assert py is not None


def test_scaffold_crash_ignores_missing_deliverable_crash() -> None:
    """`X is not a function` / `X is not defined` name the deliverable the
    bead must build — the genuine gap. Must NOT be rejected."""
    not_a_fn = _test_scaffold_crash_output(
        "test_spawn.js",
        "TypeError: game.spawnPed is not a function\n    at /ws/test_spawn.js:8:6\n",
    )
    assert not_a_fn is None
    not_defined = _test_scaffold_crash_output(
        "test_spawn.js",
        "ReferenceError: drawTile is not defined\n    at /ws/test_spawn.js:8:6\n",
    )
    assert not_defined is None


def test_scaffold_crash_ignores_crash_outside_test_file() -> None:
    """A property-of-undefined TypeError whose only frame is inside the
    source under test (not the test file) is a source-side error, not a
    harness crash — left to other guards."""
    source_side = _test_scaffold_crash_output(
        "test_x.js",
        "TypeError: Cannot read properties of undefined (reading 'y')\n"
        "    at update (/ws/game.js:140:3)\n"
        "    at [eval]:9:1\n",
    )
    assert source_side is None


def test_scaffold_crash_ignores_assertion_failure() -> None:
    """A genuine assertion failure (AssertionError / FAIL, no TypeError)
    passes the check unflagged."""
    assert (
        _test_scaffold_crash_output(
            "test_x.js", "FAIL: handbrake does not reduce speed\nexpected 0 got 12\n"
        )
        is None
    )
    assert _test_scaffold_crash_output("test_x.js", "AssertionError: expected 8 got 7") is None


def test_lint_submitted_gate_rejects_scaffold_crash(tmp_path: Path) -> None:
    """End-to-end through the combined lint: the df358902 crash output is
    rejected even though the test file loads the source (so the
    never-loads guard would pass it)."""
    test_file = tmp_path / "test_handbrake_and_collision.js"
    test_file.write_text(
        "const fs = require('fs');\neval(fs.readFileSync('game.js','utf8'));\n"
        "global.player.speed = 0;\n"
    )
    msg = _lint_submitted_gate(
        tmp_path,
        "test_handbrake_and_collision.js",
        "node test_handbrake_and_collision.js",
        1,
        _SCAFFOLD_CRASH_TAIL,
    )
    assert msg is not None
    assert "scaffolding" in msg
