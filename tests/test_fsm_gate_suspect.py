"""Always-red-gate defenses at the turn level — loop_run=dae002aa.

Run dae002aa parked all three of its beads with the work already done in
the workspace: every WRITE_TEST gate was red for a reason decoupled from
the implementation (a scope-only test that never loads the source, or a
bare-Node run of browser JS crashing on `document`), so VERIFY burned its
retry ceiling and the attempt budget against tests that could never go
green. These tests pin the two turn-level defenses:

  1. Gate-suspect detection: a test step that fails byte-identically
     across IMPLEMENT passes that touched the source halts the turn,
     marks `gate_suspect`, and drops the carried test_cmd so the next
     attempt re-authors instead of reusing the broken gate.
  2. Behavioral already_satisfied → VERIFY arbitration: a behavioral
     bead claiming "no gap" is arbitrated by the carried test + verify
     steps (green → CLOSE, red → IMPLEMENT) instead of being forced to
     manufacture a failing test against a done bead.

Plus the WRITE_TEST browser-JS hint: browser-authored workspaces get the
stub-the-DOM recipe appended to the WRITE_TEST user prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.driver.fsm_executor import _exec_test_cmd_capture, run_fsm_turn
from harness.driver.handoff import Handoff
from harness.driver.planner import VerifyStep
from harness.driver.turn_fsm import TurnPhase
from harness.model.adapter import ChatMessage
from harness.orchestrator import ToolLoopResult
from harness.orchestrator.tool_loop import ToolLoopEvent
from harness.tools.base import ToolCall, ToolResult


class _FakeCharacter:
    def system_prompt(
        self,
        *,
        include_samples: Any = (),
        include_style_rules: bool = True,
    ) -> str:
        return "you are an executor."


class _OpenBd:
    """CLOSE never reached in the halting scenarios."""

    def show(self, issue_id: str) -> Any:  # pragma: no cover - not reached
        raise AssertionError("bd.show should not be called")


class _ClosedBd:
    """CLOSE resolver post-condition: report the issue closed."""

    def show(self, issue_id: str) -> Any:
        return type("Issue", (), {"status": "closed"})()


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


def _write_event() -> ToolLoopEvent:
    """A successful edit_file tool_call_end — what the relay sniffs to
    mark the IMPLEMENT pass as having touched the source."""
    return ToolLoopEvent(
        kind="tool_call_end",
        call=ToolCall(name="edit_file", arguments={"path": "game.js"}),
        result=ToolResult(tool_name="edit_file", output="ok", success=True),
    )


def _scripted_tool_loop(
    phases_seen: list[str],
    user_prompts: dict[str, str],
    *,
    assessment_kwargs: dict[str, Any] | None = None,
    write_test_action: str = "barren",
    implement_edits: bool = True,
) -> Any:
    """Scripted run_tool_loop:

    - ASSESS: submit_assessment (extra fields via assessment_kwargs).
    - WRITE_TEST: 'barren' leaves no capture (the carried prior_test_cmd
      is reused); 'skip' calls skip_test_phase.
    - IMPLEMENT: submit_implementation_complete + (when implement_edits)
      a successful edit_file event via the observe relay, so the pass
      counts as touching source. With implement_edits=False the pass
      submits complete WITHOUT any write — the harness-smplj
      complete-without-edit shape.
    - VERIFY / CLOSE: no-op reply.

    Records the per-phase user prompt so tests can assert on hints.
    """

    def fake_run_tool_loop(
        _adapter: Any, messages: Any, registry: Any, **kwargs: Any
    ) -> ToolLoopResult:
        names = registry.names()
        user_text = messages[1].content
        if "submit_assessment" in names:
            phases_seen.append("assess")
            user_prompts["assess"] = user_text
            registry.call(
                "submit_assessment",
                {
                    "current_state": "c" * 30,
                    "gap": "g" * 30,
                    "approach": "a" * 30,
                    **(assessment_kwargs or {}),
                },
            )
        elif "submit_failing_test" in names:
            phases_seen.append("write_test")
            user_prompts["write_test"] = user_text
            if write_test_action == "skip":
                registry.call("skip_test_phase", {"reason": "no test runner in this workspace"})
        elif "submit_implementation_complete" in names:
            phases_seen.append("implement")
            user_prompts["implement"] = user_text
            observe = kwargs.get("observe")
            if observe is not None and implement_edits:
                observe(_write_event())
            registry.call(
                "submit_implementation_complete",
                {"summary": "re-applied the change (scripted)"},
            )
        else:
            phases_seen.append("verify_or_close")
        content = "ok."
        return ToolLoopResult(
            content=content,
            messages=[ChatMessage(role="assistant", content=content)],
            rounds=1,
            events=[],
        )

    return fake_run_tool_loop


def _const_red_script(workspace: Path) -> str:
    """A real (non-degenerate, runnable) test that fails with the SAME
    tail every run — the implementation-insensitive shape."""
    script = workspace / "fail_const.py"
    script.write_text("import sys\nsys.stderr.write('gap: drawTile missing')\nsys.exit(1)\n")
    return "python3 fail_const.py"


def _varying_red_script(workspace: Path) -> str:
    """A test that fails with a DIFFERENT tail every run — converging or
    not, it's observing something, so it must not be flagged suspect."""
    script = workspace / "fail_vary.py"
    script.write_text(
        "import pathlib, sys\n"
        "p = pathlib.Path('n.txt')\n"
        "n = int(p.read_text()) if p.exists() else 0\n"
        "p.write_text(str(n + 1))\n"
        "sys.stderr.write(f'fail variant {n}')\n"
        "sys.exit(1)\n"
    )
    return "python3 fail_vary.py"


def test_identical_test_failures_across_edits_mark_gate_suspect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """loop_run=dae002aa (harness-8i9): the carried test fails with the
    same tail after an IMPLEMENT pass that edited the source — the gate
    never observes the code under change. The turn halts suspect, drops
    the carried test_cmd, and does NOT burn the full verify ceiling."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]  # never reached; run_tool_loop is stubbed
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd=_const_red_script(tmp_path),
    )

    assert not result.succeeded
    assert result.final_phase == TurnPhase.HALTED
    assert "suspect verify gate" in result.reason
    assert result.gate_suspect
    assert result.last_test_cmd is None
    # Two IMPLEMENT passes were enough — identical tail + touched source.
    assert phases_seen.count("implement") == 2


def test_varying_test_failures_are_not_suspect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A gate whose failure output changes run-to-run is observing the
    workspace — it must reach the ordinary verify-retry ceiling with the
    carried test preserved, not be dropped as suspect."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}),
    )

    cmd = _varying_red_script(tmp_path)
    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd=cmd,
    )

    assert not result.succeeded
    assert "verify-retry ceiling" in result.reason
    assert not result.gate_suspect
    assert result.last_test_cmd == cmd


def test_behavioral_satisfied_green_gate_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """loop_run=dae002aa: a behavioral bead claiming already_satisfied
    with a green carried gate routes ASSESS → VERIFY → CLOSE — no forced
    WRITE_TEST, no manufactured always-red test."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}, assessment_kwargs={"already_satisfied": True}),
    )
    green = tmp_path / "pass_now.py"
    green.write_text("import sys\nsys.exit(0)\n")

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_ClosedBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd="python3 pass_now.py",
        structural_bead=False,
    )

    assert result.succeeded
    assert result.final_phase == TurnPhase.DONE
    assert "write_test" not in phases_seen
    assert "implement" not in phases_seen
    assert any(
        name == "assess->verify (claimed satisfied; verify arbitrates)"
        for _src, _dst, name in result.phase_trace
    )


def test_behavioral_satisfied_red_gate_routes_to_implement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wrong already_satisfied claim fails verification and lands in
    IMPLEMENT — the claim is arbitrated, never trusted."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}, assessment_kwargs={"already_satisfied": True}),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        verify_steps=[VerifyStep(cmd="exit 1")],
        structural_bead=False,
    )

    assert not result.succeeded
    assert "write_test" not in phases_seen
    assert "implement" in phases_seen


def test_write_test_prompt_carries_browser_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A browser-authored workspace appends the stub-the-DOM recipe to
    the WRITE_TEST user prompt; a plain workspace does not."""
    (tmp_path / "game.js").write_text("const canvas = document.getElementById('gameCanvas');\n")
    phases_seen: list[str] = []
    prompts: dict[str, str] = {}
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, prompts, write_test_action="skip"),
    )

    run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_ClosedBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
    )
    assert "WORKSPACE NOTE — browser JS" in prompts["write_test"]
    assert "process.exit(1)" in prompts["write_test"]
    # harness-vsv: the hint steers physics/behavioral gates to source-text
    # regex (runtime state is empty headless) and authorizes skip_test_phase
    # when neither a runtime nor a source gate fits.
    assert "RUNTIME STATE IS OFTEN EMPTY HEADLESS" in prompts["write_test"]
    assert "skip_test_phase" in prompts["write_test"]
    # The hint is WRITE_TEST-scoped, not sprayed across phases.
    assert "WORKSPACE NOTE" not in prompts["assess"]


def test_write_test_prompt_clean_without_browser_js(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "lib.py").write_text("def f() -> int:\n    return 1\n")
    phases_seen: list[str] = []
    prompts: dict[str, str] = {}
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, prompts, write_test_action="skip"),
    )

    run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_ClosedBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
    )
    assert "WORKSPACE NOTE" not in prompts["write_test"]


# --- harness-smplj: complete-without-edit + cross-turn seeding ------


def test_complete_without_edit_identical_tail_is_suspect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-smplj hole 1 (loop_run=df358902, harness-491j5): the model
    diagnosed 'test env broken, implementation present' and re-submitted
    complete WITHOUT editing. The old detector required touched_source, so
    it never armed and the hs50i ceiling parked the turn KEEPING the
    broken gate. A complete-without-edit pass followed by a byte-identical
    verify fail must now arm gate-suspect."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}, implement_edits=False),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd=_const_red_script(tmp_path),
    )

    assert result.gate_suspect
    assert "suspect verify gate" in result.reason
    assert result.last_test_cmd is None
    # Armed on the 2nd verify (identical tail) without burning the ceiling.
    assert phases_seen.count("implement") == 2


def test_cross_turn_identical_tail_trips_on_first_verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-smplj hole 2: seed the prior attempt's fail tail (as the
    loop does from state.last_test_fail_tail). A byte-identical failure on
    THIS turn's first verify trips gate-suspect immediately — the
    cross-turn signal the per-turn-local variable missed when each turn
    hit the hs50i ceiling first (df358902: 4 turns x 3 verifies, none
    detected)."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}),
    )
    cmd = _const_red_script(tmp_path)
    # The exact tail the carried gate produces — what attempt N persisted.
    _exit, prior_tail, _out = _exec_test_cmd_capture(cmd, tmp_path)

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd=cmd,
        prior_test_fail_tail=prior_tail,
    )

    assert result.gate_suspect
    assert result.last_test_cmd is None
    # Tripped on the FIRST verify — only one IMPLEMENT pass ran.
    assert phases_seen.count("implement") == 1


def test_ceiling_halt_persists_fail_tail_for_next_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-suspect ceiling halt (varying gate) returns its last fail
    tail so the loop can seed the next attempt; a suspect halt clears it
    (the gate is being dropped)."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}),
    )
    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd=_varying_red_script(tmp_path),
    )
    assert not result.gate_suspect
    assert "verify-retry ceiling" in result.reason
    assert result.last_test_fail_tail is not None  # carried to next attempt
