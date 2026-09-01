"""Turn-level FSM ceilings — harness-hs50i (loop_run=d45fd2f7).

Run d45fd2f7 turn 3 ran 2h50m without terminating: IMPLEMENT exited to
VERIFY (submit_implementation_complete.latest() stays non-None once
called, so every later IMPLEMENT pass auto-resolves implement_complete),
verify failed, and `verify->implement (retry)` looped — 114
wrap_up_forced events and 1,179 rounds inside one turn. The per-phase
round budgets bound each tool loop; nothing bounded the cycle.

These tests pin the two new ceilings:

  1. _MAX_VERIFY_RETRIES halts the turn after N verify_failed outcomes,
     handing control back to the loop's attempt/park machinery.
  2. _MAX_PHASE_EXECUTIONS bounds total phase runs per turn so ANY
     transition-table cycle degrades to a halted turn, not an unbounded
     one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.driver.fsm_executor import (
    _MAX_PHASE_EXECUTIONS,
    _MAX_VERIFY_RETRIES,
    run_fsm_turn,
)
from harness.driver.handoff import Handoff
from harness.driver.planner import VerifyStep
from harness.driver.turn_fsm import TurnPhase
from harness.model.adapter import ChatMessage
from harness.orchestrator import ToolLoopResult


class _FakeCharacter:
    def system_prompt(
        self,
        *,
        include_samples: Any = (),
        include_style_rules: bool = True,
    ) -> str:
        return "you are an executor."


class _FakeBd:
    """CLOSE-phase resolver surface; never reached — every scripted
    turn halts before CLOSE."""

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


def _verify_cycle_tool_loop(phases_seen: list[str]) -> Any:
    """Scripted run_tool_loop reproducing the d45fd2f7 cycle: ASSESS
    submits, WRITE_TEST skips, IMPLEMENT submits completion once (the
    capture persists, so later IMPLEMENT passes auto-resolve
    implement_complete without any new calls), and the caller supplies
    a failing VerifyStep so VERIFY always emits verify_failed."""

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        names = registry.names()
        if "submit_assessment" in names:
            phases_seen.append("assess")
            registry.call(
                "submit_assessment",
                {"current_state": "c" * 30, "gap": "g" * 30, "approach": "a" * 30},
            )
        elif "skip_test_phase" in names:
            phases_seen.append("write_test")
            registry.call("skip_test_phase", {"reason": "no test runner in this workspace"})
        elif "submit_implementation_complete" in names:
            phases_seen.append("implement")
            registry.call(
                "submit_implementation_complete",
                {"summary": "implemented the thing (scripted)"},
            )
        else:
            phases_seen.append("verify_or_other")
        content = "ok."
        return ToolLoopResult(
            content=content,
            messages=[ChatMessage(role="assistant", content=content)],
            rounds=1,
            events=[],
        )

    return fake_run_tool_loop


def test_verify_retry_ceiling_halts_the_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """IMPLEMENT↔VERIFY with an always-failing verify step halts after
    _MAX_VERIFY_RETRIES failed verifies instead of looping forever."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop", _verify_cycle_tool_loop(phases_seen)
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]  # never reached; run_tool_loop is stubbed
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_FakeBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        verify_steps=[VerifyStep(cmd="exit 1")],
    )

    assert not result.succeeded
    assert result.final_phase == TurnPhase.HALTED
    assert "verify-retry ceiling" in result.reason
    # One IMPLEMENT pass per failed verify, capped — not unbounded.
    assert phases_seen.count("implement") == _MAX_VERIFY_RETRIES


def test_verify_retry_ceiling_does_not_fire_on_eventual_green(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A verify step that goes green before the ceiling lets the turn
    proceed to CLOSE — the cap only catches non-convergence."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop", _verify_cycle_tool_loop(phases_seen)
    )
    # Fails on the first run (marker file absent), passes on the second.
    marker = tmp_path / "green.marker"
    step = VerifyStep(cmd=f"test -f {marker} || {{ touch {marker}; exit 1; }}")

    class _CloseBd:
        """CLOSE resolver checks the post-condition via bd.show — report
        the issue closed so the turn can reach DONE."""

        def show(self, issue_id: str) -> Any:
            return type("Issue", (), {"status": "closed"})()

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_CloseBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        verify_steps=[step],
    )

    # verify failed once (< ceiling), then went green and closed.
    assert "verify-retry ceiling" not in result.reason
    assert phases_seen.count("implement") == 2


def test_phase_execution_ceiling_bounds_any_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the verify-retry cap out of the way (patched high), the
    phase-execution ceiling still bounds the turn — the backstop for
    any future transition-table cycle."""
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop", _verify_cycle_tool_loop(phases_seen)
    )
    monkeypatch.setattr("harness.driver.fsm_executor._MAX_VERIFY_RETRIES", 999)

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_FakeBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        verify_steps=[VerifyStep(cmd="exit 1")],
    )

    assert not result.succeeded
    assert result.final_phase == TurnPhase.HALTED
    assert "phase-execution ceiling" in result.reason
    assert len(phases_seen) == _MAX_PHASE_EXECUTIONS
