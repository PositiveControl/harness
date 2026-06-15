"""Mid-turn flag_blocked deliverable validation — harness-0t2f9.

loop_run=069d6172 (harness-4s2bb, scratch/gta_r2) parked a perfectly
drivable bead: the model called flag_blocked(missing="pedestrian spawning
and wandering logic in game.js") — verbatim the bead's own §6a acceptance
criteria — and `_resolve_assess_outcome` honored the claim, routing ASSESS
straight to premise_unmet → park. A build/create bead's deliverable is
absent before the bead runs; that absence IS the task, not an upstream
block.

run_fsm_turn now wires the bead's `deliverable_text` into FlagBlockedTool:
a `missing` that restates the deliverable is rejected at call time with a
corrective tool error, so the model proceeds with ASSESS. A genuine
upstream precondition shares few deliverable tokens and still parks.

These tests pin the end-to-end behavior through run_fsm_turn (the scripted
tool loop calls the REAL phase-meta tools the driver wired, so the gate
fires for real).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.driver.fsm_turn import run_fsm_turn
from harness.driver.handoff import Handoff
from harness.driver.turn_fsm import TurnPhase
from harness.model.adapter import ChatMessage
from harness.orchestrator import ToolLoopResult

_PED_DELIVERABLE = (
    "Spawn pedestrians in game.js\n"
    "Add pedestrian spawning and wandering logic so peds appear on the "
    "sidewalk and walk around game.js.\n"
    "Acceptance: pedestrians spawn periodically in game.js and wander."
)

_FLAG_MISSING_DELIVERABLE = "pedestrian spawning and wandering logic in game.js"
_FLAG_MISSING_UPSTREAM = "drawCop() render function from cop.js — never defined"


class _FakeCharacter:
    def system_prompt(
        self,
        *,
        include_samples: Any = (),
        include_style_rules: bool = True,
    ) -> str:
        return "you are an executor."


class _FakeBd:
    def show(self, issue_id: str) -> Any:  # pragma: no cover - not reached
        raise AssertionError("bd.show should not be called")


def _handoff(phase: TurnPhase, assessment: Any, test_cmd: Any) -> Handoff:
    return Handoff(
        loop_run_id="testrun1",
        epic_id="harness-e9oq",
        current_issue="harness-x (P2 task)\nTitle: spawn pedestrians in game.js",
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


def _flag_then_recover_tool_loop(missing: str, events: list[str]) -> Any:
    """Scripted run_tool_loop: in ASSESS the model first tries
    flag_blocked with `missing`; whatever the tool returns (rejected or
    honored) it records the verdict. On rejection it recovers by calling
    submit_assessment. Later phases skip the test and submit completion."""

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        names = registry.names()
        if "flag_blocked" in names:  # ASSESS
            events.append("assess")
            result = registry.call(
                "flag_blocked",
                {"missing": missing, "reason": "no such code exists in the workspace yet"},
            )
            if result.success:
                events.append("flag_honored")
            else:
                events.append("flag_rejected")
                # Corrective note in hand — proceed with the assessment.
                registry.call(
                    "submit_assessment",
                    {
                        "current_state": "no pedestrians yet" + "." * 20,
                        "gap": "need spawning + wandering" + "." * 20,
                        "approach": "add a Ped class and spawner" + "." * 20,
                    },
                )
        elif "skip_test_phase" in names:  # WRITE_TEST
            events.append("write_test")
            registry.call("skip_test_phase", {"reason": "structural change, no unit gate here"})
        elif "submit_implementation_complete" in names:  # IMPLEMENT
            events.append("implement")
            registry.call(
                "submit_implementation_complete",
                {"summary": "added pedestrian spawning + wandering (scripted)"},
            )
        else:
            events.append("other")
        content = "ok."
        return ToolLoopResult(
            content=content,
            messages=[ChatMessage(role="assistant", content=content)],
            rounds=1,
            events=[],
        )

    return fake_run_tool_loop


class _CloseBd:
    """CLOSE resolver: report the issue closed so a recovered turn reaches DONE."""

    def show(self, issue_id: str) -> Any:
        return type("Issue", (), {"status": "closed"})()


def test_flag_naming_own_deliverable_is_rejected_and_turn_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-4s2bb shape: flag_blocked whose `missing` restates the
    bead's acceptance criteria is rejected; the model recovers via
    submit_assessment and the turn drives to DONE instead of parking."""
    events: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_turn.run_tool_loop",
        _flag_then_recover_tool_loop(_FLAG_MISSING_DELIVERABLE, events),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]  # never reached; run_tool_loop is stubbed
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_CloseBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        deliverable_text=_PED_DELIVERABLE,
        tdd_required=False,
    )

    assert "flag_rejected" in events
    assert "flag_honored" not in events
    assert result.succeeded
    assert result.final_phase == TurnPhase.DONE
    assert "premise" not in result.reason.lower()


def test_flag_naming_genuine_upstream_still_parks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A flag whose `missing` names a concrete upstream symbol (few
    deliverable tokens) is honored — the turn halts PREMISE_UNMET so the
    loop parks-and-flags for operator review."""
    events: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_turn.run_tool_loop",
        _flag_then_recover_tool_loop(_FLAG_MISSING_UPSTREAM, events),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_FakeBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        deliverable_text=_PED_DELIVERABLE,
        tdd_required=False,
    )

    assert "flag_honored" in events
    assert "flag_rejected" not in events
    assert not result.succeeded
    assert result.final_phase == TurnPhase.HALTED


def test_no_deliverable_text_disables_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without bead text (deliverable_text=""), the check is off and even
    a deliverable-naming flag is honored — preserves trust-the-model for
    callers that don't supply bead text."""
    events: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_turn.run_tool_loop",
        _flag_then_recover_tool_loop(_FLAG_MISSING_DELIVERABLE, events),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_FakeBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        tdd_required=False,
    )

    assert "flag_honored" in events
    assert result.final_phase == TurnPhase.HALTED
