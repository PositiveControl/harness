"""Phase-aware unavailable-tool messaging + WRITE_TEST stall wiring —
loop_run=3a0f6368.

That run burned all six turns the same way: the model reached for
`edit_file` in ASSESS / WRITE_TEST, got the catalog's "call load_tool,
then retry" error, then load_tool (builders={}) answered "restart the
session with --tools-add" — a contradictory dead-end pair it
ping-ponged between until each phase's round budget died. Meanwhile
turn 2 produced a genuine red test (write_file + shell exit=1) that was
discarded because the model never called submit_failing_test, and every
failure was recorded as "fabrication_fallback fired during halted",
masking the real `write_test->halted (no test)` cause.

These tests pin the three fixes:

  1. Phase registries register per-phase unavailable hints on both the
     registry's unknown-tool path and load_tool, replacing the
     contradiction with the real recovery path (the phase's meta-tool).
  2. WRITE_TEST arms a NoTestSubmitStreakDetector, the ASSESS-nudge
     analog for "testing instead of submitting".
  3. A fabrication-fallback final reply annotates the trace-derived
     halt reason instead of overwriting it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.driver.fsm_turn import (
    _build_phase_registry,
    _phase_unavailable_hints,
    run_fsm_turn,
)
from harness.driver.handoff import Handoff
from harness.driver.turn_fsm import TurnPhase
from harness.model.adapter import ChatMessage
from harness.orchestrator import ToolLoopResult
from harness.orchestrator.hooks import EXHAUSTED_FABRICATION_FALLBACK
from harness.orchestrator.no_write_streak import (
    NoSubmitStreakDetector,
    NoTestSubmitStreakDetector,
    NoWriteStreakDetector,
)
from harness.tools.turn_phase_meta import (
    FlagBlockedTool,
    SkipTestPhaseTool,
    SubmitAssessmentTool,
    SubmitFailingTestTool,
    SubmitImplementationCompleteTool,
)


def _registry_for(phase: TurnPhase, workspace: Path) -> Any:
    return _build_phase_registry(
        phase,
        workspace,
        submit_assessment=SubmitAssessmentTool(),
        skip_test_phase=SkipTestPhaseTool(),
        submit_failing_test=SubmitFailingTestTool(),
        submit_implementation_complete=SubmitImplementationCompleteTool(),
        flag_blocked=FlagBlockedTool(),
    )


# --- 1. phase-aware unavailable hints ---------------------------------


def test_assess_edit_file_error_names_phase_not_load_tool(tmp_path: Path) -> None:
    """Calling edit_file in ASSESS must explain the phase gate, not
    tell the model to load_tool-and-retry (the 3a0f6368 dead end)."""
    registry = _registry_for(TurnPhase.ASSESS, tmp_path)
    result = registry.call("edit_file", {"path": "game.js", "content": "x"})
    assert not result.success
    assert result.error == "unknown_tool"
    assert "ASSESS" in result.output
    assert "submit_assessment" in result.output
    assert "then retry" not in result.output


def test_assess_load_tool_refuses_with_phase_hint(tmp_path: Path) -> None:
    """load_tool('edit_file') in ASSESS must not advise a session
    restart — it must point at the phase meta-tool instead."""
    registry = _registry_for(TurnPhase.ASSESS, tmp_path)
    result = registry.call("load_tool", {"name": "edit_file"})
    assert result.success  # load_tool reports via its output string
    assert "cannot be loaded right now" in result.output
    assert "submit_assessment" in result.output
    assert "--tools-add" not in result.output


def test_write_test_stream_edit_error_points_at_submit_failing_test(tmp_path: Path) -> None:
    # edit_file is granted in WRITE_TEST now (append-to-existing-test);
    # stream_edit is the withheld editor, and its hint must name the real
    # recovery path (finish via submit_failing_test, IMPLEMENT unlocks
    # bulk edits) rather than the load_tool dead end.
    registry = _registry_for(TurnPhase.WRITE_TEST, tmp_path)
    result = registry.call("stream_edit", {"path": "game.js", "edits": []})
    assert not result.success
    assert "submit_failing_test" in result.output
    assert "IMPLEMENT" in result.output


def test_write_test_keeps_write_file_and_shell(tmp_path: Path) -> None:
    """The hints must not shadow the tools the phase actually grants."""
    registry = _registry_for(TurnPhase.WRITE_TEST, tmp_path)
    assert "write_file" in registry.names()
    assert "shell" in registry.names()
    # And write_file genuinely dispatches.
    result = registry.call("write_file", {"path": "tests/t.js", "content": "x"})
    assert result.success


def test_write_test_grants_edit_file_for_appending(tmp_path: Path) -> None:
    """WRITE_TEST now grants edit_file so a new case can be appended to an
    existing test file instead of spawning one test_<bead>.py per bead.
    edit_file must be available and NOT shadowed by an unavailable hint."""
    registry = _registry_for(TurnPhase.WRITE_TEST, tmp_path)
    assert "edit_file" in registry.names()
    assert "edit_file" not in _phase_unavailable_hints(TurnPhase.WRITE_TEST)
    # It genuinely dispatches: append to an existing test file.
    (tmp_path / "test_feature.py").write_text("def test_a():\n    assert True\n")
    result = registry.call(
        "edit_file",
        {
            "path": "test_feature.py",
            "old_string": "    assert True\n",
            "new_string": "    assert True\n\n\ndef test_b():\n    assert False\n",
        },
    )
    assert result.success


def test_write_test_still_withholds_stream_edit() -> None:
    """stream_edit (bulk source edits) stays an IMPLEMENT-only shape; the
    hint points at the test-authoring tools instead of load_tool."""
    hints = _phase_unavailable_hints(TurnPhase.WRITE_TEST)
    assert "stream_edit" in hints
    assert "load_tool" in hints["stream_edit"]
    assert "edit_file" in hints["stream_edit"]  # steers toward the append path


def test_implement_has_no_editor_hints(tmp_path: Path) -> None:
    """IMPLEMENT grants the full write tier — no hints to register."""
    assert _phase_unavailable_hints(TurnPhase.IMPLEMENT) == {}
    registry = _registry_for(TurnPhase.IMPLEMENT, tmp_path)
    assert "edit_file" in registry.names()
    assert "write_file" in registry.names()


def test_editor_withholding_phases_hint_the_editors() -> None:
    """Each phase that withholds edit_file covers it in the hint map and
    forbids the load_tool detour. WRITE_TEST is excluded — it now grants
    edit_file (append-to-existing-test) and only withholds stream_edit."""
    for phase in (TurnPhase.ASSESS, TurnPhase.VERIFY, TurnPhase.CLOSE):
        hints = _phase_unavailable_hints(phase)
        assert "edit_file" in hints, phase
        assert "load_tool" in hints["edit_file"], phase


# --- 2 + 3. run_fsm_turn wiring: detector arming + halt reason ---------


class _FakeCharacter:
    def system_prompt(
        self,
        *,
        include_samples: Any = (),
        include_style_rules: bool = True,
    ) -> str:
        return "you are an executor."


class _FakeBd:
    """CLOSE-phase resolver surface; unused in these tests."""

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


def test_fsm_turn_arms_test_submit_detector_in_write_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ASSESS arms the submit detector, WRITE_TEST the test-submit
    detector (loop_run=3a0f6368 turn 2: red test written + run, never
    submitted, no nudge fired). The fake tool loop completes ASSESS via
    its meta-tool, then stalls in WRITE_TEST."""
    detectors: list[Any] = []

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, **kwargs: Any
    ) -> ToolLoopResult:
        detectors.append(kwargs.get("no_write_streak"))
        if "submit_assessment" in registry.names():
            registry.call(
                "submit_assessment",
                {
                    "current_state": "c" * 30,
                    "gap": "g" * 30,
                    "approach": "a" * 30,
                },
            )
            content = "assessment recorded."
        else:
            content = "stalling."
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

    assert not result.succeeded
    assert isinstance(detectors[0], NoSubmitStreakDetector)
    assert isinstance(detectors[1], NoTestSubmitStreakDetector)


def test_fsm_turn_arms_write_detector_in_implement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """skip_test_phase routes ASSESS → WRITE_TEST → IMPLEMENT; the
    IMPLEMENT invocation must arm the original no-write detector."""
    detectors: list[Any] = []

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, **kwargs: Any
    ) -> ToolLoopResult:
        detectors.append(kwargs.get("no_write_streak"))
        if "submit_assessment" in registry.names():
            registry.call(
                "submit_assessment",
                {
                    "current_state": "c" * 30,
                    "gap": "g" * 30,
                    "approach": "a" * 30,
                },
            )
        elif "skip_test_phase" in registry.names():
            registry.call("skip_test_phase", {"reason": "no test runner available here"})
        content = "ok."
        return ToolLoopResult(
            content=content,
            messages=[ChatMessage(role="assistant", content=content)],
            rounds=1,
            events=[],
        )

    monkeypatch.setattr("harness.driver.fsm_turn.run_tool_loop", fake_run_tool_loop)

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_FakeBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
    )

    assert not result.succeeded  # IMPLEMENT stalls (no writes) — fine
    assert isinstance(detectors[2], NoWriteStreakDetector)


def test_fsm_turn_passes_phase_exit_tools_to_wrap_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """loop_run=dfc38c5f: each phase's run_tool_loop call must whitelist
    that phase's exit tools for the wrap-up round, so a model that
    exhausts its budget one call short of the exit signal can still
    emit it. ASSESS → WRITE_TEST via submit_assessment, then stall."""
    wrap_up_tool_sets: list[Any] = []

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, **kwargs: Any
    ) -> ToolLoopResult:
        wrap_up_tool_sets.append(kwargs.get("wrap_up_tools"))
        if "submit_assessment" in registry.names():
            registry.call(
                "submit_assessment",
                {
                    "current_state": "c" * 30,
                    "gap": "g" * 30,
                    "approach": "a" * 30,
                },
            )
            content = "assessment recorded."
        else:
            content = "stalling."
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

    assert not result.succeeded
    assert wrap_up_tool_sets[0] == frozenset({"submit_assessment", "flag_blocked"})
    assert wrap_up_tool_sets[1] == frozenset({"submit_failing_test", "skip_test_phase"})


def test_halt_reason_keeps_fsm_cause_when_fallback_fired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """loop_run=3a0f6368: all six turns recorded only
    'fabrication_fallback fired during halted', erasing the real halt
    cause from state.last_failure + the retry handoff. The fallback is
    a symptom — the reason must keep the FSM transition name and merely
    annotate the fallback."""

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, _registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        return ToolLoopResult(
            content=EXHAUSTED_FABRICATION_FALLBACK,
            messages=[ChatMessage(role="assistant", content=EXHAUSTED_FABRICATION_FALLBACK)],
            rounds=1,
            events=[],
        )

    monkeypatch.setattr("harness.driver.fsm_turn.run_tool_loop", fake_run_tool_loop)

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_FakeBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
    )

    assert not result.succeeded
    # The FSM cause survives...
    assert "no assessment" in result.reason
    # ...with the fallback as an annotation, not a replacement.
    assert "fabrication_fallback fired" in result.reason
    assert not result.reason.startswith("fabrication_fallback")
