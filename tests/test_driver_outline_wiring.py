"""Driver wiring for symbol-aware reads — harness-nfzw.

The symbol-aware reading epic (harness-umvg) shipped `outline` + a
`read_file symbol=` mode and wired them into the chat `coding` profile.
But the driver builds its tool rosters by hand, bypassing profiles.py,
so the epic was invisible to drives: outline wasn't registered in any
drive roster and nothing steered the model off offset/limit.

These tests pin that outline is now registered in every drive roster
(legacy executor, FSM read-only baseline, planner) and that the
read-strategy steering reaches each prompt path."""

from __future__ import annotations

from pathlib import Path

from harness.driver.fsm_turn import (
    _READ_STRATEGY_HINT,
    _read_only_tools,
    _resolve_assess_outcome,
    _write_tier_tools,
    phase_instructions,
)
from harness.driver.loop import EXECUTOR_USER_MESSAGE, _build_executor_registry
from harness.driver.planner import (
    PLANNER_SYSTEM_PROMPT,
    _build_planner_registry,
    _PlannerState,
)
from harness.driver.turn_fsm import TurnPhase
from harness.tools.turn_phase_meta import FlagBlockedTool, SubmitAssessmentTool

# --- rosters expose outline ---------------------------------------


def test_legacy_executor_registry_has_outline(tmp_path: Path) -> None:
    """The default drive path (use_fsm=False) — the one that was
    observed paging files with offset/limit."""
    registry = _build_executor_registry(tmp_path)
    assert "outline" in registry.names()
    # read_file is still there (its symbol mode rides on the same tool).
    assert "read_file" in registry.names()


def test_fsm_read_only_baseline_has_outline(tmp_path: Path) -> None:
    """The FSM read-tier baseline feeds ASSESS / WRITE_TEST / IMPLEMENT /
    VERIFY, so outline is available in every read-capable phase."""
    assert "outline" in _read_only_tools(tmp_path)


def test_planner_registry_has_outline(tmp_path: Path) -> None:
    registry = _build_planner_registry(tmp_path, _PlannerState())
    assert "outline" in registry.names()


# --- steering reaches each prompt path ----------------------------


def test_executor_user_message_steers_to_symbol_reads() -> None:
    assert "outline" in EXECUTOR_USER_MESSAGE
    assert "symbol=" in EXECUTOR_USER_MESSAGE


def test_fsm_assess_and_implement_carry_read_strategy() -> None:
    hint_fragment = _READ_STRATEGY_HINT.strip()[:20]
    assert hint_fragment in phase_instructions(TurnPhase.ASSESS)
    assert hint_fragment in phase_instructions(TurnPhase.IMPLEMENT)
    # CLOSE is shell-only (no reading) — it should NOT carry the hint.
    assert hint_fragment not in phase_instructions(TurnPhase.CLOSE)


def test_implement_instructions_forbid_resurvey() -> None:
    """harness read-thrash fix (loop_run=6308eb21): IMPLEMENT must steer the
    model straight to the edit, citing the carried assessment, instead of
    re-surveying files it already assessed and halting 'no writes'."""
    impl = phase_instructions(TurnPhase.IMPLEMENT)
    assert "GO STRAIGHT TO THE EDIT" in impl
    assert "already assessed this turn" in impl
    assert "edit_file" in impl
    # ASSESS must NOT carry the IMPLEMENT-only directive.
    assert "GO STRAIGHT TO THE EDIT" not in phase_instructions(TurnPhase.ASSESS)


def test_planner_prompt_steers_to_symbol_reads() -> None:
    assert "outline" in PLANNER_SYSTEM_PROMPT
    assert "symbol=" in PLANNER_SYSTEM_PROMPT


# --- harness: stream_edit in the IMPLEMENT write-tier roster ---------


def test_write_tier_tools_include_stream_edit(tmp_path: Path) -> None:
    """IMPLEMENT's file-mutating roster carries stream_edit alongside
    edit_file — the robust in-place sed/awk path that doesn't depend on
    the model reproducing an exact old_string (the edit_file failure mode
    that spun out loop_run=498a4d79 turn 5). edit_file/write_file/shell
    stay registered too."""
    tools = _write_tier_tools(tmp_path)
    assert "stream_edit" in tools
    assert {"edit_file", "write_file", "shell"} <= set(tools)


# --- ASSESS premise-unmet precedence (harness-u1il5 follow-on) -------


def test_assess_resolver_flag_blocked_wins_over_assessment() -> None:
    """When BOTH flag_blocked and submit_assessment were called in
    ASSESS, the premise-unmet signal takes precedence — the model
    decided the bead's target doesn't exist, so we park rather than
    drive IMPLEMENT against a false premise."""
    assess = SubmitAssessmentTool()
    assess.call(
        current_state="game.js has a keydown listener for arrows only",
        gap="bead wants fire (KeyJ) gated foot-only",
        approach="add an edge-triggered KeyJ branch",
    )
    blocked = FlagBlockedTool()
    blocked.call(
        missing="fireWeapon()/KeyJ handler",
        reason="no fire handler exists in game.js to gate",
    )
    outcome = _resolve_assess_outcome(assess, blocked)
    assert outcome.kind == "premise_unmet"
    assert outcome.payload["missing"] == "fireWeapon()/KeyJ handler"


def test_assess_resolver_falls_through_to_assessment_when_not_blocked() -> None:
    """No flag_blocked → normal assessment_submitted outcome (the escape
    is opt-in; the common path is unaffected)."""
    assess = SubmitAssessmentTool()
    assess.call(
        current_state="x" * 30,
        gap="y" * 30,
        approach="z" * 30,
    )
    outcome = _resolve_assess_outcome(assess, FlagBlockedTool())
    assert outcome.kind == "assessment_submitted"
