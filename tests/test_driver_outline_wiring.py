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
    phase_instructions,
)
from harness.driver.loop import EXECUTOR_USER_MESSAGE, _build_executor_registry
from harness.driver.planner import (
    PLANNER_SYSTEM_PROMPT,
    _build_planner_registry,
    _PlannerState,
)
from harness.driver.turn_fsm import TurnPhase

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


def test_planner_prompt_steers_to_symbol_reads() -> None:
    assert "outline" in PLANNER_SYSTEM_PROMPT
    assert "symbol=" in PLANNER_SYSTEM_PROMPT
