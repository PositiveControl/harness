"""Tests for the executor turn FSM — harness-kbnl.

Drives the FSM through every transition in the default table. The
generic StateMachine machinery is tested in test_fsm.py; here we pin
the *domain* mapping: which PhaseOutcome events drive which phase
transitions, including the TDD path (default) vs the TDD-skip path
vs the test-already-green short-circuit.
"""

from __future__ import annotations

import pytest

from harness.driver.turn_fsm import (
    DEFAULT_PHASE_BUDGETS,
    PREMISE_UNMET_REASON_PREFIX,
    PhaseOutcome,
    TurnPhase,
    assessment_skipped_tdd,
    assessment_submitted,
    build_turn_fsm,
    close_failed,
    close_succeeded,
    failing_test_submitted,
    green_test_outcome,
    implement_complete,
    phase_no_progress,
    premise_unmet,
    verify_failed,
    verify_passed,
)
from harness.fsm import NoMatchingTransitionError


def _drive(fsm: object, events: list[PhaseOutcome]) -> list[str]:
    """Apply each event in order; return the trace of transition
    names. Helper to keep test bodies short."""
    names: list[str] = []
    for event in events:
        _, name = fsm.handle(event)  # type: ignore[attr-defined]
        names.append(name)
    return names


# --- happy paths --------------------------------------------------


def test_default_tdd_happy_path_assess_to_done() -> None:
    """harness-kbnl: with TDD on (default), one full lap is
    ASSESS → WRITE_TEST → IMPLEMENT → VERIFY → CLOSE → DONE."""
    fsm = build_turn_fsm()
    trace = _drive(
        fsm,
        [
            assessment_submitted(current_state="x" * 30, gap="x" * 30, approach="x" * 30),
            failing_test_submitted(
                test_path="t.py",
                test_cmd="pytest t.py",
                failure_output="x" * 30,
            ),
            implement_complete(summary="x" * 30),
            verify_passed(),
            close_succeeded(),
        ],
    )
    assert trace == [
        "assess->write_test",
        "write_test->implement",
        "implement->verify",
        "verify->close",
        "close->done",
    ]
    assert fsm.state == TurnPhase.DONE
    assert fsm.is_terminal()


def test_tdd_skipped_via_assessment_routes_assess_to_implement() -> None:
    """harness-kbnl: when submit_assessment carries tdd_applicable=False
    the FSM skips WRITE_TEST entirely — ASSESS → IMPLEMENT direct."""
    fsm = build_turn_fsm()
    trace = _drive(
        fsm,
        [
            assessment_skipped_tdd(
                current_state="x" * 30,
                gap="x" * 30,
                approach="x" * 30,
                reason="documentation-only change",
            ),
            implement_complete(summary="x" * 30),
            verify_passed(),
            close_succeeded(),
        ],
    )
    assert trace == [
        "assess->implement (tdd skipped)",
        "implement->verify",
        "verify->close",
        "close->done",
    ]
    assert fsm.state == TurnPhase.DONE


def test_test_already_passes_short_circuits_to_close() -> None:
    """harness-kbnl: when submit_failing_test reports the test
    actually passes (work was already done by a prior turn / human),
    the FSM jumps WRITE_TEST → CLOSE — no implementation needed."""
    fsm = build_turn_fsm()
    trace = _drive(
        fsm,
        [
            assessment_submitted(current_state="x" * 30, gap="x" * 30, approach="x" * 30),
            green_test_outcome(test_path="t.py", test_cmd="pytest t.py"),
            close_succeeded(),
        ],
    )
    assert trace == [
        "assess->write_test",
        "write_test->close (already green)",
        "close->done",
    ]
    assert fsm.state == TurnPhase.DONE


# --- failure / halt paths -----------------------------------------


def test_assess_no_progress_halts() -> None:
    """ASSESS ran out of budget without an assessment → HALTED."""
    fsm = build_turn_fsm()
    state, name = fsm.handle(
        phase_no_progress(TurnPhase.ASSESS, reason="3 rounds, no submit_assessment")
    )
    assert state == TurnPhase.HALTED
    assert "halted" in name
    assert fsm.is_terminal()


def test_premise_unmet_halts_from_assess() -> None:
    """harness-u1il5 follow-on: ASSESS emits premise_unmet (flag_blocked
    — the thing to verify/fix doesn't exist) → HALTED. The detail carries
    the PREMISE_UNMET_REASON_PREFIX so the loop can park-without-retry."""
    fsm = build_turn_fsm()
    outcome = premise_unmet(
        missing="fireWeapon() / KeyJ handler",
        reason="bead verifies fire-key gating but no fire handler exists",
    )
    assert outcome.detail.startswith(PREMISE_UNMET_REASON_PREFIX)
    state, name = fsm.handle(outcome)
    assert state == TurnPhase.HALTED
    assert "premise unmet" in name
    assert fsm.is_terminal()


def test_write_test_no_progress_halts() -> None:
    """WRITE_TEST budget exhausted with no failing test submitted."""
    fsm = build_turn_fsm()
    fsm.handle(assessment_submitted(current_state="x" * 30, gap="x" * 30, approach="x" * 30))
    state, _ = fsm.handle(phase_no_progress(TurnPhase.WRITE_TEST, reason="no test written"))
    assert state == TurnPhase.HALTED


def test_implement_no_progress_halts() -> None:
    """IMPLEMENT budget exhausted with zero successful writes."""
    fsm = build_turn_fsm()
    fsm.handle(assessment_submitted(current_state="x" * 30, gap="x" * 30, approach="x" * 30))
    fsm.handle(failing_test_submitted(test_path="t", test_cmd="x", failure_output="x" * 30))
    state, _ = fsm.handle(phase_no_progress(TurnPhase.IMPLEMENT, reason="no writes"))
    assert state == TurnPhase.HALTED


def test_verify_failed_re_enters_implement() -> None:
    """harness-kbnl: a verify failure routes back to IMPLEMENT for
    a retry within the same turn. The caller is responsible for
    enforcing the IMPLEMENT round budget across retries."""
    fsm = build_turn_fsm()
    # Drive to VERIFY
    fsm.handle(assessment_submitted(current_state="x" * 30, gap="x" * 30, approach="x" * 30))
    fsm.handle(failing_test_submitted(test_path="t", test_cmd="x", failure_output="x" * 30))
    fsm.handle(implement_complete(summary="x" * 30))
    assert fsm.state == TurnPhase.VERIFY
    # Verify fails → re-enter IMPLEMENT
    state, name = fsm.handle(verify_failed(failure_tail="AssertionError row count"))
    assert state == TurnPhase.IMPLEMENT
    assert "retry" in name


def test_close_failed_halts() -> None:
    """bd close errored → terminal HALTED, not DONE. Rare but
    needs a clean exit."""
    fsm = build_turn_fsm()
    fsm.force(TurnPhase.CLOSE, reason="setup")
    state, _ = fsm.handle(close_failed(reason="bd error xyz"))
    assert state == TurnPhase.HALTED


# --- error shape --------------------------------------------------


def test_unhandled_event_from_assess_raises() -> None:
    """A nonsensical event in ASSESS (e.g. a verify_passed before
    we've even assessed) raises NoMatchingTransitionError — the
    caller decides whether that's a halt or a soft no-op."""
    fsm = build_turn_fsm()
    with pytest.raises(NoMatchingTransitionError):
        fsm.handle(verify_passed())


def test_implement_complete_from_assess_does_not_match() -> None:
    """The transition table is strict about phase ordering: an
    implement_complete event from ASSESS is rejected, not silently
    advanced. Catches a class of bug where the model emits the
    wrong meta-tool in the wrong phase."""
    fsm = build_turn_fsm()
    with pytest.raises(NoMatchingTransitionError):
        fsm.handle(implement_complete(summary="x" * 30))


# --- payload preservation -----------------------------------------


def test_phase_outcome_payload_round_trips() -> None:
    """The PhaseOutcome.payload field carries structured data the
    next phase wants — assessment text for handoff render, test_cmd
    for verify. The FSM doesn't mutate it; it's the caller's
    contract with itself."""
    event = failing_test_submitted(
        test_path="tests/test_grid.py",
        test_cmd="pytest tests/test_grid.py -v",
        failure_output="AssertionError: 28 != 30",
    )
    assert event.payload["test_cmd"] == "pytest tests/test_grid.py -v"
    assert event.payload["test_path"] == "tests/test_grid.py"


def test_default_budgets_cover_every_non_terminal_phase() -> None:
    """A missing budget would cause the caller to default-zero a
    phase, halting before any work. Pin that every phase that takes
    rounds has a budget."""
    non_terminal = set(TurnPhase) - {TurnPhase.DONE, TurnPhase.HALTED}
    assert non_terminal == set(DEFAULT_PHASE_BUDGETS.keys())
    for budget in DEFAULT_PHASE_BUDGETS.values():
        assert budget >= 1


def test_write_test_budget_covers_full_tdd_sequence() -> None:
    """loop_run=dfc38c5f: the minimum honest WRITE_TEST sequence is
    read → write_file → shell run → submit_failing_test, and the model
    reliably spends 1-2 extra rounds re-reading context. A budget of 4
    exhausted exactly at the shell run, so the submit never happened
    and all 6 turns halted 'no test'. Pin ≥ 6."""
    assert DEFAULT_PHASE_BUDGETS[TurnPhase.WRITE_TEST] >= 6


def test_phase_outcome_str_is_bounded() -> None:
    """PhaseOutcome.__str__ is the source for NoMatchingTransitionError
    summaries — must be bounded. detail field is truncated to 80
    chars."""
    huge = "x" * 5000
    event = PhaseOutcome(kind="t", detail=huge)
    assert len(str(event)) < 200  # well under the FSM's 200-char cap
