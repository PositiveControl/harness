"""Unit tests for src/harness/fsm.py — generic state machine.

Covers the domain-free core: transition lookup, first-match-wins
ordering, terminal-state recognition, `force` escape hatch, and the
`NoMatchingTransitionError` shape. Application-level transition tables
(TurnFSM in driver/turn_fsm.py) have their own tests; this file
pins the machinery they ride on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import pytest

from harness.fsm import (
    NoMatchingTransitionError,
    StateMachine,
    Transition,
)


class _Color(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BROKEN = "broken"


@dataclass(frozen=True)
class _TrafficEvent:
    """Minimal stand-in for an application event. The FSM's guards
    are plain Python predicates — anything hashable+inspectable works."""

    kind: str

    def __str__(self) -> str:
        return f"event({self.kind})"


def _traffic_light() -> StateMachine[_Color, _TrafficEvent]:
    return StateMachine(
        state=_Color.RED,
        transitions=(
            Transition(_Color.RED, _Color.GREEN, guard=lambda e: e.kind == "go", name="red->green"),
            Transition(
                _Color.GREEN, _Color.YELLOW, guard=lambda e: e.kind == "warn", name="green->yellow"
            ),
            Transition(
                _Color.YELLOW, _Color.RED, guard=lambda e: e.kind == "stop", name="yellow->red"
            ),
            Transition(
                _Color.RED, _Color.BROKEN, guard=lambda e: e.kind == "fault", name="red->broken"
            ),
        ),
        terminal_states=frozenset({_Color.BROKEN}),
    )


# --- happy-path transitions ---------------------------------------


def test_handle_advances_through_matching_transition() -> None:
    """A matching event advances state and records the transition in
    the trace. Returns (new_state, transition_name) to the caller."""
    fsm = _traffic_light()
    state, name = fsm.handle(_TrafficEvent("go"))
    assert state == _Color.GREEN
    assert name == "red->green"
    assert fsm.trace == [(_Color.RED, _Color.GREEN, "red->green")]


def test_handle_chains_across_multiple_events() -> None:
    """Multiple consecutive handles drive the machine through the full
    expected sequence; trace records every step."""
    fsm = _traffic_light()
    fsm.handle(_TrafficEvent("go"))
    fsm.handle(_TrafficEvent("warn"))
    fsm.handle(_TrafficEvent("stop"))
    assert fsm.state == _Color.RED
    assert [t[2] for t in fsm.trace] == ["red->green", "green->yellow", "yellow->red"]


# --- no-match + ordering ------------------------------------------


def test_handle_raises_when_no_transition_matches() -> None:
    """An event with no matching guard from the current state raises
    NoMatchingTransitionError. The FSM is left in its prior state —
    the caller decides what to do."""
    fsm = _traffic_light()
    with pytest.raises(NoMatchingTransitionError) as exc:
        fsm.handle(_TrafficEvent("warn"))  # warn is only valid from GREEN
    assert exc.value.current_state == _Color.RED
    assert "event(warn)" in exc.value.event_summary
    # State unchanged.
    assert fsm.state == _Color.RED
    assert fsm.trace == []


def test_first_matching_transition_wins() -> None:
    """When two transitions from the current state would match,
    the first declared wins. Lets callers express priority by
    table ordering."""
    fsm = StateMachine[_Color, _TrafficEvent](
        state=_Color.RED,
        transitions=(
            Transition(_Color.RED, _Color.GREEN, guard=lambda e: True, name="first-match-any"),
            Transition(_Color.RED, _Color.YELLOW, guard=lambda e: True, name="never-fires"),
        ),
    )
    state, name = fsm.handle(_TrafficEvent("anything"))
    assert state == _Color.GREEN
    assert name == "first-match-any"


# --- terminal states ----------------------------------------------


def test_is_terminal_recognizes_configured_terminal_set() -> None:
    """is_terminal() returns True iff the current state is in the
    configured set. Generic — doesn't assume any particular state
    name means 'done'."""
    fsm = _traffic_light()
    assert fsm.is_terminal() is False
    fsm.handle(_TrafficEvent("fault"))
    assert fsm.state == _Color.BROKEN
    assert fsm.is_terminal() is True


def test_terminal_state_with_no_outgoing_transitions_is_dead_end() -> None:
    """A terminal state with no further matching transitions must
    raise on handle — the FSM doesn't auto-stop just because is_terminal
    is True. Application code checks is_terminal() before handling
    the next event."""
    fsm = _traffic_light()
    fsm.handle(_TrafficEvent("fault"))
    assert fsm.is_terminal()
    with pytest.raises(NoMatchingTransitionError):
        fsm.handle(_TrafficEvent("go"))


# --- force escape hatch ------------------------------------------


def test_force_bypasses_guards_and_records_trace() -> None:
    """force() lets the caller jump to any state without consulting
    guards. Used for resume-from-persisted-state, SIGINT halting,
    test setup. Trace records the jump with a [forced] prefix."""
    fsm = _traffic_light()
    fsm.force(_Color.GREEN, reason="test setup")
    assert fsm.state == _Color.GREEN
    assert fsm.trace == [(_Color.RED, _Color.GREEN, "[forced] test setup")]


def test_force_followed_by_handle_continues_from_forced_state() -> None:
    """After force(), subsequent handle() calls operate on the new
    state. Combined with persistence: load state from JSON, force()
    into it, resume normal flow."""
    fsm = _traffic_light()
    fsm.force(_Color.GREEN, reason="resume from disk")
    # GREEN's only out-transition is 'warn'.
    state, name = fsm.handle(_TrafficEvent("warn"))
    assert state == _Color.YELLOW
    assert name == "green->yellow"


# --- candidates_from inspection ----------------------------------


def test_candidates_from_returns_only_transitions_from_state() -> None:
    """candidates_from() is read-only introspection — used by tests
    + diagnostics to inspect the graph without driving it."""
    fsm = _traffic_light()
    red_candidates = fsm.candidates_from(_Color.RED)
    assert {t.to_state for t in red_candidates} == {_Color.GREEN, _Color.BROKEN}
    green_candidates = fsm.candidates_from(_Color.GREEN)
    assert {t.to_state for t in green_candidates} == {_Color.YELLOW}


# --- generic over arbitrary state + event types -------------------


def test_fsm_works_with_string_states_and_dict_events() -> None:
    """The generic parameters are TypeVars without constraints —
    callers can use strings, ints, dataclasses, whatever. Pin that
    nothing in the implementation forces an Enum or a specific event
    shape."""

    def has_kind(kind: str) -> Callable[[dict[str, str]], bool]:
        return lambda e: e.get("kind") == kind

    fsm = StateMachine[str, dict[str, str]](
        state="a",
        transitions=(Transition[str, dict[str, str]]("a", "b", guard=has_kind("go"), name="a->b"),),
        terminal_states=frozenset({"b"}),
    )
    fsm.handle({"kind": "go"})
    assert fsm.state == "b"
    assert fsm.is_terminal()


# --- event summary in error ----------------------------------------


def test_no_match_error_summary_is_truncated() -> None:
    """The error's `event_summary` truncates at 200 chars so a runaway
    event payload (e.g. an LLM reply embedded in the event) doesn't
    blow up the operator log."""

    @dataclass(frozen=True)
    class _BigEvent:
        payload: str

        def __str__(self) -> str:
            return f"big({self.payload})"

    fsm = StateMachine[str, _BigEvent](
        state="a",
        transitions=(),
    )
    big = _BigEvent(payload="x" * 5000)
    with pytest.raises(NoMatchingTransitionError) as exc:
        fsm.handle(big)
    assert len(exc.value.event_summary) <= 200
    assert exc.value.event_summary.endswith("...")
