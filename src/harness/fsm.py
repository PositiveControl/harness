"""Generic finite state machine — harness-kbnl.

A small, pure-Python FSM with no domain knowledge. The driver's
turn-level phase machine (ASSESS → IMPLEMENT → VERIFY → CLOSE) is one
caller; other callers can layer their own state enums + transition
tables on top. Designed for clarity, not throughput — these machines
fire at human-scale rates (handfuls of transitions per second at
most), so the cost of a linear scan over candidate transitions is
irrelevant.

Design:

  - **State type is caller-defined.** Typically an Enum, but anything
    hashable works.
  - **Events are caller-defined.** The FSM is event-driven: the caller
    constructs an event (whatever shape they want), feeds it to
    `handle`, and the FSM picks the first matching transition.
  - **Transitions carry a guard predicate.** Per-event arbitrary
    Python; the caller picks the granularity (exact-match shape,
    regex over a content field, complex multi-field check).
  - **Terminal states are explicit.** `is_terminal()` answers "are
    we done?" without the caller hard-coding which states count.
  - **No automatic event consumption.** The caller drives the loop;
    the FSM never runs on its own. Keeps surprise low and testing
    trivial.
  - **Transitions are ordered.** First matching guard wins. Lets
    callers express priority ("if X AND Y, go to A; else if X, go
    to B; else error") by ordering transitions in the table.

What the FSM is NOT:

  - Not async-aware. Transitions are synchronous; an async caller
    wraps `handle` in their own coroutine if needed.
  - Not persistent. Callers serialize the current `state` themselves
    (typical pattern: an Enum value, easy to ship through JSON).
  - Not concurrent-safe. One FSM per logical flow; if you need
    multiple concurrent flows, instantiate one FSM per flow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


class FSMError(RuntimeError):
    """Base for FSM-level errors. Specific subclasses below."""


class NoMatchingTransitionError(FSMError):
    """Raised when `handle(event)` can't find any transition from the
    current state whose guard matches the supplied event. The caller
    decides whether this is a halt condition or a soft "stay put" —
    most callers should treat it as a halt; an FSM with a dead-end
    state is a configuration bug worth raising on.

    Carries the current state and a one-line event summary for the
    operator log. We deliberately avoid stringifying the event itself
    — events can be large; `event_summary` is the caller's chance to
    keep it bounded."""

    def __init__(self, current_state: object, event_summary: str = "") -> None:
        self.current_state = current_state
        self.event_summary = event_summary
        msg = f"no transition from state {current_state!r} matches event"
        if event_summary:
            msg = f"{msg}: {event_summary}"
        super().__init__(msg)


@dataclass(frozen=True)
class Transition[S, E]:
    """One arrow in the state graph.

    - `from_state` / `to_state`: source and destination.
    - `guard`: predicate run against the event. First-match wins, so
      the order transitions are declared in matters when multiple
      could match.
    - `name`: short label used in logs / event trace. Doesn't
      affect dispatch; purely operator-facing.

    Frozen so they can be put in tuples + reused across machine
    instances. Transition tables are typically declared at module
    scope once."""

    from_state: S
    to_state: S
    guard: Callable[[E], bool]
    name: str


@dataclass
class StateMachine[S, E]:
    """Pure state container + transition table.

    Mutates `state` in place on each successful `handle` call.
    Trace records every transition for debugging — operators
    reading a halted run want to see "the FSM went A → B → C
    before halting" rather than just the final state.

    Construction example:

        machine = StateMachine(
            state=Phase.ASSESS,
            transitions=(
                Transition(Phase.ASSESS, Phase.IMPLEMENT,
                           guard=lambda e: e.kind == "assessment_submitted",
                           name="assess->implement"),
                Transition(Phase.IMPLEMENT, Phase.VERIFY,
                           guard=lambda e: e.kind == "implement_complete",
                           name="implement->verify"),
                ...
            ),
            terminal_states=frozenset({Phase.DONE, Phase.HALTED}),
        )

        new_state, transition_name = machine.handle(event)
    """

    state: S
    transitions: Sequence[Transition[S, E]]
    terminal_states: frozenset[S] = field(default_factory=frozenset)
    trace: list[tuple[S, S, str]] = field(default_factory=list)

    def handle(self, event: E) -> tuple[S, str]:
        """Dispatch `event` against transitions whose `from_state` is
        the current state. Apply the first whose `guard(event)` returns
        True. Returns (new_state, transition_name).

        Raises NoMatchingTransitionError if no transition matches —
        the FSM is left in its prior state, the caller decides what
        to do. Most application code should treat this as a halt
        signal."""
        prior = self.state
        for t in self.transitions:
            if t.from_state != prior:
                continue
            if t.guard(event):
                self.state = t.to_state
                self.trace.append((prior, t.to_state, t.name))
                return self.state, t.name
        # Best-effort one-line event summary for the operator log.
        # Falls back to repr(); the application layer is encouraged
        # to define `__str__` on its event type for readable trace.
        summary = self._summarize_event(event)
        raise NoMatchingTransitionError(prior, summary)

    def force(self, new_state: S, *, reason: str) -> None:
        """Bypass guards and jump to `new_state`. Records the forced
        transition in the trace with a `[forced]` prefix in the name
        so audit can distinguish it from guard-driven transitions.

        Intended uses: resume-from-saved-state (loading a persisted
        state value), halting on an out-of-band condition (SIGINT,
        timeout), or tests that want to set up a mid-flow state
        without replaying the prior events. NOT for normal flow —
        guard-based transitions remain the load-bearing path."""
        prior = self.state
        self.state = new_state
        self.trace.append((prior, new_state, f"[forced] {reason}"))

    def is_terminal(self) -> bool:
        """True when the current state is in the configured terminal
        set. Callers loop `while not fsm.is_terminal():` to drive
        flow until done."""
        return self.state in self.terminal_states

    def candidates_from(self, state: S) -> tuple[Transition[S, E], ...]:
        """Inspect-only: transitions whose `from_state` matches.
        Useful for tests + diagnostics that want to introspect the
        graph without driving it."""
        return tuple(t for t in self.transitions if t.from_state == state)

    @staticmethod
    def _summarize_event(event: E) -> str:
        # Lean on the event's __str__ if defined; otherwise repr().
        # Truncated to 200 chars so a runaway event payload doesn't
        # blow up the error message.
        text = str(event) if event is not None else "None"
        return text if len(text) <= 200 else text[:197] + "..."


__all__ = [
    "FSMError",
    "NoMatchingTransitionError",
    "StateMachine",
    "Transition",
]
