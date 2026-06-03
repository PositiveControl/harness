"""Turn-level FSM for the executor loop — harness-kbnl.

Models the implicit "phase" of one executor turn as an explicit state
machine over `TurnPhase`. The generic FSM machinery lives in
`harness.fsm`; this module supplies the domain piece: the phase enum,
the event shape that drives transitions, and the transition table.

Phase chain (default):

    ASSESS  →  WRITE_TEST  →  IMPLEMENT  →  VERIFY  →  CLOSE  →  DONE
       │           │
       │           └─ (skipped when tdd_required=False or assessment marks
       │              tdd_applicable=False)
       └─ HALTED (any phase can short-circuit here)

Why TDD by default: surfaced in loop runs d4e01d68 / 028d605b where the
model wrote a correct-looking tile grid into a SIDE file (game_fixed.js)
and claimed completion without ever overwriting game.js. A failing-test
gate gives us a per-phase artifact that proves the gap exists *before*
implementation, and the same test then becomes the verify step — no
gap between "the model thinks it's done" and "the artifact actually
works."

Why generic FSM underneath: the same machinery should be reusable for
planner phases (recon → decompose → commit), verify-pipeline composition,
and future agentic flows. See `harness.fsm` — domain-free.

Events: `PhaseOutcome` carries the result of one phase's `run_tool_loop`
call. The FSM transitions are guard predicates over its `kind` field.
The kind names match the meta-tools the model uses to signal phase
completion (`assessment_submitted`, `failing_test_submitted`, etc.) —
that 1-to-1 correspondence keeps the trace readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from harness.fsm import StateMachine, Transition


class TurnPhase(Enum):
    """The phases one executor turn passes through.

    Ordered roughly by chronological position in a successful turn,
    though the FSM is the load-bearing definition — `is_terminal()`
    answers "are we done" by checking `terminal_states`, not by
    ordinal position.

    Terminal phases (DONE / HALTED) don't take events; the FSM
    `is_terminal()` returns True and the caller exits the loop.
    """

    ASSESS = "assess"  # read + reason; emit assessment
    WRITE_TEST = "write_test"  # red test for TDD path
    IMPLEMENT = "implement"  # main coding work
    VERIFY = "verify"  # run tests + verify cmds
    CLOSE = "close"  # bd close
    DONE = "done"  # terminal — success
    HALTED = "halted"  # terminal — failure


# Per-phase round budget. Tunable; conservative defaults targeting
# ~8-12 rounds total for an end-to-end pass, matching the existing
# `executor_max_rounds=12` (which we keep as the IMPLEMENT budget
# floor since that's where the work actually happens).
DEFAULT_PHASE_BUDGETS: dict[TurnPhase, int] = {
    TurnPhase.ASSESS: 3,
    TurnPhase.WRITE_TEST: 4,
    TurnPhase.IMPLEMENT: 8,
    TurnPhase.VERIFY: 2,
    TurnPhase.CLOSE: 1,
}


@dataclass(frozen=True)
class PhaseOutcome:
    """One phase's result, fed to the FSM to pick the next state.

    - `kind` is the load-bearing field; transition guards key off it.
      Values mirror the meta-tools / fail modes:
        "assessment_submitted"          — submit_assessment succeeded
        "assessment_skipped"            — submit_assessment(tdd_applicable=False)
        "failing_test_submitted"        — submit_failing_test succeeded
        "test_already_passes"           — submit_failing_test reported no-op
        "test_phase_skipped"            — skip_test_phase invoked
        "implement_complete"            — submit_implementation_complete
        "implement_some_writes"         — IMPLEMENT rounds exhausted with ≥1 successful write
        "verify_passed"                 — all verify steps exit 0 (or no verify steps)
        "verify_failed"                 — at least one verify step exit != 0
        "close_succeeded"               — bd close returned ok
        "close_failed"                  — bd close errored
        "phase_budget_exhausted"        — round budget hit with no other signal
        "phase_no_progress"             — budget exhausted with zero useful tool calls

    - `detail` is operator-facing text describing what happened
      (verify command stderr tail, assessment-validation error, etc.).
      Used in handoff renders + audit; not consumed by guards.

    - `payload` is structured data the next phase might want (the
      failing test cmd to feed into VERIFY, the assessment text to
      render in the handoff). Open dict — callers pick the shape.
    """

    kind: str
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        # NoMatchingTransitionError summarizes the event via str();
        # bound the output so a long detail doesn't bloat the error.
        text = f"kind={self.kind}"
        if self.detail:
            tail = self.detail if len(self.detail) <= 80 else self.detail[:77] + "..."
            text = f"{text} detail={tail!r}"
        return text


# Convenience PhaseOutcome constructors so call sites don't repeat the
# kind string and risk typos. Each maps to one transition in the
# default table below.
def assessment_submitted(*, current_state: str, gap: str, approach: str) -> PhaseOutcome:
    return PhaseOutcome(
        kind="assessment_submitted",
        detail=f"gap: {gap[:120]}",
        payload={"current_state": current_state, "gap": gap, "approach": approach},
    )


def assessment_skipped_tdd(
    *, current_state: str, gap: str, approach: str, reason: str
) -> PhaseOutcome:
    """Assessment landed with tdd_applicable=False. Routes directly to
    IMPLEMENT, skipping WRITE_TEST. The reason is captured so the
    handoff records why TDD was skipped."""
    return PhaseOutcome(
        kind="assessment_skipped",
        detail=f"tdd skipped: {reason[:100]}",
        payload={
            "current_state": current_state,
            "gap": gap,
            "approach": approach,
            "tdd_skip_reason": reason,
        },
    )


def failing_test_submitted(*, test_path: str, test_cmd: str, failure_output: str) -> PhaseOutcome:
    return PhaseOutcome(
        kind="failing_test_submitted",
        detail=f"red: {test_path}",
        payload={
            "test_path": test_path,
            "test_cmd": test_cmd,
            "failure_output": failure_output,
        },
    )


def green_test_outcome(*, test_path: str, test_cmd: str) -> PhaseOutcome:
    """submit_failing_test reported the test passes. The work is
    already done — skip implementation and go straight to CLOSE.

    Function name avoids the `test*` prefix because pytest's default
    discovery (`python_functions = test*`) would otherwise collect this
    constructor as a test fixture site."""
    return PhaseOutcome(
        kind="test_already_passes",
        detail=f"already green: {test_path}",
        payload={"test_path": test_path, "test_cmd": test_cmd},
    )


def implement_complete(*, summary: str) -> PhaseOutcome:
    return PhaseOutcome(
        kind="implement_complete",
        detail=summary[:120],
        payload={"summary": summary},
    )


def phase_no_progress(phase: TurnPhase, *, reason: str) -> PhaseOutcome:
    return PhaseOutcome(
        kind="phase_no_progress",
        detail=f"{phase.value}: {reason[:140]}",
        payload={"phase": phase.value, "reason": reason},
    )


# Marker prefix on a halt reason that signals the bead's PREMISE is unmet
# — the thing the bead asks to verify/fix doesn't exist yet because an
# upstream dependency never landed it. The loop keys off this prefix to
# PARK the issue immediately (with the `blocked` flag) instead of burning
# the full retry budget on attempts that face the identical false premise
# every time (loop_run=498a4d79: §15a-iii gating parked after 3 attempts
# trying to gate fire/walk/weapon inputs that §9b-i/§10 closed blind).
PREMISE_UNMET_REASON_PREFIX = "premise unmet:"


def premise_unmet(*, missing: str, reason: str) -> PhaseOutcome:
    """ASSESS determined the bead can't be done because a concrete
    precondition is absent (a named function/file/symbol the bead's work
    presupposes). Routes ASSESS straight to HALTED with a marked reason so
    the loop parks-and-flags instead of retrying a futile premise.

    `missing` is the concrete absent artifact (operator-checkable);
    `reason` is the one-line explanation of why that blocks the bead."""
    return PhaseOutcome(
        kind="premise_unmet",
        detail=f"{PREMISE_UNMET_REASON_PREFIX} {missing} — {reason[:140]}",
        payload={"missing": missing, "reason": reason},
    )


def verify_passed(*, verify_summary: str = "") -> PhaseOutcome:
    return PhaseOutcome(
        kind="verify_passed",
        detail=verify_summary[:120],
        payload={},
    )


def verify_failed(*, failure_tail: str) -> PhaseOutcome:
    return PhaseOutcome(
        kind="verify_failed",
        detail=failure_tail[:140],
        payload={"failure_tail": failure_tail},
    )


def close_succeeded() -> PhaseOutcome:
    return PhaseOutcome(kind="close_succeeded")


def close_failed(*, reason: str) -> PhaseOutcome:
    return PhaseOutcome(kind="close_failed", detail=reason[:120])


# --- transition table --------------------------------------------------


# Guard predicates kept as small `lambda e: e.kind == "..."` so the
# table reads top-to-bottom as "from→to when ...". Compound predicates
# (e.g. check kind AND a payload field) get a named helper above.
def _kind(name: str) -> Any:
    """Closure factory — needed because lambdas in a comprehension would
    all capture the same loop var. Returns a fresh closure per call."""
    return lambda e: e.kind == name


def _default_transitions() -> tuple[Transition[TurnPhase, PhaseOutcome], ...]:
    """The default TurnPhase transition table.

    Order matters when multiple guards could match — first match wins.
    Today no two rows share a from_state + overlapping guard, but the
    ordering convention is documented for future additions.
    """
    return (
        # ASSESS exits
        Transition(
            TurnPhase.ASSESS,
            TurnPhase.WRITE_TEST,
            guard=_kind("assessment_submitted"),
            name="assess->write_test",
        ),
        Transition(
            TurnPhase.ASSESS,
            TurnPhase.IMPLEMENT,
            guard=_kind("assessment_skipped"),
            name="assess->implement (tdd skipped)",
        ),
        Transition(
            TurnPhase.ASSESS,
            TurnPhase.HALTED,
            guard=_kind("phase_no_progress"),
            name="assess->halted (no assessment)",
        ),
        Transition(
            TurnPhase.ASSESS,
            TurnPhase.HALTED,
            guard=_kind("premise_unmet"),
            name="assess->halted (premise unmet)",
        ),
        # WRITE_TEST exits
        Transition(
            TurnPhase.WRITE_TEST,
            TurnPhase.IMPLEMENT,
            guard=_kind("failing_test_submitted"),
            name="write_test->implement",
        ),
        Transition(
            TurnPhase.WRITE_TEST,
            TurnPhase.CLOSE,
            guard=_kind("test_already_passes"),
            name="write_test->close (already green)",
        ),
        Transition(
            TurnPhase.WRITE_TEST,
            TurnPhase.IMPLEMENT,
            guard=_kind("test_phase_skipped"),
            name="write_test->implement (skipped)",
        ),
        Transition(
            TurnPhase.WRITE_TEST,
            TurnPhase.HALTED,
            guard=_kind("phase_no_progress"),
            name="write_test->halted (no test)",
        ),
        # IMPLEMENT exits
        Transition(
            TurnPhase.IMPLEMENT,
            TurnPhase.VERIFY,
            guard=_kind("implement_complete"),
            name="implement->verify",
        ),
        Transition(
            TurnPhase.IMPLEMENT,
            TurnPhase.VERIFY,
            guard=_kind("implement_some_writes"),
            name="implement->verify (budget+writes)",
        ),
        Transition(
            TurnPhase.IMPLEMENT,
            TurnPhase.HALTED,
            guard=_kind("phase_no_progress"),
            name="implement->halted (no writes)",
        ),
        # VERIFY exits
        Transition(
            TurnPhase.VERIFY,
            TurnPhase.CLOSE,
            guard=_kind("verify_passed"),
            name="verify->close",
        ),
        Transition(
            TurnPhase.VERIFY,
            TurnPhase.IMPLEMENT,
            guard=_kind("verify_failed"),
            name="verify->implement (retry)",
        ),
        # CLOSE exits
        Transition(
            TurnPhase.CLOSE,
            TurnPhase.DONE,
            guard=_kind("close_succeeded"),
            name="close->done",
        ),
        Transition(
            TurnPhase.CLOSE,
            TurnPhase.HALTED,
            guard=_kind("close_failed"),
            name="close->halted",
        ),
    )


_TERMINAL_PHASES: frozenset[TurnPhase] = frozenset({TurnPhase.DONE, TurnPhase.HALTED})


def build_turn_fsm(
    *,
    initial: TurnPhase = TurnPhase.ASSESS,
    transitions: tuple[Transition[TurnPhase, PhaseOutcome], ...] | None = None,
) -> StateMachine[TurnPhase, PhaseOutcome]:
    """Construct a fresh TurnFSM at the supplied initial state.

    `transitions` defaults to `_default_transitions()`. Callers
    wanting to override (e.g. a TDD-disabled run that bypasses
    WRITE_TEST entirely) can pass a custom tuple; the generic
    StateMachine doesn't care about which transitions are
    "domain-default" vs custom — it just walks the table."""
    return StateMachine(
        state=initial,
        transitions=transitions if transitions is not None else _default_transitions(),
        terminal_states=_TERMINAL_PHASES,
    )


__all__ = [
    "DEFAULT_PHASE_BUDGETS",
    "PREMISE_UNMET_REASON_PREFIX",
    "_TERMINAL_PHASES",
    "PhaseOutcome",
    "TurnPhase",
    "assessment_skipped_tdd",
    "assessment_submitted",
    "build_turn_fsm",
    "close_failed",
    "close_succeeded",
    "failing_test_submitted",
    "green_test_outcome",
    "implement_complete",
    "phase_no_progress",
    "premise_unmet",
    "verify_failed",
    "verify_passed",
]
