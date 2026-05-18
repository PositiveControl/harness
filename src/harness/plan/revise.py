"""Precondition evaluator + plan revision — harness-jige.

The reasoning step that turns the static Plan structure (qm9n) +
external state (bd, time, eventually memory) into a new Plan with
subgoal statuses advanced. Pure function: no I/O, no side effects,
fully testable from in-memory fixtures.

State machine (per Subgoal):

  pending  → active     when every precondition evaluates true
  active   → pending    when a precondition newly evaluates false
                        (rare, but possible — e.g. a bd bead reopens)
  active   → achieved   for bd-sourced subgoals whose bead just
                        closed (the natural completion signal)
  achieved → (sticky)   terminal
  abandoned → (sticky)  terminal (set externally; never auto-set)
  root     → never demoted (stays active throughout the plan's life)

Bd-sourced subgoals (id prefix `bd:`) ride a second completion path:
when the underlying bead lands in `world.bd_closed_beads`, the
subgoal moves directly to `achieved` regardless of precondition
state. Non-bd subgoals stay in `active` until something else marks
them done (a future tool or explicit caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from harness.plan.model import Plan, Precondition, Status, Subgoal


@dataclass(frozen=True)
class WorldSnapshot:
    """External state the evaluator queries. Frozen so a single
    snapshot can be passed to multiple `revise_plan` calls without
    accidental mutation.

    `bd_closed_beads` / `bd_open_beads` are sets for O(1) membership
    checks. `now` is the wall-clock used to evaluate
    `timestamp_past` preconditions; tests pass a fixed value.

    Future fields (Phase 1 dependency): memory-state hooks for
    fact-based preconditions. Out of scope here — extend the snapshot
    + evaluator together when they land."""

    bd_closed_beads: frozenset[str] = field(default_factory=frozenset)
    bd_open_beads: frozenset[str] = field(default_factory=frozenset)
    now: datetime = field(default_factory=lambda: datetime.now(UTC))


# --- evaluator ---------------------------------------------------------


def evaluate_precondition(precondition: Precondition, world: WorldSnapshot) -> bool:
    """Evaluate one precondition against the world snapshot.

    Unknown kinds (a future runtime added one the current binary
    doesn't know) evaluate as False — safer than True because a
    Subgoal stuck in pending is recoverable; one prematurely activated
    isn't."""
    if precondition.kind == "bd_closed":
        bead = precondition.payload.get("bead")
        return isinstance(bead, str) and bead in world.bd_closed_beads
    if precondition.kind == "bd_open":
        bead = precondition.payload.get("bead")
        return isinstance(bead, str) and bead in world.bd_open_beads
    if precondition.kind == "timestamp_past":
        when_raw = precondition.payload.get("when")
        if not isinstance(when_raw, str):
            return False
        try:
            when = datetime.fromisoformat(when_raw)
        except ValueError:
            return False
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return world.now >= when
    return False


def all_preconditions_satisfied(subgoal: Subgoal, world: WorldSnapshot) -> bool:
    """True iff every precondition on `subgoal` evaluates true.
    A subgoal with no preconditions is trivially satisfied."""
    return all(evaluate_precondition(p, world) for p in subgoal.preconditions)


# --- revision ----------------------------------------------------------


def _bd_id_of(subgoal: Subgoal) -> str | None:
    """Extract the bd id from a `bd:<id>` Subgoal id. Returns None
    for non-bd-sourced subgoals (the agent created the subgoal directly
    or the id doesn't follow the convention)."""
    if subgoal.id.startswith("bd:") and not subgoal.id.endswith(":root"):
        return subgoal.id[len("bd:") :]
    return None


def _next_status(subgoal: Subgoal, world: WorldSnapshot, *, is_root: bool) -> Status:
    """Compute the next status for a subgoal under the world snapshot.
    Pure function; the caller diffs against `subgoal.status` to
    decide whether to emit a new Subgoal record."""
    # Terminal states stay put. Root never demotes.
    if subgoal.status in ("achieved", "abandoned"):
        return subgoal.status
    if is_root:
        return subgoal.status

    bd_id = _bd_id_of(subgoal)
    # bd-sourced subgoal whose bead is now closed: completion signal.
    # Applies regardless of preconditions — closing the bead is the
    # human/agent action that signals the work is done.
    if bd_id is not None and bd_id in world.bd_closed_beads:
        return "achieved"

    satisfied = all_preconditions_satisfied(subgoal, world)
    if subgoal.status == "pending" and satisfied:
        return "active"
    if subgoal.status == "active" and not satisfied:
        # Reversible demotion: a precondition newly false drops the
        # subgoal back to pending rather than abandoning it.
        return "pending"
    return subgoal.status


def revise_plan(plan: Plan, world: WorldSnapshot) -> Plan:
    """Return a new Plan with each Subgoal's status advanced under
    `world`. The input Plan is unchanged (frozen dataclasses + dict
    copy).

    Idempotent: revise_plan(revise_plan(plan, world), world) == revise_plan(plan, world).
    """
    new_subgoals: dict[str, Subgoal] = {}
    changed = False
    for sid, sg in plan.subgoals.items():
        is_root = sid == plan.root_subgoal_id
        nxt = _next_status(sg, world, is_root=is_root)
        if nxt == sg.status:
            new_subgoals[sid] = sg
            continue
        new_subgoals[sid] = replace(
            sg,
            status=nxt,
            updated_at=world.now.isoformat(timespec="seconds"),
        )
        changed = True
    if not changed:
        return plan
    return replace(
        plan,
        subgoals=new_subgoals,
        updated_at=world.now.isoformat(timespec="seconds"),
    )
