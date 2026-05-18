"""Tests for the precondition evaluator + plan revision — harness-jige."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from harness.plan import (
    Precondition,
    Subgoal,
    WorldSnapshot,
    all_preconditions_satisfied,
    evaluate_precondition,
    new_plan,
    new_subgoal,
    revise_plan,
)

_NOW = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)


def _world(
    *,
    closed: Iterable[str] = (),
    open: Iterable[str] = (),
    now: datetime = _NOW,
) -> WorldSnapshot:
    """Build a WorldSnapshot anchored to _NOW with optional overrides."""
    return WorldSnapshot(
        bd_closed_beads=frozenset(closed),
        bd_open_beads=frozenset(open),
        now=now,
    )


# --- evaluate_precondition ------------------------------------------------


def test_bd_closed_satisfied_when_bead_in_closed_set() -> None:
    p = Precondition(kind="bd_closed", payload={"bead": "harness-x"})
    assert evaluate_precondition(p, _world(closed=["harness-x"])) is True


def test_bd_closed_unsatisfied_when_bead_not_in_closed_set() -> None:
    p = Precondition(kind="bd_closed", payload={"bead": "harness-x"})
    assert evaluate_precondition(p, _world(closed=["harness-y"])) is False
    assert evaluate_precondition(p, _world(closed=[])) is False


def test_bd_open_satisfied_when_bead_in_open_set() -> None:
    p = Precondition(kind="bd_open", payload={"bead": "harness-x"})
    assert evaluate_precondition(p, _world(open=["harness-x"])) is True


def test_bd_open_unsatisfied_when_bead_absent() -> None:
    p = Precondition(kind="bd_open", payload={"bead": "harness-x"})
    assert evaluate_precondition(p, _world(open=["harness-y"])) is False


def test_timestamp_past_satisfied_when_now_at_or_after() -> None:
    when = (_NOW - timedelta(hours=1)).isoformat()
    p = Precondition(kind="timestamp_past", payload={"when": when})
    assert evaluate_precondition(p, _world()) is True


def test_timestamp_past_unsatisfied_when_now_before_target() -> None:
    when = (_NOW + timedelta(hours=1)).isoformat()
    p = Precondition(kind="timestamp_past", payload={"when": when})
    assert evaluate_precondition(p, _world()) is False


def test_timestamp_past_naive_treated_as_utc() -> None:
    """A persisted ISO string without an offset is read as UTC."""
    when = (_NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat()
    p = Precondition(kind="timestamp_past", payload={"when": when})
    assert evaluate_precondition(p, _world()) is True


def test_timestamp_past_malformed_evaluates_false() -> None:
    """A corrupt ISO string should fail closed, not crash."""
    p = Precondition(kind="timestamp_past", payload={"when": "not-iso"})
    assert evaluate_precondition(p, _world()) is False


def test_unknown_precondition_kind_fails_closed() -> None:
    """A future precondition kind the current binary doesn't understand
    evaluates False — safer than True (a stuck pending subgoal is
    recoverable; a prematurely activated one isn't)."""
    p = Precondition(kind="future_kind", payload={"x": 1})
    assert evaluate_precondition(p, _world()) is False


def test_bd_closed_missing_bead_field_fails_closed() -> None:
    """Defensive: precondition with no bead field is unsatisfiable."""
    p = Precondition(kind="bd_closed", payload={})
    assert evaluate_precondition(p, _world(closed=["harness-x"])) is False


# --- all_preconditions_satisfied -----------------------------------------


def test_all_preconditions_satisfied_empty_list_is_true() -> None:
    """A subgoal with no preconditions is trivially achievable."""
    sg = new_subgoal("noop")
    assert all_preconditions_satisfied(sg, _world()) is True


def test_all_preconditions_satisfied_requires_every_one() -> None:
    sg = new_subgoal(
        "needs both",
        preconditions=(
            Precondition(kind="bd_closed", payload={"bead": "x"}),
            Precondition(kind="bd_closed", payload={"bead": "y"}),
        ),
    )
    assert all_preconditions_satisfied(sg, _world(closed=["x"])) is False
    assert all_preconditions_satisfied(sg, _world(closed=["x", "y"])) is True


# --- revise_plan: state transitions --------------------------------------


def test_pending_to_active_when_preconditions_satisfied() -> None:
    p = new_plan("p", plan_id="plan-1")
    sg = new_subgoal(
        "sg",
        parent_id=p.root_subgoal_id,
        status="pending",
        preconditions=(Precondition(kind="bd_closed", payload={"bead": "x"}),),
    )
    p = p.with_subgoal(sg)
    revised = revise_plan(p, _world(closed=["x"]))
    assert revised.subgoal(sg.id).status == "active"


def test_pending_stays_pending_when_preconditions_unsatisfied() -> None:
    p = new_plan("p", plan_id="plan-1")
    sg = new_subgoal(
        "sg",
        parent_id=p.root_subgoal_id,
        status="pending",
        preconditions=(Precondition(kind="bd_closed", payload={"bead": "x"}),),
    )
    p = p.with_subgoal(sg)
    revised = revise_plan(p, _world(closed=[]))
    assert revised.subgoal(sg.id).status == "pending"


def test_active_to_pending_when_precondition_newly_false() -> None:
    """A bd bead reopens (rare but possible) — the dependent active
    subgoal drops back to pending. Reversible transitions are the
    point of separating pending from abandoned."""
    p = new_plan("p", plan_id="plan-1")
    sg = new_subgoal(
        "sg",
        parent_id=p.root_subgoal_id,
        status="active",
        preconditions=(Precondition(kind="bd_open", payload={"bead": "x"}),),
    )
    p = p.with_subgoal(sg)
    # x is currently NOT in open set → precondition false → demote.
    revised = revise_plan(p, _world(open=[]))
    assert revised.subgoal(sg.id).status == "pending"


def test_bd_sourced_active_to_achieved_when_bead_closes() -> None:
    """A subgoal with id 'bd:<bead-id>' transitions to achieved when
    that bead is now closed — the natural completion signal."""
    p = new_plan("p", plan_id="plan-1")
    sg = Subgoal(
        id="bd:harness-x",
        title="x",
        status="active",
        parent_id=p.root_subgoal_id,
    )
    p = p.with_subgoal(sg)
    revised = revise_plan(p, _world(closed=["harness-x"]))
    assert revised.subgoal("bd:harness-x").status == "achieved"


def test_bd_sourced_completion_overrides_unsatisfied_preconditions() -> None:
    """Closing the bead trumps everything else. A subgoal with an
    unsatisfied precondition still achieves if its bead closes —
    the human/agent action of closing the bead is the load-bearing
    signal."""
    p = new_plan("p", plan_id="plan-1")
    sg = Subgoal(
        id="bd:harness-x",
        title="x",
        status="active",
        parent_id=p.root_subgoal_id,
        preconditions=(Precondition(kind="bd_open", payload={"bead": "harness-y"}),),
    )
    p = p.with_subgoal(sg)
    # Precondition unsatisfied but bd is closed.
    revised = revise_plan(p, _world(closed=["harness-x"]))
    assert revised.subgoal("bd:harness-x").status == "achieved"


def test_achieved_is_sticky() -> None:
    """Once achieved, always achieved — even if preconditions go
    unsatisfied or the bd state changes."""
    p = new_plan("p", plan_id="plan-1")
    sg = Subgoal(
        id="bd:harness-x",
        title="x",
        status="achieved",
        parent_id=p.root_subgoal_id,
    )
    p = p.with_subgoal(sg)
    # Bead reopens — but achieved stays.
    revised = revise_plan(p, _world(open=["harness-x"]))
    assert revised.subgoal("bd:harness-x").status == "achieved"


def test_abandoned_is_sticky() -> None:
    p = new_plan("p", plan_id="plan-1")
    sg = new_subgoal("sg", parent_id=p.root_subgoal_id, status="abandoned")
    p = p.with_subgoal(sg)
    revised = revise_plan(p, _world(closed=["x"]))
    assert revised.subgoal(sg.id).status == "abandoned"


def test_root_subgoal_never_demoted() -> None:
    """The root subgoal stays in its initial 'active' state regardless
    of preconditions (it has none) and bd closures."""
    p = new_plan("p", plan_id="plan-1")
    revised = revise_plan(p, _world(closed=[]))
    assert revised.subgoal(p.root_subgoal_id).status == "active"


def test_root_with_bd_prefix_not_treated_as_bd_completion_signal() -> None:
    """The root id contains the substring 'bd:' when plan_id starts
    with 'bd:' (build_plan_from_bd does this). Make sure the root
    isn't mistakenly auto-achieved when some unrelated bead closes."""
    p = new_plan("p", plan_id="bd:mark")  # root id = 'bd:mark:root'
    # Lookup logic: _bd_id_of returns None for ':root' suffix.
    revised = revise_plan(p, _world(closed=["mark"]))
    assert revised.subgoal(p.root_subgoal_id).status == "active"


# --- revise_plan: idempotence + immutability ------------------------------


def test_revise_plan_idempotent() -> None:
    """A no-op world (no state changes) returns a Plan equal to the
    second-pass result."""
    p = new_plan("p", plan_id="plan-1")
    sg = new_subgoal(
        "sg",
        parent_id=p.root_subgoal_id,
        status="pending",
        preconditions=(Precondition(kind="bd_closed", payload={"bead": "x"}),),
    )
    p = p.with_subgoal(sg)
    world = _world(closed=["x"])
    once = revise_plan(p, world)
    twice = revise_plan(once, world)
    # Same statuses on both passes (the second is the no-op).
    for sid, original in once.subgoals.items():
        assert twice.subgoal(sid).status == original.status


def test_revise_plan_with_no_changes_returns_same_instance() -> None:
    """Optimization: when nothing transitions, the function returns
    the input Plan unchanged (object-identical) so downstream code
    can short-circuit."""
    p = new_plan("p", plan_id="plan-1")
    revised = revise_plan(p, _world())
    assert revised is p


def test_revise_plan_does_not_mutate_input() -> None:
    """Frozen dataclasses guarantee this at runtime but a regression
    here would still be expensive — pin the contract explicitly."""
    p = new_plan("p", plan_id="plan-1")
    sg = new_subgoal(
        "sg",
        parent_id=p.root_subgoal_id,
        status="pending",
        preconditions=(Precondition(kind="bd_closed", payload={"bead": "x"}),),
    )
    p = p.with_subgoal(sg)
    snapshot = p.to_dict()
    revise_plan(p, _world(closed=["x"]))
    assert p.to_dict() == snapshot


def test_revise_plan_updates_timestamps_only_for_changed_subgoals() -> None:
    p = new_plan("p", plan_id="plan-1")
    a = new_subgoal(
        "a",
        parent_id=p.root_subgoal_id,
        status="pending",
        preconditions=(Precondition(kind="bd_closed", payload={"bead": "x"}),),
    )
    b = new_subgoal("b", parent_id=p.root_subgoal_id, status="pending")
    p = p.with_subgoal(a).with_subgoal(b)
    # x closes → a transitions; b has no preconditions but starts
    # pending → also transitions (precondition list empty == satisfied).
    # Re-target the test: give b a never-true precondition so it
    # stays pending.
    b_pinned = replace(
        p.subgoal(b.id),
        preconditions=(Precondition(kind="bd_open", payload={"bead": "nope"}),),
    )
    p = p.with_subgoal(b_pinned)
    revised = revise_plan(p, _world(closed=["x"]))
    # a transitions → updated_at advanced. b stays pending → unchanged.
    assert revised.subgoal(a.id).updated_at == _NOW.isoformat(timespec="seconds")
    assert revised.subgoal(b.id).updated_at == b_pinned.updated_at
