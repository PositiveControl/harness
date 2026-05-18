"""Tests for the runtime-typed plan data model — harness-qm9n."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from harness.plan import (
    STATUS_VALUES,
    Action,
    Plan,
    Precondition,
    Subgoal,
    new_plan,
    new_subgoal,
)

# --- record construction ---------------------------------------------------


def test_status_values_enumerated() -> None:
    """STATUS_VALUES is the canonical tuple; future status additions
    update this pin so consumers can match against the full set."""
    assert STATUS_VALUES == ("pending", "active", "achieved", "abandoned")


def test_precondition_defaults() -> None:
    p = Precondition(kind="bd_closed")
    assert p.kind == "bd_closed"
    assert p.payload == {}


def test_action_defaults() -> None:
    a = Action(kind="tool_call")
    assert a.kind == "tool_call"
    assert a.payload == {}


def test_subgoal_defaults() -> None:
    s = Subgoal(id="sg-1", title="something")
    assert s.status == "pending"
    assert s.preconditions == ()
    assert s.actions == ()
    assert s.parent_id is None


def test_new_subgoal_generates_id_and_timestamps() -> None:
    s = new_subgoal("write the doc")
    assert s.id.startswith("sg-")
    assert s.title == "write the doc"
    assert s.created_at  # non-empty
    assert s.updated_at == s.created_at


def test_new_plan_creates_root_subgoal_and_anchors_it() -> None:
    p = new_plan("ship harness 1.0")
    assert p.title == "ship harness 1.0"
    assert p.root_subgoal_id in p.subgoals
    root = p.subgoal(p.root_subgoal_id)
    assert root.status == "active"
    assert root.parent_id is None


# --- helpers ---------------------------------------------------------------


def test_subgoal_lookup_raises_for_unknown() -> None:
    p = new_plan("p")
    with pytest.raises(KeyError, match="no subgoal"):
        p.subgoal("does-not-exist")


def test_with_subgoal_inserts_and_replaces() -> None:
    p = new_plan("p")
    root_id = p.root_subgoal_id
    child = new_subgoal("first child", parent_id=root_id)
    p2 = p.with_subgoal(child)
    assert child.id in p2.subgoals
    # Mutating again replaces.
    updated = replace(child, title="renamed")
    p3 = p2.with_subgoal(updated)
    assert p3.subgoal(child.id).title == "renamed"
    # Original plan unchanged (immutability).
    assert p.subgoals.get(child.id) is None


def test_with_subgoal_rejects_self_parent() -> None:
    p = new_plan("p")
    bad = new_subgoal("loop", parent_id=None)
    bad = replace(bad, parent_id=bad.id)
    with pytest.raises(ValueError, match="cannot be its own parent"):
        p.with_subgoal(bad)


def test_with_subgoal_rejects_unknown_parent() -> None:
    p = new_plan("p")
    orphan = new_subgoal("nope", parent_id="sg-nonexistent")
    with pytest.raises(ValueError, match="is not in the plan"):
        p.with_subgoal(orphan)


def test_with_subgoal_removed_drops_node() -> None:
    p = new_plan("p")
    child = new_subgoal("child", parent_id=p.root_subgoal_id)
    p = p.with_subgoal(child)
    assert child.id in p.subgoals
    p = p.with_subgoal_removed(child.id)
    assert child.id not in p.subgoals


def test_with_subgoal_removed_refuses_to_orphan_children() -> None:
    p = new_plan("p")
    parent = new_subgoal("parent", parent_id=p.root_subgoal_id)
    p = p.with_subgoal(parent)
    leaf = new_subgoal("leaf", parent_id=parent.id)
    p = p.with_subgoal(leaf)
    with pytest.raises(ValueError, match="re-parent before removing"):
        p.with_subgoal_removed(parent.id)


def test_children_of_returns_direct_descendants_only() -> None:
    p = new_plan("p")
    a = new_subgoal("a", parent_id=p.root_subgoal_id)
    p = p.with_subgoal(a)
    b = new_subgoal("b", parent_id=a.id)
    p = p.with_subgoal(b)
    # b is a grandchild of the root; not a child.
    root_children = p.children_of(p.root_subgoal_id)
    assert [c.id for c in root_children] == [a.id]
    a_children = p.children_of(a.id)
    assert [c.id for c in a_children] == [b.id]


def test_active_returns_only_active_status_subgoals() -> None:
    p = new_plan("p")
    pending_sg = new_subgoal("p", parent_id=p.root_subgoal_id, status="pending")
    active_sg = new_subgoal("a", parent_id=p.root_subgoal_id, status="active")
    achieved_sg = new_subgoal("d", parent_id=p.root_subgoal_id, status="achieved")
    p = p.with_subgoal(pending_sg).with_subgoal(active_sg).with_subgoal(achieved_sg)
    active_ids = {s.id for s in p.active()}
    # The root is also `active` by default; both ours + the root land here.
    assert active_sg.id in active_ids
    assert p.root_subgoal_id in active_ids
    assert pending_sg.id not in active_ids
    assert achieved_sg.id not in active_ids


def test_achievable_is_stub_returning_empty_list() -> None:
    """ptdw.4 ships the real evaluator; for now the stub keeps the
    callable available so downstream code can wire it today."""
    p = new_plan("p")
    assert p.achievable() == []


# --- serialization ---------------------------------------------------------


def test_plan_json_round_trip_preserves_all_fields() -> None:
    p = new_plan("ship it", plan_id="plan-abc")
    a = new_subgoal(
        "a",
        parent_id=p.root_subgoal_id,
        preconditions=(
            Precondition(kind="bd_closed", payload={"bead": "harness-x"}),
            Precondition(kind="timestamp_past", payload={"when": "2026-06-01T00:00:00+00:00"}),
        ),
        actions=(Action(kind="tool_call", payload={"tool": "now", "args": {}}),),
    )
    p = p.with_subgoal(a)
    payload = json.dumps(p.to_dict())
    p2 = Plan.from_dict(json.loads(payload))
    assert p2.id == p.id
    assert p2.title == p.title
    assert p2.root_subgoal_id == p.root_subgoal_id
    assert set(p2.subgoals) == set(p.subgoals)
    a2 = p2.subgoal(a.id)
    assert a2.preconditions == a.preconditions
    assert a2.actions == a.actions


def test_from_dict_tolerates_missing_optional_fields() -> None:
    """A minimally-specified Plan dict (no subgoals, no timestamps)
    parses without raising."""
    p = Plan.from_dict({"id": "p", "title": "t", "root_subgoal_id": "p:root"})
    assert p.id == "p"
    assert p.subgoals == {}


def test_from_dict_rejects_non_mapping_top_level() -> None:
    with pytest.raises(ValueError, match="expected dict"):
        Plan.from_dict([])  # type: ignore[arg-type]


def test_from_dict_rejects_non_mapping_subgoals() -> None:
    with pytest.raises(ValueError, match="subgoals must be a mapping"):
        Plan.from_dict({"id": "p", "subgoals": "oops"})


def test_from_dict_rejects_non_mapping_subgoal_entry() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        Plan.from_dict({"id": "p", "subgoals": {"sg-1": "not-a-mapping"}})


def test_from_dict_coerces_unknown_status_to_pending() -> None:
    """A persisted Plan with an unknown status value (because a future
    revision added one) parses as 'pending' rather than crashing."""
    p = Plan.from_dict(
        {
            "id": "p",
            "subgoals": {"sg-1": {"id": "sg-1", "title": "t", "status": "unknown-future-status"}},
        }
    )
    assert p.subgoal("sg-1").status == "pending"


def test_from_dict_drops_non_mapping_preconditions_and_actions() -> None:
    """Defensive: a corrupt list entry inside preconditions shouldn't
    crash the loader; it's just skipped."""
    p = Plan.from_dict(
        {
            "id": "p",
            "subgoals": {
                "sg-1": {
                    "id": "sg-1",
                    "preconditions": [
                        "garbage",
                        {"kind": "bd_closed", "payload": {"bead": "harness-1"}},
                    ],
                    "actions": ["also-garbage"],
                }
            },
        }
    )
    sg = p.subgoal("sg-1")
    assert sg.preconditions == (Precondition(kind="bd_closed", payload={"bead": "harness-1"}),)
    assert sg.actions == ()


def test_asdict_works_on_plan() -> None:
    """The dataclasses.asdict() escape hatch should still produce a
    JSON-safe shape (no enums, no datetimes) so callers that prefer
    asdict over to_dict get the same guarantees."""
    p = new_plan("p")
    raw = asdict(p)
    # round-trips through json without TypeError.
    json.dumps(raw)


def test_frozen_dataclasses_reject_mutation() -> None:
    """Sanity check that Subgoal / Precondition / Action are
    immutable — replacement is the only mutation path. Frozen
    dataclasses raise FrozenInstanceError on attribute assignment."""
    from dataclasses import FrozenInstanceError

    s = Subgoal(id="sg-1", title="t")
    with pytest.raises(FrozenInstanceError):
        s.title = "renamed"  # type: ignore[misc]


def test_subgoal_replace_preserves_other_fields() -> None:
    """dataclasses.replace is the canonical mutation pattern; the
    fields the caller doesn't name are preserved."""
    a = new_subgoal("a", preconditions=(Precondition(kind="bd_closed"),))
    b = replace(a, status="active")
    assert b.status == "active"
    assert b.preconditions == a.preconditions
    assert b.id == a.id
