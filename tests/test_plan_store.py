"""Tests for the JSON PlanStore implementation — harness-yrwi."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.plan import (
    JsonPlanStore,
    PlanStoreError,
    Precondition,
    new_plan,
    new_subgoal,
)

# --- save / load round trip ----------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = JsonPlanStore(tmp_path)
    plan = new_plan("ship harness", plan_id="plan-abc")
    plan = plan.with_subgoal(
        new_subgoal(
            "subgoal one",
            parent_id=plan.root_subgoal_id,
            preconditions=(Precondition(kind="bd_closed", payload={"bead": "harness-x"}),),
        )
    )
    store.save(plan)
    loaded = store.load("plan-abc")
    assert loaded is not None
    assert loaded.id == plan.id
    assert loaded.title == plan.title
    assert set(loaded.subgoals) == set(plan.subgoals)


def test_load_returns_none_for_missing_plan(tmp_path: Path) -> None:
    """A request for a never-saved plan is not an error."""
    store = JsonPlanStore(tmp_path)
    assert store.load("plan-does-not-exist") is None


# --- atomic write --------------------------------------------------------


def test_save_writes_through_tmp_file(tmp_path: Path) -> None:
    """After a successful save, only the final file exists — the .tmp
    sibling is renamed, not left as artifact."""
    store = JsonPlanStore(tmp_path)
    plan = new_plan("p", plan_id="plan-1")
    store.save(plan)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["plan-1.json"]


def test_save_creates_root_directory_lazily(tmp_path: Path) -> None:
    """First-launch case: the root doesn't exist yet."""
    nested = tmp_path / "deeply" / "nested" / "plans"
    assert not nested.exists()
    store = JsonPlanStore(nested)
    store.save(new_plan("p", plan_id="plan-1"))
    assert (nested / "plan-1.json").exists()


def test_save_overwrites_existing_file(tmp_path: Path) -> None:
    """A second save under the same plan id replaces the file
    contents — the previous Plan version is gone."""
    store = JsonPlanStore(tmp_path)
    v1 = new_plan("first", plan_id="plan-1")
    v2 = new_plan("second", plan_id="plan-1")
    store.save(v1)
    store.save(v2)
    loaded = store.load("plan-1")
    assert loaded is not None
    assert loaded.title == "second"


def test_save_does_not_merge_with_old_file_contents(tmp_path: Path) -> None:
    """A pre-existing file from a different schema must NOT be
    partially merged. .tmp + replace fully overwrites."""
    path = tmp_path / "plan-1.json"
    path.write_text(json.dumps({"old": "schema", "stale_field": True}))
    store = JsonPlanStore(tmp_path)
    store.save(new_plan("new", plan_id="plan-1"))
    on_disk = json.loads(path.read_text())
    assert "old" not in on_disk
    assert "stale_field" not in on_disk
    assert on_disk["title"] == "new"


# --- list_ids ------------------------------------------------------------


def test_list_ids_returns_sorted_filenames(tmp_path: Path) -> None:
    store = JsonPlanStore(tmp_path)
    for pid in ("plan-c", "plan-a", "plan-b"):
        store.save(new_plan(pid, plan_id=pid))
    assert store.list_ids() == ["plan-a", "plan-b", "plan-c"]


def test_list_ids_missing_root_is_empty(tmp_path: Path) -> None:
    """A never-saved-to root that doesn't exist on disk yields []."""
    store = JsonPlanStore(tmp_path / "absent")
    assert store.list_ids() == []


def test_list_ids_ignores_non_json_files(tmp_path: Path) -> None:
    """Stray files (.tmp leftovers, manual notes) are filtered out by
    the `*.json` glob."""
    store = JsonPlanStore(tmp_path)
    store.save(new_plan("p", plan_id="plan-1"))
    (tmp_path / "notes.md").write_text("operator scratchpad")
    (tmp_path / "plan-1.json.tmp").write_text("{}")  # stray
    assert store.list_ids() == ["plan-1"]


def test_list_ids_does_not_parse_files(tmp_path: Path) -> None:
    """A corrupt JSON file shouldn't crash list_ids — it only walks
    filenames. (load() is where the corruption is caught.)"""
    store = JsonPlanStore(tmp_path)
    store.save(new_plan("good", plan_id="plan-good"))
    (tmp_path / "plan-bad.json").write_text("{not valid json")
    assert store.list_ids() == ["plan-bad", "plan-good"]


# --- delete --------------------------------------------------------------


def test_delete_removes_file(tmp_path: Path) -> None:
    store = JsonPlanStore(tmp_path)
    store.save(new_plan("p", plan_id="plan-1"))
    store.delete("plan-1")
    assert store.load("plan-1") is None
    assert store.list_ids() == []


def test_delete_is_idempotent(tmp_path: Path) -> None:
    """Deleting an absent plan must not raise."""
    store = JsonPlanStore(tmp_path)
    store.delete("plan-never-existed")  # no raise
    store.save(new_plan("p", plan_id="plan-1"))
    store.delete("plan-1")
    store.delete("plan-1")  # no raise


# --- error paths ---------------------------------------------------------


def test_load_malformed_json_raises_plan_store_error(tmp_path: Path) -> None:
    """A corrupt JSON file should surface a PlanStoreError naming
    the path so the operator can decide whether to delete + rebuild."""
    (tmp_path / "plan-bad.json").write_text("{not valid json")
    store = JsonPlanStore(tmp_path)
    with pytest.raises(PlanStoreError, match="malformed JSON"):
        store.load("plan-bad")


def test_load_wrong_shape_raises_plan_store_error(tmp_path: Path) -> None:
    """Valid JSON but wrong top-level shape (e.g., a list instead of
    a dict) raises PlanStoreError rather than crashing on Plan.from_dict."""
    (tmp_path / "plan-shape.json").write_text(json.dumps(["not", "a", "plan"]))
    store = JsonPlanStore(tmp_path)
    with pytest.raises(PlanStoreError, match="malformed Plan shape"):
        store.load("plan-shape")


# --- protocol surface ----------------------------------------------------


def test_json_plan_store_satisfies_protocol(tmp_path: Path) -> None:
    """Static and runtime conformance to the PlanStore Protocol — the
    runtime never code-paths on the concrete class, so the substitution
    contract has to hold."""
    from harness.plan import PlanStore

    store: PlanStore = JsonPlanStore(tmp_path)
    # Every Protocol method dispatches without TypeError.
    store.save(new_plan("p", plan_id="plan-1"))
    assert store.load("plan-1") is not None
    assert store.list_ids() == ["plan-1"]
    store.delete("plan-1")
    assert store.list_ids() == []
