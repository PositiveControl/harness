"""Tests for heartbeat-state persistence (load/save) + Heartbeat
snapshot/restore — harness-m64i."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from harness.runtime.heartbeat import Heartbeat
from harness.runtime.state import (
    HeartbeatState,
    TaskState,
    load_state,
    save_state,
)

# --- load_state / save_state -------------------------------------------------


def test_load_state_missing_file_returns_empty(tmp_path: Path) -> None:
    """First-ever daemon launch: no sidecar yet. Empty state, no
    raise."""
    state = load_state(tmp_path / "no-such-file.json")
    assert state.tasks == {}


def test_load_state_malformed_json_returns_empty(tmp_path: Path) -> None:
    """A truncated / hand-edited corrupt file should NOT crash the
    daemon — empty state lets the next save regenerate it."""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    state = load_state(p)
    assert state.tasks == {}


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    original = HeartbeatState(
        tasks={
            "compaction": TaskState(
                last_success_at="2026-05-18T19:00:00+00:00",
                last_error_at=None,
                consecutive_errors=0,
                quarantined=False,
            ),
            "consolidation": TaskState(
                last_error_at="2026-05-18T19:05:00+00:00",
                last_error_msg="RuntimeError('store dead')",
                consecutive_errors=2,
                quarantined=False,
            ),
        }
    )
    save_state(original, p)
    loaded = load_state(p)
    assert loaded.tasks == original.tasks


def test_save_state_is_atomic(tmp_path: Path) -> None:
    """save_state writes through a .tmp sibling and then replaces.
    A pre-existing file with a different schema should be fully
    overwritten, not partially merged."""
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"tasks": {"old": {"consecutive_errors": 99}}}))
    save_state(HeartbeatState(tasks={"new": TaskState(consecutive_errors=1)}), p)
    on_disk = json.loads(p.read_text())
    assert "old" not in on_disk["tasks"]
    assert on_disk["tasks"]["new"]["consecutive_errors"] == 1


def test_load_state_drops_unknown_fields_for_forward_compat(tmp_path: Path) -> None:
    """A future daemon might add a field; an older daemon reading the
    same file should skip it instead of crashing on TypeError."""
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps(
            {
                "tasks": {
                    "task1": {
                        "last_success_at": "2026-05-18T19:00:00+00:00",
                        "future_field": "ignored",
                        "consecutive_errors": 1,
                    }
                }
            }
        )
    )
    state = load_state(p)
    assert "task1" in state.tasks
    assert state.tasks["task1"].consecutive_errors == 1


def test_load_state_skips_non_dict_task_entries(tmp_path: Path) -> None:
    """Defensive: a hand-edit that replaces a task record with a
    string shouldn't crash the loader."""
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"tasks": {"good": {"consecutive_errors": 1}, "bad": "oops"}}))
    state = load_state(p)
    assert set(state.tasks) == {"good"}


def test_state_for_task_returns_empty_for_missing() -> None:
    state = HeartbeatState(tasks={"a": TaskState(consecutive_errors=3)})
    assert state.for_task("a").consecutive_errors == 3
    assert state.for_task("missing").consecutive_errors == 0


def test_save_state_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "state.json"
    save_state(HeartbeatState(tasks={"a": TaskState()}), nested)
    assert nested.exists()


# --- Heartbeat snapshot / restore -------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_captures_wall_clock_after_fire() -> None:
    """A successful fire updates last_success_at; snapshot() exposes
    it as an ISO string."""
    hb = Heartbeat()
    fired = 0

    async def task_a() -> None:
        nonlocal fired
        fired += 1

    hb.register("a", task_a, interval_s=999.0)
    await hb.tick_once()
    snap = hb.snapshot()
    assert "a" in snap.tasks
    assert snap.tasks["a"].last_success_at is not None
    # Round-trips as an ISO datetime.
    parsed = datetime.fromisoformat(snap.tasks["a"].last_success_at)
    assert parsed.tzinfo is not None
    assert snap.tasks["a"].consecutive_errors == 0
    assert snap.tasks["a"].quarantined is False


@pytest.mark.asyncio
async def test_snapshot_captures_error_state() -> None:
    """A raising task: snapshot records last_error_at + last_error_msg
    + the error counter."""
    hb = Heartbeat()

    def boom() -> None:
        raise ValueError("explicit failure")

    hb.register("boom", boom, interval_s=999.0)
    await hb.tick_once()
    snap = hb.snapshot()
    assert snap.tasks["boom"].last_error_at is not None
    assert "ValueError" in (snap.tasks["boom"].last_error_msg or "")
    assert "explicit failure" in (snap.tasks["boom"].last_error_msg or "")
    assert snap.tasks["boom"].consecutive_errors == 1
    assert snap.tasks["boom"].quarantined is False  # below default threshold


def test_restore_applies_persisted_state_to_registered_task() -> None:
    """A daemon restart scenario: the sidecar says 'task X failed 3
    times and is quarantined.' After register() + restore(), the
    in-memory task carries those flags."""
    hb = Heartbeat()

    async def t() -> None:
        pass

    hb.register("flaky", t, interval_s=999.0)
    hb.restore(
        HeartbeatState(
            tasks={
                "flaky": TaskState(
                    last_error_at="2026-05-18T19:00:00+00:00",
                    last_error_msg="ValueError('boom')",
                    consecutive_errors=3,
                    quarantined=True,
                )
            }
        )
    )
    rec = hb.task("flaky")
    assert rec.consecutive_errors == 3
    assert rec.quarantined is True
    assert rec.last_error_msg == "ValueError('boom')"
    assert rec.last_error_at is not None


def test_restore_ignores_unregistered_task_names() -> None:
    """A persisted record for a task the current daemon doesn't
    register (config change, removed task) is silently skipped — not
    an error."""
    hb = Heartbeat()

    async def t() -> None:
        pass

    hb.register("kept", t, interval_s=999.0)
    hb.restore(
        HeartbeatState(
            tasks={
                "kept": TaskState(consecutive_errors=2),
                "stale": TaskState(consecutive_errors=99, quarantined=True),
            }
        )
    )
    assert hb.task("kept").consecutive_errors == 2
    # `stale` was never registered; it doesn't appear.
    assert "stale" not in hb.names()


def test_restore_leaves_unmentioned_registered_tasks_alone() -> None:
    """A newly added task (no entry in the sidecar yet) keeps its
    fresh defaults."""
    hb = Heartbeat()

    async def t() -> None:
        pass

    hb.register("with_history", t, interval_s=999.0)
    hb.register("brand_new", t, interval_s=999.0)
    hb.restore(HeartbeatState(tasks={"with_history": TaskState(consecutive_errors=2)}))
    assert hb.task("with_history").consecutive_errors == 2
    assert hb.task("brand_new").consecutive_errors == 0


@pytest.mark.asyncio
async def test_state_path_autosaves_after_each_fire(tmp_path: Path) -> None:
    """A Heartbeat with state_path set writes the sidecar on every
    fire — so a process crash between fires loses at most one tick of
    state, not the whole task history."""
    path = tmp_path / "state.json"
    hb = Heartbeat(state_path=path)

    async def t() -> None:
        pass

    hb.register("t", t, interval_s=999.0)
    assert not path.exists()  # no save yet
    await hb.tick_once()
    assert path.exists()
    loaded = load_state(path)
    assert loaded.tasks["t"].last_success_at is not None


@pytest.mark.asyncio
async def test_state_path_autosaves_quarantine_flag(tmp_path: Path) -> None:
    """Quarantine after max_consecutive_errors must survive a restart —
    that's the whole point of persisting it. Fire a failing task
    enough times to quarantine; verify the sidecar carries the flag."""
    path = tmp_path / "state.json"
    hb = Heartbeat(state_path=path, max_consecutive_errors=2)

    def boom() -> None:
        raise RuntimeError("dead")

    hb.register("boom", boom, interval_s=999.0)
    await hb.tick_once()
    await hb.tick_once()
    loaded = load_state(path)
    assert loaded.tasks["boom"].quarantined is True
    assert loaded.tasks["boom"].consecutive_errors >= 2


def test_restore_from_disk_no_state_path_is_noop() -> None:
    """A Heartbeat constructed without state_path: restore_from_disk
    is a no-op (no crash, no read)."""
    hb = Heartbeat()
    hb.restore_from_disk()  # must not raise
    assert hb.names() == []


def test_restore_from_disk_missing_file_is_noop(tmp_path: Path) -> None:
    """state_path set but file doesn't exist: restore_from_disk is a
    no-op — first-ever daemon launch."""
    hb = Heartbeat(state_path=tmp_path / "absent.json")

    async def t() -> None:
        pass

    hb.register("t", t, interval_s=999.0)
    hb.restore_from_disk()  # must not raise
    assert hb.task("t").consecutive_errors == 0


@pytest.mark.asyncio
async def test_full_daemon_restart_round_trip(tmp_path: Path) -> None:
    """End-to-end: process 1 fires a failing task, persists state.
    Process 2 (fresh Heartbeat, same state_path) reads + restores +
    the quarantine flag carries across."""
    path = tmp_path / "state.json"

    # First "process"
    hb1 = Heartbeat(state_path=path, max_consecutive_errors=2)

    def boom() -> None:
        raise RuntimeError("dead")

    hb1.register("boom", boom, interval_s=999.0)
    await hb1.tick_once()
    await hb1.tick_once()
    assert hb1.task("boom").quarantined

    # Second "process" — fresh Heartbeat, same state_path.
    hb2 = Heartbeat(state_path=path, max_consecutive_errors=2)
    hb2.register("boom", boom, interval_s=999.0)
    # Before restore, fresh defaults.
    assert hb2.task("boom").quarantined is False
    hb2.restore_from_disk()
    assert hb2.task("boom").quarantined is True
    assert "boom" in hb2.quarantined_tasks()
