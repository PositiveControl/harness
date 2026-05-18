"""Tests for the scheduled-tool-calls heartbeat task — harness-6dnf."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from harness.runtime.tasks.scheduled_tools import (
    ScheduledToolsTaskOutcome,
    ScheduleError,
    build_scheduled_tools_task,
    load_schedule,
)


def _fixed_clock(when: datetime) -> Callable[[], datetime]:
    return lambda: when


def _write_schedule(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump(entries))


# --- load_schedule ---------------------------------------------------------


def test_load_schedule_missing_file_returns_empty(tmp_path: Path) -> None:
    """First-launch case: schedule file doesn't exist yet."""
    entries = load_schedule(tmp_path / "no-such-schedule.yaml")
    assert entries == []


def test_load_schedule_parses_valid_entries(tmp_path: Path) -> None:
    """A well-formed YAML file produces ScheduleEntry records with the
    documented defaults — `id` falls back to `<tool>#<index>`,
    `description` is None when absent."""
    p = tmp_path / "sched.yaml"
    _write_schedule(
        p,
        [
            {"tool": "search_web", "args": {"query": "X"}, "interval": 3600},
            {"id": "notes", "tool": "scribe_session", "args": {}, "interval": 1800},
        ],
    )
    entries = load_schedule(p)
    assert len(entries) == 2
    assert entries[0].id == "search_web#0"
    assert entries[0].tool == "search_web"
    assert entries[0].args == {"query": "X"}
    assert entries[0].interval_s == 3600.0
    assert entries[0].description is None
    assert entries[1].id == "notes"


def test_load_schedule_rejects_top_level_non_list(tmp_path: Path) -> None:
    p = tmp_path / "sched.yaml"
    p.write_text("tool: search_web\ninterval: 60\n")  # mapping at root
    with pytest.raises(ScheduleError, match="top-level must be a list"):
        load_schedule(p)


def test_load_schedule_rejects_missing_tool(tmp_path: Path) -> None:
    p = tmp_path / "sched.yaml"
    _write_schedule(p, [{"args": {}, "interval": 60}])  # no tool
    with pytest.raises(ScheduleError, match="'tool' must be a non-empty string"):
        load_schedule(p)


def test_load_schedule_rejects_non_positive_interval(tmp_path: Path) -> None:
    p = tmp_path / "sched.yaml"
    _write_schedule(p, [{"tool": "now", "interval": 0}])
    with pytest.raises(ScheduleError, match="'interval' must be positive"):
        load_schedule(p)


def test_load_schedule_rejects_bool_interval(tmp_path: Path) -> None:
    """bool is a subclass of int in Python — would silently pass an
    isinstance(_, int) check. Reject explicitly so 'interval: true' in
    YAML fails loudly."""
    p = tmp_path / "sched.yaml"
    _write_schedule(p, [{"tool": "now", "interval": True}])
    with pytest.raises(ScheduleError, match="'interval' must be a number"):
        load_schedule(p)


def test_load_schedule_rejects_duplicate_ids(tmp_path: Path) -> None:
    p = tmp_path / "sched.yaml"
    _write_schedule(
        p,
        [
            {"id": "dup", "tool": "now", "args": {}, "interval": 60},
            {"id": "dup", "tool": "calc", "args": {"expr": "1+1"}, "interval": 60},
        ],
    )
    with pytest.raises(ScheduleError, match="duplicate id 'dup'"):
        load_schedule(p)


def test_load_schedule_rejects_malformed_yaml(tmp_path: Path) -> None:
    p = tmp_path / "sched.yaml"
    p.write_text("- tool: search_web\n  args: {unclosed: \n")
    with pytest.raises(ScheduleError, match="YAML parse error"):
        load_schedule(p)


# --- task firing -----------------------------------------------------------


def test_task_fires_entry_with_no_prior_state(tmp_path: Path) -> None:
    """First tick: no state file yet, every entry is due, all fire and
    the state sidecar records the timestamps."""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    _write_schedule(schedule, [{"tool": "now", "args": {}, "interval": 60}])

    executor_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_executor(tool: str, args: dict[str, Any]) -> str:
        executor_calls.append((tool, args))
        return "now=2026-05-18"

    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    sink_records: list[ScheduledToolsTaskOutcome] = []
    task = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=fake_executor,
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()

    assert executor_calls == [("now", {})]
    out = sink_records[0]
    assert out.entries_loaded == 1
    assert len(out.fires) == 1
    assert out.fires[0].success is True
    assert out.fires[0].output_preview == "now=2026-05-18"
    assert out.fires[0].entry_id == "now#0"

    persisted = json.loads(state.read_text())
    assert persisted["last_fire_at"]["now#0"].startswith("2026-05-18T12:00:00")


def test_task_skips_entry_within_interval(tmp_path: Path) -> None:
    """Second tick before the interval elapses: entry is in skipped_ids,
    not fires, and the executor isn't called."""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    _write_schedule(schedule, [{"tool": "now", "args": {}, "interval": 3600}])

    calls = 0

    def exec_fn(tool: str, args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return "ok"

    sink_records: list[ScheduledToolsTaskOutcome] = []
    t0 = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    task = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=exec_fn,
        sink=sink_records.append,
        clock=_fixed_clock(t0),
    )
    task()  # fires
    # 5 min later — interval is 1 h, so still within window.
    t1 = t0 + timedelta(minutes=5)
    task2 = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=exec_fn,
        sink=sink_records.append,
        clock=_fixed_clock(t1),
    )
    task2()
    assert calls == 1
    assert sink_records[1].fires == []
    assert sink_records[1].skipped_ids == ["now#0"]


def test_task_refires_after_interval(tmp_path: Path) -> None:
    """Three ticks: t0 fires, t0+30min skips (interval=60min), t0+90min
    fires again. Validates the math + persistence."""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    _write_schedule(schedule, [{"tool": "now", "args": {}, "interval": 3600}])

    fires = 0

    def exec_fn(tool: str, args: dict[str, Any]) -> str:
        nonlocal fires
        fires += 1
        return "ok"

    t0 = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    for now in (t0, t0 + timedelta(minutes=30), t0 + timedelta(minutes=90)):
        task = build_scheduled_tools_task(
            schedule_path=schedule,
            state_path=state,
            tool_executor=exec_fn,
            clock=_fixed_clock(now),
        )
        task()
    assert fires == 2


def test_task_continues_through_executor_errors(tmp_path: Path) -> None:
    """One failing entry + one passing entry: both processed, error
    captured into the failing fire record, the rest still runs."""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    _write_schedule(
        schedule,
        [
            {"tool": "broken", "args": {}, "interval": 60},
            {"tool": "now", "args": {}, "interval": 60},
        ],
    )

    def exec_fn(tool: str, args: dict[str, Any]) -> str:
        if tool == "broken":
            raise RuntimeError("tool dead")
        return "ok"

    sink_records: list[ScheduledToolsTaskOutcome] = []
    task = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=exec_fn,
        sink=sink_records.append,
        clock=_fixed_clock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC)),
    )
    task()
    out = sink_records[0]
    assert len(out.fires) == 2
    bad = next(f for f in out.fires if f.tool == "broken")
    good = next(f for f in out.fires if f.tool == "now")
    assert bad.success is False
    assert "RuntimeError" in (bad.error or "")
    assert good.success is True


def test_task_advances_timestamp_on_error_to_avoid_hammering(tmp_path: Path) -> None:
    """A permanently broken entry shouldn't fire every tick — the
    timestamp advances on failure too. (Quarantine on repeated failure
    is a follow-up; this is the minimum protection against tight
    error loops.)"""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    _write_schedule(schedule, [{"tool": "broken", "args": {}, "interval": 60}])

    fires = 0

    def exec_fn(tool: str, args: dict[str, Any]) -> str:
        nonlocal fires
        fires += 1
        raise RuntimeError("dead")

    t0 = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    task1 = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=exec_fn,
        clock=_fixed_clock(t0),
    )
    task1()  # fires, errors, advances ts
    # 5s later (well within 60s interval): should be skipped.
    task2 = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=exec_fn,
        clock=_fixed_clock(t0 + timedelta(seconds=5)),
    )
    task2()
    assert fires == 1


def test_task_load_error_surfaces_via_outcome(tmp_path: Path) -> None:
    """A malformed schedule lands in outcome.load_error rather than
    crashing the heartbeat. Existing state is preserved (we never
    saved this tick)."""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    schedule.write_text("not: a: list")  # parseable YAML but wrong shape

    sink_records: list[ScheduledToolsTaskOutcome] = []
    task = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=lambda *_a: "ok",
        sink=sink_records.append,
        clock=_fixed_clock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC)),
    )
    task()
    out = sink_records[0]
    assert out.load_error is not None
    assert out.entries_loaded == 0
    assert out.fires == []


def test_task_truncates_long_output_in_preview(tmp_path: Path) -> None:
    """Large tool output isn't blown into the outcome record verbatim —
    `output_preview` truncates to `preview_chars` (200 by default)."""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    _write_schedule(schedule, [{"tool": "noisy", "args": {}, "interval": 60}])

    big_blob = "x" * 5000

    sink_records: list[ScheduledToolsTaskOutcome] = []
    task = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=lambda *_a: big_blob,
        sink=sink_records.append,
        clock=_fixed_clock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC)),
    )
    task()
    preview = sink_records[0].fires[0].output_preview or ""
    assert len(preview) <= 200
    assert preview.endswith("…")


def test_task_reloads_schedule_each_tick(tmp_path: Path) -> None:
    """Editing the schedule between ticks should take effect on the
    next tick — no daemon restart needed."""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    _write_schedule(schedule, [{"tool": "now", "args": {}, "interval": 60}])

    sink_records: list[ScheduledToolsTaskOutcome] = []
    t0 = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    task = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=lambda *_a: "ok",
        sink=sink_records.append,
        clock=_fixed_clock(t0),
    )
    task()
    assert sink_records[0].entries_loaded == 1
    _write_schedule(
        schedule,
        [
            {"tool": "now", "args": {}, "interval": 60},
            {"tool": "calc", "args": {"expr": "1+1"}, "interval": 60},
        ],
    )
    # Far enough out that both fire.
    task2 = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=lambda *_a: "ok",
        sink=sink_records.append,
        clock=_fixed_clock(t0 + timedelta(hours=1)),
    )
    task2()
    assert sink_records[1].entries_loaded == 2


def test_task_no_sink_runs_silently(tmp_path: Path) -> None:
    """sink=None: task does its work, state is persisted, no records
    appended."""
    schedule = tmp_path / "sched.yaml"
    state = tmp_path / "state.json"
    _write_schedule(schedule, [{"tool": "now", "args": {}, "interval": 60}])
    task = build_scheduled_tools_task(
        schedule_path=schedule,
        state_path=state,
        tool_executor=lambda *_a: "ok",
        sink=None,
        clock=_fixed_clock(datetime(2026, 5, 18, 12, 0, tzinfo=UTC)),
    )
    task()
    persisted = json.loads(state.read_text())
    assert "now#0" in persisted["last_fire_at"]
