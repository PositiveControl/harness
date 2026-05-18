"""Tests for the bd-based drift-detection heartbeat task — harness-c32m."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from harness.runtime.tasks.drift import (
    DriftTaskOutcome,
    build_drift_task,
)
from harness.store._bd_types import BeadsIssue


@dataclass
class _StubBdAdapter:
    """Minimal duck-typed BeadsAdapter for drift tests.

    Returns a fixed list of in_progress beads for the configured
    assignee; can be set to raise to exercise the error path. We
    deliberately don't subclass BeadsAdapter — the drift task only
    calls `list_issues(status=..., assignee=...)` and the duck-typed
    stub keeps the test surface tiny.
    """

    in_progress_by_assignee: dict[str, list[BeadsIssue]] = field(default_factory=dict)
    raise_on_list: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def list_issues(
        self,
        *,
        status: str | None = None,
        assignee: str | None = None,
        **_: Any,
    ) -> list[BeadsIssue]:
        self.calls.append({"status": status, "assignee": assignee})
        if self.raise_on_list is not None:
            raise self.raise_on_list
        if status != "in_progress":
            return []
        return list(self.in_progress_by_assignee.get(assignee or "", []))


def _bead(
    id: str,
    *,
    updated_at: str | None = None,
    assignee: str = "mark",
    status: str = "in_progress",
) -> BeadsIssue:
    raw: dict[str, Any] = {"id": id, "title": id, "status": status, "assignee": assignee}
    if updated_at is not None:
        raw["updated_at"] = updated_at
    return BeadsIssue(
        id=id,
        title=id,
        status=status,
        priority=2,
        issue_type="task",
        labels=(),
        raw=raw,
        assignee=assignee,
    )


def _fixed_clock(when: datetime) -> Callable[[], datetime]:
    return lambda: when


# --- happy-path heuristics -------------------------------------------------


def test_clean_state_emits_no_drift() -> None:
    """Below overload threshold + fresh updates → no DriftIssues."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    fresh = (now - timedelta(hours=1)).isoformat()
    adapter = _StubBdAdapter(
        in_progress_by_assignee={
            "mark": [_bead("harness-a", updated_at=fresh)],
        }
    )
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        assignee="mark",
        max_in_progress=3,
        stale_after_days=7,
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    out = sink_records[0]
    assert out.issues == []
    assert out.inspected == {"in_progress_count": 1}
    assert out.error is None


def test_overload_fires_when_in_progress_exceeds_threshold() -> None:
    """4 in_progress + threshold=3 → one 'overload' DriftIssue with
    severity 'warn'."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    fresh = (now - timedelta(hours=1)).isoformat()
    beads = [_bead(f"harness-{i}", updated_at=fresh) for i in range(4)]
    adapter = _StubBdAdapter(in_progress_by_assignee={"mark": beads})
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        assignee="mark",
        max_in_progress=3,
        stale_after_days=7,
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    out = sink_records[0]
    overloads = [i for i in out.issues if i.kind == "overload"]
    assert len(overloads) == 1
    assert "4 in_progress beads" in overloads[0].summary
    assert overloads[0].severity == "warn"
    assert overloads[0].bead_id is None  # overload is a roll-up, not a single bead


def test_stale_in_progress_fires_when_updated_at_older_than_threshold() -> None:
    """A bead with updated_at older than stale_after_days → one
    'stale_in_progress' DriftIssue, named by bead_id, with the parsed
    timestamp echoed back."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    stale_ts = (now - timedelta(days=10)).isoformat()
    fresh_ts = (now - timedelta(hours=1)).isoformat()
    adapter = _StubBdAdapter(
        in_progress_by_assignee={
            "mark": [
                _bead("harness-fresh", updated_at=fresh_ts),
                _bead("harness-stale", updated_at=stale_ts),
            ]
        }
    )
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        assignee="mark",
        stale_after_days=7,
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    out = sink_records[0]
    stale_issues = [i for i in out.issues if i.kind == "stale_in_progress"]
    assert len(stale_issues) == 1
    assert stale_issues[0].bead_id == "harness-stale"
    assert stale_issues[0].last_activity_at == stale_ts
    assert "10 day(s)" in stale_issues[0].summary


def test_both_heuristics_can_fire_in_one_tick() -> None:
    """Overload AND staleness independently — same tick reports both."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    stale_ts = (now - timedelta(days=10)).isoformat()
    fresh_ts = (now - timedelta(hours=1)).isoformat()
    beads = [_bead(f"harness-fresh-{i}", updated_at=fresh_ts) for i in range(3)] + [
        _bead("harness-stale", updated_at=stale_ts)
    ]
    adapter = _StubBdAdapter(in_progress_by_assignee={"mark": beads})
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        assignee="mark",
        max_in_progress=2,
        stale_after_days=7,
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    kinds = {i.kind for i in sink_records[0].issues}
    assert kinds == {"overload", "stale_in_progress"}


def test_bead_with_missing_updated_at_is_not_flagged_as_stale() -> None:
    """A bead whose raw dict has no `updated_at` key shouldn't crash
    or be flagged as stale (we don't know its age)."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    adapter = _StubBdAdapter(
        in_progress_by_assignee={"mark": [_bead("harness-no-ts", updated_at=None)]}
    )
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        assignee="mark",
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    assert sink_records[0].issues == []


def test_bead_with_malformed_updated_at_is_skipped_not_crashed() -> None:
    """A corrupt ISO string in `updated_at` shouldn't crash the task."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    adapter = _StubBdAdapter(
        in_progress_by_assignee={"mark": [_bead("harness-bad-ts", updated_at="not-a-timestamp")]}
    )
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        assignee="mark",
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    out = sink_records[0]
    # No crash, no stale flag (we couldn't determine the age).
    assert out.issues == []
    assert out.error is None


def test_naive_updated_at_is_treated_as_utc() -> None:
    """bd export occasionally emits naive ISO strings. Assume UTC
    rather than tossing them — staleness math still needs to work."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    # Naive (no offset), 10 days old in UTC.
    naive_old = (now - timedelta(days=10)).replace(tzinfo=None).isoformat()
    adapter = _StubBdAdapter(
        in_progress_by_assignee={"mark": [_bead("harness-old", updated_at=naive_old)]}
    )
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        assignee="mark",
        stale_after_days=7,
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    stale_kinds = [i.kind for i in sink_records[0].issues]
    assert stale_kinds == ["stale_in_progress"]


# --- error + observability paths -------------------------------------------


def test_adapter_error_captured_into_outcome() -> None:
    """A raising bd adapter (init missing, db locked) lands in
    outcome.error with empty issues/inspected — heartbeat continues."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    adapter = _StubBdAdapter(raise_on_list=RuntimeError("bd not initialized"))
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    out = sink_records[0]
    assert out.error is not None
    assert "bd not initialized" in out.error
    assert out.issues == []
    assert out.inspected == {}


def test_no_sink_runs_silently() -> None:
    """sink=None: no records appended, but the task doesn't raise."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    adapter = _StubBdAdapter(
        in_progress_by_assignee={"mark": [_bead("harness-a", updated_at=now.isoformat())]}
    )
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        sink=None,
        clock=_fixed_clock(now),
    )
    task()  # must not raise


def test_inspected_counts_reflect_inspected_volume() -> None:
    """The inspected dict reports the count even when no drift fires —
    daemon-status can show 'no drift, inspected 5'."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    fresh = now.isoformat()
    adapter = _StubBdAdapter(
        in_progress_by_assignee={"mark": [_bead(f"h-{i}", updated_at=fresh) for i in range(5)]}
    )
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        max_in_progress=10,
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    assert sink_records[0].inspected == {"in_progress_count": 5}


def test_assignee_filter_passes_through_to_adapter() -> None:
    """Sanity: the task asks the adapter for the configured assignee,
    not the default."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    adapter = _StubBdAdapter()
    sink_records: list[DriftTaskOutcome] = []
    task = build_drift_task(
        bd_adapter=adapter,  # type: ignore[arg-type]
        assignee="airton_b",
        sink=sink_records.append,
        clock=_fixed_clock(now),
    )
    task()
    assert adapter.calls == [{"status": "in_progress", "assignee": "airton_b"}]
