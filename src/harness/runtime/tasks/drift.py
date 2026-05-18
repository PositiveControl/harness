"""Bd-based drift detection heartbeat task — harness-c32m.

Pre-Phase-2 degraded version of the plan-revision task (rbj9, blocked
on the typed goal graph). Uses bd as the source of truth and runs a
small set of heuristics on each tick:

  * Overload — too many in_progress beads for the watched assignee.
  * Stale in_progress — an in_progress bead with no `updated_at`
    activity within the configured window.

No autonomous action: the task flags drift, doesn't fix it. A future
rbj9 task can replace this with a typed-graph walk that understands
subgoal preconditions; until then, the bd-based heuristic is the
cheapest useful signal available.

Per-tick output is a `DriftTaskOutcome` record carrying the flagged
issues + the inspected counts; daemon surfaces it via the sink
callback the same way compaction/consolidation do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.store.bd_adapter import BeadsAdapter


@dataclass(frozen=True)
class DriftIssue:
    """One flagged drift item.

    `kind` is a short tag the daemon/status command can group by:
      - 'overload': the assignee carries more in_progress work than
        the threshold allows.
      - 'stale_in_progress': a specific bead has been in_progress
        without activity for too long.

    `severity` is 'warn' for actionable drift (something the operator
    should look at) and 'info' for soft signals.
    """

    kind: str
    summary: str
    bead_id: str | None
    last_activity_at: str | None
    severity: str = "warn"


@dataclass(frozen=True)
class DriftTaskOutcome:
    """One tick's drift record.

    `inspected` carries the raw counts the heuristics looked at so a
    'no drift detected' tick is still informative — daemon-status can
    show 'last checked: 3 in_progress, no drift' rather than going
    silent.

    `error` is set when the bd adapter failed (init missing, db
    locked, etc.); in that case `issues` and `inspected` are empty.
    """

    tick_at: datetime
    issues: list[DriftIssue] = field(default_factory=list)
    inspected: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def _parse_iso(value: object) -> datetime | None:
    """Best-effort ISO-8601 parse from a bd `updated_at` field. bd
    emits tz-aware ISO strings; older exports may be naive. Returns
    None on anything we can't decode rather than raising — drift
    detection shouldn't crash on a single weird row."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def build_drift_task(
    *,
    bd_adapter: BeadsAdapter,
    assignee: str = "mark",
    max_in_progress: int = 3,
    stale_after_days: float = 7.0,
    sink: Callable[[DriftTaskOutcome], None] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Callable[[], None]:
    """Return a zero-arg callable for `Heartbeat.register()`.

    Args:
        bd_adapter: constructed adapter pointing at the character's
            bd directory.
        assignee: bd assignee whose work to inspect. Defaults to
            'mark' — for the ab daemon, override to 'airton_b'.
        max_in_progress: flag overload when the assignee's
            in_progress count exceeds this.
        stale_after_days: flag staleness when an in_progress bead's
            `updated_at` is older than this.
        sink: optional callback for the outcome record.
        clock: injected for tests.
    """

    def task() -> None:
        now = clock()
        try:
            in_progress = bd_adapter.list_issues(
                status="in_progress",
                assignee=assignee,
            )
        except Exception as exc:
            # BeadsAdapterError + anything else: surface as a clean
            # error on the outcome so the heartbeat loop continues.
            if sink is not None:
                sink(
                    DriftTaskOutcome(
                        tick_at=now,
                        issues=[],
                        inspected={},
                        error=repr(exc),
                    )
                )
            return

        inspected = {"in_progress_count": len(in_progress)}
        issues: list[DriftIssue] = []

        if len(in_progress) > max_in_progress:
            issues.append(
                DriftIssue(
                    kind="overload",
                    summary=(
                        f"{len(in_progress)} in_progress beads for "
                        f"assignee={assignee!r} exceed threshold "
                        f"{max_in_progress}"
                    ),
                    bead_id=None,
                    last_activity_at=None,
                    severity="warn",
                )
            )

        cutoff = now - timedelta(days=stale_after_days)
        for bead in in_progress:
            updated_at = _parse_iso(bead.raw.get("updated_at"))
            if updated_at is None:
                continue
            if updated_at >= cutoff:
                continue
            days_idle = (now - updated_at).days
            issues.append(
                DriftIssue(
                    kind="stale_in_progress",
                    summary=(
                        f"{bead.id}: in_progress with no update in "
                        f"{days_idle} day(s) (last {updated_at.isoformat()})"
                    ),
                    bead_id=bead.id,
                    last_activity_at=updated_at.isoformat(),
                    severity="warn",
                )
            )

        if sink is not None:
            sink(
                DriftTaskOutcome(
                    tick_at=now,
                    issues=issues,
                    inspected=inspected,
                )
            )

    return task
