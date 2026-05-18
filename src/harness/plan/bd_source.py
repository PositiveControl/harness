"""Bd → Plan read adapter — harness-8jhq.

Materializes a Plan from a `BeadsAdapter`'s view of the assignee's
work. Read-only for v0: the writeback half of the bridge (Plan
mutations → bd updates) is ptdw.5.

Subgoal id convention: `bd:<bead-id>` so the bd id round-trips
losslessly. Writeback (ptdw.5) splits on the prefix to recover the
bd id without needing a side lookup table.

Status mapping (bd → Plan.Status):
  in_progress → active
  open        → pending
  closed      → achieved   (only included when include_closed=True)
  blocked     → pending    (still has unmet preconditions; included)
  deferred    → skipped    (not a current goal; intentionally absent)

Precondition extraction: each Subgoal carries one `bd_closed`
precondition per bd dependency with `dependency_type == 'blocks'`.
The precondition evaluator (ptdw.4) will satisfy these against a
WorldSnapshot.bd_closed_beads set.

Determinism: same bd state → identical Plan (modulo timestamps).
Beads are sorted by id before iteration so iteration order doesn't
leak into the Plan structure. The Plan id defaults to
`bd:<assignee>` so re-running on the same assignee produces the
same Plan id; pass `plan_id=` explicitly for a different anchor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from harness.plan.model import Plan, Precondition, Subgoal

# Status mapping — bd statuses we accept into the Plan.
_STATUS_BY_BD: dict[str, str] = {
    "in_progress": "active",
    "open": "pending",
    "blocked": "pending",
}


def _bead_status_to_plan_status(bd_status: str, include_closed: bool) -> str | None:
    """Map a bd status string to a Plan.Status, or None if the bead
    should be excluded from the Plan."""
    if bd_status == "closed":
        return "achieved" if include_closed else None
    return _STATUS_BY_BD.get(bd_status)


def _subgoal_id_for(bd_id: str) -> str:
    """Stable Subgoal id derived from a bd id. ptdw.5 (writeback)
    splits on the prefix to recover the bd id."""
    return f"bd:{bd_id}"


def _preconditions_from_deps(raw_dependencies: object) -> tuple[Precondition, ...]:
    """Extract `bd_closed` preconditions from a bead's `dependencies`
    JSON list. Each entry is a dict with at least {id, dependency_type};
    we keep only entries with `dependency_type == 'blocks'`."""
    if not isinstance(raw_dependencies, list):
        return ()
    out: list[Precondition] = []
    for dep in raw_dependencies:
        if not isinstance(dep, dict):
            continue
        if dep.get("dependency_type") != "blocks":
            continue
        bead_id = dep.get("id")
        if not isinstance(bead_id, str) or not bead_id:
            continue
        out.append(Precondition(kind="bd_closed", payload={"bead": bead_id}))
    # Sort by bead id so the precondition order is deterministic across
    # runs (bd's `dependencies` field doesn't promise ordering).
    out.sort(key=lambda p: str(p.payload.get("bead", "")))
    return tuple(out)


def build_plan_from_bd(
    adapter: Any,
    *,
    assignee: str,
    root_title: str = "active work",
    plan_id: str | None = None,
    include_closed: bool = False,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Plan:
    """Build a Plan from bd state.

    Args:
        adapter: BeadsAdapter (or any duck-typed object with
            `list_issues` + `show`).
        assignee: bd assignee whose beads enter the Plan.
        root_title: title of the synthesized root Subgoal.
        plan_id: explicit Plan id. Defaults to `bd:<assignee>` for
            reproducibility.
        include_closed: when True, closed beads become `achieved`
            Subgoals so the Plan carries the completion history.
        clock: injected for tests.
    """
    pid = plan_id or f"bd:{assignee}"
    now_iso = clock().isoformat(timespec="seconds")
    root_id = f"{pid}:root"
    root = Subgoal(
        id=root_id,
        title=root_title,
        status="active",
        parent_id=None,
        created_at=now_iso,
        updated_at=now_iso,
    )

    # Fetch every status we might care about. bd returns BeadsIssue
    # objects, not dicts; we touch `.id`, `.title`, `.status`, `.raw`.
    statuses_to_fetch = ["in_progress", "open", "blocked"]
    if include_closed:
        statuses_to_fetch.append("closed")

    seen: dict[str, object] = {}
    for status in statuses_to_fetch:
        for bead in adapter.list_issues(status=status, assignee=assignee):
            bead_id = getattr(bead, "id", None)
            if not isinstance(bead_id, str) or not bead_id:
                continue
            if bead_id not in seen:
                seen[bead_id] = bead

    # Deterministic iteration: sort bd ids alphabetically.
    plan = Plan(
        id=pid,
        title=f"bd: {assignee}",
        root_subgoal_id=root_id,
        subgoals={root_id: root},
        created_at=now_iso,
        updated_at=now_iso,
    )
    for bead_id in sorted(seen):
        bead = seen[bead_id]
        bd_status = str(getattr(bead, "status", "") or "")
        plan_status = _bead_status_to_plan_status(bd_status, include_closed=include_closed)
        if plan_status is None:
            continue
        # Fetch dependencies via show — bd's list_issues doesn't carry
        # the dep edges. Per-bead cost is O(1) calls; acceptable for
        # the bootstrap path (ptdw.8) which runs once.
        shown = adapter.show(bead_id)
        raw = getattr(shown, "raw", None) or {}
        deps_raw = raw.get("dependencies") if isinstance(raw, dict) else None
        preconditions = _preconditions_from_deps(deps_raw)
        title = str(getattr(bead, "title", "") or bead_id)
        subgoal_id = _subgoal_id_for(bead_id)
        subgoal = Subgoal(
            id=subgoal_id,
            title=title,
            status=plan_status,  # type: ignore[arg-type]
            preconditions=preconditions,
            parent_id=root_id,
            created_at=now_iso,
            updated_at=now_iso,
        )
        plan = plan.with_subgoal(subgoal)

    # `with_subgoal` updates `plan.updated_at` to wall-clock-now via
    # _now_iso(); for determinism in tests, freeze it to our injected
    # clock by replacing once at the end.
    return replace(plan, updated_at=now_iso)
