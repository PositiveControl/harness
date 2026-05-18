"""Tests for the typed-plan-revision heartbeat task — harness-rbj9."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.plan import (
    JsonPlanStore,
    Plan,
    Precondition,
    Subgoal,
    new_plan,
)
from harness.runtime.tasks.plan_revision import (
    PlanRevisionTaskOutcome,
    build_plan_revision_task,
)
from harness.store._bd_types import BeadsIssue

_NOW = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)


def _clock() -> Callable[[], datetime]:
    return lambda: _NOW


@dataclass
class _StubBdAdapter:
    """Duck-typed BeadsAdapter. Returns fixed lists per status; tests
    set bd state up-front. Mirror of test_runtime_drift_task pattern."""

    issues_by_status: dict[str, list[BeadsIssue]] = field(default_factory=dict)
    close_calls: list[tuple[str, str | None]] = field(default_factory=list)
    reopen_calls: list[tuple[str, str | None]] = field(default_factory=list)
    raise_on_list: Exception | None = None

    def list_issues(
        self,
        *,
        status: str | None = None,
        assignee: str | None = None,
        **_: Any,
    ) -> list[BeadsIssue]:
        if self.raise_on_list is not None:
            raise self.raise_on_list
        return list(self.issues_by_status.get(status or "", []))

    def close(self, issue_id: str, *, reason: str | None = None) -> None:
        self.close_calls.append((issue_id, reason))

    def reopen(self, issue_id: str, *, reason: str | None = None) -> None:
        self.reopen_calls.append((issue_id, reason))


def _bead(bd_id: str, status: str = "open") -> BeadsIssue:
    return BeadsIssue(
        id=bd_id,
        title=bd_id,
        status=status,
        priority=2,
        issue_type="task",
        labels=(),
        raw={"id": bd_id},
        assignee="mark",
    )


def _plan_with_pending_bead_dep(plan_id: str, *, dep_bead: str) -> Plan:
    """A Plan whose subgoal is pending until <dep_bead> closes."""
    p = new_plan("test", plan_id=plan_id)
    sg = Subgoal(
        id="sg-1",
        title="needs the dep",
        status="pending",
        preconditions=(Precondition(kind="bd_closed", payload={"bead": dep_bead}),),
        parent_id=p.root_subgoal_id,
    )
    return p.with_subgoal(sg)


def _plan_with_bd_sourced_active(plan_id: str, *, bead_id: str) -> Plan:
    p = new_plan("test", plan_id=plan_id)
    sg = Subgoal(
        id=f"bd:{bead_id}",
        title=bead_id,
        status="active",
        parent_id=p.root_subgoal_id,
    )
    return p.with_subgoal(sg)


# --- happy path -----------------------------------------------------------


def test_revises_plan_when_preconditions_satisfied(tmp_path: Path) -> None:
    """Pending subgoal with bd_closed(harness-x) precondition; bd
    state has harness-x closed → revise promotes to active and saves."""
    store = JsonPlanStore(tmp_path)
    plan = _plan_with_pending_bead_dep("plan-1", dep_bead="harness-x")
    store.save(plan)

    adapter = _StubBdAdapter(issues_by_status={"closed": [_bead("harness-x", status="closed")]})
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    assert len(sink) == 1
    out = sink[0]
    assert out.loaded is True
    assert out.revised is True
    assert len(out.transitions) == 1
    assert out.transitions[0].subgoal_id == "sg-1"
    assert out.transitions[0].before == "pending"
    assert out.transitions[0].after == "active"

    # On-disk Plan updated.
    reloaded = store.load("plan-1")
    assert reloaded is not None
    assert reloaded.subgoal("sg-1").status == "active"


def test_no_transitions_leaves_on_disk_plan_untouched(tmp_path: Path) -> None:
    """When revise_plan returns the input unchanged, the on-disk
    Plan isn't rewritten — saves are avoided on no-op ticks."""
    store = JsonPlanStore(tmp_path)
    plan = _plan_with_pending_bead_dep("plan-1", dep_bead="harness-x")
    store.save(plan)

    # bd state: harness-x still open → no transition.
    adapter = _StubBdAdapter(issues_by_status={"open": [_bead("harness-x", status="open")]})
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    assert sink[0].loaded is True
    assert sink[0].revised is False
    assert sink[0].transitions == []


def test_bd_sourced_active_subgoal_achieved_when_bead_closes(tmp_path: Path) -> None:
    """A bd-sourced subgoal (id 'bd:harness-x') in active status →
    achieved when bd shows harness-x closed."""
    store = JsonPlanStore(tmp_path)
    plan = _plan_with_bd_sourced_active("plan-1", bead_id="harness-x")
    store.save(plan)
    adapter = _StubBdAdapter(issues_by_status={"closed": [_bead("harness-x", status="closed")]})
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    assert sink[0].revised is True
    assert sink[0].transitions[0].after == "achieved"


def test_world_snapshot_unions_open_in_progress_blocked(tmp_path: Path) -> None:
    """bd_open in the snapshot is open + in_progress + blocked, so a
    subgoal with a bd_open(in_progress_bead) precondition is satisfied."""
    store = JsonPlanStore(tmp_path)
    p = new_plan("p", plan_id="plan-1")
    sg = Subgoal(
        id="sg-1",
        title="needs in_progress",
        status="pending",
        preconditions=(Precondition(kind="bd_open", payload={"bead": "harness-y"}),),
        parent_id=p.root_subgoal_id,
    )
    store.save(p.with_subgoal(sg))

    adapter = _StubBdAdapter(
        issues_by_status={"in_progress": [_bead("harness-y", status="in_progress")]}
    )
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    assert sink[0].revised is True
    assert sink[0].transitions[0].after == "active"


# --- writeback path -------------------------------------------------------


def test_writeback_off_by_default(tmp_path: Path) -> None:
    """Default config doesn't push the bd close on active→achieved
    even when one is implied. Operator opts in explicitly."""
    store = JsonPlanStore(tmp_path)
    store.save(_plan_with_bd_sourced_active("plan-1", bead_id="harness-x"))
    adapter = _StubBdAdapter(issues_by_status={"closed": [_bead("harness-x", status="closed")]})
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    assert sink[0].writeback_op_count == 0
    assert adapter.close_calls == []


def test_writeback_on_propagates_active_to_achieved_close(tmp_path: Path) -> None:
    """With apply_bd_writeback=True, the active→achieved transition
    propagates a close call to bd (the bead is *already* closed in
    this scenario — apply_writeback's idempotent path catches that)."""
    store = JsonPlanStore(tmp_path)
    store.save(_plan_with_bd_sourced_active("plan-1", bead_id="harness-x"))
    adapter = _StubBdAdapter(issues_by_status={"closed": [_bead("harness-x", status="closed")]})
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        apply_bd_writeback=True,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    # active→achieved with bd_id 'harness-x' produces a to_close entry,
    # which apply_writeback calls — even though bd shows it already
    # closed, the close call is fired (idempotent at the bd layer).
    assert adapter.close_calls == [("harness-x", "closed via plan writeback")]
    assert sink[0].writeback_op_count == 1


# --- error / missing-plan paths ------------------------------------------


def test_missing_plan_id_logs_loaded_false_and_returns(tmp_path: Path) -> None:
    """Pointing the daemon at a plan_id that doesn't exist on disk
    yet (operator hasn't run bootstrap) is not an error — sink
    records loaded=False and the tick is a no-op."""
    store = JsonPlanStore(tmp_path)
    adapter = _StubBdAdapter()
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-never-bootstrapped",
        bd_adapter=adapter,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    assert sink[0].loaded is False
    assert sink[0].error is None  # missing is not error
    assert sink[0].transitions == []


def test_bd_adapter_failure_lands_in_error(tmp_path: Path) -> None:
    """A raising adapter (bd not initialized, lock contention) yields
    a clean outcome.error with loaded=True but no transitions."""
    store = JsonPlanStore(tmp_path)
    store.save(_plan_with_pending_bead_dep("plan-1", dep_bead="harness-x"))
    adapter = _StubBdAdapter(raise_on_list=RuntimeError("bd lock held"))
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    assert sink[0].loaded is True
    assert sink[0].error is not None
    assert "bd lock held" in sink[0].error


def test_no_sink_runs_silently(tmp_path: Path) -> None:
    """sink=None: task completes; store advanced."""
    store = JsonPlanStore(tmp_path)
    store.save(_plan_with_pending_bead_dep("plan-1", dep_bead="harness-x"))
    adapter = _StubBdAdapter(issues_by_status={"closed": [_bead("harness-x", status="closed")]})
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        sink=None,
        clock=_clock(),
    )
    task()
    reloaded = store.load("plan-1")
    assert reloaded is not None
    assert reloaded.subgoal("sg-1").status == "active"


def test_transitions_sorted_by_subgoal_id_for_determinism(tmp_path: Path) -> None:
    """Multiple transitions in one tick come back sorted by subgoal_id
    so the audit log doesn't drift between runs."""
    store = JsonPlanStore(tmp_path)
    p = new_plan("p", plan_id="plan-1")
    for sid, dep in (("sg-c", "harness-z"), ("sg-a", "harness-x"), ("sg-b", "harness-y")):
        p = p.with_subgoal(
            Subgoal(
                id=sid,
                title=sid,
                status="pending",
                preconditions=(Precondition(kind="bd_closed", payload={"bead": dep}),),
                parent_id=p.root_subgoal_id,
            )
        )
    store.save(p)
    adapter = _StubBdAdapter(
        issues_by_status={
            "closed": [
                _bead("harness-x", "closed"),
                _bead("harness-y", "closed"),
                _bead("harness-z", "closed"),
            ]
        }
    )
    sink: list[PlanRevisionTaskOutcome] = []
    task = build_plan_revision_task(
        plan_store=store,
        plan_id="plan-1",
        bd_adapter=adapter,
        sink=sink.append,
        clock=_clock(),
    )
    task()
    ids = [t.subgoal_id for t in sink[0].transitions]
    assert ids == ["sg-a", "sg-b", "sg-c"]
