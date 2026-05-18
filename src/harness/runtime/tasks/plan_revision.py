"""Typed-plan revision heartbeat task — harness-rbj9.

The cross-phase capstone for swvf + ptdw. Each tick:

  1. Load the active Plan from the PlanStore.
  2. Build a `WorldSnapshot` of current bd state (closed + open bead
     id sets + now).
  3. Call `revise_plan` to advance subgoal statuses.
  4. Save the revised Plan back to the PlanStore.
  5. Optionally diff (old, new) and apply the resulting bd writeback
     so bd stays in lockstep with the Plan. Off by default — the
     operator opts in once they trust the revisions.

This is the read-update-write loop ptdw was building toward. With it,
a daemon can keep its plan progressing across days without per-turn
intervention: preconditions get checked, achieved subgoals get retired,
newly-ready subgoals get promoted to active.

Per-tick costs: 3 `list_issues` calls on the bd adapter (closed +
open + in_progress + blocked merged client-side) regardless of plan
size. The closed-set call dominates for old projects; if perf becomes
an issue we can switch to per-precondition `show` calls (cheaper for
small plans) or cache the closed set across ticks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from harness.plan import (
    Plan,
    PlanStoreError,
    Status,
    WorldSnapshot,
    apply_writeback,
    diff_plans,
    revise_plan,
)

if TYPE_CHECKING:
    from harness.plan import PlanStore


@dataclass(frozen=True)
class SubgoalTransition:
    """One subgoal's status change in a revision tick. Captured for
    the outcome audit + the optional episodic-procedural-tier write
    (a future PR — for now the daemon's sink callback prints it)."""

    subgoal_id: str
    before: Status
    after: Status


@dataclass(frozen=True)
class PlanRevisionTaskOutcome:
    """One tick's audit record.

    `loaded=False` means the configured plan_id wasn't on disk — the
    operator hasn't run `harness plan bootstrap` yet (or pointed the
    daemon at the wrong --plan-id). `revised=True` means the Plan
    changed and was saved; `revised=False` means no transitions
    fired and the on-disk Plan is unchanged.

    `writeback_op_count` is non-zero only when writeback is enabled
    AND transitions emitted bd-writable ops (e.g. active→achieved
    on a bd-sourced Subgoal).
    """

    tick_at: datetime
    plan_id: str
    loaded: bool
    revised: bool = False
    transitions: list[SubgoalTransition] = field(default_factory=list)
    writeback_op_count: int = 0
    writeback_errors: list[str] = field(default_factory=list)
    error: str | None = None


def build_plan_revision_task(
    *,
    plan_store: PlanStore,
    plan_id: str,
    bd_adapter: Any,
    assignee: str = "mark",
    apply_bd_writeback: bool = False,
    sink: Callable[[PlanRevisionTaskOutcome], None] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Callable[[], None]:
    """Return a zero-arg callable for `Heartbeat.register()`.

    Args:
        plan_store: source + destination for the Plan.
        plan_id: which plan to load + revise. Daemons typically point
            this at the bootstrap default `bd:<assignee>`.
        bd_adapter: BeadsAdapter for the WorldSnapshot build.
        assignee: bd assignee for filtering open/closed sets. Defaults
            to 'mark' (the user); 'airton_b' for the ab daemon.
        apply_bd_writeback: when True, diff (old, new) and call
            `apply_writeback` so bd-sourced Subgoal transitions
            (active→achieved closes the bead, etc.) propagate to bd.
            Default False — the operator opts in after watching a
            few ticks via the sink.
        sink: optional outcome callback. Daemon logs to console;
            tests assert via list-append.
        clock: injected for tests.
    """

    def task() -> None:
        now = clock()
        try:
            old_plan = plan_store.load(plan_id)
        except PlanStoreError as exc:
            if sink is not None:
                sink(
                    PlanRevisionTaskOutcome(
                        tick_at=now,
                        plan_id=plan_id,
                        loaded=False,
                        error=repr(exc),
                    )
                )
            return

        if old_plan is None:
            if sink is not None:
                sink(
                    PlanRevisionTaskOutcome(
                        tick_at=now,
                        plan_id=plan_id,
                        loaded=False,
                    )
                )
            return

        # Build the WorldSnapshot: closed + open bead id sets +
        # current time. open = open + in_progress + blocked from bd's
        # perspective. Three list_issues calls per tick — cheaper than
        # per-precondition show calls for any plan with more than ~3
        # preconditions.
        try:
            world = _world_snapshot(bd_adapter, assignee=assignee, now=now)
        except Exception as exc:
            if sink is not None:
                sink(
                    PlanRevisionTaskOutcome(
                        tick_at=now,
                        plan_id=plan_id,
                        loaded=True,
                        error=f"world snapshot failed: {exc!r}",
                    )
                )
            return

        new_plan = revise_plan(old_plan, world)
        transitions = _collect_transitions(old_plan, new_plan)

        if new_plan is old_plan or not transitions:
            # No-op tick — leave the on-disk Plan untouched.
            if sink is not None:
                sink(
                    PlanRevisionTaskOutcome(
                        tick_at=now,
                        plan_id=plan_id,
                        loaded=True,
                        revised=False,
                    )
                )
            return

        try:
            plan_store.save(new_plan)
        except Exception as exc:
            if sink is not None:
                sink(
                    PlanRevisionTaskOutcome(
                        tick_at=now,
                        plan_id=plan_id,
                        loaded=True,
                        revised=False,
                        transitions=transitions,
                        error=f"plan save failed: {exc!r}",
                    )
                )
            return

        # Optional: apply the writeback so bd reflects the new plan
        # state. Off by default — opting in is an explicit trust call.
        writeback_count = 0
        writeback_errors: list[str] = []
        if apply_bd_writeback:
            wb = diff_plans(old_plan, new_plan)
            if not wb.is_empty:
                result = apply_writeback(bd_adapter, wb)
                writeback_count = len(result.ops)
                writeback_errors = [op.error or "" for op in result.errors if op.error]

        if sink is not None:
            sink(
                PlanRevisionTaskOutcome(
                    tick_at=now,
                    plan_id=plan_id,
                    loaded=True,
                    revised=True,
                    transitions=transitions,
                    writeback_op_count=writeback_count,
                    writeback_errors=writeback_errors,
                )
            )

    return task


# --- internals --------------------------------------------------------


def _world_snapshot(
    bd_adapter: Any,
    *,
    assignee: str,
    now: datetime,
) -> WorldSnapshot:
    """Pull the bead-id sets we need from bd in three list_issues
    calls. `open` for the WorldSnapshot is the union of bd's open +
    in_progress + blocked statuses (all states where the bead exists
    and hasn't been closed)."""
    closed = bd_adapter.list_issues(status="closed", assignee=assignee)
    open_ = bd_adapter.list_issues(status="open", assignee=assignee)
    in_progress = bd_adapter.list_issues(status="in_progress", assignee=assignee)
    blocked = bd_adapter.list_issues(status="blocked", assignee=assignee)

    def _ids(beads: Any) -> list[str]:
        out: list[str] = []
        for b in beads or ():
            bid = getattr(b, "id", None)
            if isinstance(bid, str) and bid:
                out.append(bid)
        return out

    return WorldSnapshot(
        bd_closed_beads=frozenset(_ids(closed)),
        bd_open_beads=frozenset(_ids(open_) + _ids(in_progress) + _ids(blocked)),
        now=now,
    )


def _collect_transitions(old: Plan, new: Plan) -> list[SubgoalTransition]:
    """Per-subgoal status diff between two Plans. Only emits an entry
    when status changed; preserved subgoals are absent. Output sorted
    by subgoal_id for deterministic logs."""
    out: list[SubgoalTransition] = []
    for sid, old_sg in old.subgoals.items():
        new_sg = new.subgoals.get(sid)
        if new_sg is None:
            continue
        if old_sg.status == new_sg.status:
            continue
        out.append(SubgoalTransition(subgoal_id=sid, before=old_sg.status, after=new_sg.status))
    out.sort(key=lambda t: t.subgoal_id)
    return out
