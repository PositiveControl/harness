"""Runtime-typed plan / subgoal / action graph — harness-ptdw Phase 2.

This package is the source of truth for the agent's own plan
structure. Bd becomes one backend among N (see ptdw.3 / ptdw.5 / ptdw.7
for bd-source, bd-writeback, and sqlite-store slices); the orchestrator
reasons over `Plan` values directly, no subprocess hop per turn.

All records are frozen dataclasses. Mutation is done via
`dataclasses.replace()` or the helper methods on `Plan` — there's no
in-place state.
"""

from harness.plan.model import (
    STATUS_VALUES,
    Action,
    Plan,
    Precondition,
    Status,
    Subgoal,
    new_plan,
    new_subgoal,
)
from harness.plan.store import (
    JsonPlanStore,
    PlanStore,
    PlanStoreError,
)

__all__ = [
    "STATUS_VALUES",
    "Action",
    "JsonPlanStore",
    "Plan",
    "PlanStore",
    "PlanStoreError",
    "Precondition",
    "Status",
    "Subgoal",
    "new_plan",
    "new_subgoal",
]
