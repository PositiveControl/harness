"""Runtime-typed plan / subgoal / action graph — harness-ptdw Phase 2.

This package is the source of truth for the agent's own plan
structure. Bd becomes one backend among N (see ptdw.3 / ptdw.5 / ptdw.7
for bd-source, bd-writeback, and sqlite-store slices); the orchestrator
reasons over `Plan` values directly, no subprocess hop per turn.

All records are frozen dataclasses. Mutation is done via
`dataclasses.replace()` or the helper methods on `Plan` — there's no
in-place state.
"""

from harness.plan.bd_source import build_plan_from_bd
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
from harness.plan.revise import (
    WorldSnapshot,
    all_preconditions_satisfied,
    evaluate_precondition,
    revise_plan,
)
from harness.plan.store import (
    JsonPlanStore,
    PlanStore,
    PlanStoreError,
)
from harness.plan.writeback import (
    BeadsIssueDraft,
    PlanWriteback,
    WritebackOpResult,
    WritebackResult,
    apply_writeback,
    diff_plans,
)

__all__ = [
    "STATUS_VALUES",
    "Action",
    "BeadsIssueDraft",
    "JsonPlanStore",
    "Plan",
    "PlanStore",
    "PlanStoreError",
    "PlanWriteback",
    "Precondition",
    "Status",
    "Subgoal",
    "WorldSnapshot",
    "WritebackOpResult",
    "WritebackResult",
    "all_preconditions_satisfied",
    "apply_writeback",
    "build_plan_from_bd",
    "diff_plans",
    "evaluate_precondition",
    "new_plan",
    "new_subgoal",
    "revise_plan",
]
