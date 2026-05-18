"""Heartbeat-compatible task builders.

Each module here exports a `build_<name>_task` factory that takes the
stores + adapters the task depends on and returns a zero-arg callable
suitable for `Heartbeat.register()`. Outcomes flow through optional
sink callbacks so the daemon (and tests) can observe per-tick activity
without diving into the task internals.
"""

from harness.runtime.tasks.compaction import (
    CompactionTaskOutcome,
    build_compaction_task,
)
from harness.runtime.tasks.consolidation import (
    ConsolidationTaskOutcome,
    build_consolidation_task,
)
from harness.runtime.tasks.drift import (
    DriftIssue,
    DriftTaskOutcome,
    build_drift_task,
)
from harness.runtime.tasks.plan_revision import (
    PlanRevisionTaskOutcome,
    SubgoalTransition,
    build_plan_revision_task,
)
from harness.runtime.tasks.scheduled_tools import (
    ScheduledFire,
    ScheduledToolsTaskOutcome,
    ScheduleEntry,
    ScheduleError,
    build_scheduled_tools_task,
    load_schedule,
)

__all__ = [
    "CompactionTaskOutcome",
    "ConsolidationTaskOutcome",
    "DriftIssue",
    "DriftTaskOutcome",
    "PlanRevisionTaskOutcome",
    "ScheduleEntry",
    "ScheduleError",
    "ScheduledFire",
    "ScheduledToolsTaskOutcome",
    "SubgoalTransition",
    "build_compaction_task",
    "build_consolidation_task",
    "build_drift_task",
    "build_plan_revision_task",
    "build_scheduled_tools_task",
    "load_schedule",
]
