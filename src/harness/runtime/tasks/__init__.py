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

__all__ = [
    "CompactionTaskOutcome",
    "ConsolidationTaskOutcome",
    "build_compaction_task",
    "build_consolidation_task",
]
