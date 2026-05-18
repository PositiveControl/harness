"""Runtime-loop primitives: the agent-side counterpart to user-driven
sessions.

The first inhabitant is the heartbeat — a clock-driven loop separate
from chat turns that runs maintenance tasks (compaction, consolidation,
scheduled tool calls) at configured intervals. See `heartbeat.py` and
the harness-swvf epic for the larger plan.
"""

from harness.runtime.heartbeat import Heartbeat, HeartbeatTask

__all__ = ["Heartbeat", "HeartbeatTask"]
