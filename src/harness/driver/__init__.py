"""Multi-turn driver — harness-e9oq.

Phase 1 is `state.LoopRunState`: crash-safe persistence for an in-progress
loop run. Subsequent phases (handoff, executor, planner) compose around it.
"""

from harness.driver.state import LoopRunState

__all__ = ["LoopRunState"]
