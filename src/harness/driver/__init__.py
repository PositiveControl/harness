"""Multi-turn driver — harness-e9oq.

Composed in phases:
  * `state.LoopRunState` — crash-safe per-run persistence.
  * `bd.DriverBd`         — main-project bd CLI client.
  * `handoff.Handoff` + `build_handoff` — per-turn context block.

Subsequent phases (executor, planner) compose around these.
"""

from harness.driver.bd import DriverBd, DriverBdError
from harness.driver.handoff import Handoff, build_handoff
from harness.driver.state import LoopRunState

__all__ = [
    "DriverBd",
    "DriverBdError",
    "Handoff",
    "LoopRunState",
    "build_handoff",
]
