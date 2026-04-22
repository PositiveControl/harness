"""bd → episodic bridge.

Two substrates feed the same destination (episodic store, tier=
'procedural', shared scope) so the existing retrieval path surfaces
both on relevant user turns.

- **Thought-graph skills** (`harvester.py`, harness-vu3). Closed
  `thought:decision` and `thought:observation` beads — intentional
  choices and post-mortem learnings that describe how the agent
  acted in similar situations.
- **bd memories** (`memory_harvester.py`, harness-9yd). Free-text
  insights stored via `bd remember` / `retro record` — durable
  knowledge about the user, the character, or the world that would
  otherwise hallucinate when asked.

Both harvesters are idempotent on `external_id` so re-running is
safe: bead id for skills, `bd-mem:<key>` for memories.
"""

from harness.skills.harvester import (
    DEFAULT_HARVEST_LABELS,
    HarvestReport,
    harvest_bd_skills,
)
from harness.skills.memory_harvester import (
    EXTERNAL_ID_PREFIX,
    MEMORY_PRINCIPLE,
    MEMORY_SOURCE,
    MEMORY_TAGS,
    MemoryHarvestReport,
    harvest_bd_memories,
)

__all__ = [
    "DEFAULT_HARVEST_LABELS",
    "EXTERNAL_ID_PREFIX",
    "MEMORY_PRINCIPLE",
    "MEMORY_SOURCE",
    "MEMORY_TAGS",
    "HarvestReport",
    "MemoryHarvestReport",
    "harvest_bd_memories",
    "harvest_bd_skills",
]
