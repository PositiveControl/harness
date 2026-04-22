"""Skill library built on ab's bd thought-graph (sota punch #7,
harness-vu3).

Closed-focus `thought:decision` and `thought:observation` beads are the
substrate Voyager would call "skills" — intentional choices and
post-mortem learnings that describe how the agent acted in similar
situations. This module harvests them into the episodic store as
`tier="procedural"` records so the existing retrieval path surfaces
them on relevant user turns ("last time X came up, you decided Y").

`harvest_bd_skills` is idempotent: external_id = bead id means a
second run sees the first run's rows and no-ops.
"""

from harness.skills.harvester import (
    DEFAULT_HARVEST_LABELS,
    HarvestReport,
    harvest_bd_skills,
)

__all__ = ["DEFAULT_HARVEST_LABELS", "HarvestReport", "harvest_bd_skills"]
