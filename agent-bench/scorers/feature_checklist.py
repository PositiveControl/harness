"""`feature_checklist` — static probe of milestones M1..M7.

Cheap, honest heuristic: scan generated source for signals that each milestone
was attempted. This is a *static* approximation — it detects presence of the
relevant APIs/patterns, not that they work. Runtime probes (driving the game and
asserting behavior) are a TODO; see the plan's quality section.
"""

from __future__ import annotations

import re
from pathlib import Path

from adapters.base import RunArtifacts

from scorers.base import Scores

# milestone -> regex signals (any match -> attempted). Deliberately conservative.
MILESTONES: dict[str, list[str]] = {
    "m1_window_loop": [r"set_mode", r"pygame\.event\.get", r"while .*:"],
    "m2_player_car": [r"K_(UP|DOWN|LEFT|RIGHT|w|a|s|d)\b", r"get_pressed"],
    "m3_world": [r"camera|offset|scroll", r"blit"],
    "m4_physics": [r"accel|velocity|friction|momentum"],
    "m5_npc": [r"npc|traffic|enemy|ai_", r"class .*(Car|Vehicle|NPC)"],
    "m6_collision": [r"colliderect|collide|Rect\(", r"collision"],
    "m7_objective": [r"score|mission|waypoint|wanted|objective", r"render.*font|HUD|hud"],
}


class FeatureChecklistScorer:
    name = "feature_checklist"

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        src = "\n".join(
            p.read_text(errors="replace") for p in workspace.rglob("*.py") if ".git" not in p.parts
        )
        scores: Scores = {}
        reached = 0
        for milestone, patterns in MILESTONES.items():
            hit = all(re.search(pat, src, re.IGNORECASE) for pat in patterns)
            scores[milestone] = hit
            reached += int(hit)
        scores["milestones_reached"] = reached
        scores["milestones_total"] = len(MILESTONES)
        return scores
