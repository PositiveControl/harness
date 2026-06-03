"""`feature_checklist` — static probe of web-game milestones M1..M8.

Cheap heuristic: scan index.html + every .js for signals that each milestone
was attempted. Static approximation — detects the relevant canvas/JS patterns,
not that they work. Runtime behavior is graded by `runs_headless` (headless
browser). All-patterns-must-match per milestone, so it stays conservative.
"""

from __future__ import annotations

import re
from pathlib import Path

from adapters.base import RunArtifacts

from scorers.base import Scores

# milestone -> regex signals (all must match -> attempted). Conservative.
MILESTONES: dict[str, list[str]] = {
    "m1_loop": [r"requestAnimationFrame", r"addEventListener\(\s*['\"]key(down|up)"],
    "m2_world": [r"getContext\(\s*['\"]2d", r"drawTile|tile|grid"],
    "m3_player": [r"player\s*=\s*\{", r"\b(translate|rotate)\b", r"\bangle\b"],
    "m4_driving": [r"\bspeed\b", r"Math\.(floor|cos|sin)", r"collision|collide|revert|bounce"],
    "m5_camera": [r"camera", r"translate\(\s*-"],
    "m6_peds": [r"ped|pedestrian", r"\bscore\b"],
    "m7_hud": [r"fillText", r"setTransform\(\s*1\s*,\s*0\s*,\s*0\s*,\s*1"],
    "m8_police": [r"police|cop", r"wanted"],
}


class FeatureChecklistScorer:
    name = "feature_checklist"

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        sources = [
            p
            for p in workspace.rglob("*")
            if p.suffix in {".js", ".html"} and ".git" not in p.parts
        ]
        src = "\n".join(p.read_text(errors="replace") for p in sources)

        scores: Scores = {}
        reached = 0
        for milestone, patterns in MILESTONES.items():
            hit = all(re.search(pat, src, re.IGNORECASE) for pat in patterns)
            scores[milestone] = hit
            reached += int(hit)
        scores["milestones_reached"] = reached
        scores["milestones_total"] = len(MILESTONES)
        return scores
