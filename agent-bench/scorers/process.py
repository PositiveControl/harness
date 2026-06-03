"""`process` — how the agent worked, not whether it succeeded.

Strong signal is the unified diff (deterministic): files touched/created and
lines added/removed. Softer signals (turns, tool calls, tracebacks) come from
the captured artifacts and are framework-dependent — reported as 0 when the
framework doesn't surface them, never guessed.
"""

from __future__ import annotations

import re
from pathlib import Path

from adapters.base import RunArtifacts

from scorers.base import Scores

_DIFF_GIT = re.compile(r"^diff --git ", re.MULTILINE)
_NEW_FILE = re.compile(r"^new file mode ", re.MULTILINE)
# Tool-call markers: opencode emits `"type":"tool_use"` JSONL; tolerate spacing/dashes.
_TOOL_USE = re.compile(r'"type"\s*:\s*"tool[_-]?use"')
_TRACEBACK = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)


class ProcessScorer:
    name = "process"

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        added, removed = _count_diff_lines(artifacts.diff)
        return {
            "files_touched": len(_DIFF_GIT.findall(artifacts.diff)),
            "files_created": len(_NEW_FILE.findall(artifacts.diff)),
            "lines_added": added,
            "lines_removed": removed,
            "diff_loc": added + removed,
            "turns": artifacts.turns if artifacts.turns is not None else 0,
            "tool_calls": len(_TOOL_USE.findall(artifacts.transcript)),
            "tracebacks": len(_TRACEBACK.findall(artifacts.transcript)),
        }


def _count_diff_lines(diff: str) -> tuple[int, int]:
    """Added/removed content lines in a unified diff, excluding +++/--- headers."""
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed
