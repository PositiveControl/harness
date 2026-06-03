"""Scorer protocol. Add a metric by dropping a module here and registering it."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from adapters.base import RunArtifacts

ScoreValue = float | int | bool | str
Scores = dict[str, ScoreValue]


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        """Return metric keys -> values for this run. Keys are namespaced by caller."""
        ...
