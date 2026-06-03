"""`builds` — does the generated project compile? py_compile over every .py."""

from __future__ import annotations

import py_compile
from pathlib import Path

from adapters.base import RunArtifacts

from scorers.base import Scores


class BuildsScorer:
    name = "builds"

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        py_files = [p for p in workspace.rglob("*.py") if ".git" not in p.parts]
        failed: list[str] = []
        for path in py_files:
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError:
                failed.append(str(path.relative_to(workspace)))
        return {
            "builds": len(py_files) > 0 and not failed,
            "py_files": len(py_files),
            "compile_failures": len(failed),
        }
