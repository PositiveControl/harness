"""`builds` — does the generated web game parse + have a valid entry page?

JS syntax via `node --check` over every .js file; HTML sanity on index.html
(has a <canvas> and a <script>). Static — proves the code parses and the page is
wired, not that it runs (that's runs_headless).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from adapters.base import RunArtifacts

from scorers.base import Scores

_CANVAS_RE = "<canvas"
_SCRIPT_RE = "<script"


class BuildsScorer:
    name = "builds"

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        js_files = [p for p in workspace.rglob("*.js") if ".git" not in p.parts]
        failed = [str(p.relative_to(workspace)) for p in js_files if not _node_check(p)]

        index = workspace / "index.html"
        html = index.read_text(errors="replace").lower() if index.exists() else ""
        html_ok = bool(html) and _CANVAS_RE in html and _SCRIPT_RE in html

        return {
            "builds": bool(js_files) and not failed and html_ok,
            "js_files": len(js_files),
            "js_syntax_failures": len(failed),
            "index_html": index.exists(),
            "html_wired": html_ok,  # has <canvas> + <script>
        }


def _node_check(path: Path) -> bool:
    """`node --check <file>` — True if the file is syntactically valid JS."""
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["node", "--check", str(path)],  # noqa: S607 — `node` from PATH
            env={**os.environ},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0
