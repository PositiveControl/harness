"""Deterministic cross-workspace premise check (harness-y0hu5).

A bead that references concrete files which exist nowhere under the
workspace is mis-wired — filed against the wrong epic or the wrong
checkout — and no amount of driving can close it there. harness-rorj
("score tracking in snake_game.py, PLAN.md Phase 3.2") sat under the
GTAII epic in scratch/gta_r2: the drive burned 4 WRITE_TEST attempts in
a workspace with neither file, scattering junk test files, before
parking with the opaque "write_test->halted (no test)".

This check runs BEFORE the turn fires (cheaper than an ASSESS round —
no model call) and parks the bead PREMISE_UNMET with the missing-file
list in the note, the same escape ASSESS's flag_blocked takes for other
false premises.

Deliberately conservative — every guard below biases toward driving:

- ALL extracted tokens must be absent. A bead that names one existing
  file ("add drawCop to game.js, mirror cop.js") still drives even if a
  sibling token is missing.
- At least TWO distinct tokens are required. A creation bead typically
  names exactly its one deliverable ("create utils.js with …"); parking
  on a single absent token would block legitimate new-file work.
- The workspace must already contain at least one source file. A
  from-scratch build's first beads name files that don't exist yet —
  that's the premise, not a violation.
"""

from __future__ import annotations

import re
from pathlib import Path

# Suffixes that make a token "path-shaped". Mirrors the artifact types
# the driver builds and references in bead text (source + docs).
_PATH_SUFFIXES = (
    "py|js|mjs|ts|tsx|jsx|html|css|json|yaml|yml|toml|md|txt|sh|sql|c|h|cpp|hpp|rs|go|java|rb"
)
_PATH_TOKEN_RE = re.compile(rf"\b[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:{_PATH_SUFFIXES})\b")

# Bound the workspace walk and the token set — this is a cheap pre-turn
# guard, not an index.
_MAX_TOKENS = 8
_MAX_WALK_FILES = 5000


def _extract_path_tokens(text: str) -> list[str]:
    """Unique path-shaped tokens from bead text, in first-seen order,
    capped at `_MAX_TOKENS`. URLs are skipped — `example.com/x.html`
    inside a link is not a workspace reference."""
    seen: list[str] = []
    for match in _PATH_TOKEN_RE.finditer(text):
        token = match.group(0)
        prefix = text[max(0, match.start() - 8) : match.start()]
        if "://" in prefix:
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) >= _MAX_TOKENS:
            break
    return seen


def _walk_workspace_files(workspace: Path) -> tuple[set[str], set[str]]:
    """One bounded walk: (relative posix paths, basenames) of every
    non-hidden file. Hidden directories (.harness, .git, …) are skipped
    at every depth — driver scratch must not satisfy a premise."""
    rel_paths: set[str] = set()
    basenames: set[str] = set()
    stack = [workspace]
    while stack and len(rel_paths) < _MAX_WALK_FILES:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                rel_paths.add(entry.relative_to(workspace).as_posix())
                basenames.add(entry.name)
    return rel_paths, basenames


def referenced_missing_files(text: str, workspace: Path) -> list[str] | None:
    """The missing-file list when `text` trips the cross-workspace
    premise check, else None (= drive normally).

    Trips only when: >=2 distinct path-shaped tokens were extracted,
    the workspace already has at least one source-suffixed file, and
    EVERY token matches nothing — neither as a workspace-relative path
    nor by basename anywhere under the workspace.
    """
    tokens = _extract_path_tokens(text)
    if len(tokens) < 2:
        return None
    rel_paths, basenames = _walk_workspace_files(workspace)
    source_re = re.compile(rf"\.(?:{_PATH_SUFFIXES})$")
    if not any(source_re.search(p) for p in rel_paths):
        return None  # from-scratch workspace: absent files ARE the work
    missing = [
        t for t in tokens if t.lstrip("./") not in rel_paths and Path(t).name not in basenames
    ]
    if len(missing) == len(tokens):
        return missing
    return None


__all__ = ["referenced_missing_files"]
