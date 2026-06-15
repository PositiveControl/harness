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
from collections.abc import Iterable
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


# harness-0t2f9: mid-turn flag_blocked validation. The pre-turn guard
# above catches a bead wired against the wrong WORKSPACE (named files
# absent everywhere). This sibling catches a different false premise the
# model declares MID-turn: flag_blocked(missing=...) whose `missing` names
# the bead's OWN deliverable rather than an upstream precondition. A
# build/create bead's deliverable is absent before the bead runs — that
# absence IS the task, not a block. loop_run=069d6172 (harness-4s2bb)
# parked a perfectly drivable "spawn pedestrians in game.js" bead because
# the model flagged missing="pedestrian spawning and wandering logic in
# game.js" — verbatim the §6a acceptance criteria.
#
# Cheap, deterministic discriminator: content-token overlap between
# `missing` and the bead's own text (title + description + acceptance). A
# high overlap means the absent thing IS what the bead exists to build —
# reject the flag. A genuine upstream precondition (a named symbol/file the
# bead presupposes but does not itself produce) shares few tokens with the
# deliverable text, so it stays under threshold and still parks.

# English function words + bead-boilerplate that carry no deliverable
# signal. Kept deliberately small — over-pruning would let a genuine
# upstream `missing` slip below threshold on its remaining content words.
_OVERLAP_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "for",
        "nor",
        "yet",
        "so",
        "in",
        "on",
        "at",
        "to",
        "of",
        "by",
        "as",
        "is",
        "are",
        "be",
        "was",
        "were",
        "with",
        "into",
        "from",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "no",
        "not",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "must",
        "should",
        "will",
        "would",
        "can",
        "could",
        "may",
        "there",
        "here",
        "then",
        "than",
        "use",
        "used",
        "using",
        "via",
    }
)


def _stem(token: str) -> str:
    """Crude suffix strip so inflected forms collide (spawning↔spawn,
    wanders↔wander). Longest suffix first; never strips below 3 chars."""
    for suffix in ("ings", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _content_tokens(text: str) -> set[str]:
    """Stemmed content tokens: lowercased alnum words, minus stopwords,
    pure numbers, and tokens under 3 chars."""
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9_]+", text.lower()):
        if len(raw) < 3 or raw.isdigit() or raw in _OVERLAP_STOPWORDS:
            continue
        out.add(_stem(raw))
    return out


# Threshold tuned on harness-4s2bb (overlap 1.0) vs a genuine upstream
# precondition (a named foreign symbol shares few deliverable tokens,
# overlap well under 0.5). 0.7 leaves margin against both.
_DELIVERABLE_OVERLAP_THRESHOLD = 0.7


def flag_blocked_names_own_deliverable(missing: str, deliverable_text: str) -> bool:
    """True when `missing` substantially restates the bead's own
    deliverable (so flag_blocked should be rejected, not honored).

    Conservative — biases toward honoring the flag (parking):
    - Requires >=2 content tokens in `missing`. A one-word `missing`
      (a bare symbol name) carries too little signal to call.
    - Requires the deliverable text to have content tokens at all.
    - Trips only when >=70% of `missing`'s content tokens appear in the
      deliverable text. A genuine upstream artifact shares few.
    """
    m = _content_tokens(missing)
    if len(m) < 2:
        return False
    d = _content_tokens(deliverable_text)
    if not d:
        return False
    overlap = len(m & d) / len(m)
    return overlap >= _DELIVERABLE_OVERLAP_THRESHOLD


# harness-vsv: a third false-premise shape. ASSESS withholds the write-tier
# editor tools by design — they're handed to IMPLEMENT (see
# fsm_turn._build_phase_registry). A model that flag_blocks citing
# "edit_file tool not available" (loop_run=065ff3c1, harness-vsv: parked
# PREMISE_UNMET on attempt 1, no retry, no gate → no s0el9/6zjjm revival →
# stranded forever) has mistaken a phase boundary for an unmet premise. The
# recovery is submit_assessment, which routes the turn to IMPLEMENT where the
# editor exists — not a park. Match is word-boundary on the tool identifier:
# these are distinctive underscore tokens that don't surface in natural
# premise prose, so a genuine "edit the foo_file config" stays clear.


def flag_blocked_names_withheld_tool(missing: str, tool_names: Iterable[str]) -> str | None:
    """The withheld-tool name `missing` blames, when it cites a phase-scoped
    editor tool as the blocker — else None (= a real premise, honor the flag).

    `tool_names` is the set of editors the ASSESS phase withholds but a later
    phase provides. A `missing` naming any of them is a phase-confusion
    hallucination, not an upstream precondition gap.
    """
    lowered = missing.lower()
    for name in tool_names:
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            return name
    return None


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


__all__ = [
    "flag_blocked_names_own_deliverable",
    "flag_blocked_names_withheld_tool",
    "referenced_missing_files",
]
