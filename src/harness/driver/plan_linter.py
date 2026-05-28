"""Plan linter — flag over-scoped beads for decomposition (harness-bpix).

Every bead the GTA2 drive hand-decomposed (§7 Police, §11 Traffic, §9b
Weapons) walled the 32B coder and cost 5+ wasted turns *discovering* it
was too big; the beads that closed cleanly (§4 Driving, §10 On-foot, and
the §7/§11 sub-beads after splitting) didn't. The discriminating signal
is structural, not semantic — a bead that references many spec
sub-sections, runs long, and stacks many independent acceptance clauses
is doing too much — so a cheap text-shape score predicts it with no model
call.

This module is pure (text in, score out); the CLI (`harness drive
lint-epic`) wires it to bd. Thresholds are calibrated from the GTA beads
and exposed as module constants so they're easy to retune against real
attempt/park telemetry later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Distinct §X.Y[a] sub-section references in the text — the strongest
# signal. §7 cited 5 (§7.1 through §7.5), §11 cited 11, §9b cited 4; the
# closers cited 0-1.
_SUBSECTION_RE = re.compile(r"§\s*(\d+\.\d+[a-z]?)")
# Top-level §N[a] refs that are NOT the head of a §N.Y (negative
# lookahead on `.<digit>`). Cross-section coupling — informational, not a
# primary flag, but high coupling correlates with integration surface.
_SECTION_RE = re.compile(r"§\s*(\d+[a-z]?)(?!\s*\.\d)")
# List items: "- ", "* ", "• ", "1. ", "2) " — a proxy for distinct
# concerns bundled into one bead.
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

# Thresholds (calibrated from §7/§11/§9b vs §4/§10). Any one trips the
# flag; the reason names which. Tunable — bump precision by raising, or
# recall by lowering.
SUBSECTION_THRESHOLD = 3
ACCEPTANCE_CLAUSE_THRESHOLD = 4
DESCRIPTION_CHARS_THRESHOLD = 1500
BULLET_THRESHOLD = 6
# harness-yzg8: structural-wrap heuristic. When the bead text uses any
# of these phrases AND a referenced function in the workspace is
# >LONG_FUNCTION_THRESHOLD lines, the drive will get stuck trying to
# patch around a long function with regex/sed primitives (see
# loop_run=3e295564 turns 16-19 on harness-yd3m). Suggest the model
# read+write_file the whole file from the start.
LONG_FUNCTION_THRESHOLD = 100
_WRAP_PATTERN_RE = re.compile(
    r"\b(wrap|freeze|skip\s+(?:the\s+)?update|pause\s+(?:toggle|state)|"
    r"skip\s+update\s+step|gate\s+\w+\s+on|short-circuit)\b",
    re.IGNORECASE,
)
# Function-reference extraction: backticked names (preferred — the bead
# author marked them as code) and call-shaped identifiers (`fooBar(`).
# camelCase / snake_case both qualify; PascalCase rejected (class names).
_BACKTICK_REF_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_CALL_REF_RE = re.compile(r"(?<![A-Za-z_])([a-z][a-zA-Z0-9_]*)\s*\(")
# Function-definition shapes worth measuring. JS top-level + indented
# functions, TS class methods, Python def. Each is a per-language regex
# the workspace scan compiles ONCE per language.
_FUNCTION_DEF_PATTERNS: dict[str, tuple[str, ...]] = {
    ".js": (r"^\s*function\s+{name}\s*\(", r"^\s*(?:const|let|var)\s+{name}\s*=\s*function\s*\("),
    ".mjs": (r"^\s*function\s+{name}\s*\(", r"^\s*(?:const|let|var)\s+{name}\s*=\s*function\s*\("),
    ".cjs": (r"^\s*function\s+{name}\s*\(", r"^\s*(?:const|let|var)\s+{name}\s*=\s*function\s*\("),
    ".ts": (r"^\s*function\s+{name}\s*\(",),
    ".py": (r"^\s*def\s+{name}\s*\(",),
}
# Top-level files we'll scan. Tests/.git/node_modules excluded — those
# never carry the bead's target function and slow the lint down.
_SCAN_SUFFIXES: frozenset[str] = frozenset(_FUNCTION_DEF_PATTERNS.keys())
_SCAN_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".git", ".harness", "node_modules", "__pycache__", "venv", ".venv"}
)


@dataclass(frozen=True)
class BeadComplexity:
    """One bead's structural complexity readout. `flagged` means at least
    one signal cleared its threshold; `reasons` says which (and what to
    tell the operator)."""

    bead_id: str
    title: str
    subsection_refs: int
    cross_section_refs: int
    acceptance_clauses: int
    description_chars: int
    bullets: int
    flagged: bool
    reasons: tuple[str, ...]


def _extract_function_references(text: str) -> list[str]:
    """Pull function-name candidates from `text`. Backticked names rank
    first (the author flagged them as code); call-shape names follow.
    Deduped, lowercase-leading filter applied, length cap so a runaway
    word doesn't blow the scan."""
    seen: dict[str, None] = {}
    for m in _BACKTICK_REF_RE.findall(text):
        if 2 <= len(m) <= 60 and m[0].islower():
            seen[m] = None
    for m in _CALL_REF_RE.findall(text):
        if 2 <= len(m) <= 60 and m not in seen:
            seen[m] = None
    return list(seen)


def _function_line_count_in_file(name: str, source: str, suffix: str) -> int | None:
    """Find a definition of `name` in `source` (a single file's text)
    and return the line count from header to terminating brace / indent
    drop. Returns None when no matching definition exists OR the closing
    boundary can't be located."""
    patterns = _FUNCTION_DEF_PATTERNS.get(suffix)
    if patterns is None:
        return None
    escaped = re.escape(name)
    lines = source.splitlines()
    for pat in patterns:
        regex = re.compile(pat.format(name=escaped))
        for i, line in enumerate(lines):
            if regex.search(line):
                end = _find_function_end(lines, i, suffix)
                if end is not None:
                    return end - i + 1
    return None


def _find_function_end(lines: list[str], start: int, suffix: str) -> int | None:
    """Return the (0-indexed) line of the function-body terminator
    starting at `lines[start]`. For brace languages: brace-balance until
    depth returns to 0 (with depth seeded from the opening brace on the
    header line). For Python: scan until indent drops back to or below
    the def's indent."""
    if suffix == ".py":
        header_indent = len(lines[start]) - len(lines[start].lstrip())
        for j in range(start + 1, len(lines)):
            body = lines[j]
            if not body.strip():
                continue
            indent = len(body) - len(body.lstrip())
            if indent <= header_indent:
                return j - 1
        return len(lines) - 1
    depth = lines[start].count("{") - lines[start].count("}")
    if depth <= 0:
        # Brace not opened on header line (e.g. `function X(\n  arg\n) {`)
        # — find the line that opens it.
        for j in range(start + 1, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth > 0:
                start_body = j
                break
        else:
            return None
        for k in range(start_body + 1, len(lines)):
            depth += lines[k].count("{") - lines[k].count("}")
            if depth == 0:
                return k
        return None
    for j in range(start + 1, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth == 0:
            return j
    return None


def _measure_referenced_functions(names: list[str], workspace: Path) -> list[tuple[str, str, int]]:
    """For each `name`, walk the workspace and return all
    `(name, file_relpath, line_count)` triples where `line_count` >=
    LONG_FUNCTION_THRESHOLD. Short or unmatched defs are dropped — the
    linter only surfaces ones worth warning about."""
    hits: list[tuple[str, str, int]] = []
    if not names or not workspace.is_dir():
        return hits
    name_set = set(names)
    stack: list[Path] = [workspace]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in _SCAN_EXCLUDE_DIRS and not entry.name.startswith("."):
                    stack.append(entry)
                continue
            if entry.suffix not in _SCAN_SUFFIXES:
                continue
            try:
                source = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for name in name_set:
                count = _function_line_count_in_file(name, source, entry.suffix)
                if count is not None and count >= LONG_FUNCTION_THRESHOLD:
                    rel = entry.relative_to(workspace).as_posix()
                    hits.append((name, rel, count))
    return hits


def _count_acceptance_clauses(acceptance: str) -> int:
    """Independent clauses in an acceptance string — split on sentence
    ends, semicolons, newlines, and the conjunction ``AND``. Fragments
    under 12 chars are dropped so trailing scraps don't inflate the
    count."""
    if not acceptance.strip():
        return 0
    parts = re.split(r"(?:\.\s+|;\s*|\n+|\bAND\b)", acceptance)
    return sum(1 for p in parts if len(p.strip()) >= 12)


def _count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if _BULLET_RE.match(line))


def score_bead(
    bead_id: str,
    title: str,
    description: str,
    acceptance: str = "",
    *,
    workspace: Path | None = None,
    check_long_functions: bool = True,
) -> BeadComplexity:
    """Score one bead's decomposition risk from its text. `acceptance`
    is optional — when bd doesn't surface it separately, the description
    signals (sub-sections / length / bullets) carry the call.

    `workspace` + `check_long_functions` (harness-yzg8) enable the
    structural-wrap heuristic: when the bead text matches the wrap-
    pattern AND a referenced function in the workspace is
    >=LONG_FUNCTION_THRESHOLD lines, surface a hint to use whole-file
    rewrite from the start. Pass `workspace=None` (or pass the flag
    False) to skip — keeps the linter pure-text when callers don't
    want to walk the filesystem."""
    blob = f"{title}\n{description}\n{acceptance}"
    subsection_refs = len(set(_SUBSECTION_RE.findall(blob)))
    cross_section_refs = len(set(_SECTION_RE.findall(blob)))
    acceptance_clauses = _count_acceptance_clauses(acceptance)
    description_chars = len(description)
    bullets = _count_bullets(description)

    reasons: list[str] = []
    if subsection_refs >= SUBSECTION_THRESHOLD:
        reasons.append(
            f"{subsection_refs} distinct spec sub-sections (>= {SUBSECTION_THRESHOLD}) "
            f"— each is a candidate sub-bead"
        )
    if acceptance_clauses >= ACCEPTANCE_CLAUSE_THRESHOLD:
        reasons.append(
            f"{acceptance_clauses} independent acceptance clauses "
            f"(>= {ACCEPTANCE_CLAUSE_THRESHOLD})"
        )
    if description_chars > DESCRIPTION_CHARS_THRESHOLD:
        reasons.append(f"{description_chars}-char description (> {DESCRIPTION_CHARS_THRESHOLD})")
    if bullets >= BULLET_THRESHOLD:
        reasons.append(f"{bullets} bullet items (>= {BULLET_THRESHOLD}) — many bundled concerns")

    # harness-yzg8: structural-wrap detection. Only fires when wrap-
    # pattern wording AND a referenced function in the workspace is
    # large. Both signals required — wrap-language alone is too noisy.
    if check_long_functions and workspace is not None and _WRAP_PATTERN_RE.search(blob):
        names = _extract_function_references(blob)
        long_hits = _measure_referenced_functions(names, workspace)
        for name, rel, lines in long_hits:
            reasons.append(
                f"`{name}` in {rel} is {lines} lines (>= {LONG_FUNCTION_THRESHOLD}) "
                f"and the task wraps/freezes/skips parts of it — "
                f"use read_file + write_file (whole-file rewrite) instead "
                f"of regex/sed/edit_file patches"
            )

    return BeadComplexity(
        bead_id=bead_id,
        title=title,
        subsection_refs=subsection_refs,
        cross_section_refs=cross_section_refs,
        acceptance_clauses=acceptance_clauses,
        description_chars=description_chars,
        bullets=bullets,
        flagged=bool(reasons),
        reasons=tuple(reasons),
    )


__all__ = [
    "ACCEPTANCE_CLAUSE_THRESHOLD",
    "BULLET_THRESHOLD",
    "DESCRIPTION_CHARS_THRESHOLD",
    "LONG_FUNCTION_THRESHOLD",
    "SUBSECTION_THRESHOLD",
    "BeadComplexity",
    "score_bead",
]
