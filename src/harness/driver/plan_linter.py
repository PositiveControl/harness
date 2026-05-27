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
) -> BeadComplexity:
    """Score one bead's decomposition risk from its text. `acceptance`
    is optional — when bd doesn't surface it separately, the description
    signals (sub-sections / length / bullets) carry the call."""
    blob = f"{description}\n{acceptance}"
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
    "SUBSECTION_THRESHOLD",
    "BeadComplexity",
    "score_bead",
]
