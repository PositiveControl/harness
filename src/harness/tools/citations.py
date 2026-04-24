"""Citation extraction for tool-declared grounding (harness-ywp.5).

A grounding-tier tool (search_memory today; fetch_url / search_facts
in future as their consumers land) declares the citations present in
the text it's returning so downstream hooks and the per-turn audit
log can ask structured questions like 'did any tool actually ground
this §-reference?' without regex-parsing the tool output.

Normalisation is load-bearing: the corpus mixes ASCII hyphens,
en-dashes, and unicode minus signs, and the model sometimes emits a
space after the section sign. Canonical form is `§N-N-N` (or the
broader two-segment `§N-N`) with ASCII hyphens and no internal
whitespace, so set-membership comparisons against the hook-side
regex behave.
"""

from __future__ import annotations

import re

# Matches the same citation shapes the orchestrator's
# _CITATION_PRESENT_RE recognises (hooks.py ~line 819) — kept in sync
# by convention, not by import, to avoid a tools → hooks cycle:
#
#   §N-N-N                  (primary — JO chapter-section-paragraph)
#   §N-N                    (broader chapter-section)
#   TBL/Table/FIG/Figure N-N-N (tables/figures are section-scoped)
#
# Tolerates ASCII hyphen, en-dash (U+2013), and unicode minus
# (U+2212) — all three appear in the corpus.
CITATION_RE = re.compile(
    r"§\s*\d+[-–−]\d+(?:[-–−]\d+)?"  # noqa: RUF001 — dash variants load-bearing
    r"|\b(?:TBL|Table|FIG|Figure)\s+\d+[-–−]\d+[-–−]\d+\b",  # noqa: RUF001
    re.IGNORECASE,
)


# All dash variants the corpus uses — unified to ASCII hyphen in the
# canonical form.
_DASH_VARIANTS = ("–", "−")  # noqa: RUF001 — en-dash + unicode minus load-bearing


def _canonicalise(raw: str) -> str:
    """Normalise a matched citation to `§N-N-N` / `§N-N` / `TBL N-N-N`
    form. Strips all whitespace, converts dash variants to ASCII
    hyphens, title-cases the TBL/FIG prefix so downstream consumers
    don't have to deal with case variance."""

    stripped = re.sub(r"\s+", "", raw)
    for variant in _DASH_VARIANTS:
        stripped = stripped.replace(variant, "-")
    # Title-case the leading label for TBL/Table/FIG/Figure forms.
    # Uppercase §-forms need no transform (the symbol is its own prefix).
    if stripped[0] == "§":
        return stripped
    # Re-insert a single space between the label and the number for
    # readability: "TBL 4-1-2" not "TBL4-1-2".
    match = re.match(r"([A-Za-z]+)(.+)", stripped)
    if not match:
        return stripped
    label, rest = match.group(1), match.group(2)
    return f"{label.upper()} {rest}"


def extract_citations(text: str) -> frozenset[str]:
    """Return the canonicalised set of citations present in `text`.
    Empty frozenset when nothing matches — tools that find no
    citations report `citations_grounded=frozenset()` on their
    ToolResult (the default)."""

    out: set[str] = set()
    for match in CITATION_RE.finditer(text):
        out.add(_canonicalise(match.group(0)))
    return frozenset(out)
