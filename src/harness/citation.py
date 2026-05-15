"""Per-character citation grammar (harness-jaqe).

A character that speaks against a corpus where citation discipline is
load-bearing (airton_c1 with JO 7110.65; future legal / RFC / paper
personas) declares the regex shapes its corpus uses + the string
tokens that appear in nudge text. Every consumer that previously
imported FAA-shaped regex constants from `persona/rewriter.py`,
`persona/cite_grounding.py`, or `orchestrator/hooks.py` now reads
from `Character.citation_grammar` (a `CitationGrammar | None`).

When a character does not declare citation_grammar in core.yaml,
every citation-aware function passes through (returns empty list /
empty string / no-op) and the citation-related fabrication-catcher
hooks (MissingCitationHook etc.) silently no-op for that character.
That is the desired generalization — non-corpus characters never
needed those checks, and now we can prove they don't run.

YAML shape (under `core.yaml: citation_grammar:`) — see
`character/airton_c1/core.yaml` for the canonical FAA example.
Required keys: `surface_patterns` (non-empty list of regex strings),
`anchor_pattern` (regex), `document_reference` (regex),
`document_name` (str), `example_anchor` (str). Optional:
`strip_prefix` (regex; scrub before anchor extraction).

All patterns compile case-insensitive (matches the original constants'
behavior — every pattern was `re.IGNORECASE`). If a future grammar
needs case-sensitive matching, extend with a per-pattern flags
dict; today every consumer wants ignore-case so the loader bakes it
in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CitationGrammar:
    """Compiled citation patterns + nudge-text tokens for one character.

    Each field maps 1:1 to a former FAA-coded module constant:
      - `surface_patterns` ← persona/rewriter.py:_CITATION_PATTERNS
      - `strip_prefix` + `anchor_pattern` ← persona/cite_grounding.py:
        _JO_PREFIX_RE / _SECTION_RE
      - `document_reference` ← orchestrator/hooks.py:_ORDER_REFERENCE_RE
      - `document_name` + `example_anchor` ← embedded literals in the
        MissingCitationHook nudge string
    """

    surface_patterns: tuple[re.Pattern[str], ...]
    strip_prefix: re.Pattern[str] | None
    anchor_pattern: re.Pattern[str]
    document_reference: re.Pattern[str]
    document_name: str
    example_anchor: str
    # Whether this character's "is the reply cited?" check should also
    # accept the FAA hyphen-form fallback regex (`§N-N` / `§N-N-N` /
    # `TBL N-N-N`) in `hooks._CITATION_PRESENT_RE`. FAA-flavored
    # characters (airton_c, airton_c1, airton_c_tfr) set this true so
    # the model can use bare `§3-10-3` without forcing the `JO 7110.65`
    # prefix. Non-FAA characters (airton_f and future legal / RFC
    # personas) leave it false — for them, the global FAA shape is a
    # FALSE POSITIVE (smoke 2026-05-15: airton_f's model produced
    # `§1-5 explicitly disclaims security`; surface_patterns required
    # `§<anchor> (<doc>)` so didn't match, but the global FAA regex
    # accepted it and let the reply through uncited).
    accept_faa_bare_anchor: bool = False


def load_citation_grammar(raw: object | None, path: Path) -> CitationGrammar | None:
    """Parse the `citation_grammar:` block from a character's core.yaml.

    Returns None when the block is absent or null — character has no
    citation discipline, citation-aware functions short-circuit.
    Raises ValueError with a path-anchored message when the block is
    present but malformed (unknown keys allowed; required keys
    enforced).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}/core.yaml: citation_grammar must be a mapping; got {type(raw).__name__}"
        )
    surface_raw = raw.get("surface_patterns")
    if not isinstance(surface_raw, list) or not surface_raw:
        raise ValueError(
            f"{path}/core.yaml: citation_grammar.surface_patterns must be a "
            f"non-empty list of regex strings"
        )
    surface_patterns: list[re.Pattern[str]] = []
    for idx, pat in enumerate(surface_raw):
        if not isinstance(pat, str):
            raise ValueError(
                f"{path}/core.yaml: citation_grammar.surface_patterns[{idx}] "
                f"must be a string regex; got {type(pat).__name__}"
            )
        try:
            surface_patterns.append(re.compile(pat, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(
                f"{path}/core.yaml: citation_grammar.surface_patterns[{idx}] "
                f"failed to compile: {exc}"
            ) from exc
    strip_prefix_raw = raw.get("strip_prefix")
    strip_prefix: re.Pattern[str] | None = None
    if strip_prefix_raw is not None:
        if not isinstance(strip_prefix_raw, str):
            raise ValueError(
                f"{path}/core.yaml: citation_grammar.strip_prefix must be a string regex or null"
            )
        try:
            strip_prefix = re.compile(strip_prefix_raw, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"{path}/core.yaml: citation_grammar.strip_prefix failed to compile: {exc}"
            ) from exc
    anchor_raw = raw.get("anchor_pattern")
    if not isinstance(anchor_raw, str):
        raise ValueError(
            f"{path}/core.yaml: citation_grammar.anchor_pattern must be a string regex"
        )
    try:
        anchor_pattern = re.compile(anchor_raw)
    except re.error as exc:
        raise ValueError(
            f"{path}/core.yaml: citation_grammar.anchor_pattern failed to compile: {exc}"
        ) from exc
    docref_raw = raw.get("document_reference")
    if not isinstance(docref_raw, str):
        raise ValueError(
            f"{path}/core.yaml: citation_grammar.document_reference must be a string regex"
        )
    try:
        document_reference = re.compile(docref_raw, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(
            f"{path}/core.yaml: citation_grammar.document_reference failed to compile: {exc}"
        ) from exc
    document_name = raw.get("document_name")
    if not isinstance(document_name, str) or not document_name.strip():
        raise ValueError(
            f"{path}/core.yaml: citation_grammar.document_name must be a non-empty string"
        )
    example_anchor = raw.get("example_anchor")
    if not isinstance(example_anchor, str) or not example_anchor.strip():
        raise ValueError(
            f"{path}/core.yaml: citation_grammar.example_anchor must be a non-empty string"
        )
    # Optional FAA bare-anchor fallback flag. Defaults False so non-
    # FAA characters don't accidentally inherit the JO/AIM hyphen-form
    # acceptance in hooks._CITATION_PRESENT_RE.
    accept_faa_bare = raw.get("accept_faa_bare_anchor", False)
    if not isinstance(accept_faa_bare, bool):
        raise ValueError(
            f"{path}/core.yaml: citation_grammar.accept_faa_bare_anchor "
            f"must be a bool; got {type(accept_faa_bare).__name__}"
        )
    return CitationGrammar(
        surface_patterns=tuple(surface_patterns),
        strip_prefix=strip_prefix,
        anchor_pattern=anchor_pattern,
        document_reference=document_reference,
        document_name=document_name.strip(),
        example_anchor=example_anchor.strip(),
        accept_faa_bare_anchor=accept_faa_bare,
    )
