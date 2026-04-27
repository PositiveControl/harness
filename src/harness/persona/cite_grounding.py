"""Cite-grounding catcher (lane F-prime, harness-11ha).

Detects citations in a generated reply that are real-but-wrong-section
fabrications: the cited section exists in the corpus but doesn't match
the question's topic. Voice-sample farming addresses these per-case;
this catcher addresses them structurally.

Algorithm — rank-based, not cosine-threshold-based. Empirical cosines
on the airton_c1 JO 7110.65 corpus run 0.016-0.033 (dense BGE-small);
no absolute threshold separates grounded from ungrounded reliably.
Instead we run hybrid retrieval on the question, take top-K anchors,
and ask: does the cited section appear in the question's top-K?

  - Yes → grounded. Retrieval already validated the cite.
  - No  → ungrounded. The model's cite isn't what hybrid retrieval
          would surface for this question; it's likely a real-but-
          wrong-section fab.

Returns per-citation status. Caller decides what to do — drop the
cite, replace with the suggested top-1, re-roll the model, or just
log the detection. Detect-only is the MVP; replacement strategy is
a follow-up bead once we have data on how often + which cases the
catcher fires on.

Threshold knob: K. Smaller K (3-5) is strict — false positives on
sections that are valid alternatives but not retrieval-best. Larger K
(10-15) is loose — catches fewer real fabs. Default K=10 balances
both for the airton_c1 atc fixture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from harness.persona.rewriter import extract_citations
from harness.store.episodic import EpisodicStore

# Strip the JO version-number prefix before pulling section anchors —
# otherwise the section regex matches '7110.65' (the document version)
# instead of the section that follows it. Handles 'JO 7110.65',
# 'JO_7110.65' (corpus principle form), 'JO 7110.65BB' (versioned).
_JO_PREFIX_RE = re.compile(r"JO[_ ]7110\.65[A-Z]*", re.IGNORECASE)
# `−` (U+2212, unicode minus) is intentional — the corpus source uses  # noqa: RUF003
# it interchangeably with ASCII hyphen in section anchors.
_SECTION_RE = re.compile(r"§?\s*(\d+(?:[-−]\d+){1,2}|\d+\.\d+[a-z]*)")  # noqa: RUF001


def _extract_anchor(text: str) -> str:
    """Pull the section number out of a citation or principle string.

    'JO 7110.65 §10-2-5' → '10-2-5'
    'JO_7110.65 §10-2-5 (Emergency Assistance — EMERGENCY SITUATIONS)' → '10-2-5'
    'AIM 5-3-8' → '5-3-8'
    '14 CFR §91.155' → '91.155'

    Returns '' when no section anchor matches. Folds U+2212 (unicode
    minus) to ASCII hyphen so cite forms from corpus + model agree.

    Strips JO version-number prefix first so '7110.65' isn't mistaken
    for a section anchor (the dotted form is also CFR-shaped — but the
    JO prefix disambiguates)."""
    if not text:
        return ""
    cleaned = _JO_PREFIX_RE.sub("", text)
    match = _SECTION_RE.search(cleaned)
    if match is None:
        return ""
    return match.group(1).replace("−", "-")  # noqa: RUF001


@dataclass(frozen=True)
class CiteCheck:
    """Result of grounding one citation against the question.

    `cite` is the surface form as the model wrote it (e.g.
    'JO 7110.65 §10-2-5'). `section` is the normalized anchor used
    for set comparisons. `grounded` is True iff `section` appears in
    the question's top-K retrieval results. `rank` is the retrieval
    rank of the cited section (None when it doesn't appear in top-K).
    `suggested` is the top-1 anchor when the cite is ungrounded (None
    when grounded, when no replacement exists, or when the suggested
    anchor IS the cited one)."""

    cite: str
    section: str
    grounded: bool
    rank: int | None
    suggested: str | None


@dataclass(frozen=True)
class CiteGroundingResult:
    checks: tuple[CiteCheck, ...]

    @property
    def has_ungrounded(self) -> bool:
        return any(not c.grounded for c in self.checks)

    @property
    def ungrounded(self) -> tuple[CiteCheck, ...]:
        return tuple(c for c in self.checks if not c.grounded)


def check_cite_groundedness(
    question: str,
    reply: str,
    *,
    episodic_store: EpisodicStore,
    k: int = 10,
    user_id: str | None = None,
) -> CiteGroundingResult:
    """Detect ungrounded citations in `reply`.

    Runs hybrid retrieval on `question`, extracts section anchors from
    the top-K results, and checks whether each citation in `reply`
    appears in that anchor set. Returns per-citation grounded /
    ungrounded status with a suggested replacement for ungrounded
    cites (top-1 retrieved section).

    Detect-only — does not mutate `reply`. Caller decides whether to
    drop, replace, or re-roll on the result.

    `user_id` scopes retrieval the same way the chat path does:
    `user_id=None` is the owner view; otherwise filters to shared +
    that user's rows. Should match the user_id the model saw when
    generating the reply."""
    cites = extract_citations(reply)
    if not cites:
        return CiteGroundingResult(checks=())

    hits = episodic_store.search(question, k=k, mode="hybrid", user_id=user_id)
    # Anchor → rank in question retrieval. Index of first occurrence
    # wins on duplicates (multiple chunks under one section).
    retrieved_ranks: dict[str, int] = {}
    for idx, (rec, _score) in enumerate(hits):
        anchor = _extract_anchor(rec.principle or "")
        if anchor and anchor not in retrieved_ranks:
            retrieved_ranks[anchor] = idx
    suggested_top1: str | None = next(iter(retrieved_ranks), None)

    seen: set[str] = set()
    checks: list[CiteCheck] = []
    for cite in cites:
        anchor = _extract_anchor(cite)
        if not anchor or anchor in seen:
            continue
        seen.add(anchor)
        rank = retrieved_ranks.get(anchor)
        grounded = rank is not None
        suggested = None if grounded or suggested_top1 == anchor else suggested_top1
        checks.append(
            CiteCheck(
                cite=cite,
                section=anchor,
                grounded=grounded,
                rank=rank,
                suggested=suggested,
            )
        )
    return CiteGroundingResult(checks=tuple(checks))
