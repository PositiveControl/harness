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

from dataclasses import dataclass

from harness.citation import CitationGrammar
from harness.store.episodic import EpisodicStore


def _extract_anchor(text: str, grammar: CitationGrammar | None) -> str:
    """Pull the section number out of a citation or principle string,
    using the active character's citation grammar (harness-jaqe).

    'JO 7110.65 §10-2-5' → '10-2-5' (with the JO grammar's strip_prefix)
    'AIM 5-3-8' → '5-3-8'
    '14 CFR §91.155' → '91.155'

    Returns '' when grammar is None (character has no citation
    discipline), when text is empty, or when no anchor matches. Folds
    U+2212 (unicode minus) to ASCII hyphen so cite forms from corpus
    + model agree.

    The optional strip_prefix scrub runs first so the document-version
    number (e.g. JO '7110.65') isn't mistaken for a section anchor in
    grammars where the dotted form is also a valid section shape."""
    if not text or grammar is None:
        return ""
    cleaned = grammar.strip_prefix.sub("", text) if grammar.strip_prefix is not None else text
    match = grammar.anchor_pattern.search(cleaned)
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
    grammar: CitationGrammar | None,
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

    `grammar` is the active character's citation grammar — drives
    both the surface-form citation extractor and the section-anchor
    extractor. None ⇒ empty result, no work to do.

    `user_id` scopes retrieval the same way the chat path does:
    `user_id=None` is the owner view; otherwise filters to shared +
    that user's rows. Should match the user_id the model saw when
    generating the reply."""
    if grammar is None:
        return CiteGroundingResult(checks=())
    # Lazy import: persona.rewriter pulls in model.adapter which can
    # cycle back through tools.__init__ → tools.phraseology_lint →
    # persona.cite_grounding when imports start from the model side
    # (e.g. the ablation_validate gate). Local import here breaks the
    # cycle without changing the public call signature.
    from harness.persona.rewriter import extract_citations

    cites = extract_citations(reply, grammar)
    if not cites:
        return CiteGroundingResult(checks=())

    hits = episodic_store.search(question, k=k, mode="hybrid", user_id=user_id)
    # Anchor → rank in question retrieval. Index of first occurrence
    # wins on duplicates (multiple chunks under one section).
    retrieved_ranks: dict[str, int] = {}
    for idx, (rec, _score) in enumerate(hits):
        anchor = _extract_anchor(rec.principle or "", grammar)
        if anchor and anchor not in retrieved_ranks:
            retrieved_ranks[anchor] = idx
    suggested_top1: str | None = next(iter(retrieved_ranks), None)

    seen: set[str] = set()
    checks: list[CiteCheck] = []
    for cite in cites:
        anchor = _extract_anchor(cite, grammar)
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
