from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Openers that violate Airton's register regardless of substance.
# Extend deliberately — each entry costs a false negative somewhere.
_BANNED_OPENERS: tuple[str, ...] = (
    "That's a solid",
    "That sounds like a solid",
    "That's a great",
    "Great question",
    "Certainly",
    "Sure thing",
    "I'd be happy to",
    "I cannot comply",
    "I will do my best",
    "I'll do my best",
    "As an AI",
    "As a language model",
    "Here are a few",
    "Here's a breakdown",
    "Let me break",
    "Here's how",
)

# Numbered lists ("1. foo\n2. bar") are Airton-forbidden when the gold
# uses prose. Bullet dashes ("- foo\n- bar") are allowed by the style rules.
_NUMBERED_LIST_PATTERN = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)

# Mid-sentence assistant tells. A phrase here is a weak signal alone; in
# aggregate, three or four of them in a response is very strong. Gold
# responses in the canonical suite avoid all of these, so any excess
# over the gold's own count is counted as filler.
_FILLER_PHRASES: tuple[str, ...] = (
    "ensure that",
    "ensures that",
    "to ensure",
    "make sure that",
    "make sure to",
    "it's important to",
    "it is important to",
    "comprehensive",
    "maintainability",
    "maintainable",
    "maintains",
    "robustness",
    "various scenarios",
    "various conditions",
    "in this context",
    "given the complexity",
    "let me know",
    "feel free to",
    "happy to help",
    "to summarize",
    "in summary",
    "let me break",
    "here's a breakdown",
    "consider the following",
    "keep in mind",
    "clearly defined",
    "thoroughly",
    "seamlessly",
    "proceed with caution",
    "from my end",
    "from your end",
)


@dataclass(frozen=True)
class VoiceScore:
    """Sub-metrics each scored 0..1; aggregate is the mean. Sub-scores
    are reported separately so we can see *where* a drift happens, not
    just that it did."""

    length_match: float
    no_banned_openers: float
    bullet_discipline: float
    filler_discipline: float
    aggregate: float
    notes: tuple[str, ...]
    judge_score: int | None = None  # set by the optional LLM judge; None if not run


def _length_score(actual: str, gold: str) -> tuple[float, str]:
    gold_chars = max(len(gold.strip()), 1)
    actual_chars = max(len(actual.strip()), 1)
    ratio = actual_chars / gold_chars
    # Symmetric log distance: ratio=1 → 1.0; ratio=2 or 0.5 → ~0.59;
    # ratio=4 or 0.25 → ~0.42; ratio=10 → ~0.30.
    score = 1.0 / (1.0 + abs(math.log(ratio)))
    return round(score, 3), f"length_ratio={ratio:.2f}"


def _banned_opener_score(actual: str) -> tuple[float, str | None]:
    first_line = actual.strip().split("\n", 1)[0].strip()
    # Strip markdown fence openers for fairness on commit-message-shaped gold
    if first_line.startswith("```"):
        return 1.0, None
    for opener in _BANNED_OPENERS:
        if first_line.lower().startswith(opener.lower()):
            return 0.0, f"banned_opener={opener!r}"
    return 1.0, None


def _bullet_discipline_score(actual: str, gold: str) -> tuple[float, str | None]:
    gold_has_numbered = bool(_NUMBERED_LIST_PATTERN.search(gold))
    actual_has_numbered = bool(_NUMBERED_LIST_PATTERN.search(actual))
    if gold_has_numbered == actual_has_numbered:
        return 1.0, None
    if actual_has_numbered and not gold_has_numbered:
        return 0.0, "unexpected_numbered_list"
    return 0.5, "missing_expected_numbered_list"


def _filler_score(actual: str, gold: str) -> tuple[float, list[str]]:
    actual_lower = actual.lower()
    gold_lower = gold.lower()
    hits: list[str] = []
    excess_total = 0
    for phrase in _FILLER_PHRASES:
        actual_count = actual_lower.count(phrase)
        gold_count = gold_lower.count(phrase)
        excess = actual_count - gold_count
        if excess > 0:
            hits.append(f"filler:{phrase!r}x{excess}")
            excess_total += excess
    # Each excess filler phrase costs 0.15 of the score; cap at zero.
    # Gold uses none of these, so a clean response scores 1.0; two
    # excess phrases → 0.70; five excess phrases → 0.25; seven+ → 0.
    score = max(0.0, 1.0 - 0.15 * excess_total)
    return round(score, 3), hits


def score_actual_against_gold(actual: str, gold: str) -> VoiceScore:
    """Compare a model output to its gold response along four axes:
    length, opener register, bullet shape, and mid-sentence filler.
    Cheap, deterministic, interpretable. A perfect score is 1.0 across
    the board. The `judge_score` field is populated separately by the
    optional LLM judge when running evals with --judge."""
    length_match, length_note = _length_score(actual, gold)
    no_banned, banned_note = _banned_opener_score(actual)
    bullet_discipline, bullet_note = _bullet_discipline_score(actual, gold)
    filler_discipline, filler_notes = _filler_score(actual, gold)

    notes: list[str] = [length_note]
    if banned_note:
        notes.append(banned_note)
    if bullet_note:
        notes.append(bullet_note)
    notes.extend(filler_notes)

    aggregate = (length_match + no_banned + bullet_discipline + filler_discipline) / 4.0
    return VoiceScore(
        length_match=length_match,
        no_banned_openers=no_banned,
        bullet_discipline=bullet_discipline,
        filler_discipline=filler_discipline,
        aggregate=round(aggregate, 3),
        notes=tuple(notes),
    )
