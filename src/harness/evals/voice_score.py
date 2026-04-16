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


@dataclass(frozen=True)
class VoiceScore:
    """Three sub-metrics, aggregate is the mean. Each sub-score is 0..1
    so the aggregate is easy to reason about. Sub-scores are separately
    readable so we can see *where* a drift happens, not just that it
    did."""

    length_match: float
    no_banned_openers: float
    bullet_discipline: float
    aggregate: float
    notes: tuple[str, ...]


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


def score_actual_against_gold(actual: str, gold: str) -> VoiceScore:
    """Compare a model output to its gold response along three axes:
    length, opener register, and bullet shape. Cheap, deterministic,
    interpretable. A perfect score is 1.0 across the board."""
    length_match, length_note = _length_score(actual, gold)
    no_banned, banned_note = _banned_opener_score(actual)
    bullet_discipline, bullet_note = _bullet_discipline_score(actual, gold)

    notes: list[str] = [length_note]
    if banned_note:
        notes.append(banned_note)
    if bullet_note:
        notes.append(bullet_note)

    aggregate = (length_match + no_banned + bullet_discipline) / 3.0
    return VoiceScore(
        length_match=length_match,
        no_banned_openers=no_banned,
        bullet_discipline=bullet_discipline,
        aggregate=round(aggregate, 3),
        notes=tuple(notes),
    )
