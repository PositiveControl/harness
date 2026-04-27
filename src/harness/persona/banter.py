"""Banter detection + ATC joke deflection (epic harness-jjm9).

When the user sends a low-signal prompt — empty, "test", "ping", a
fragment with no domain anchor, the literal string "this page
intentionally left blank" — Airton shouldn't fabricate JO 7110.65
content from thin retrieval. Run 12 lesson: hybrid retrieval always
returns *something*, and the model treats it as gold even when the
top-1 cosine is 0.02.

This module is the shared substrate. Two consumers wire it in:

- Router 'banter' intent (harness-q7ff) — preferred path. Router
  classifies the prompt before round 0; banter intent short-circuits
  to the joke reply with no model round.
- Tool-loop pre-model intercept (harness-vadq) — safety net for
  router-off / misclassified turns. Owns the per-session streak state
  that drives the 1-joke-then-3-redirects cycle.

Detection is deliberately conservative: a real domain question with a
§-citation, a controller-vocabulary noun, or a callsign-shaped token
exits the 'short-prompt' branch immediately. The cost of a false
negative (fabricated answer) is lower than the cost of a false
positive (snarky joke when the user actually had a real question).
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------- domain anchors ----------

# §N-N-N (and §N-NN-N) citations are an unambiguous anchor. Hyphen or
# en-dash (U+2013), optional § prefix, optional spaces.
_SECTION_RE = re.compile(r"§?\s*\d+\s*[-–]\s*\d+\s*[-–]\s*\d+")

# Controller / aviation vocabulary that exempts a short prompt from
# banter classification. Intentionally narrow — only words that would
# never appear in casual chat. Greetings, common verbs, and stop-words
# are NOT here on purpose.
_DOMAIN_ANCHORS: frozenset[str] = frozenset(
    {
        "altitude",
        "approach",
        "atc",
        "atis",
        "callsign",
        "clearance",
        "controller",
        "departure",
        "downwind",
        "etops",
        "final",
        "ground",
        "heading",
        "hold",
        "holding",
        "ifr",
        "ils",
        "imc",
        "metar",
        "minima",
        "minimum",
        "missed",
        "notam",
        "phraseology",
        "pilot",
        "radar",
        "runway",
        "rvsm",
        "section",
        "separation",
        "squawk",
        "taf",
        "tcas",
        "tower",
        "transmission",
        "transponder",
        "vector",
        "vfr",
        "vmc",
        "wake",
    }
)

# Explicit smartass / meta / empty-signal patterns. Each fires
# independently of token count — even a long "this page intentionally
# left blank but here are the section headers" still trips the meta
# pattern because the substring is present.
_META_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bthis\s+page\s+(?:is\s+)?(?:intentionally\s+)?(?:left\s+)?blank\b",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*test+\s*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*ping\s*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"\bare\s+you\s+(?:there|alive|on|awake|real)\b", re.IGNORECASE),
    re.compile(r"\b(?:lorem\s+ipsum|foo\s*bar|hello\s+world)\b", re.IGNORECASE),
    # Repeat-character mash: "aaaa", ".....", "!!!!". Min 4 to avoid
    # "ok!" tripping. Single-char only.
    re.compile(r"^\s*(.)\1{3,}\s*$"),
)

# Greetings get a separate path (one-line acknowledgement, not a joke).
# Detector returns False for these so banter logic stays out of the way.
_GREETING_RE = re.compile(
    r"^\s*(?:hi|hey+|hello+|morning|afternoon|evening|sup|yo|howdy)\s*[.!?,]?\s*$",
    re.IGNORECASE,
)


def _token_count(text: str) -> int:
    return sum(1 for t in text.split() if t)


def _has_domain_anchor(text: str) -> bool:
    if _SECTION_RE.search(text):
        return True
    words = re.findall(r"[a-z][a-z-]+", text.lower())
    return any(w in _DOMAIN_ANCHORS for w in words)


def is_banter_prompt(
    prompt: str,
    *,
    prior_prompt: str | None = None,
    short_token_threshold: int = 8,
) -> bool:
    """True when `prompt` reads as banter, smartass, or empty signal.

    Fires on (any one):
      - explicit meta pattern (blank-page, test, ping, lorem ipsum,
        repeat-char, are-you-there)
      - empty / whitespace-only
      - exact repeat of `prior_prompt` (verbatim re-send is the user
        testing whether we'll actually reread)
      - <`short_token_threshold` tokens AND zero domain anchors AND
        not a greeting

    Greetings exit False so the caller can route them to a brief
    acknowledgement path instead of a joke.
    """
    text = prompt.strip()
    if not text:
        return True
    if any(p.search(text) for p in _META_PATTERNS):
        return True
    if prior_prompt is not None and text == prior_prompt.strip():
        return True
    if _GREETING_RE.match(text):
        return False
    return _token_count(text) < short_token_threshold and not _has_domain_anchor(text)


# ---------- joke corpus ----------


@dataclass(frozen=True)
class JokeEntry:
    id: str
    text: str
    tags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class BanterCorpus:
    """Loaded character/<name>/jokes.yaml. Immutable; safe to share
    across concurrent turns. Per-session 'seen' set is the caller's
    responsibility (lives on the streak tracker)."""

    jokes: tuple[JokeEntry, ...]

    @classmethod
    def load(cls, path: Path | str) -> BanterCorpus:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
        jokes = tuple(
            JokeEntry(
                id=str(entry["id"]),
                text=str(entry["text"]),
                tags=frozenset(str(t) for t in (entry.get("tags") or ())),
            )
            for entry in raw
        )
        return cls(jokes=jokes)

    def pick(
        self,
        seen: Iterable[str],
        *,
        rng: random.Random | None = None,
    ) -> JokeEntry:
        """Random pick excluding ids in `seen`. When `seen` covers the
        full corpus, reshuffle from the full set — this guarantees no
        eternal silence at the cost of one possible immediate repeat
        only when the corpus is fully exhausted (acceptable; corpus
        starts at 20)."""
        if not self.jokes:
            raise ValueError("BanterCorpus is empty")
        seen_set = set(seen)
        candidates = [j for j in self.jokes if j.id not in seen_set]
        if not candidates:
            candidates = list(self.jokes)
        chooser = rng if rng is not None else random
        return chooser.choice(candidates)


# ---------- redirect ladder ----------

# Three-tier redirect copy used by D's streak tracker after a joke
# fires. Indexed by (streak % 4) - 1 when streak % 4 != 0 (joke turn).
# Tone deliberately escalates from patient → terse → flat.
REDIRECT_LADDER: tuple[str, str, str] = (
    "Got a § or topic?",
    "I cite JO 7110.65. Send a section.",
    "No content without an anchor.",
)
