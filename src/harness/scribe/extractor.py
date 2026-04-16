from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from harness.model.adapter import ChatMessage

if TYPE_CHECKING:
    from harness.character import Character
    from harness.model.adapter import ModelAdapter
    from harness.store.transcript import TranscriptMessage


@dataclass(frozen=True)
class EpisodicCandidate:
    title: str
    body: str
    principle: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SemanticCandidate:
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass(frozen=True)
class ScribeResult:
    episodic: tuple[EpisodicCandidate, ...]
    semantic: tuple[SemanticCandidate, ...]
    parse_error: str | None = None


_SCRIBE_SYSTEM = """\
You are the memory scribe for {name}, a persistent-character agent.

Given a window of recent conversation, extract memories worth keeping.

EPISODIC candidates — notable events, decisions, or exchanges worth
remembering as a short narrative. For each:
  title: 3-8 words
  body: 2-4 sentences in first person from {name}'s perspective
  principle: a one-sentence takeaway, or null if there isn't one
  tags: 2-5 single-word lowercase tags

SEMANTIC candidates — atomic facts about participants, the project, or
recurring patterns. For each:
  subject: a short noun phrase (e.g. "mark", "{name}", "harness")
  predicate: the relation (e.g. "prefers", "located_at", "uses")
  object: the value (e.g. "raw sqlite", "M4 Pro 48GB", "Metal")
  confidence: float 0.0-1.0 — how sure you are from the text alone

RULES:
- Be conservative. Empty or trivial windows should produce empty lists.
- Never invent facts not grounded in the conversation.
- Prefer 1-3 high-signal extractions over 5-10 trivial ones.
- Episodic bodies read as {name} telling the story to itself, not a
  transcript dump.
- Return STRICT JSON only. No preamble, no trailing text, no markdown
  code fences.

Schema:
{{
  "episodic": [
    {{"title": str, "body": str, "principle": str or null, "tags": [str]}}
  ],
  "semantic": [
    {{"subject": str, "predicate": str, "object": str, "confidence": number}}
  ]
}}"""


def format_window(turns: Sequence[TranscriptMessage]) -> str:
    """Render a window of TranscriptMessages as lines the scribe can
    reason over. Speaker + content only — no timestamps, no channel
    metadata."""
    return "\n".join(f"{t.speaker}: {t.content}" for t in turns)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_scribe_output(raw: str) -> ScribeResult:
    """Parse the scribe model's response into structured candidates.
    Tolerates leading/trailing text and markdown fences; returns an
    empty result with a `parse_error` note when JSON extraction fails.

    Any individual candidate that doesn't match the schema is dropped,
    not fatal — the scribe might emit 5 good items and 1 malformed, and
    we want to keep the 5."""
    text = raw.strip()
    # Strip markdown code fences if the model added them despite the rule
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text)

    match = _JSON_BLOCK.search(text)
    if match is None:
        return ScribeResult(episodic=(), semantic=(), parse_error="no JSON block found")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return ScribeResult(episodic=(), semantic=(), parse_error=f"JSON decode: {exc}")

    if not isinstance(data, dict):
        return ScribeResult(episodic=(), semantic=(), parse_error="top-level not an object")

    episodic = tuple(_coerce_episodic(item) for item in data.get("episodic", []) or [])
    semantic = tuple(_coerce_semantic(item) for item in data.get("semantic", []) or [])
    return ScribeResult(
        episodic=tuple(e for e in episodic if e is not None),
        semantic=tuple(s for s in semantic if s is not None),
    )


def _coerce_episodic(item: Any) -> EpisodicCandidate | None:
    if not isinstance(item, dict):
        return None
    title = item.get("title")
    body = item.get("body")
    if not (isinstance(title, str) and isinstance(body, str) and title.strip() and body.strip()):
        return None
    principle = item.get("principle")
    if principle is not None and not isinstance(principle, str):
        principle = None
    tags_raw = item.get("tags") or []
    if not isinstance(tags_raw, list):
        tags_raw = []
    tags = tuple(str(t).lower().strip() for t in tags_raw if isinstance(t, str) and t.strip())
    return EpisodicCandidate(
        title=title.strip(),
        body=body.strip(),
        principle=principle.strip() if isinstance(principle, str) and principle.strip() else None,
        tags=tags,
    )


def _coerce_semantic(item: Any) -> SemanticCandidate | None:
    if not isinstance(item, dict):
        return None
    subject = item.get("subject")
    predicate = item.get("predicate")
    obj = item.get("object")
    if not (isinstance(subject, str) and subject.strip()):
        return None
    if not (isinstance(predicate, str) and predicate.strip()):
        return None
    if not (isinstance(obj, str) and obj.strip()):
        return None
    try:
        confidence = float(item.get("confidence", 0.5))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))
    return SemanticCandidate(
        subject=subject.strip(),
        predicate=predicate.strip(),
        object=obj.strip(),
        confidence=confidence,
    )


def extract_candidates(
    adapter: ModelAdapter,
    character: Character,
    *,
    turns: Sequence[TranscriptMessage],
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> ScribeResult:
    """Ask the model to extract episodic + semantic candidates from a
    window of conversation. Returns parsed candidates, or an empty
    result with a `parse_error` note when the model's output can't be
    decoded. Pure beyond the adapter call."""
    if not turns:
        return ScribeResult(episodic=(), semantic=())
    system = ChatMessage(role="system", content=_SCRIBE_SYSTEM.format(name=character.name))
    window = format_window(turns)
    user = ChatMessage(role="user", content=f"Conversation window:\n{window}\n\nJSON:")
    raw = adapter.complete([system, user], max_tokens=max_tokens, temperature=temperature)
    return parse_scribe_output(raw)
