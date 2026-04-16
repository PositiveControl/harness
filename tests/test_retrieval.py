from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harness.character import load_character
from harness.retrieval import VoiceRetriever
from harness.retrieval.embed import Embedder

REPO = Path(__file__).resolve().parents[1]
AIRTON = REPO / "character" / "airton"


@dataclass
class _FakeEmbedder:
    """Deterministic pseudo-embedder for testing. Maps each string to a
    4-dim vector via character-code arithmetic. Similarity is only
    vaguely correlated with lexical overlap — fine for testing pipeline
    mechanics, useless for semantic quality checks."""

    id: str = "fake"
    dimension: int = 4
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        self.calls.append(texts_list)
        vectors: list[np.ndarray] = []
        for text in texts_list:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            vectors.append(v / norm if norm > 0 else v)
        return np.stack(vectors)


def test_fake_embedder_satisfies_protocol() -> None:
    assert isinstance(_FakeEmbedder(), Embedder)


def test_voice_retriever_returns_exactly_k() -> None:
    character = load_character(AIRTON)
    retriever = VoiceRetriever(embedder=_FakeEmbedder(), character=character)

    results = retriever.top_k("What should I do?", k=5)
    assert len(results) == 5
    assert len({r.id for r in results}) == 5


def test_voice_retriever_caps_k_at_available() -> None:
    character = load_character(AIRTON)
    retriever = VoiceRetriever(embedder=_FakeEmbedder(), character=character)

    huge = retriever.top_k("anything", k=10_000)
    assert len(huge) == len(character.voice_samples)


def test_voice_retriever_respects_exclude_ids() -> None:
    character = load_character(AIRTON)
    retriever = VoiceRetriever(embedder=_FakeEmbedder(), character=character)

    excluded = frozenset({s.id for s in character.voice_samples[:3]})
    results = retriever.top_k("anything", k=100, exclude_ids=excluded)
    returned = {r.id for r in results}
    assert not (returned & excluded)


def test_voice_retriever_embeds_samples_once_per_construction() -> None:
    character = load_character(AIRTON)
    embedder = _FakeEmbedder()
    retriever = VoiceRetriever(embedder=embedder, character=character)

    # One call on construction (for all sample prompts)
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == len(character.voice_samples)

    retriever.top_k("first")
    retriever.top_k("second")
    # Each top_k embeds one query
    assert len(embedder.calls) == 3
    assert embedder.calls[1] == ["first"]
    assert embedder.calls[2] == ["second"]
