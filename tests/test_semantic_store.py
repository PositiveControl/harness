from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.store.semantic import SemanticStore


@dataclass
class _FakeEmbedder:
    id: str = "fake"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            vectors.append(v / norm if norm > 0 else v)
        return np.stack(vectors)


@pytest.fixture
def store(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "harness.sqlite", embedder=_FakeEmbedder())


def test_add_and_get_round_trip(store: SemanticStore) -> None:
    fact = store.add(
        subject="mark",
        predicate="prefers",
        object="raw sqlite",
        confidence=0.9,
        source="user",
        tier="working",
    )
    assert fact.id > 0
    assert fact.subject == "mark"
    assert fact.predicate == "prefers"
    assert fact.object == "raw sqlite"
    assert fact.confidence == 0.9
    assert fact.tier == "working"

    fetched = store.get(fact.id)
    assert fetched == fact


def test_all_filters_by_tier_and_subject(store: SemanticStore) -> None:
    store.add(subject="mark", predicate="uses", object="m4 pro", source="yaml", tier="seed")
    store.add(subject="mark", predicate="prefers", object="raw sqlite", source="user")
    store.add(subject="harness", predicate="targets", object="darwin", source="user")

    all_facts = store.all()
    assert len(all_facts) == 3

    seeds = store.all(tier="seed")
    assert len(seeds) == 1
    assert seeds[0].predicate == "uses"

    mark_facts = store.all(subject="mark")
    assert {f.predicate for f in mark_facts} == {"uses", "prefers"}


def test_search_returns_top_k(store: SemanticStore) -> None:
    store.add(subject="a", predicate="x", object="one", source="u")
    store.add(subject="b", predicate="y", object="two", source="u")
    store.add(subject="c", predicate="z", object="three", source="u")

    hits = store.search("anything", k=2)
    assert len(hits) == 2
    for _fact, score in hits:
        assert -1.0 <= score <= 1.0


def test_search_respects_min_confidence(store: SemanticStore) -> None:
    store.add(subject="a", predicate="b", object="low", confidence=0.3, source="u")
    store.add(subject="a", predicate="b", object="high", confidence=0.9, source="u")

    hits_all = store.search("query", k=10, min_confidence=0.0)
    assert len(hits_all) == 2

    hits_high = store.search("query", k=10, min_confidence=0.5)
    assert len(hits_high) == 1
    assert hits_high[0][0].object == "high"


def test_search_on_empty_store_returns_empty(store: SemanticStore) -> None:
    assert store.search("anything") == []


def test_supersedes_is_nullable_and_stored(store: SemanticStore) -> None:
    first = store.add(subject="s", predicate="p", object="old", source="u")
    second = store.add(subject="s", predicate="p", object="new", source="u", supersedes=first.id)

    assert second.supersedes == first.id
    refetched = store.get(first.id)
    assert refetched.supersedes is None
