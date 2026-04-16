from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.character import load_character
from harness.store.episodic import EpisodicStore, ensure_seeds_ingested

REPO = Path(__file__).resolve().parents[1]
AIRTON = REPO / "character" / "airton"


@dataclass
class _FakeEmbedder:
    """Deterministic pseudo-embeddings from character-code arithmetic."""

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
def store(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "harness.sqlite", embedder=_FakeEmbedder())


def test_ingest_and_retrieve_round_trip(store: EpisodicStore) -> None:
    rec = store.ingest(
        external_id="test-1",
        title="A Test Memory",
        body="Something happened once.",
        principle="Things can happen.",
        tags=["test"],
        tier="working",
        source="user",
    )
    assert rec.id > 0
    assert rec.external_id == "test-1"
    assert rec.tier == "working"
    assert rec.tags == ("test",)

    fetched = store.get(rec.id)
    assert fetched == rec


def test_ingest_is_idempotent_on_external_id(store: EpisodicStore) -> None:
    first = store.ingest(
        external_id="dupe",
        title="First",
        body="body",
        tier="working",
        source="user",
    )
    second = store.ingest(
        external_id="dupe",
        title="Second",  # different title — should be ignored
        body="body",
        tier="working",
        source="user",
    )
    assert first.id == second.id
    assert second.title == "First"  # original wins


def test_search_returns_top_k_with_scores(store: EpisodicStore) -> None:
    store.ingest(external_id="a", title="alpha", body="A", tier="seed", source="yaml")
    store.ingest(external_id="b", title="bravo", body="B", tier="seed", source="yaml")
    store.ingest(external_id="c", title="charlie", body="C", tier="seed", source="yaml")

    hits = store.search("alpha", k=2)
    assert len(hits) == 2
    for rec, score in hits:
        assert isinstance(rec.title, str)
        assert -1.0 <= score <= 1.0


def test_search_on_empty_store_returns_empty(store: EpisodicStore) -> None:
    assert store.search("anything") == []


def test_all_filters_by_tier(store: EpisodicStore) -> None:
    store.ingest(external_id="seed1", title="s1", body="", tier="seed", source="yaml")
    store.ingest(external_id="work1", title="w1", body="", tier="working", source="user")

    assert {r.tier for r in store.all()} == {"seed", "working"}
    seeds_only = store.all(tier="seed")
    assert len(seeds_only) == 1
    assert seeds_only[0].tier == "seed"


def test_ensure_seeds_ingested_loads_character_seeds(store: EpisodicStore) -> None:
    character = load_character(AIRTON)
    inserted = ensure_seeds_ingested(character, store)

    assert inserted == len(character.seed_memories)
    assert len(store.all(tier="seed")) == len(character.seed_memories)

    # Second call is a no-op
    reinserted = ensure_seeds_ingested(character, store)
    assert reinserted == 0


def test_ensure_seeds_preserves_principle_and_tags(store: EpisodicStore) -> None:
    character = load_character(AIRTON)
    ensure_seeds_ingested(character, store)

    capacitor = next(r for r in store.all(tier="seed") if r.external_id == "01-missing-capacitor")
    assert capacitor.principle == "Root cause or nothing."
    assert "root_cause" in capacitor.tags
