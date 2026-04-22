"""Tests for the bd → episodic memory harvester (harness-9yd).

Parallel to tests/test_skills_harvester.py — shares the embedder +
store setup pattern but stubs `memories_json()` rather than
`list_issues()`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from harness.skills import (
    EXTERNAL_ID_PREFIX,
    MEMORY_PRINCIPLE,
    MEMORY_SOURCE,
    MEMORY_TAGS,
    harvest_bd_memories,
)
from harness.store.episodic import EpisodicStore


@dataclass
class _Embedder:
    id: str = "fake"
    dimension: int = 4
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        batch = list(texts)
        self.calls.append(batch)
        return np.stack([np.ones(4, dtype=np.float32) / 2.0 for _ in batch])


@dataclass
class _StubAdapter:
    """Minimal stand-in for BeadsAdapter — only the surface the
    memory harvester reaches for. Tests pass it positionally with
    `# type: ignore[arg-type]` so mypy doesn't demand the full mixin
    MRO."""

    memories: dict[str, str]
    calls: list[str] = field(default_factory=list)

    def memories_json(self, query: str = "") -> dict[str, str]:
        self.calls.append(query)
        return dict(self.memories)


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "h.sqlite", embedder=_Embedder())


def test_harvest_ingests_memories_as_procedural(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        memories={
            "mark-is-the-creator-and-primary-user": "Mark is the creator and primary user.",
            "mark-first-computer": "Mark's first computer was a Macintosh Performa 405.",
        }
    )
    report = harvest_bd_memories(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]

    assert report.scanned == 2
    assert report.newly_ingested == 2
    assert report.already_present == 0
    assert set(report.ingested_keys) == {
        "mark-is-the-creator-and-primary-user",
        "mark-first-computer",
    }

    # Confirm the written rows carry the expected metadata.
    rows = episodic.all()
    assert {r.external_id for r in rows} == {
        "bd-mem:mark-is-the-creator-and-primary-user",
        "bd-mem:mark-first-computer",
    }
    for row in rows:
        assert row.tier == "procedural"
        assert row.principle == MEMORY_PRINCIPLE
        assert row.source == MEMORY_SOURCE
        assert row.user_id is None
        assert set(row.tags) == set(MEMORY_TAGS)


def test_harvest_is_idempotent(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(
        memories={"mark-is-the-creator-and-primary-user": "Mark is the creator and primary user."}
    )
    embedder_calls_before = len(episodic.embedder.calls)  # type: ignore[attr-defined]

    first = harvest_bd_memories(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    embedder_calls_after_first = len(episodic.embedder.calls)  # type: ignore[attr-defined]
    second = harvest_bd_memories(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    embedder_calls_after_second = len(episodic.embedder.calls)  # type: ignore[attr-defined]

    assert first.newly_ingested == 1
    assert second.newly_ingested == 0
    assert second.already_present == 1
    # Second run must not touch the embedder — the has(external_id)
    # short-circuit is the whole point of idempotency.
    assert embedder_calls_after_first > embedder_calls_before
    assert embedder_calls_after_second == embedder_calls_after_first


def test_harvest_empty_store_is_noop(episodic: EpisodicStore) -> None:
    adapter = _StubAdapter(memories={})
    report = harvest_bd_memories(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    assert report.scanned == 0
    assert report.newly_ingested == 0
    assert report.already_present == 0
    assert report.ingested_keys == ()
    assert episodic.all() == []


def test_external_id_carries_bd_mem_prefix(episodic: EpisodicStore) -> None:
    """Namespacing external_id avoids even theoretical collision
    with bead ids (harness-xyz)."""
    adapter = _StubAdapter(memories={"some-key": "some body"})
    harvest_bd_memories(ab_adapter=adapter, episodic=episodic)  # type: ignore[arg-type]
    rows = episodic.all()
    assert len(rows) == 1
    assert rows[0].external_id is not None
    assert rows[0].external_id.startswith(EXTERNAL_ID_PREFIX)
    assert rows[0].external_id == f"{EXTERNAL_ID_PREFIX}some-key"


def test_harvested_memory_surfaces_in_episodic_search(tmp_path: Path) -> None:
    """Closes the loop: after harvest, the mirrored memory ranks for a
    related query through the standard hybrid retrieval path — which
    is what the chat loop already injects into the prompt."""

    @dataclass
    class _KeywordEmbedder:
        id: str = "kw"
        dimension: int = 4

        def embed(self, texts: Iterable[str]) -> np.ndarray:
            vecs: list[np.ndarray] = []
            for text in texts:
                low = text.lower()
                v = np.array(
                    [
                        1.0 if "creator" in low or "created" in low else 0.0,
                        1.0 if "mark" in low else 0.0,
                        1.0 if "memory" in low else 0.0,
                        0.1,
                    ],
                    dtype=np.float32,
                )
                n = float(np.linalg.norm(v))
                vecs.append(v / n if n > 0 else v)
            return np.stack(vecs)

    store = EpisodicStore(tmp_path / "h.sqlite", embedder=_KeywordEmbedder())
    # An unrelated seed — mustn't outrank the harvested memory on the
    # creator query.
    store.ingest(
        external_id="seed-1",
        title="Unrelated",
        body="Something about weather.",
        tier="seed",
        source="yaml",
    )

    adapter = _StubAdapter(
        memories={
            "mark-is-the-creator-and-primary-user": "Mark is the creator and primary user.",
        }
    )
    harvest_bd_memories(ab_adapter=adapter, episodic=store)  # type: ignore[arg-type]

    hits = store.search("who created airton", k=3, mode="hybrid")
    ids = {r.external_id for r, _ in hits}
    assert "bd-mem:mark-is-the-creator-and-primary-user" in ids
