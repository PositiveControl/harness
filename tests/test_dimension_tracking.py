from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore


@dataclass
class _DimNEmbedder:
    dimension: int
    id: str = "test-embedder"

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        vecs: list[np.ndarray] = []
        for text in texts_list:
            base = sum(ord(c) for c in text.lower())
            v = np.array([(base + i) % 7 for i in range(self.dimension)], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            vecs.append(v / norm if norm > 0 else v)
        return np.stack(vecs)


# ---------- Episodic ----------


def test_episodic_search_skips_mismatched_dim(tmp_path: Path) -> None:
    db = tmp_path / "h.sqlite"
    # Write a record with a 4-dim embedder
    store_a = EpisodicStore(db, embedder=_DimNEmbedder(dimension=4, id="a"))
    store_a.ingest(external_id="first", title="one", body="alpha", tier="working", source="t")
    store_a.close()

    # Open with a DIFFERENT-dim embedder; the existing row should be
    # filtered out of search rather than crash on dim mismatch.
    store_b = EpisodicStore(db, embedder=_DimNEmbedder(dimension=8, id="b"))
    try:
        assert store_b.count_mismatched_embeddings() == 1
        # New row at dim=8 participates
        store_b.ingest(external_id="second", title="two", body="beta", tier="working", source="t")
        hits = store_b.search("query", k=10)
        ids = {r.id for r, _ in hits}
        # Only the new record (dim 8) is retrievable
        assert len(ids) == 1
        first = next(r for r in store_b.all(include_superseded=True) if r.external_id == "first")
        assert first.id not in ids
    finally:
        store_b.close()


def test_episodic_rebuild_refreshes_dims(tmp_path: Path) -> None:
    db = tmp_path / "h.sqlite"
    store_a = EpisodicStore(db, embedder=_DimNEmbedder(dimension=4, id="a"))
    store_a.ingest(external_id="first", title="t", body="b", tier="working", source="t")
    store_a.close()

    store_b = EpisodicStore(db, embedder=_DimNEmbedder(dimension=8, id="b"))
    try:
        assert store_b.count_mismatched_embeddings() == 1
        updated, _ = store_b.rebuild_embeddings()
        assert updated == 1
        assert store_b.count_mismatched_embeddings() == 0
        # Now the rebuilt row is retrievable
        hits = store_b.search("query", k=10)
        assert len(hits) == 1
    finally:
        store_b.close()


def test_episodic_new_writes_carry_embedder_metadata(tmp_path: Path) -> None:
    embedder = _DimNEmbedder(dimension=5, id="tagged")
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        rec = store.ingest(external_id="x", title="t", body="b", tier="working", source="t")
        row = store._conn.execute(
            "SELECT embedder_id, embedding_dim FROM episodic WHERE id = ?", (rec.id,)
        ).fetchone()
        assert row == ("tagged", 5)
    finally:
        store.close()


# ---------- Semantic ----------


def test_semantic_search_skips_mismatched_dim(tmp_path: Path) -> None:
    db = tmp_path / "h.sqlite"
    store_a = SemanticStore(db, embedder=_DimNEmbedder(dimension=4, id="a"))
    store_a.add(subject="x", predicate="y", object="z", source="t")
    store_a.close()

    store_b = SemanticStore(db, embedder=_DimNEmbedder(dimension=6, id="b"))
    try:
        assert store_b.count_mismatched_embeddings() == 1
        store_b.add(subject="p", predicate="q", object="r", source="t")
        hits = store_b.search("query", k=10)
        assert len(hits) == 1  # only the new one
    finally:
        store_b.close()


def test_semantic_rebuild_refreshes_dims(tmp_path: Path) -> None:
    db = tmp_path / "h.sqlite"
    store_a = SemanticStore(db, embedder=_DimNEmbedder(dimension=4, id="a"))
    store_a.add(subject="x", predicate="y", object="z", source="t")
    store_a.close()

    store_b = SemanticStore(db, embedder=_DimNEmbedder(dimension=6, id="b"))
    try:
        updated, _ = store_b.rebuild_embeddings()
        assert updated == 1
        assert store_b.count_mismatched_embeddings() == 0
        hits = store_b.search("query", k=10)
        assert len(hits) == 1
    finally:
        store_b.close()
