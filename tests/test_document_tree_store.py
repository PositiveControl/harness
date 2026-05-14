"""Tests for harness.store.document_tree (harness-h5ly / Phase 1).

Real SQLite (`tmp_path` per convention). Deterministic fake embedder
that hashes text into a stable unit vector so dense cosine has real
signal without pulling sentence-transformers into the test run."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.store.document_tree import DocumentTreeStore, TreeNode, _build_embed_text


@dataclass
class _HashEmbedder:
    """Fake embedder that maps each text to a deterministic, normalized
    8-dim vector via byte-hashing. Different text → different vector;
    identical text → identical vector — enough signal for cosine-based
    retrieval tests without loading a real model."""

    id: str = "hash-test"
    dimension: int = 8

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # 8 floats out of 32 bytes — take 4 bytes per float and
            # interpret as a [-1, 1] coordinate.
            coords = np.array(
                [(digest[i * 4] - 128) / 128.0 for i in range(8)],
                dtype=np.float32,
            )
            norm = np.linalg.norm(coords) or 1.0
            rows.append(coords / norm)
        return np.stack(rows)


# ---------- documents + nodes ----------


def test_upsert_document_is_idempotent(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    first = store.upsert_document(name="JO 7110.65", source_uri="faa.gov/...")
    second = store.upsert_document(name="JO 7110.65", source_uri="ignored-on-re-upsert")
    assert first.id == second.id
    assert second.source_uri == "faa.gov/..."  # original survives


def test_ingest_node_writes_embedded_node(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    doc = store.upsert_document(name="doc")
    node = store.ingest_node(
        document_id=doc.id,
        path="2-4-3",
        ordinal=0,
        depth=3,
        node_type="section",
        heading="Read-back of an ATC clearance",
        body="When a pilot reads back a clearance, the controller must verify it.",
    )
    assert node.path == "2-4-3"
    assert node.node_type == "section"
    assert store.count_embedded() == 1


def test_ingest_node_skips_embedding_when_embed_false(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    doc = store.upsert_document(name="doc")
    structural = store.ingest_node(
        document_id=doc.id,
        path="2",
        ordinal=0,
        depth=1,
        node_type="chapter",
        heading="General Control",
        embed=False,
    )
    leaf = store.ingest_node(
        document_id=doc.id,
        parent_id=structural.id,
        path="2-4",
        ordinal=0,
        depth=2,
        node_type="section_group",
        heading="ATC Clearances",
        body="Detail about clearances...",
    )
    # Only the embedded leaf surfaces.
    assert store.count_embedded() == 1
    assert leaf.parent_id == structural.id


def test_ingest_node_is_idempotent_on_path(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    doc = store.upsert_document(name="doc")
    first = store.ingest_node(
        document_id=doc.id,
        path="2-4-3",
        ordinal=0,
        depth=3,
        node_type="section",
        heading="Read-back",
        body="x",
    )
    second = store.ingest_node(
        document_id=doc.id,
        path="2-4-3",
        ordinal=0,
        depth=3,
        node_type="section",
        heading="changed heading — should not overwrite",
        body="changed body",
    )
    assert first.id == second.id
    refetched = store.get_node(first.id)
    assert refetched.heading == "Read-back"  # first write wins


def test_get_node_by_path_returns_none_when_absent(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    doc = store.upsert_document(name="doc")
    assert store.get_node_by_path(doc.id, "nope") is None


def test_children_returns_in_ordinal_order(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    doc = store.upsert_document(name="doc")
    parent = store.ingest_node(
        document_id=doc.id,
        path="2",
        ordinal=0,
        depth=1,
        node_type="chapter",
        heading="Chapter 2",
        embed=False,
    )
    # Ingest in non-monotonic order to confirm ORDER BY ordinal kicks in.
    store.ingest_node(
        document_id=doc.id,
        parent_id=parent.id,
        path="2-3",
        ordinal=2,
        depth=2,
        node_type="section_group",
        heading="third",
        body="b",
    )
    store.ingest_node(
        document_id=doc.id,
        parent_id=parent.id,
        path="2-1",
        ordinal=0,
        depth=2,
        node_type="section_group",
        heading="first",
        body="b",
    )
    store.ingest_node(
        document_id=doc.id,
        parent_id=parent.id,
        path="2-2",
        ordinal=1,
        depth=2,
        node_type="section_group",
        heading="second",
        body="b",
    )
    children = store.children(parent.id)
    assert [c.path for c in children] == ["2-1", "2-2", "2-3"]


# ---------- search ----------


def _seed_corpus(store: DocumentTreeStore) -> tuple[TreeNode, TreeNode, TreeNode]:
    """Three section-grain nodes about distinct topics so BM25 + dense
    both surface the right one for a topical query."""
    doc = store.upsert_document(name="seed-corpus")
    readback = store.ingest_node(
        document_id=doc.id,
        path="2-4-3",
        ordinal=0,
        depth=3,
        node_type="section",
        heading="Read-back of an ATC clearance",
        body=(
            "When a pilot reads back a clearance, the controller must verify "
            "the read-back. Hold-short instructions must be acknowledged."
        ),
    )
    separation = store.ingest_node(
        document_id=doc.id,
        path="3-10-3",
        ordinal=1,
        depth=3,
        node_type="section",
        heading="Same Runway Separation",
        body=(
            "Separate an arriving aircraft from another aircraft using the "
            "same runway by ensuring the trailing aircraft does not cross "
            "the landing threshold until separation exists."
        ),
    )
    wake = store.ingest_node(
        document_id=doc.id,
        path="2-1-19",
        ordinal=2,
        depth=3,
        node_type="section",
        heading="Wake Turbulence",
        body=(
            "Wake turbulence separation procedures apply when a smaller "
            "aircraft follows a larger one on departure or approach."
        ),
    )
    return readback, separation, wake


def test_search_hybrid_finds_the_right_section(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    readback, _, _ = _seed_corpus(store)
    hits = store.search("readback clearance hold-short", k=3, mode="hybrid")
    assert hits, "hybrid should return at least one node"
    # The read-back section's heading + body share most distinctive
    # tokens with the query; BM25 should surface it at rank 0.
    assert hits[0][0].id == readback.id


def test_search_text_only_returns_bm25_matches(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    _, _, wake = _seed_corpus(store)
    hits = store.search("wake turbulence separation procedures", k=3, mode="text")
    assert hits[0][0].id == wake.id


def test_search_excludes_structural_only_nodes(tmp_path: Path) -> None:
    """Structural-only nodes (no embedding) must never show up in
    search results — they're metadata, not retrieval targets."""
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    doc = store.upsert_document(name="doc")
    chapter = store.ingest_node(
        document_id=doc.id,
        path="2",
        ordinal=0,
        depth=1,
        node_type="chapter",
        heading="Chapter 2",
        body="overview that mentions readback prominently",
        embed=False,
    )
    section = store.ingest_node(
        document_id=doc.id,
        parent_id=chapter.id,
        path="2-4-3",
        ordinal=0,
        depth=3,
        node_type="section",
        heading="Read-back of an ATC clearance",
        body="A pilot's read-back must be verified.",
    )
    hits = store.search("readback clearance", k=5, mode="hybrid")
    assert hits, "should still get the embedded section"
    assert {h[0].id for h in hits} == {section.id}


def test_search_empty_query_returns_no_hits(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    _seed_corpus(store)
    # sanitize_fts_query strips to empty on pure punctuation; text path
    # short-circuits. Hybrid still has dense to fall back on, but
    # whitespace-only should produce no usable query at all.
    hits_text = store.search("!!!", k=3, mode="text")
    assert hits_text == []


def test_search_dense_ranks_by_cosine_descending_with_stable_id_tiebreak(
    tmp_path: Path,
) -> None:
    """Dense path must return nodes sorted by cosine descending. With
    the hash embedder, ties are rare but the tiebreaker on node.id is
    the contract that pins determinism — assert it explicitly."""
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    _seed_corpus(store)
    hits = store.search("airspace", k=10, mode="dense")
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


# ---------- embed-text composition ----------


def test_build_embed_text_includes_path_tag() -> None:
    out = _build_embed_text(heading="Read-back", body="A pilot read-back...", path="2-4-3")
    assert "[path: 2-4-3]" in out
    assert "Read-back" in out
    assert "A pilot read-back..." in out


def test_build_embed_text_skips_empty_body() -> None:
    out = _build_embed_text(heading="Chapter 2", body="", path="2")
    assert out.startswith("[path: 2]")
    assert "Chapter 2" in out
    # No trailing double-newline-then-empty body.
    assert not out.endswith("\n\n")


# ---------- error paths ----------


def test_get_node_raises_keyerror_on_missing_id(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    with pytest.raises(KeyError):
        store.get_node(9999)


def test_get_document_raises_keyerror_on_missing_id(tmp_path: Path) -> None:
    store = DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())
    with pytest.raises(KeyError):
        store.get_document(9999)
