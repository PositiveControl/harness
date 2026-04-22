from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.consolidate import (
    cluster_by_similarity,
    consolidate_episodic,
    consolidate_semantic,
    run_consolidation,
)
from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore


@dataclass
class _ControlledEmbedder:
    """Returns embeddings taken from a stable dict keyed by input text.
    Lets tests force specific similarity relationships between records.

    The table keys match the post-contextual-chunking format — the
    episodic store prepends a `[tier: X; principle: Y; date: Z]`
    header (harness-2am) before embedding. `_lookup_key` strips that
    header so test fixtures can key on logical body text without
    pinning the current date."""

    table: dict[str, np.ndarray]
    id: str = "controlled"
    dimension: int = 4

    @staticmethod
    def _lookup_key(text: str) -> str:
        # Drop a leading "[...]\n\n" tag header if present so the
        # table matches on the stable content suffix.
        if text.startswith("["):
            end = text.find("]")
            if end != -1 and text[end + 1 : end + 3] == "\n\n":
                return text[end + 3 :]
        return text

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for text in texts:
            key = self._lookup_key(text)
            vec = self.table.get(key)
            if vec is None:
                # Default to a stable deterministic vector for unknown text.
                h = sum(ord(c) for c in key.lower())
                v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
                norm = float(np.linalg.norm(v))
                vec = v / norm if norm > 0 else v
            out.append(vec.astype(np.float32))
        return np.stack(out)


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 0 else arr


# ---------- cluster_by_similarity ----------


def test_cluster_transitive_closure() -> None:
    # A-B similar, B-C similar → A, B, C all in one cluster even if A-C < threshold
    a = _unit([1.0, 0.0, 0.0, 0.0])
    b = _unit([0.9, 0.1, 0.0, 0.0])
    c = _unit([0.7, 0.3, 0.0, 0.0])
    clusters = cluster_by_similarity([1, 2, 3], [a, b, c], threshold=0.9)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [1, 2, 3]


def test_cluster_isolates_unrelated() -> None:
    a = _unit([1.0, 0.0, 0.0, 0.0])
    b = _unit([0.0, 1.0, 0.0, 0.0])
    c = _unit([0.0, 0.0, 1.0, 0.0])
    clusters = cluster_by_similarity([1, 2, 3], [a, b, c], threshold=0.9)
    assert sorted([sorted(c) for c in clusters]) == [[1], [2], [3]]


# ---------- consolidate_episodic ----------


def test_consolidate_episodic_merges_near_duplicates(tmp_path: Path) -> None:
    dupe_vec = _unit([1.0, 0.0, 0.0, 0.0])
    near_vec = _unit([0.98, 0.05, 0.0, 0.0])
    lone_vec = _unit([0.0, 1.0, 0.0, 0.0])

    embedder = _ControlledEmbedder(
        table={
            "alpha-1\n\ndoes foo\n\nbody one": dupe_vec,
            "alpha-2\n\ndoes foo\n\nbody two": near_vec,
            "beta\n\ndoes bar\n\nbody three": lone_vec,
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(
            external_id="alpha-1",
            title="alpha-1",
            body="body one",
            principle="does foo",
            tier="working",
            source="test",
        )
        store.ingest(
            external_id="alpha-2",
            title="alpha-2",
            body="body two",
            principle="does foo",
            tier="working",
            source="test",
        )
        store.ingest(
            external_id="beta",
            title="beta",
            body="body three",
            principle="does bar",
            tier="working",
            source="test",
        )

        considered, merged, superseded = consolidate_episodic(store, threshold=0.90)
        assert considered == 3
        assert merged == 1  # alpha cluster merged, beta alone
        assert superseded == 2

        active = store.all()
        # 1 consolidated + 1 lone working = 2
        assert len(active) == 2
        assert any(r.tier == "consolidated" for r in active)
        assert any(r.external_id == "beta" for r in active)

        # Superseded originals still exist when asked for them
        all_including = store.all(include_superseded=True)
        assert len(all_including) == 4  # 3 working + 1 consolidated
    finally:
        store.close()


def test_consolidate_episodic_skips_when_fewer_than_two(tmp_path: Path) -> None:
    embedder = _ControlledEmbedder(table={})
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(
            external_id="only",
            title="only",
            body="body",
            tier="working",
            source="test",
        )
        considered, merged, superseded = consolidate_episodic(store, threshold=0.80)
        assert considered == 1
        assert merged == 0
        assert superseded == 0
    finally:
        store.close()


def test_superseded_records_drop_out_of_search(tmp_path: Path) -> None:
    dupe_a = _unit([1.0, 0.0, 0.0, 0.0])
    dupe_b = _unit([0.99, 0.01, 0.0, 0.0])
    embedder = _ControlledEmbedder(
        table={
            "a\n\n\n\nbody a": dupe_a,
            "b\n\n\n\nbody b": dupe_b,
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(external_id="a", title="a", body="body a", tier="working", source="t")
        store.ingest(external_id="b", title="b", body="body b", tier="working", source="t")

        consolidate_episodic(store, threshold=0.90)

        # Search should only return the consolidated record, not the superseded originals
        hits = store.search("anything", k=10)
        active_ids = {r.id for r, _ in hits}
        assert all(store.get(i).superseded_by is None for i in active_ids)
    finally:
        store.close()


# ---------- consolidate_semantic ----------


def test_consolidate_semantic_merges_same_subject_predicate(tmp_path: Path) -> None:
    embedder = _ControlledEmbedder(table={})
    store = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.add(subject="mark", predicate="prefers", object="A", confidence=0.6, source="t")
        store.add(subject="Mark", predicate="Prefers", object="B", confidence=0.9, source="t")
        store.add(subject="harness", predicate="runs_on", object="M4", confidence=0.8, source="t")

        considered, merged, superseded = consolidate_semantic(store)
        assert considered == 3
        assert merged == 1  # the mark/prefers group
        assert superseded == 2

        active = store.all()
        # 1 consolidated (with highest-conf object "B") + 1 lone (harness runs_on) = 2
        assert len(active) == 2
        consolidated_row = next(r for r in active if r.tier == "consolidated")
        assert consolidated_row.object == "B"  # highest confidence won
    finally:
        store.close()


def test_consolidate_semantic_single_fact_no_op(tmp_path: Path) -> None:
    embedder = _ControlledEmbedder(table={})
    store = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.add(subject="s", predicate="p", object="o", source="t")
        _considered, merged, superseded = consolidate_semantic(store)
        assert merged == 0
        assert superseded == 0
    finally:
        store.close()


def test_semantic_superseded_facts_drop_out_of_search(tmp_path: Path) -> None:
    embedder = _ControlledEmbedder(table={})
    store = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.add(subject="x", predicate="y", object="old", confidence=0.5, source="t")
        store.add(subject="x", predicate="y", object="new", confidence=0.95, source="t")

        consolidate_semantic(store)

        hits = store.search("anything", k=10)
        for fact, _score in hits:
            assert fact.superseded_by is None
    finally:
        store.close()


# ---------- run_consolidation top-level ----------


def test_run_consolidation_returns_summary(tmp_path: Path) -> None:
    embedder = _ControlledEmbedder(table={})
    ep = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    sem = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        summary = run_consolidation(ep, sem)
        assert summary.episodic_considered == 0
        assert summary.semantic_considered == 0
    finally:
        ep.close()
        sem.close()


def test_consolidation_is_idempotent(tmp_path: Path) -> None:
    """Running twice should be a no-op the second time — the
    consolidator only touches working-tier, and the first pass moves
    everything out."""
    dupe_a = _unit([1.0, 0.0, 0.0, 0.0])
    dupe_b = _unit([0.99, 0.01, 0.0, 0.0])
    embedder = _ControlledEmbedder(table={"a\n\n\n\nbody a": dupe_a, "b\n\n\n\nbody b": dupe_b})
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(external_id="a", title="a", body="body a", tier="working", source="t")
        store.ingest(external_id="b", title="b", body="body b", tier="working", source="t")

        first = consolidate_episodic(store, threshold=0.90)
        second = consolidate_episodic(store, threshold=0.90)

        assert first == (2, 1, 2)
        assert second == (0, 0, 0)  # nothing left in working tier
    finally:
        store.close()


def test_store_schema_migration_adds_superseded_by(tmp_path: Path) -> None:
    """Opening an EpisodicStore against a DB that lacks the
    superseded_by column (simulated by creating a minimal table first)
    should ALTER TABLE to add it rather than crash."""
    import sqlite3

    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE episodic (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               external_id TEXT UNIQUE,
               title TEXT NOT NULL,
               body TEXT NOT NULL,
               principle TEXT,
               tags TEXT NOT NULL DEFAULT '[]',
               tier TEXT NOT NULL,
               source TEXT NOT NULL,
               session_id TEXT,
               user_id TEXT,
               created_at TEXT NOT NULL,
               last_accessed TEXT,
               embedding BLOB NOT NULL
           )"""
    )
    conn.commit()
    conn.close()

    embedder = _ControlledEmbedder(table={})
    store = EpisodicStore(db, embedder=embedder)
    try:
        # Doesn't raise → migration worked
        rec = store.ingest(
            external_id="post-migration",
            title="t",
            body="b",
            tier="working",
            source="t",
        )
        assert rec.superseded_by is None
    finally:
        store.close()


def test_run_consolidation_does_not_touch_consolidated_tier(tmp_path: Path) -> None:
    """Only working-tier records are eligible for consolidation. A
    previously-consolidated record must not be clobbered by a second
    run."""
    pytest.importorskip("numpy")

    vec = _unit([1.0, 0.0, 0.0, 0.0])
    embedder = _ControlledEmbedder(
        table={
            "x\n\n\n\nbody x": vec,
            "y\n\n\n\nbody y": vec,
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(external_id="x", title="x", body="body x", tier="working", source="t")
        store.ingest(external_id="y", title="y", body="body y", tier="working", source="t")
        consolidate_episodic(store, threshold=0.90)

        # One consolidated record now exists
        consolidated = [r for r in store.all() if r.tier == "consolidated"]
        assert len(consolidated) == 1
        consolidated_before_id = consolidated[0].id

        # Run again — consolidated record should still be there, untouched
        consolidate_episodic(store, threshold=0.90)
        after = [r for r in store.all() if r.tier == "consolidated"]
        assert len(after) == 1
        assert after[0].id == consolidated_before_id
    finally:
        store.close()


# ---------- user-awareness (harness-4uh) ----------


def test_consolidate_episodic_does_not_merge_across_users(tmp_path: Path) -> None:
    """Two near-duplicate records belonging to different users must NOT
    be merged. Shared (user_id IS NULL) counts as its own user for
    this purpose — shared stays shared."""
    pytest.importorskip("numpy")
    dupe = _unit([1.0, 0.0, 0.0, 0.0])
    embedder = _ControlledEmbedder(
        table={
            "mark's note\n\n\n\nsame body": dupe,
            "alice's note\n\n\n\nsame body": dupe,
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(
            external_id="mark-1",
            title="mark's note",
            body="same body",
            tier="working",
            source="t",
            user_id="mark",
        )
        store.ingest(
            external_id="alice-1",
            title="alice's note",
            body="same body",
            tier="working",
            source="t",
            user_id="alice",
        )

        considered, merged, superseded = consolidate_episodic(store, threshold=0.90)
        assert considered == 2
        assert merged == 0
        assert superseded == 0

        # Both original records still active and scoped to their owners.
        active = store.all()
        by_user = {r.user_id: r for r in active}
        assert by_user["mark"].external_id == "mark-1"
        assert by_user["alice"].external_id == "alice-1"
    finally:
        store.close()


def test_consolidate_episodic_merges_within_same_user(tmp_path: Path) -> None:
    """Within one user's partition the normal clustering still fires."""
    pytest.importorskip("numpy")
    dupe = _unit([1.0, 0.0, 0.0, 0.0])
    near = _unit([0.98, 0.05, 0.0, 0.0])
    embedder = _ControlledEmbedder(
        table={
            "mark-1\n\n\n\nsame body": dupe,
            "mark-2\n\n\n\nsame body": near,
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(
            external_id="mark-1",
            title="mark-1",
            body="same body",
            tier="working",
            source="t",
            user_id="mark",
        )
        store.ingest(
            external_id="mark-2",
            title="mark-2",
            body="same body",
            tier="working",
            source="t",
            user_id="mark",
        )
        considered, merged, superseded = consolidate_episodic(store, threshold=0.90)
        assert considered == 2
        assert merged == 1
        assert superseded == 2

        # Consolidated record inherits user_id=mark.
        consolidated = [r for r in store.all() if r.tier == "consolidated"]
        assert len(consolidated) == 1
        assert consolidated[0].user_id == "mark"
    finally:
        store.close()


def test_consolidate_episodic_shared_stays_shared(tmp_path: Path) -> None:
    """Shared records (user_id IS NULL) form their own partition and
    must not be merged with a user's private memory."""
    pytest.importorskip("numpy")
    dupe = _unit([1.0, 0.0, 0.0, 0.0])
    embedder = _ControlledEmbedder(
        table={
            "shared-1\n\n\n\nsame body": dupe,
            "mark-1\n\n\n\nsame body": dupe,
        }
    )
    store = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.ingest(
            external_id="shared-1",
            title="shared-1",
            body="same body",
            tier="working",
            source="t",
            user_id=None,
        )
        store.ingest(
            external_id="mark-1",
            title="mark-1",
            body="same body",
            tier="working",
            source="t",
            user_id="mark",
        )
        _considered, merged, superseded = consolidate_episodic(store, threshold=0.90)
        # Different partitions, no merge.
        assert merged == 0
        assert superseded == 0

        active = store.all()
        by_user = {r.user_id: r for r in active}
        assert by_user[None].external_id == "shared-1"
        assert by_user["mark"].external_id == "mark-1"
    finally:
        store.close()


def test_consolidate_semantic_does_not_merge_across_users(tmp_path: Path) -> None:
    """One user's (subject, predicate) partition must never merge with
    another user's, even if the subject and predicate text match."""
    embedder = _ControlledEmbedder(table={})
    store = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.add(
            subject="mark",
            predicate="prefers",
            object="espresso",
            confidence=0.9,
            source="t",
            user_id="mark",
        )
        store.add(
            subject="mark",
            predicate="prefers",
            object="drip coffee",
            confidence=0.9,
            source="t",
            user_id="alice",
        )
        considered, merged, superseded = consolidate_semantic(store)
        assert considered == 2
        assert merged == 0
        assert superseded == 0

        active = store.all()
        by_user = {f.user_id: f.object for f in active}
        assert by_user["mark"] == "espresso"
        assert by_user["alice"] == "drip coffee"
    finally:
        store.close()


def test_consolidate_semantic_merges_within_same_user(tmp_path: Path) -> None:
    embedder = _ControlledEmbedder(table={})
    store = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.add(
            subject="mark",
            predicate="prefers",
            object="A",
            confidence=0.6,
            source="t",
            user_id="mark",
        )
        store.add(
            subject="Mark",
            predicate="Prefers",
            object="B",
            confidence=0.9,
            source="t",
            user_id="mark",
        )
        considered, merged, superseded = consolidate_semantic(store)
        assert considered == 2
        assert merged == 1
        assert superseded == 2

        # Promoted fact carries user_id=mark and keeps the higher-confidence object.
        consolidated = [f for f in store.all() if f.tier == "consolidated"]
        assert len(consolidated) == 1
        assert consolidated[0].user_id == "mark"
        assert consolidated[0].object == "B"
    finally:
        store.close()


def test_consolidate_semantic_shared_stays_shared(tmp_path: Path) -> None:
    embedder = _ControlledEmbedder(table={})
    store = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    try:
        store.add(
            subject="airton",
            predicate="uses",
            object="mlx",
            confidence=0.9,
            source="t",
            user_id=None,
        )
        store.add(
            subject="airton",
            predicate="uses",
            object="ollama",
            confidence=0.95,
            source="t",
            user_id="alice",
        )
        _considered, merged, superseded = consolidate_semantic(store)
        # Different partitions — the same (subject, predicate) doesn't
        # bridge the user_id boundary.
        assert merged == 0
        assert superseded == 0
    finally:
        store.close()
