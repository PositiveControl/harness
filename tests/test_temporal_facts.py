"""Tests for temporal fields on SemanticStore (sota punch #4,
harness-kr2).

Facts carry an optional validity window (`valid_from`, `valid_to`)
and an `asserted_at` timestamp distinct from `created_at`. Retrieval
filters by `as_of` so 'what did we believe about X last March?' is a
first-class query shape.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from harness.store.semantic import SemanticStore


@dataclass
class _Embedder:
    id: str = "fake"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([np.ones(4, dtype=np.float32) / 2.0 for _ in texts])


@pytest.fixture
def store(tmp_path: Path) -> SemanticStore:
    return SemanticStore(tmp_path / "s.sqlite", embedder=_Embedder())


# ---------- schema + add() ----------


def test_add_defaults_asserted_at_to_now(store: SemanticStore) -> None:
    fact = store.add(
        subject="mark",
        predicate="lives_in",
        object="SF",
        source="user",
    )
    assert fact.asserted_at is not None
    # asserted_at and created_at should be within a few microseconds
    # (same clock read).
    assert abs((fact.asserted_at - fact.created_at).total_seconds()) < 1
    assert fact.valid_from is None
    assert fact.valid_to is None


def test_add_accepts_explicit_temporal_fields(store: SemanticStore) -> None:
    vf = datetime(2020, 1, 1, tzinfo=UTC)
    vt = datetime(2024, 6, 1, tzinfo=UTC)
    asserted = datetime(2026, 3, 15, tzinfo=UTC)
    fact = store.add(
        subject="mark",
        predicate="lives_in",
        object="SF",
        source="user",
        valid_from=vf,
        valid_to=vt,
        asserted_at=asserted,
    )
    assert fact.valid_from == vf
    assert fact.valid_to == vt
    assert fact.asserted_at == asserted


def test_migration_backfills_asserted_at_from_created_at(tmp_path: Path) -> None:
    """Install created before temporal fields landed: rows have
    created_at but no asserted_at. Reopening through the new store
    must backfill asserted_at = created_at so provenance isn't lost."""
    db = tmp_path / "s.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE semantic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.8,
            source TEXT NOT NULL,
            attributed_to TEXT,
            session_id TEXT,
            user_id TEXT,
            supersedes INTEGER,
            superseded_by INTEGER,
            tier TEXT NOT NULL DEFAULT 'working',
            created_at TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedder_id TEXT,
            embedding_dim INTEGER
        );
        """
    )
    conn.execute(
        """INSERT INTO semantic (
            subject, predicate, object, source, tier, created_at,
            embedding, embedder_id, embedding_dim
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "mark",
            "uses",
            "BeadsAdapter",
            "user",
            "working",
            "2024-01-15T12:00:00+00:00",
            np.ones(4, dtype=np.float32).tobytes(),
            "fake",
            4,
        ),
    )
    conn.commit()
    conn.close()

    store = SemanticStore(db, embedder=_Embedder())
    fact = store.get(1)
    assert fact.asserted_at is not None
    assert fact.asserted_at == fact.created_at
    # Validity window stays unbounded; legacy fact keeps participating
    # in retrieval without manual fixups.
    assert fact.valid_from is None
    assert fact.valid_to is None


# ---------- search with as_of ----------


def test_search_filters_out_not_yet_valid_facts(store: SemanticStore) -> None:
    """A fact with valid_from in the future should be filtered out
    when `as_of` is earlier than valid_from."""
    future = datetime.now(UTC) + timedelta(days=30)
    store.add(
        subject="mark",
        predicate="will_use",
        object="NewTool",
        source="user",
        valid_from=future,
    )
    hits = store.search("NewTool", k=5, mode="text")
    assert hits == []


def test_search_filters_out_expired_facts(store: SemanticStore) -> None:
    """A fact with valid_to in the past is stale; default search
    must drop it."""
    past = datetime.now(UTC) - timedelta(days=30)
    store.add(
        subject="mark",
        predicate="lived_in",
        object="Austin",
        source="user",
        valid_to=past,
    )
    hits = store.search("Austin", k=5, mode="text")
    assert hits == []


def test_search_includes_currently_valid_facts(store: SemanticStore) -> None:
    past = datetime.now(UTC) - timedelta(days=365)
    future = datetime.now(UTC) + timedelta(days=365)
    store.add(
        subject="mark",
        predicate="lives_in",
        object="SF",
        source="user",
        valid_from=past,
        valid_to=future,
    )
    hits = store.search("SF", k=5, mode="text")
    assert len(hits) == 1
    assert hits[0][0].object == "SF"


def test_as_of_shifts_the_temporal_lens(store: SemanticStore) -> None:
    """Classic use case: 'where did mark live in March 2024?' with
    a fact that was valid only 2020-2024."""
    vf = datetime(2020, 1, 1, tzinfo=UTC)
    vt = datetime(2024, 6, 1, tzinfo=UTC)
    store.add(
        subject="mark",
        predicate="lived_in",
        object="Austin",
        source="user",
        valid_from=vf,
        valid_to=vt,
    )
    # Default search (now) → not returned, window has closed.
    assert store.search("Austin", k=5, mode="text") == []
    # as_of inside the window → returned.
    march_2024 = datetime(2024, 3, 15, tzinfo=UTC)
    hits = store.search("Austin", k=5, mode="text", as_of=march_2024)
    assert len(hits) == 1


def test_null_valid_from_treated_as_unbounded_past(store: SemanticStore) -> None:
    vt = datetime(2024, 6, 1, tzinfo=UTC)
    store.add(
        subject="mark",
        predicate="worked_on",
        object="ProjectX",
        source="user",
        valid_from=None,
        valid_to=vt,
    )
    # as_of = 1990 should still find it — valid_from=NULL means always-was.
    old = datetime(1990, 1, 1, tzinfo=UTC)
    hits = store.search("ProjectX", k=5, mode="text", as_of=old)
    assert len(hits) == 1


def test_null_valid_to_treated_as_still_valid(store: SemanticStore) -> None:
    vf = datetime(2020, 1, 1, tzinfo=UTC)
    store.add(
        subject="mark",
        predicate="uses",
        object="Tailscale",
        source="user",
        valid_from=vf,
        valid_to=None,
    )
    hits = store.search("Tailscale", k=5, mode="text")
    assert len(hits) == 1


def test_hybrid_mode_also_respects_temporal_window(store: SemanticStore) -> None:
    """Default mode='hybrid' must filter identically to text/dense —
    otherwise the window is load-bearing only for direct-mode callers,
    which would be a footgun."""
    past = datetime.now(UTC) - timedelta(days=30)
    store.add(
        subject="mark",
        predicate="lived_in",
        object="OldPlace",
        source="user",
        valid_to=past,
    )
    store.add(
        subject="mark",
        predicate="lives_in",
        object="NewPlace",
        source="user",
    )
    hits = store.search("lives", k=5)  # default hybrid
    objs = {f.object for f, _ in hits}
    assert "NewPlace" in objs
    assert "OldPlace" not in objs


def test_dense_mode_respects_temporal_window(store: SemanticStore) -> None:
    past = datetime.now(UTC) - timedelta(days=30)
    store.add(
        subject="mark",
        predicate="lived_in",
        object="OldPlace",
        source="user",
        valid_to=past,
    )
    # mode='dense' skips FTS entirely — confirm the WHERE clause also
    # fires in the dense path.
    hits = store.search("OldPlace", k=5, mode="dense")
    assert hits == []
