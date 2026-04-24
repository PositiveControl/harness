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


@dataclass
class _LazyFakeEmbedder:
    """Mirrors the real SentenceTransformersEmbedder's lazy-init: starts
    with dimension=0 and populates it on the first embed() call.
    Without the harness-m35 fix, _search_dense reads the zero dimension
    before calling embed(), and the SQL filter matches nothing."""

    id: str = "fake-lazy"
    dimension: int = 0

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        self.dimension = 4
        vectors: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            vectors.append(v / norm if norm > 0 else v)
        return np.stack(vectors)


def test_search_dense_warms_embedder_before_dim_filter(tmp_path: Path) -> None:
    """Regression for harness-m35: a lazy embedder that only knows its
    dimension after the first embed() call must still yield results on
    the very first search — _search_dense embeds the query first so the
    dimension is populated before the SQL filter reads it."""
    # Seed with a warmed embedder so rows land at dimension=4.
    warm = _FakeEmbedder()
    warm_store = EpisodicStore(tmp_path / "harness.sqlite", embedder=warm)
    warm_store.ingest(external_id="a", title="alpha", body="A", tier="seed", source="yaml")

    # Fresh store instance with a lazy embedder — dimension starts at 0.
    lazy = _LazyFakeEmbedder()
    assert lazy.dimension == 0
    fresh_store = EpisodicStore(tmp_path / "harness.sqlite", embedder=lazy)
    hits = fresh_store.search("alpha", k=1, mode="dense")
    assert len(hits) == 1, "dense search must surface the row on first call with a lazy embedder"
    assert lazy.dimension == 4, "embed() must have been called and populated dimension"


def test_all_filters_by_tier(store: EpisodicStore) -> None:
    store.ingest(external_id="seed1", title="s1", body="", tier="seed", source="yaml")
    store.ingest(external_id="work1", title="w1", body="", tier="working", source="user")

    assert {r.tier for r in store.all()} == {"seed", "working"}
    seeds_only = store.all(tier="seed")
    assert len(seeds_only) == 1
    assert seeds_only[0].tier == "seed"


def test_count_matches_all_and_scopes_by_user(store: EpisodicStore) -> None:
    """harness-qw4: count() is a cheap alternative to len(all()) for the
    introspect tool. Honours user_id scoping the same way search does —
    a per-user count sees shared rows plus that user's private rows but
    not other users'."""
    store.ingest(external_id="shared", title="s", body="", tier="seed", source="yaml")
    store.ingest(
        external_id="mine", title="m", body="", tier="working", source="user", user_id="mark"
    )
    store.ingest(
        external_id="yours", title="y", body="", tier="working", source="user", user_id="alex"
    )

    assert store.count() == len(store.all()) == 3
    assert store.count(tier="seed") == 1
    # Scoped count: shared (1) + mark's own (1) = 2, excludes alex's row.
    assert store.count(user_id="mark") == 2
    assert store.count(user_id="alex") == 2  # shared + alex's


def test_count_excludes_superseded_by_default(store: EpisodicStore) -> None:
    first = store.ingest(external_id="a", title="a", body="", tier="working", source="user")
    second = store.ingest(external_id="b", title="b", body="", tier="consolidated", source="user")
    store.mark_superseded(first.id, by=second.id)

    assert store.count() == 1
    assert store.count(include_superseded=True) == 2


def test_last_created_at_returns_newest_row(store: EpisodicStore) -> None:
    """Newest row's created_at wins. None on an empty filter."""
    assert store.last_created_at() is None
    first = store.ingest(external_id="first", title="a", body="", tier="working", source="user")
    second = store.ingest(external_id="second", title="b", body="", tier="working", source="user")
    last = store.last_created_at()
    assert last is not None
    assert last == max(first.created_at, second.created_at)


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


# ---------------------------------------------------------------------------
# Porter-stemming regression (harness-cpf)
# ---------------------------------------------------------------------------


def test_text_search_matches_stemmed_form(tmp_path: Path) -> None:
    """FTS5 porter stemming: querying 'declare' must match a row whose body
    contains 'declared'. Without porter tokenizer, unicode61 treats these as
    distinct tokens and the row is missed by BM25."""
    store = EpisodicStore(tmp_path / "harness.sqlite", embedder=_FakeEmbedder())
    store.ingest(
        external_id="stem-1",
        title="Emergency Situations",
        body=(
            "An emergency is declared by any of the following:"
            " pilot, facility personnel, officials."
        ),
        tier="seed",
        source="yaml",
    )
    hits = store.search("declare", k=5, mode="text")
    assert len(hits) >= 1, "porter stemming must match 'declared' body on query 'declare'"
    assert hits[0][0].external_id == "stem-1"


def test_fts_migration_from_old_tokenizer(tmp_path: Path) -> None:
    """Migration regression (harness-cpf): a database created with the old
    unicode61 FTS tokenizer must be transparently migrated to porter on the
    next open, and the stemmed search must work on the migrated data."""
    import sqlite3

    db = tmp_path / "harness.sqlite"

    # --- Phase 1: create a store manually with the OLD tokenizer DDL -------
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodic (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id    TEXT    UNIQUE,
            title          TEXT    NOT NULL,
            body           TEXT    NOT NULL,
            principle      TEXT,
            tags           TEXT    NOT NULL DEFAULT '[]',
            tier           TEXT    NOT NULL,
            source         TEXT    NOT NULL,
            session_id     TEXT,
            user_id        TEXT,
            created_at     TEXT    NOT NULL,
            last_accessed  TEXT,
            embedding      BLOB    NOT NULL,
            embedder_id    TEXT,
            embedding_dim  INTEGER,
            superseded_by  INTEGER REFERENCES episodic(id)
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
            title, body, principle,
            content='episodic',
            content_rowid='id',
            tokenize='unicode61'
        );
        CREATE TRIGGER IF NOT EXISTS episodic_fts_ai
        AFTER INSERT ON episodic BEGIN
            INSERT INTO episodic_fts(rowid, title, body, principle)
            VALUES (new.id, new.title, new.body, COALESCE(new.principle, ''));
        END;
    """)
    # Insert a row directly, bypassing EpisodicStore so the FTS index is
    # populated with the old unicode61 tokenizer.
    import json
    from datetime import UTC, datetime

    import numpy as np

    now = datetime.now(UTC).isoformat()
    vec = np.zeros(4, dtype=np.float32)
    conn.execute(
        """INSERT INTO episodic
               (external_id, title, body, principle, tags, tier, source,
                session_id, user_id, created_at, embedding, embedder_id, embedding_dim)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "old-stem-1",
            "Emergency Situations",
            "An emergency is declared by pilot or officials.",
            None,
            json.dumps([]),
            "seed",
            "yaml",
            None,
            None,
            now,
            vec.tobytes(),
            "fake",
            4,
        ),
    )
    conn.close()

    # --- Phase 2: open via EpisodicStore (triggers migration) ---------------
    store = EpisodicStore(db, embedder=_FakeEmbedder())
    hits = store.search("declare", k=5, mode="text")
    assert len(hits) >= 1, "after porter migration, query 'declare' must match stored 'declared'"
    assert hits[0][0].external_id == "old-stem-1"
