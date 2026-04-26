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


@dataclass
class _LazyFakeEmbedder:
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
    """Regression for harness-m35: mirrors the episodic-store fix. A
    lazy embedder with dimension=0 until first embed() must still yield
    dense results on first search — _search_dense embeds the query
    before the SQL dim filter reads self.embedder.dimension."""
    warm = _FakeEmbedder()
    warm_store = SemanticStore(tmp_path / "harness.sqlite", embedder=warm)
    warm_store.add(subject="a", predicate="b", object="c", source="u")

    lazy = _LazyFakeEmbedder()
    assert lazy.dimension == 0
    fresh_store = SemanticStore(tmp_path / "harness.sqlite", embedder=lazy)
    hits = fresh_store.search("anything", k=1, mode="dense")
    assert len(hits) == 1, "dense search must surface rows on first call with a lazy embedder"
    assert lazy.dimension == 4


def test_count_matches_all_and_scopes_by_user(store: SemanticStore) -> None:
    """harness-qw4: count() is a cheap alternative to len(all()) for the
    introspect tool. Honours user_id scoping — a per-user count sees
    shared rows plus that user's private facts but not other users'."""
    store.add(subject="airton", predicate="knows", object="python", source="yaml", tier="seed")
    store.add(subject="mark", predicate="uses", object="macOS", source="user", user_id="mark")
    store.add(subject="alex", predicate="uses", object="linux", source="user", user_id="alex")

    assert store.count() == len(store.all()) == 3
    assert store.count(tier="seed") == 1
    assert store.count(subject="mark") == 1
    # Scoped count: shared (1) + mark's own (1) = 2.
    assert store.count(user_id="mark") == 2
    assert store.count(user_id="alex") == 2


def test_count_excludes_superseded_by_default(store: SemanticStore) -> None:
    first = store.add(subject="s", predicate="p", object="old", source="user")
    second = store.add(subject="s", predicate="p", object="new", source="user")
    store.mark_superseded(first.id, by=second.id)

    assert store.count() == 1
    assert store.count(include_superseded=True) == 2


def test_last_created_at_returns_newest_row(store: SemanticStore) -> None:
    assert store.last_created_at() is None
    first = store.add(subject="a", predicate="b", object="x", source="u")
    second = store.add(subject="c", predicate="d", object="y", source="u")
    last = store.last_created_at()
    assert last is not None
    assert last == max(first.created_at, second.created_at)


def test_supersedes_is_nullable_and_stored(store: SemanticStore) -> None:
    first = store.add(subject="s", predicate="p", object="old", source="u")
    second = store.add(subject="s", predicate="p", object="new", source="u", supersedes=first.id)

    assert second.supersedes == first.id
    refetched = store.get(first.id)
    assert refetched.supersedes is None


# ---------------------------------------------------------------------------
# Porter-stemming regression (harness-cpf)
# ---------------------------------------------------------------------------


def test_text_search_matches_stemmed_form(tmp_path: Path) -> None:
    """FTS5 porter stemming: querying 'declare' must match a row whose object
    contains 'declared'. Without porter tokenizer, unicode61 treats these as
    distinct tokens and BM25 misses the row."""
    store = SemanticStore(tmp_path / "harness.sqlite", embedder=_FakeEmbedder())
    store.add(
        subject="emergency",
        predicate="declared by",
        object="pilot, facility personnel, officials",
        source="yaml",
        tier="seed",
    )
    hits = store.search("declare", k=5, mode="text")
    assert len(hits) >= 1, "porter stemming must match 'declared' predicate on query 'declare'"
    assert hits[0][0].subject == "emergency"


def test_fts_migration_from_old_tokenizer(tmp_path: Path) -> None:
    """Migration regression (harness-cpf): a database created with the old
    unicode61 tokenizer must be transparently migrated to porter on the next
    open, and the stemmed search must work on the migrated data."""
    import sqlite3
    from datetime import UTC, datetime

    import numpy as np

    db = tmp_path / "harness.sqlite"

    # --- Phase 1: create manually with OLD unicode61 DDL -------------------
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS semantic (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            subject        TEXT    NOT NULL,
            predicate      TEXT    NOT NULL,
            object         TEXT    NOT NULL,
            confidence     REAL    NOT NULL DEFAULT 0.8,
            source         TEXT    NOT NULL,
            attributed_to  TEXT,
            session_id     TEXT,
            user_id        TEXT,
            supersedes     INTEGER REFERENCES semantic(id),
            superseded_by  INTEGER REFERENCES semantic(id),
            tier           TEXT    NOT NULL DEFAULT 'working',
            created_at     TEXT    NOT NULL,
            embedding      BLOB    NOT NULL,
            embedder_id    TEXT,
            embedding_dim  INTEGER,
            valid_from     TEXT,
            valid_to       TEXT,
            asserted_at    TEXT
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
            subject, predicate, object,
            content='semantic',
            content_rowid='id',
            tokenize='unicode61'
        );
        CREATE TRIGGER IF NOT EXISTS semantic_fts_ai
        AFTER INSERT ON semantic BEGIN
            INSERT INTO semantic_fts(rowid, subject, predicate, object)
            VALUES (new.id, new.subject, new.predicate, new.object);
        END;
    """)
    now = datetime.now(UTC).isoformat()
    vec = np.zeros(4, dtype=np.float32)
    conn.execute(
        """INSERT INTO semantic
               (subject, predicate, object, confidence, source, attributed_to,
                session_id, user_id, supersedes, superseded_by, tier, created_at,
                embedding, embedder_id, embedding_dim, valid_from, valid_to, asserted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "emergency",
            "declared by",
            "pilot or officials",
            0.9,
            "yaml",
            None,
            None,
            None,
            None,
            None,
            "seed",
            now,
            vec.tobytes(),
            "fake",
            4,
            None,
            None,
            now,
        ),
    )
    conn.close()

    # --- Phase 2: open via SemanticStore (triggers migration) ---------------
    store = SemanticStore(db, embedder=_FakeEmbedder())
    hits = store.search("declare", k=5, mode="text")
    assert len(hits) >= 1, "after porter migration, query 'declare' must match stored 'declared by'"
    assert hits[0][0].subject == "emergency"


# ---------------------------------------------------------------------------
# delete_working_for_session (harness-k7m9)
# ---------------------------------------------------------------------------


def test_delete_working_for_session_targets_only_matching_working(
    tmp_path: Path,
) -> None:
    """`delete_working_for_session` removes only tier=working rows
    tagged with the given session_id. Seeds, consolidated rows, and
    other sessions' rows are untouched. Used by `harness session
    reset` to prune scribed facts (harness-k7m9)."""
    store = SemanticStore(tmp_path / "harness.sqlite", embedder=_FakeEmbedder())
    try:
        store.add(
            subject="airton",
            predicate="is",
            object="program",
            confidence=1.0,
            source="yaml",
            tier="seed",
        )
        store.add(
            subject="mark",
            predicate="prefers",
            object="caveman",
            confidence=0.8,
            source="scribe",
            tier="working",
            session_id="A",
        )
        store.add(
            subject="mark",
            predicate="works_on",
            object="harness",
            confidence=0.8,
            source="scribe",
            tier="working",
            session_id="A",
        )
        store.add(
            subject="ada",
            predicate="prefers",
            object="dvorak",
            confidence=0.7,
            source="scribe",
            tier="working",
            session_id="B",
        )
        deleted = store.delete_working_for_session("A")
        assert deleted == 2

        survivors = {(f.subject, f.predicate, f.object) for f in store.all()}
        assert ("airton", "is", "program") in survivors
        assert ("ada", "prefers", "dvorak") in survivors
        assert ("mark", "prefers", "caveman") not in survivors
    finally:
        store.close()


def test_delete_working_for_session_purges_semantic_fts_sidecar(
    tmp_path: Path,
) -> None:
    """semantic_fts_ad trigger (harness-k7m9) must drop deleted rows
    from the FTS sidecar so BM25 doesn't return rowids that no
    longer exist in the main table."""
    store = SemanticStore(tmp_path / "harness.sqlite", embedder=_FakeEmbedder())
    try:
        store.add(
            subject="orbit",
            predicate="contains",
            object="capacitor",
            confidence=0.8,
            source="scribe",
            tier="working",
            session_id="X",
        )
        before = store.search("capacitor", k=5, mode="text")
        assert before, "fixture: row should match before delete"

        store.delete_working_for_session("X")

        after = store.search("capacitor", k=5, mode="text")
        assert after == [], "semantic_fts sidecar must drop the deleted row's rowid"
    finally:
        store.close()
