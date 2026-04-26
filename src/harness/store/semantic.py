from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from harness.store._hybrid import (
    reciprocal_rank_fusion,
    sanitize_fts_query,
    session_scope_filter,
)

if TYPE_CHECKING:
    from harness.retrieval.embed import Embedder


SearchMode = Literal["hybrid", "dense", "text"]

_CREATE_TABLE = """
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
    embedding_dim  INTEGER
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS semantic_subject_idx       ON semantic (subject);
CREATE INDEX IF NOT EXISTS semantic_tier_idx          ON semantic (tier);
CREATE INDEX IF NOT EXISTS semantic_superseded_by_idx ON semantic (superseded_by);
-- Speeds up `--memory-scope=current-session` retrieval and
-- `delete_working_for_session` (harness-w3mo / harness-k7m9).
CREATE INDEX IF NOT EXISTS semantic_session_id_idx    ON semantic (session_id);
"""

# FTS5 sidecar — same design as episodic_fts. Indexes the triple
# (subject, predicate, object) so queries on identifier-heavy data
# ("BeadsAdapter.get_focus", "mark@ucollect.com") don't lose to dense
# cosine's semantic smoothing.
#
# Porter stemming (harness-cpf): same rationale as episodic_fts —
# tense/morphology mismatches between stored triples and natural-language
# queries tank BM25 recall and drag down RRF-fused hybrid ranks.
_FTS_TOKENIZE = "porter unicode61 remove_diacritics 1"

_CREATE_FTS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
    subject, predicate, object,
    content='semantic',
    content_rowid='id',
    tokenize='{_FTS_TOKENIZE}'
);

CREATE TRIGGER IF NOT EXISTS semantic_fts_ai
AFTER INSERT ON semantic BEGIN
    INSERT INTO semantic_fts(rowid, subject, predicate, object)
    VALUES (new.id, new.subject, new.predicate, new.object);
END;

-- Mirror deletes into the external-content FTS sidecar so
-- `delete_working_for_session` (harness-k7m9) doesn't leave dangling
-- BM25 entries pointing at vanished rowids. Idempotent.
CREATE TRIGGER IF NOT EXISTS semantic_fts_ad
AFTER DELETE ON semantic BEGIN
    INSERT INTO semantic_fts(semantic_fts, rowid, subject, predicate, object)
    VALUES ('delete', old.id, old.subject, old.predicate, old.object);
END;
"""  # noqa: S608 — module-level constant; _FTS_TOKENIZE is never user-supplied


@dataclass(frozen=True)
class SemanticFact:
    id: int
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    attributed_to: str | None
    session_id: str | None
    user_id: str | None
    supersedes: int | None
    tier: str
    created_at: datetime
    superseded_by: int | None = None
    # Temporal validity window (sota punch #4, harness-kr2). None means
    # unbounded on that side:
    #   valid_from=None — the fact has always been true (or the
    #     start isn't known / doesn't matter).
    #   valid_to=None — still true as of last assertion.
    # asserted_at is when the user / scribe communicated the fact to
    # the agent; separate from created_at so backfilled facts ("mark
    # moved last March") can carry accurate provenance.
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    asserted_at: datetime | None = None


class SemanticStore:
    """Atomic-fact memory. Each row is a (subject, predicate, object)
    triple with provenance. Same SQLite-plus-BLOB-embedding pattern as
    the episodic store; same 'graduate to LanceDB past 10k rows' plan."""

    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # See transcript.py for the `check_same_thread=False` rationale.
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_CREATE_TABLE)
        # Schema migrations.
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(semantic)")}
        if "superseded_by" not in cols:
            self._conn.execute(
                "ALTER TABLE semantic ADD COLUMN superseded_by INTEGER REFERENCES semantic(id)"
            )
        if "embedder_id" not in cols:
            self._conn.execute("ALTER TABLE semantic ADD COLUMN embedder_id TEXT")
        if "embedding_dim" not in cols:
            self._conn.execute("ALTER TABLE semantic ADD COLUMN embedding_dim INTEGER")
        # Temporal fields (sota punch #4). NULL on legacy rows = the
        # window is unbounded on that side; asserted_at backfills to
        # created_at so provenance is preserved without losing the
        # "when the agent heard about this" signal.
        if "valid_from" not in cols:
            self._conn.execute("ALTER TABLE semantic ADD COLUMN valid_from TEXT")
        if "valid_to" not in cols:
            self._conn.execute("ALTER TABLE semantic ADD COLUMN valid_to TEXT")
        if "asserted_at" not in cols:
            self._conn.execute("ALTER TABLE semantic ADD COLUMN asserted_at TEXT")
            self._conn.execute(
                "UPDATE semantic SET asserted_at = created_at WHERE asserted_at IS NULL"
            )
        self._conn.execute(
            "UPDATE semantic SET embedding_dim = LENGTH(embedding) / 4 WHERE embedding_dim IS NULL"
        )
        self._conn.execute("UPDATE semantic SET embedder_id = 'legacy' WHERE embedder_id IS NULL")
        self._conn.executescript(_CREATE_INDEXES)
        # FTS5 sidecar — rebuild on first create so pre-existing rows
        # are indexed. See episodic.py for the schema-existence
        # rationale (COUNT(*) on external-content FTS mirrors the
        # main table count, so a row-count check can't see the empty-
        # index case).
        #
        # Porter-stemming migration (harness-cpf): same pattern as
        # episodic.py — probe sqlite_master DDL for "porter"; if absent
        # on an existing table, drop + recreate + rebuild.
        fts_row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='semantic_fts'"
        ).fetchone()
        fts_existed_before = fts_row is not None
        needs_tokenizer_migration = fts_existed_before and (
            fts_row[0] is None or "porter" not in fts_row[0].lower()
        )
        if needs_tokenizer_migration:
            self._conn.executescript(
                "DROP TABLE IF EXISTS semantic_fts;DROP TRIGGER IF EXISTS semantic_fts_ai;"
            )
            fts_existed_before = False
        self._conn.executescript(_CREATE_FTS)
        if not fts_existed_before:
            self._conn.execute("INSERT INTO semantic_fts(semantic_fts) VALUES('rebuild')")

    def add(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.8,
        source: str,
        attributed_to: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        supersedes: int | None = None,
        tier: str = "working",
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        asserted_at: datetime | None = None,
    ) -> SemanticFact:
        """Insert a fact. Always appends — deduplication and supersession
        are the consolidator's job, not this method's.

        Temporal fields (sota punch #4): `valid_from` / `valid_to`
        define the window during which the fact was / is true. Both
        default to None = unbounded. `asserted_at` is when the user
        communicated the fact to the agent (defaults to now); this
        is separate from `created_at` so backfilled facts can keep
        accurate provenance."""
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        asserted_iso = (asserted_at or now_dt).isoformat()
        valid_from_iso = valid_from.isoformat() if valid_from is not None else None
        valid_to_iso = valid_to.isoformat() if valid_to is not None else None
        # Embed "subject predicate object" so natural-language search hits
        # all three axes. Tuple-stringification keeps it simple.
        text = f"{subject} {predicate} {object}"
        vec = self.embedder.embed([text])[0].astype(np.float32)
        cur = self._conn.execute(
            """INSERT INTO semantic (
                subject, predicate, object, confidence, source,
                attributed_to, session_id, user_id, supersedes, tier,
                created_at, embedding, embedder_id, embedding_dim,
                valid_from, valid_to, asserted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                subject,
                predicate,
                object,
                confidence,
                source,
                attributed_to,
                session_id,
                user_id,
                supersedes,
                tier,
                now,
                vec.tobytes(),
                self.embedder.id,
                self.embedder.dimension,
                valid_from_iso,
                valid_to_iso,
                asserted_iso,
            ),
        )
        return self.get(cur.lastrowid or 0)

    def get(self, record_id: int) -> SemanticFact:
        row = self._conn.execute(
            """SELECT id, subject, predicate, object, confidence, source,
                      attributed_to, session_id, user_id, supersedes, tier,
                      created_at, superseded_by,
                      valid_from, valid_to, asserted_at
               FROM semantic WHERE id = ?""",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"semantic fact {record_id} not found")
        return _row_to_fact(row)

    def all(
        self,
        *,
        tier: str | None = None,
        subject: str | None = None,
        include_superseded: bool = False,
    ) -> list[SemanticFact]:
        query = """SELECT id, subject, predicate, object, confidence, source,
                      attributed_to, session_id, user_id, supersedes, tier,
                      created_at, superseded_by,
                      valid_from, valid_to, asserted_at FROM semantic"""
        conditions: list[str] = []
        params: list[object] = []
        if tier is not None:
            conditions.append("tier = ?")
            params.append(tier)
        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)
        if not include_superseded:
            conditions.append("superseded_by IS NULL")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_fact(r) for r in rows]

    def mark_superseded(self, record_id: int, *, by: int) -> None:
        self._conn.execute(
            "UPDATE semantic SET superseded_by = ? WHERE id = ?",
            (by, record_id),
        )

    def delete_working_for_session(self, session_id: str) -> int:
        """Hard-delete every tier=working row tagged with `session_id`.

        Mirror of EpisodicStore.delete_working_for_session — used by
        `harness session reset` to prune the scribed facts for one
        session without disturbing shared seeds, consolidated facts,
        or other sessions' working-tier writes (harness-k7m9)."""
        cur = self._conn.execute(
            "DELETE FROM semantic WHERE tier = 'working' AND session_id = ?",
            (session_id,),
        )
        return cur.rowcount or 0

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        min_confidence: float = 0.0,
        min_score: float = 0.0,
        user_id: str | None = None,
        mode: SearchMode = "hybrid",
        as_of: datetime | None = None,
        bm25_min_score: float | None = None,
        allowed_sessions: tuple[str, ...] | None = None,
    ) -> list[tuple[SemanticFact, float]]:
        """Return up to `k` active facts (not superseded) where stored
        confidence >= `min_confidence`, ranked by `mode` and filtered
        to the time window containing `as_of`.

        `mode='hybrid'` (default) — fuse dense cosine + FTS5 BM25 via
        RRF (k=60). `min_score` still filters the dense component
        pre-fusion; the returned score is the RRF score.

        `mode='dense'` — legacy pure-cosine path. `min_score` is the
        cosine threshold.

        `mode='text'` — BM25 only. `min_confidence` still applies.

        `as_of` (sota punch #4) selects which time the retrieval is
        asking about: facts whose `valid_from > as_of` (future-only)
        or `valid_to <= as_of` (expired) are filtered out. Default
        None means now. NULL on either column is treated as unbounded
        on that side.

        Confidence gates by trustworthiness; the score gate gates by
        relevance. Both matter — a high-confidence fact about an
        unrelated topic still pollutes the prompt.

        `user_id` scopes to relationship memory across all modes:
        when given, returns rows where `user_id IS NULL` OR
        `user_id = <this user>`."""
        as_of_iso = (as_of or datetime.now(UTC)).isoformat()
        if mode == "dense":
            return self._search_dense(
                query,
                k=k,
                min_confidence=min_confidence,
                min_score=min_score,
                user_id=user_id,
                as_of_iso=as_of_iso,
                allowed_sessions=allowed_sessions,
            )
        if mode == "text":
            return self._search_text(
                query,
                k=k,
                min_confidence=min_confidence,
                user_id=user_id,
                as_of_iso=as_of_iso,
                allowed_sessions=allowed_sessions,
            )
        bm25_floor = bm25_min_score if bm25_min_score is not None else max(min_score - 0.15, 0.0)
        return self._search_hybrid(
            query,
            k=k,
            min_confidence=min_confidence,
            min_score=min_score,
            bm25_min_score=bm25_floor,
            user_id=user_id,
            as_of_iso=as_of_iso,
            allowed_sessions=allowed_sessions,
        )

    def _search_dense(
        self,
        query: str,
        *,
        k: int,
        min_confidence: float,
        min_score: float,
        user_id: str | None,
        as_of_iso: str,
        allowed_sessions: tuple[str, ...] | None = None,
    ) -> list[tuple[SemanticFact, float]]:
        # Embed the query FIRST so a lazy embedder (dimension=0 until
        # first embed() call) populates its real dimension before the
        # SQL filter reads it (harness-m35). Mirrors the episodic-store
        # fix.
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        session_clause, session_params = session_scope_filter(allowed_sessions)
        if user_id is None:
            rows = self._conn.execute(
                f"""SELECT id, subject, predicate, object, confidence, source,
                          attributed_to, session_id, user_id, supersedes, tier,
                          created_at, superseded_by,
                          valid_from, valid_to, asserted_at, embedding
                   FROM semantic
                   WHERE confidence >= ? AND superseded_by IS NULL
                     AND embedding_dim = ?
                     AND (valid_from IS NULL OR valid_from <= ?)
                     AND (valid_to IS NULL OR valid_to > ?){session_clause}""",  # noqa: S608 — session_clause is a static fragment with bind placeholders
                (
                    min_confidence,
                    self.embedder.dimension,
                    as_of_iso,
                    as_of_iso,
                    *session_params,
                ),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""SELECT id, subject, predicate, object, confidence, source,
                          attributed_to, session_id, user_id, supersedes, tier,
                          created_at, superseded_by,
                          valid_from, valid_to, asserted_at, embedding
                   FROM semantic
                   WHERE confidence >= ? AND superseded_by IS NULL
                     AND embedding_dim = ?
                     AND (user_id IS NULL OR user_id = ?)
                     AND (valid_from IS NULL OR valid_from <= ?)
                     AND (valid_to IS NULL OR valid_to > ?){session_clause}""",  # noqa: S608 — session_clause is a static fragment with bind placeholders
                (
                    min_confidence,
                    self.embedder.dimension,
                    user_id,
                    as_of_iso,
                    as_of_iso,
                    *session_params,
                ),
            ).fetchall()
        if not rows:
            return []
        scored: list[tuple[SemanticFact, float]] = []
        for row in rows:
            # Embedding is at index 16 now that temporal fields precede it.
            vec = np.frombuffer(row[16], dtype=np.float32)
            sim = float(np.dot(q_vec, vec))
            if sim >= min_score:
                scored.append((_row_to_fact(row[:16]), sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def _search_text(
        self,
        query: str,
        *,
        k: int,
        min_confidence: float,
        user_id: str | None,
        as_of_iso: str,
        allowed_sessions: tuple[str, ...] | None = None,
    ) -> list[tuple[SemanticFact, float]]:
        match = sanitize_fts_query(query)
        if not match:
            return []
        session_clause, session_params = session_scope_filter(
            allowed_sessions, column="s.session_id"
        )
        if user_id is None:
            rows = self._conn.execute(
                f"""SELECT s.id, s.subject, s.predicate, s.object, s.confidence,
                          s.source, s.attributed_to, s.session_id, s.user_id,
                          s.supersedes, s.tier, s.created_at, s.superseded_by,
                          s.valid_from, s.valid_to, s.asserted_at,
                          bm25(semantic_fts) AS bm25_score
                   FROM semantic_fts
                   JOIN semantic s ON s.id = semantic_fts.rowid
                   WHERE semantic_fts MATCH ?
                     AND s.superseded_by IS NULL
                     AND s.confidence >= ?
                     AND (s.valid_from IS NULL OR s.valid_from <= ?)
                     AND (s.valid_to IS NULL OR s.valid_to > ?){session_clause}
                   ORDER BY bm25_score
                   LIMIT ?""",  # noqa: S608 — session_clause is a static fragment with bind placeholders
                (match, min_confidence, as_of_iso, as_of_iso, *session_params, k),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""SELECT s.id, s.subject, s.predicate, s.object, s.confidence,
                          s.source, s.attributed_to, s.session_id, s.user_id,
                          s.supersedes, s.tier, s.created_at, s.superseded_by,
                          s.valid_from, s.valid_to, s.asserted_at,
                          bm25(semantic_fts) AS bm25_score
                   FROM semantic_fts
                   JOIN semantic s ON s.id = semantic_fts.rowid
                   WHERE semantic_fts MATCH ?
                     AND s.superseded_by IS NULL
                     AND s.confidence >= ?
                     AND (s.user_id IS NULL OR s.user_id = ?)
                     AND (s.valid_from IS NULL OR s.valid_from <= ?)
                     AND (s.valid_to IS NULL OR s.valid_to > ?){session_clause}
                   ORDER BY bm25_score
                   LIMIT ?""",  # noqa: S608 — session_clause is a static fragment with bind placeholders
                (
                    match,
                    min_confidence,
                    user_id,
                    as_of_iso,
                    as_of_iso,
                    *session_params,
                    k,
                ),
            ).fetchall()
        return [(_row_to_fact(row[:16]), -float(row[16])) for row in rows]

    def _search_hybrid(
        self,
        query: str,
        *,
        k: int,
        min_confidence: float,
        min_score: float,
        bm25_min_score: float,
        user_id: str | None,
        as_of_iso: str,
        allowed_sessions: tuple[str, ...] | None = None,
    ) -> list[tuple[SemanticFact, float]]:
        candidate_k = max(k * 4, 20)
        dense_hits = self._search_dense(
            query,
            k=candidate_k,
            min_confidence=min_confidence,
            min_score=min_score,
            user_id=user_id,
            as_of_iso=as_of_iso,
            allowed_sessions=allowed_sessions,
        )
        text_hits = self._search_text(
            query,
            k=candidate_k,
            min_confidence=min_confidence,
            user_id=user_id,
            as_of_iso=as_of_iso,
            allowed_sessions=allowed_sessions,
        )
        if bm25_min_score > 0.0 and text_hits:
            text_hits = self._gate_text_hits_by_cosine(
                query=query, hits=text_hits, threshold=bm25_min_score
            )
        if not dense_hits and not text_hits:
            return []
        record_map: dict[int, SemanticFact] = {}
        for fact, _ in dense_hits:
            record_map[fact.id] = fact
        for fact, _ in text_hits:
            record_map.setdefault(fact.id, fact)
        fused = reciprocal_rank_fusion(
            [[fact.id for fact, _ in dense_hits], [fact.id for fact, _ in text_hits]]
        )
        return [(record_map[rid], score) for rid, score in fused[:k] if rid in record_map]

    def _gate_text_hits_by_cosine(
        self,
        *,
        query: str,
        hits: list[tuple[SemanticFact, float]],
        threshold: float,
    ) -> list[tuple[SemanticFact, float]]:
        """Drop BM25 hits whose dense cosine to the query is below
        `threshold` so a lexical match on a common token doesn't
        rank a semantically unrelated fact into hybrid top-K
        (harness-dffh, mirror of EpisodicStore._gate_text_hits_by_cosine)."""
        if not hits:
            return []
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        ids = [fact.id for fact, _ in hits]
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, embedding, embedding_dim FROM semantic WHERE id IN ({placeholders})",  # noqa: S608 — placeholders param-bind, not user data
            ids,
        ).fetchall()
        embed_map = {row[0]: (row[1], int(row[2])) for row in rows}
        out: list[tuple[SemanticFact, float]] = []
        for fact, bm in hits:
            payload = embed_map.get(fact.id)
            if payload is None:
                continue
            blob, dim = payload
            if dim != self.embedder.dimension:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            cos = float(np.dot(q_vec, vec))
            if cos >= threshold:
                out.append((fact, bm))
        return out

    def fetch_embedding(self, record_id: int) -> np.ndarray:
        row = self._conn.execute(
            "SELECT embedding FROM semantic WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"semantic fact {record_id} not found")
        return np.frombuffer(row[0], dtype=np.float32)

    def count(
        self,
        *,
        tier: str | None = None,
        subject: str | None = None,
        user_id: str | None = None,
        include_superseded: bool = False,
    ) -> int:
        """Row count scoped the same way `search()` is: when `user_id` is
        given, counts shared rows plus that user's private rows. Used by
        the introspect tool to report fact-store size without pulling
        every record via `all()`."""
        conditions: list[str] = []
        params: list[object] = []
        if tier is not None:
            conditions.append("tier = ?")
            params.append(tier)
        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)
        if user_id is not None:
            conditions.append("(user_id IS NULL OR user_id = ?)")
            params.append(user_id)
        if not include_superseded:
            conditions.append("superseded_by IS NULL")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM semantic{where}",  # noqa: S608
            params,
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def last_created_at(
        self,
        *,
        tier: str | None = None,
        user_id: str | None = None,
        source: str | None = None,
    ) -> datetime | None:
        """Most recent `created_at` among active facts matching the same
        scope as `count()`. `source` narrows further (e.g. 'scribe',
        'consolidator'). Returns None when the filter matches no rows."""
        conditions: list[str] = ["superseded_by IS NULL"]
        params: list[object] = []
        if tier is not None:
            conditions.append("tier = ?")
            params.append(tier)
        if user_id is not None:
            conditions.append("(user_id IS NULL OR user_id = ?)")
            params.append(user_id)
        if source is not None:
            conditions.append("source = ?")
            params.append(source)
        where = " WHERE " + " AND ".join(conditions)
        row = self._conn.execute(
            f"SELECT MAX(created_at) FROM semantic{where}",  # noqa: S608
            params,
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return datetime.fromisoformat(row[0])

    def count_mismatched_embeddings(self) -> int:
        row = self._conn.execute(
            """SELECT COUNT(*) FROM semantic
               WHERE superseded_by IS NULL AND embedding_dim != ?""",
            (self.embedder.dimension,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def rebuild_embeddings(self) -> tuple[int, int]:
        """Re-embed every active fact with the current embedder."""
        cur = self._conn.execute(
            """SELECT id, subject, predicate, object FROM semantic
               WHERE superseded_by IS NULL ORDER BY id"""
        )
        rows = cur.fetchall()
        if not rows:
            return 0, 0
        texts = [f"{r[1]} {r[2]} {r[3]}" for r in rows]
        vectors = self.embedder.embed(texts)
        updated = 0
        for (record_id, *_), vec in zip(rows, vectors, strict=True):
            self._conn.execute(
                """UPDATE semantic
                      SET embedding = ?, embedder_id = ?, embedding_dim = ?
                    WHERE id = ?""",
                (
                    vec.astype(np.float32).tobytes(),
                    self.embedder.id,
                    self.embedder.dimension,
                    record_id,
                ),
            )
            updated += 1
        return updated, 0

    def close(self) -> None:
        self._conn.close()


def _row_to_fact(row: Iterable[Any]) -> SemanticFact:
    r = list(row)
    return SemanticFact(
        id=int(r[0]),
        subject=str(r[1]),
        predicate=str(r[2]),
        object=str(r[3]),
        confidence=float(r[4]),
        source=str(r[5]),
        attributed_to=str(r[6]) if r[6] is not None else None,
        session_id=str(r[7]) if r[7] is not None else None,
        user_id=str(r[8]) if r[8] is not None else None,
        supersedes=int(r[9]) if r[9] is not None else None,
        tier=str(r[10]),
        created_at=datetime.fromisoformat(str(r[11])),
        superseded_by=int(r[12]) if len(r) > 12 and r[12] is not None else None,
        valid_from=(
            datetime.fromisoformat(str(r[13])) if len(r) > 13 and r[13] is not None else None
        ),
        valid_to=(
            datetime.fromisoformat(str(r[14])) if len(r) > 14 and r[14] is not None else None
        ),
        asserted_at=(
            datetime.fromisoformat(str(r[15])) if len(r) > 15 and r[15] is not None else None
        ),
    )
