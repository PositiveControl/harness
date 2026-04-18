from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from harness.retrieval.embed import Embedder

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
"""


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
        self._conn.execute(
            "UPDATE semantic SET embedding_dim = LENGTH(embedding) / 4 WHERE embedding_dim IS NULL"
        )
        self._conn.execute("UPDATE semantic SET embedder_id = 'legacy' WHERE embedder_id IS NULL")
        self._conn.executescript(_CREATE_INDEXES)

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
    ) -> SemanticFact:
        """Insert a fact. Always appends — deduplication and supersession
        are the consolidator's job, not this method's."""
        now = datetime.now(UTC).isoformat()
        # Embed "subject predicate object" so natural-language search hits
        # all three axes. Tuple-stringification keeps it simple.
        text = f"{subject} {predicate} {object}"
        vec = self.embedder.embed([text])[0].astype(np.float32)
        cur = self._conn.execute(
            """INSERT INTO semantic (
                subject, predicate, object, confidence, source,
                attributed_to, session_id, user_id, supersedes, tier,
                created_at, embedding, embedder_id, embedding_dim
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        return self.get(cur.lastrowid or 0)

    def get(self, record_id: int) -> SemanticFact:
        row = self._conn.execute(
            """SELECT id, subject, predicate, object, confidence, source,
                      attributed_to, session_id, user_id, supersedes, tier,
                      created_at, superseded_by
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
                      created_at, superseded_by FROM semantic"""
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

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        min_confidence: float = 0.0,
        min_score: float = 0.0,
        user_id: str | None = None,
    ) -> list[tuple[SemanticFact, float]]:
        """Return up to `k` active facts (not superseded) where stored
        confidence >= `min_confidence` AND retrieval cosine similarity
        >= `min_score`. Confidence gates by trustworthiness; the score
        gate gates by relevance. Both matter — a high-confidence fact
        about an unrelated topic still pollutes the prompt.

        `user_id` scopes to relationship memory: when given, returns
        rows where `user_id IS NULL` OR `user_id = <this user>`.
        Other users' private facts are never returned."""
        if user_id is None:
            rows = self._conn.execute(
                """SELECT id, subject, predicate, object, confidence, source,
                          attributed_to, session_id, user_id, supersedes, tier,
                          created_at, superseded_by, embedding
                   FROM semantic
                   WHERE confidence >= ? AND superseded_by IS NULL
                     AND embedding_dim = ?""",
                (min_confidence, self.embedder.dimension),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT id, subject, predicate, object, confidence, source,
                          attributed_to, session_id, user_id, supersedes, tier,
                          created_at, superseded_by, embedding
                   FROM semantic
                   WHERE confidence >= ? AND superseded_by IS NULL
                     AND embedding_dim = ?
                     AND (user_id IS NULL OR user_id = ?)""",
                (min_confidence, self.embedder.dimension, user_id),
            ).fetchall()
        if not rows:
            return []
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        scored: list[tuple[SemanticFact, float]] = []
        for row in rows:
            vec = np.frombuffer(row[13], dtype=np.float32)
            sim = float(np.dot(q_vec, vec))
            if sim >= min_score:
                scored.append((_row_to_fact(row[:13]), sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

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
    )
