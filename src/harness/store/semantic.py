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

_SCHEMA = """
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
    tier           TEXT    NOT NULL DEFAULT 'working',
    created_at     TEXT    NOT NULL,
    embedding      BLOB    NOT NULL
);

CREATE INDEX IF NOT EXISTS semantic_subject_idx ON semantic (subject);
CREATE INDEX IF NOT EXISTS semantic_tier_idx    ON semantic (tier);
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


class SemanticStore:
    """Atomic-fact memory. Each row is a (subject, predicate, object)
    triple with provenance. Same SQLite-plus-BLOB-embedding pattern as
    the episodic store; same 'graduate to LanceDB past 10k rows' plan."""

    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.executescript(_SCHEMA)

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
                created_at, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        return self.get(cur.lastrowid or 0)

    def get(self, record_id: int) -> SemanticFact:
        row = self._conn.execute(
            """SELECT id, subject, predicate, object, confidence, source,
                      attributed_to, session_id, user_id, supersedes, tier,
                      created_at
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
    ) -> list[SemanticFact]:
        query = """SELECT id, subject, predicate, object, confidence, source,
                      attributed_to, session_id, user_id, supersedes, tier,
                      created_at FROM semantic"""
        conditions: list[str] = []
        params: list[object] = []
        if tier is not None:
            conditions.append("tier = ?")
            params.append(tier)
        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_fact(r) for r in rows]

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        min_confidence: float = 0.0,
        min_score: float = 0.0,
    ) -> list[tuple[SemanticFact, float]]:
        """Return up to `k` facts where stored confidence >= `min_confidence`
        AND retrieval cosine similarity >= `min_score`. The confidence
        gate filters by trustworthiness; the score gate filters by
        actual relevance to the query. Both matter — a high-confidence
        fact about an unrelated topic still pollutes the prompt."""
        rows = self._conn.execute(
            """SELECT id, subject, predicate, object, confidence, source,
                      attributed_to, session_id, user_id, supersedes, tier,
                      created_at, embedding
               FROM semantic WHERE confidence >= ?""",
            (min_confidence,),
        ).fetchall()
        if not rows:
            return []
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        scored: list[tuple[SemanticFact, float]] = []
        for row in rows:
            vec = np.frombuffer(row[12], dtype=np.float32)
            sim = float(np.dot(q_vec, vec))
            if sim >= min_score:
                scored.append((_row_to_fact(row[:12]), sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

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
    )
