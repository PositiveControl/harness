from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from harness.character import Character
    from harness.retrieval.embed import Embedder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id  TEXT    UNIQUE,
    title        TEXT    NOT NULL,
    body         TEXT    NOT NULL,
    principle    TEXT,
    tags         TEXT    NOT NULL DEFAULT '[]',
    tier         TEXT    NOT NULL,
    source       TEXT    NOT NULL,
    session_id   TEXT,
    user_id      TEXT,
    created_at   TEXT    NOT NULL,
    last_accessed TEXT,
    embedding    BLOB    NOT NULL
);

CREATE INDEX IF NOT EXISTS episodic_external_id_idx ON episodic (external_id);
CREATE INDEX IF NOT EXISTS episodic_tier_idx ON episodic (tier);
"""


@dataclass(frozen=True)
class EpisodicRecord:
    id: int
    external_id: str | None
    title: str
    body: str
    principle: str | None
    tags: tuple[str, ...]
    tier: str  # "seed" | "consolidated" | "working"
    source: str  # "yaml" | "scribe" | "user"
    session_id: str | None
    user_id: str | None
    created_at: datetime


class EpisodicStore:
    """Append-mostly episodic memory. SQLite holds the canonical record
    plus a BLOB of the embedding (float32 bytes). Search is a full scan
    with cosine similarity — fast enough for thousands of records. When
    we cross ~10k we graduate to LanceDB with the same external API.

    The store owns an Embedder because every write must produce a
    vector; caller-provided vectors would split responsibility
    awkwardly."""

    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.executescript(_SCHEMA)

    def has(self, external_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM episodic WHERE external_id = ? LIMIT 1", (external_id,)
        ).fetchone()
        return row is not None

    def ingest(
        self,
        *,
        external_id: str | None,
        title: str,
        body: str,
        principle: str | None = None,
        tags: Iterable[str] = (),
        tier: str = "working",
        source: str = "user",
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> EpisodicRecord:
        """Insert a new episodic record. If `external_id` is given and
        already exists, returns the existing record unchanged —
        idempotent by design so seed ingestion is safe to run on every
        startup."""
        if external_id is not None:
            existing = self._conn.execute(
                "SELECT id FROM episodic WHERE external_id = ?", (external_id,)
            ).fetchone()
            if existing is not None:
                return self.get(existing[0])

        now = datetime.now(UTC).isoformat()
        # Embed title + principle + body — so searches on the lesson or
        # the title hit too, not just the narrative body.
        parts = [title]
        if principle:
            parts.append(principle)
        parts.append(body)
        content = "\n\n".join(parts)
        vec = self.embedder.embed([content])[0].astype(np.float32)

        cur = self._conn.execute(
            """INSERT INTO episodic (
                external_id, title, body, principle, tags, tier, source,
                session_id, user_id, created_at, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                external_id,
                title,
                body,
                principle,
                json.dumps(list(tags)),
                tier,
                source,
                session_id,
                user_id,
                now,
                vec.tobytes(),
            ),
        )
        return self.get(cur.lastrowid or 0)

    def get(self, record_id: int) -> EpisodicRecord:
        row = self._conn.execute(
            """SELECT id, external_id, title, body, principle, tags, tier,
                      source, session_id, user_id, created_at
               FROM episodic WHERE id = ?""",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"episodic record {record_id} not found")
        return _row_to_record(row)

    def all(self, *, tier: str | None = None) -> list[EpisodicRecord]:
        if tier is None:
            rows = self._conn.execute(
                """SELECT id, external_id, title, body, principle, tags, tier,
                          source, session_id, user_id, created_at
                   FROM episodic ORDER BY id"""
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT id, external_id, title, body, principle, tags, tier,
                          source, session_id, user_id, created_at
                   FROM episodic WHERE tier = ? ORDER BY id""",
                (tier,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        min_score: float = 0.0,
    ) -> list[tuple[EpisodicRecord, float]]:
        """Return up to `k` records with cosine similarity >= `min_score`,
        paired with the similarity. Empty list if the store is empty or
        nothing clears the threshold — memory that doesn't clear the bar
        pollutes the prompt and gives us hallucinated "relevance" where
        there is none."""
        rows = self._conn.execute(
            """SELECT id, external_id, title, body, principle, tags, tier,
                      source, session_id, user_id, created_at, embedding
               FROM episodic"""
        ).fetchall()
        if not rows:
            return []

        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        scored: list[tuple[EpisodicRecord, float]] = []
        for row in rows:
            vec = np.frombuffer(row[11], dtype=np.float32)
            # Vectors are normalized by the Embedder contract; dot == cosine.
            sim = float(np.dot(q_vec, vec))
            if sim >= min_score:
                scored.append((_row_to_record(row[:11]), sim))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def close(self) -> None:
        self._conn.close()


def _row_to_record(row: tuple) -> EpisodicRecord:  # type: ignore[type-arg]
    return EpisodicRecord(
        id=row[0],
        external_id=row[1],
        title=row[2],
        body=row[3],
        principle=row[4],
        tags=tuple(json.loads(row[5])),
        tier=row[6],
        source=row[7],
        session_id=row[8],
        user_id=row[9],
        created_at=datetime.fromisoformat(row[10]),
    )


def ensure_seeds_ingested(character: Character, store: EpisodicStore) -> int:
    """Walk `character.seed_memories` and ingest anything not already in
    the store. Returns the number of records newly inserted. Idempotent
    — safe to call on every startup."""
    inserted = 0
    for seed in character.seed_memories:
        if store.has(seed.id):
            continue
        store.ingest(
            external_id=seed.id,
            title=seed.title,
            body=seed.body,
            principle=seed.principle,
            tags=seed.tags,
            tier="seed",
            source="yaml",
        )
        inserted += 1
    return inserted
