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

_CREATE_TABLE = """
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
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS episodic_external_id_idx   ON episodic (external_id);
CREATE INDEX IF NOT EXISTS episodic_tier_idx          ON episodic (tier);
CREATE INDEX IF NOT EXISTS episodic_superseded_by_idx ON episodic (superseded_by);
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
    source: str  # "yaml" | "scribe" | "user" | "consolidator"
    session_id: str | None
    user_id: str | None
    created_at: datetime
    superseded_by: int | None = None  # non-null → retired; filter out of retrieval


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
        # See transcript.py for the `check_same_thread=False` rationale:
        # the TUI worker and main thread share this connection; SQLite
        # + Python's per-connection lock already serialize writes.
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_CREATE_TABLE)
        # Schema migrations — safe to run on fresh tables (no-op) or on
        # older tables (adds the missing column).
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(episodic)")}
        if "superseded_by" not in cols:
            self._conn.execute(
                "ALTER TABLE episodic ADD COLUMN superseded_by INTEGER REFERENCES episodic(id)"
            )
        if "embedder_id" not in cols:
            self._conn.execute("ALTER TABLE episodic ADD COLUMN embedder_id TEXT")
        if "embedding_dim" not in cols:
            self._conn.execute("ALTER TABLE episodic ADD COLUMN embedding_dim INTEGER")
        # Backfill newly-added columns for rows from older schemas.
        # Dim is inferable from the BLOB length (4 bytes per float32);
        # embedder_id can't be recovered, so legacy rows get "legacy".
        self._conn.execute(
            "UPDATE episodic SET embedding_dim = LENGTH(embedding) / 4 WHERE embedding_dim IS NULL"
        )
        self._conn.execute("UPDATE episodic SET embedder_id = 'legacy' WHERE embedder_id IS NULL")
        # Indexes created after migration so the superseded_by index can
        # reference the freshly-added column.
        self._conn.executescript(_CREATE_INDEXES)

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
                session_id, user_id, created_at, embedding,
                embedder_id, embedding_dim
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                self.embedder.id,
                self.embedder.dimension,
            ),
        )
        return self.get(cur.lastrowid or 0)

    def get(self, record_id: int) -> EpisodicRecord:
        row = self._conn.execute(
            """SELECT id, external_id, title, body, principle, tags, tier,
                      source, session_id, user_id, created_at, superseded_by
               FROM episodic WHERE id = ?""",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"episodic record {record_id} not found")
        return _row_to_record(row)

    def all(
        self,
        *,
        tier: str | None = None,
        include_superseded: bool = False,
    ) -> list[EpisodicRecord]:
        conditions: list[str] = []
        params: list[object] = []
        if tier is not None:
            conditions.append("tier = ?")
            params.append(tier)
        if not include_superseded:
            conditions.append("superseded_by IS NULL")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        # `where` is assembled from hardcoded column checks, not user
        # input — S608 is a false positive here.
        rows = self._conn.execute(
            f"""SELECT id, external_id, title, body, principle, tags, tier,
                       source, session_id, user_id, created_at, superseded_by
                FROM episodic{where} ORDER BY id""",  # noqa: S608
            params,
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def mark_superseded(self, record_id: int, *, by: int) -> None:
        """Mark `record_id` as superseded by the record with id `by`. The
        record still exists (audit trail) but drops out of retrieval."""
        self._conn.execute(
            "UPDATE episodic SET superseded_by = ? WHERE id = ?",
            (by, record_id),
        )

    def fetch_embedding(self, record_id: int) -> np.ndarray:
        """Return the stored embedding for a record, as a numpy array.
        Used by the consolidator for clustering without re-embedding."""
        row = self._conn.execute(
            "SELECT embedding FROM episodic WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"episodic record {record_id} not found")
        return np.frombuffer(row[0], dtype=np.float32)

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        min_score: float = 0.0,
        user_id: str | None = None,
    ) -> list[tuple[EpisodicRecord, float]]:
        """Return up to `k` active records with cosine similarity >=
        `min_score`. Rows whose embedding dimension doesn't match the
        current embedder are silently skipped — they belong to a
        previous embedder generation and need a `rebuild-embeddings`
        run before they'll participate in search again.

        `user_id` scopes to relationship memory: when given, returns
        rows where `user_id IS NULL` (shared / character-level) OR
        `user_id = <this user>`. Other users' private memories are
        never returned. When `user_id` is None, this is an owner-tier
        view that sees everything."""
        if user_id is None:
            rows = self._conn.execute(
                """SELECT id, external_id, title, body, principle, tags, tier,
                          source, session_id, user_id, created_at, superseded_by,
                          embedding
                   FROM episodic
                   WHERE superseded_by IS NULL AND embedding_dim = ?""",
                (self.embedder.dimension,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT id, external_id, title, body, principle, tags, tier,
                          source, session_id, user_id, created_at, superseded_by,
                          embedding
                   FROM episodic
                   WHERE superseded_by IS NULL AND embedding_dim = ?
                     AND (user_id IS NULL OR user_id = ?)""",
                (self.embedder.dimension, user_id),
            ).fetchall()
        if not rows:
            return []

        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        scored: list[tuple[EpisodicRecord, float]] = []
        for row in rows:
            vec = np.frombuffer(row[12], dtype=np.float32)
            # Vectors are normalized by the Embedder contract; dot == cosine.
            sim = float(np.dot(q_vec, vec))
            if sim >= min_score:
                scored.append((_row_to_record(row[:12]), sim))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def count(
        self,
        *,
        tier: str | None = None,
        user_id: str | None = None,
        include_superseded: bool = False,
    ) -> int:
        """Row count scoped the same way `search()` is: when `user_id` is
        given, counts shared rows plus that user's private rows. Used by
        the introspect tool to report memory size without pulling every
        record via `all()`."""
        conditions: list[str] = []
        params: list[object] = []
        if tier is not None:
            conditions.append("tier = ?")
            params.append(tier)
        if user_id is not None:
            conditions.append("(user_id IS NULL OR user_id = ?)")
            params.append(user_id)
        if not include_superseded:
            conditions.append("superseded_by IS NULL")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM episodic{where}",  # noqa: S608
            params,
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def last_created_at(
        self,
        *,
        tier: str | None = None,
        user_id: str | None = None,
    ) -> datetime | None:
        """Most recent `created_at` among active rows matching the same
        scope as `count()`. Returns None when the filter matches no rows
        (fresh store, or user hasn't produced any memories yet)."""
        conditions: list[str] = ["superseded_by IS NULL"]
        params: list[object] = []
        if tier is not None:
            conditions.append("tier = ?")
            params.append(tier)
        if user_id is not None:
            conditions.append("(user_id IS NULL OR user_id = ?)")
            params.append(user_id)
        where = " WHERE " + " AND ".join(conditions)
        row = self._conn.execute(
            f"SELECT MAX(created_at) FROM episodic{where}",  # noqa: S608
            params,
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return datetime.fromisoformat(row[0])

    def count_mismatched_embeddings(self) -> int:
        """How many active rows carry embeddings from a prior embedder
        generation. Use to tell the user whether a rebuild is worth it."""
        row = self._conn.execute(
            """SELECT COUNT(*) FROM episodic
               WHERE superseded_by IS NULL AND embedding_dim != ?""",
            (self.embedder.dimension,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def rebuild_embeddings(self) -> tuple[int, int]:
        """Re-embed every active row with the current embedder. Useful
        after an embedder switch — previous BLOBs are dim-locked to the
        old model. Returns (rows_updated, rows_skipped). Skipped rows
        are superseded ones; no point re-embedding retired data."""
        cur = self._conn.execute(
            """SELECT id, title, body, principle FROM episodic
               WHERE superseded_by IS NULL ORDER BY id"""
        )
        rows = cur.fetchall()
        if not rows:
            return 0, 0
        texts: list[str] = []
        for _id, title, body, principle in rows:
            parts = [title]
            if principle:
                parts.append(principle)
            parts.append(body)
            texts.append("\n\n".join(parts))
        vectors = self.embedder.embed(texts)
        updated = 0
        for (record_id, _title, _body, _principle), vec in zip(rows, vectors, strict=True):
            self._conn.execute(
                """UPDATE episodic
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
        superseded_by=row[11] if len(row) > 11 else None,
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
