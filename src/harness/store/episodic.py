from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from harness.store._hybrid import (
    reciprocal_rank_fusion,
    sanitize_fts_query,
    session_scope_filter,
)

if TYPE_CHECKING:
    from harness.character import Character
    from harness.retrieval.embed import Embedder


SearchMode = Literal["hybrid", "dense", "text"]


def _build_embed_text(
    *,
    title: str,
    body: str,
    principle: str | None,
    tier: str,
    created_at_iso: str | None,
) -> str:
    """Compose the text that gets fed into the embedder. Prepends a
    structured tag header so dense-cosine retrieval can match by
    lesson (`[principle: X]`) or timeframe (`[date: YYYY-MM-DD]`) —
    the seed-memory frontmatter already carries this metadata, but
    under the pre-contextual-chunking format it only reached columns,
    never the embedding. See sota punch #6 / harness-2am.

    Existing installs need one `harness memory rebuild-embeddings`
    run after this lands: rows embedded under the old format are
    still valid cosine vectors but don't get the tag-match lift.
    """
    tags: list[str] = [f"tier: {tier}"]
    if principle:
        tags.append(f"principle: {principle}")
    if created_at_iso:
        # ISO 8601 splits at T between date and time; keep only the
        # date — hour/minute/second noise rarely helps retrieval and
        # fights tokenization on small embedders.
        tags.append(f"date: {created_at_iso.split('T')[0]}")
    header = "[" + "; ".join(tags) + "]"
    parts: list[str] = [header, title]
    # Keep principle as a standalone line so it contributes to the
    # embedding semantically (full-sentence phrasing) in addition to
    # the structured tag.
    if principle:
        parts.append(principle)
    parts.append(body)
    return "\n\n".join(parts)


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
-- Speeds up `--memory-scope=current-session` retrieval and
-- `delete_working_for_session` (harness-w3mo / harness-k7m9).
CREATE INDEX IF NOT EXISTS episodic_session_id_idx    ON episodic (session_id);
"""

# FTS5 sidecar for BM25 text search. External-content table points at
# `episodic` as the source of truth; the AFTER INSERT trigger keeps
# the sidecar in sync. Text columns (title / body / principle) are
# never UPDATE'd after insert — only administrative columns like
# superseded_by / embedding / last_accessed change — so we don't need
# UPDATE or DELETE triggers. Backfill on first init handles pre-
# existing rows from the pre-FTS5 schema.
#
# Porter stemming (harness-cpf): the porter tokenizer wraps unicode61 so
# query "declare" matches stored "declared", "declaring", etc. This is
# the primary retrieval quality fix for tense-mismatch failures in the
# airton_c1 corpus — dense cosine finds the right section by semantic
# similarity, but without stemming BM25 misses it and drags down the
# RRF-fused hybrid rank.
_FTS_TOKENIZE = "porter unicode61 remove_diacritics 1"

_CREATE_FTS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
    title, body, principle,
    content='episodic',
    content_rowid='id',
    tokenize='{_FTS_TOKENIZE}'
);

CREATE TRIGGER IF NOT EXISTS episodic_fts_ai
AFTER INSERT ON episodic BEGIN
    INSERT INTO episodic_fts(rowid, title, body, principle)
    VALUES (new.id, new.title, new.body, COALESCE(new.principle, ''));
END;

-- Mirror deletes into the external-content FTS sidecar so
-- `delete_working_for_session` (harness-k7m9) doesn't leave dangling
-- BM25 entries that point at vanished rowids. Idempotent: running
-- the migration on an existing db just adds the missing trigger.
CREATE TRIGGER IF NOT EXISTS episodic_fts_ad
AFTER DELETE ON episodic BEGIN
    INSERT INTO episodic_fts(episodic_fts, rowid, title, body, principle)
    VALUES ('delete', old.id, old.title, old.body, COALESCE(old.principle, ''));
END;
"""  # noqa: S608 — module-level constant; _FTS_TOKENIZE is never user-supplied


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
        # FTS5 sidecar for the hybrid retrieval path. Create the
        # virtual table if missing; on first-install / post-migration
        # the main table can already carry rows from the pre-FTS
        # schema, so we must rebuild the index. Note: COUNT(*) on
        # external-content FTS mirrors the main table count even
        # when the index is empty, so counting can't detect the
        # migration case — check schema existence instead.
        #
        # Porter-stemming migration (harness-cpf): if the FTS table
        # exists but was created with the old tokenizer (no "porter"),
        # DROP it and recreate with the new DDL, then re-populate from
        # the canonical table. Detection is a DDL-string probe against
        # sqlite_master — cheapest approach, requires no version table.
        fts_row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='episodic_fts'"
        ).fetchone()
        fts_existed_before = fts_row is not None
        needs_tokenizer_migration = fts_existed_before and (
            fts_row[0] is None or "porter" not in fts_row[0].lower()
        )
        if needs_tokenizer_migration:
            # Drop the stale FTS table (triggers referencing it are also
            # dropped automatically by SQLite when the virtual table goes).
            self._conn.executescript(
                "DROP TABLE IF EXISTS episodic_fts;DROP TRIGGER IF EXISTS episodic_fts_ai;"
            )
            fts_existed_before = False  # force rebuild below
        self._conn.executescript(_CREATE_FTS)
        if not fts_existed_before:
            self._conn.execute("INSERT INTO episodic_fts(episodic_fts) VALUES('rebuild')")

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
        content = _build_embed_text(
            title=title,
            body=body,
            principle=principle,
            tier=tier,
            created_at_iso=now,
        )
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

    def delete_working_for_session(self, session_id: str) -> int:
        """Hard-delete every tier=working row tagged with `session_id`.

        Used by `harness session reset` (harness-k7m9) to prune the
        scribed memory for one session without touching shared seeds,
        consolidated rows, or other sessions' working-tier writes.

        Consolidated rows are intentionally left alone: they merge
        candidates from possibly multiple sessions, so removing them
        on a single-session reset would corrupt the merge. The
        scribe watermark is also untouched — without it, the next
        scribe run would re-create exactly the rows we just deleted
        from the surviving transcript turns. Returns the number of
        rows removed."""
        cur = self._conn.execute(
            "DELETE FROM episodic WHERE tier = 'working' AND session_id = ?",
            (session_id,),
        )
        return cur.rowcount or 0

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
        mode: SearchMode = "hybrid",
        bm25_min_score: float | None = None,
        allowed_sessions: tuple[str, ...] | None = None,
        recency_ranks: dict[str, int] | None = None,
        recency_weight: float = 0.0,
    ) -> list[tuple[EpisodicRecord, float]]:
        """Return up to `k` active records ranked by `mode`.

        `mode='hybrid'` (default) — fuse dense cosine + FTS5 BM25 via
        Reciprocal Rank Fusion (k=60). Best recall on queries that
        mix semantic intent with proper nouns or code identifiers
        ('when did we decide to rename BeadsAdapter.get_focus').
        `min_score` is applied to the dense component pre-fusion; the
        returned score is the RRF score, not a cosine similarity.

        `bm25_min_score` (harness-dffh) — soft cosine floor applied to
        BM25 candidates pre-fusion. Without it, a lexical match on a
        common token ('junior', 'stack') could rank a semantically
        unrelated row into the top-K via the BM25 side of RRF. When
        unset, defaults to `max(min_score - 0.15, 0.0)` — softer than
        the dense floor so identifier queries (where dense cosine is
        genuinely lower) keep working, but enough of a floor that
        common-word topical bleed is filtered out. Pass `0.0`
        explicitly to disable the gate. No effect on `dense` /
        `text` modes.

        `mode='dense'` — legacy path, pure cosine similarity. Rows
        whose embedding dim doesn't match the current embedder are
        silently skipped (run `rebuild-embeddings` to bring them back).
        Returned score is cosine similarity.

        `mode='text'` — BM25 only. Returned score is the raw FTS5
        BM25 score (negated so higher=better for consistency with the
        other modes).

        `user_id` scopes to relationship memory across all modes:
        when given, returns rows where `user_id IS NULL` (shared) OR
        `user_id = <this user>`. Other users' private memories are
        never returned. `user_id=None` is an owner-tier view.

        `allowed_sessions` (harness-w3mo) — optional session-scope
        filter that drives `--memory-scope`. None = no filter (today).
        Empty tuple = only NULL-session rows (procedural / shared
        seeds). Tuple of session ids = NULL OR session_id IN (...).
        NULL rows always pass — the C-plan treats them as 'always
        eligible' so consolidator output, seeds, and harvested
        procedural memory survive every scope.

        `recency_ranks` + `recency_weight` (harness-w3mo step 5) —
        opt-in recency boost in the hybrid path. When `recency_weight
        > 0`, hybrid fuses a third RRF ranking built from
        `recency_ranks[session_id]` so newer-session rows tilt
        higher in the fused score. NULL-session rows (seeds /
        procedural / cross-session consolidated) skip the recency
        ranking — they ride on dense + BM25 alone, so seeds aren't
        artificially aged out. Default `recency_weight=0.0` keeps
        the gate off; chat boots with `HARNESS_RETRIEVAL_RECENCY_WEIGHT`
        from settings."""
        if mode == "dense":
            return self._search_dense(
                query,
                k=k,
                min_score=min_score,
                user_id=user_id,
                allowed_sessions=allowed_sessions,
            )
        if mode == "text":
            return self._search_text(query, k=k, user_id=user_id, allowed_sessions=allowed_sessions)
        bm25_floor = bm25_min_score if bm25_min_score is not None else max(min_score - 0.15, 0.0)
        return self._search_hybrid(
            query,
            k=k,
            min_score=min_score,
            bm25_min_score=bm25_floor,
            user_id=user_id,
            allowed_sessions=allowed_sessions,
            recency_ranks=recency_ranks,
            recency_weight=recency_weight,
        )

    def _search_dense(
        self,
        query: str,
        *,
        k: int,
        min_score: float,
        user_id: str | None,
        allowed_sessions: tuple[str, ...] | None = None,
    ) -> list[tuple[EpisodicRecord, float]]:
        # Embed the query FIRST so a lazy embedder (dimension=0 until
        # first embed() call) populates its real dimension before the
        # SQL filter reads it. Otherwise `WHERE embedding_dim = 0`
        # matches zero rows and dense silently returns [] on the very
        # first call against a fresh embedder (harness-m35).
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        session_clause, session_params = session_scope_filter(allowed_sessions)
        if user_id is None:
            rows = self._conn.execute(
                f"""SELECT id, external_id, title, body, principle, tags, tier,
                          source, session_id, user_id, created_at, superseded_by,
                          embedding
                   FROM episodic
                   WHERE superseded_by IS NULL AND embedding_dim = ?{session_clause}""",  # noqa: S608 — session_clause is a static fragment with bind placeholders
                (self.embedder.dimension, *session_params),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""SELECT id, external_id, title, body, principle, tags, tier,
                          source, session_id, user_id, created_at, superseded_by,
                          embedding
                   FROM episodic
                   WHERE superseded_by IS NULL AND embedding_dim = ?
                     AND (user_id IS NULL OR user_id = ?){session_clause}""",  # noqa: S608 — session_clause is a static fragment with bind placeholders
                (self.embedder.dimension, user_id, *session_params),
            ).fetchall()
        if not rows:
            return []

        scored: list[tuple[EpisodicRecord, float]] = []
        for row in rows:
            vec = np.frombuffer(row[12], dtype=np.float32)
            # Vectors are normalized by the Embedder contract; dot == cosine.
            sim = float(np.dot(q_vec, vec))
            if sim >= min_score:
                scored.append((_row_to_record(row[:12]), sim))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def _search_text(
        self,
        query: str,
        *,
        k: int,
        user_id: str | None,
        allowed_sessions: tuple[str, ...] | None = None,
    ) -> list[tuple[EpisodicRecord, float]]:
        match = sanitize_fts_query(query)
        if not match:
            return []
        # Join FTS rowids back to the main table so we inherit the
        # same scope filters as the dense path (superseded + user_id).
        # bm25() returns lower=better; ORDER BY bm25(...) ASC plus a
        # score negation on the way out keeps the "higher is better"
        # external contract consistent with cosine. session_scope_filter
        # builds the optional `--memory-scope` AND-clause inline so the
        # LIMIT k clause still returns k rows AFTER session filtering
        # (Python-side filter would over-truncate). Reference column
        # name resolves through the JOIN — episodic alias `e` exposes
        # session_id as `e.session_id` for the placeholder.
        session_clause, session_params = session_scope_filter(
            allowed_sessions, column="e.session_id"
        )
        if user_id is None:
            rows = self._conn.execute(
                f"""SELECT e.id, e.external_id, e.title, e.body, e.principle,
                          e.tags, e.tier, e.source, e.session_id, e.user_id,
                          e.created_at, e.superseded_by,
                          bm25(episodic_fts) AS bm25_score
                   FROM episodic_fts
                   JOIN episodic e ON e.id = episodic_fts.rowid
                   WHERE episodic_fts MATCH ?
                     AND e.superseded_by IS NULL{session_clause}
                   ORDER BY bm25_score
                   LIMIT ?""",  # noqa: S608 — session_clause is a static fragment with bind placeholders
                (match, *session_params, k),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""SELECT e.id, e.external_id, e.title, e.body, e.principle,
                          e.tags, e.tier, e.source, e.session_id, e.user_id,
                          e.created_at, e.superseded_by,
                          bm25(episodic_fts) AS bm25_score
                   FROM episodic_fts
                   JOIN episodic e ON e.id = episodic_fts.rowid
                   WHERE episodic_fts MATCH ?
                     AND e.superseded_by IS NULL
                     AND (e.user_id IS NULL OR e.user_id = ?){session_clause}
                   ORDER BY bm25_score
                   LIMIT ?""",  # noqa: S608 — session_clause is a static fragment with bind placeholders
                (match, user_id, *session_params, k),
            ).fetchall()
        return [(_row_to_record(row[:12]), -float(row[12])) for row in rows]

    def _search_hybrid(
        self,
        query: str,
        *,
        k: int,
        min_score: float,
        bm25_min_score: float,
        user_id: str | None,
        allowed_sessions: tuple[str, ...] | None = None,
        recency_ranks: dict[str, int] | None = None,
        recency_weight: float = 0.0,
    ) -> list[tuple[EpisodicRecord, float]]:
        # Widen the candidate sets ~4x so RRF has room to reorder.
        # Past ~50 candidates the tail contributes <0.001 per match
        # so further widening is wasted.
        candidate_k = max(k * 4, 20)
        dense_hits = self._search_dense(
            query,
            k=candidate_k,
            min_score=min_score,
            user_id=user_id,
            allowed_sessions=allowed_sessions,
        )
        text_hits = self._search_text(
            query, k=candidate_k, user_id=user_id, allowed_sessions=allowed_sessions
        )
        if bm25_min_score > 0.0 and text_hits:
            text_hits = self._gate_text_hits_by_cosine(
                query=query, hits=text_hits, threshold=bm25_min_score
            )
        if not dense_hits and not text_hits:
            return []
        record_map: dict[int, EpisodicRecord] = {}
        for rec, _ in dense_hits:
            record_map[rec.id] = rec
        for rec, _ in text_hits:
            record_map.setdefault(rec.id, rec)

        rankings: list[list[int]] = [
            [rec.id for rec, _ in dense_hits],
            [rec.id for rec, _ in text_hits],
        ]
        weights: list[float] = [1.0, 1.0]
        if recency_weight > 0.0 and recency_ranks:
            # Build the recency ranking from the union of dense + text
            # candidates. Records with NULL session_id are excluded —
            # they shouldn't get aged out (seeds + procedural memory)
            # nor get an artificial freshness boost; the gate is for
            # session-tagged rows only. Sort by recency rank ascending
            # (lower = newer) so the most-recent records lead the list.
            session_tagged = [rec for rec in record_map.values() if rec.session_id is not None]
            recency_sorted = sorted(
                session_tagged,
                key=lambda r: recency_ranks.get(
                    r.session_id or "",
                    len(recency_ranks) + 1,
                ),
            )
            rankings.append([rec.id for rec in recency_sorted])
            weights.append(recency_weight)

        fused = reciprocal_rank_fusion(rankings, weights=weights)
        return [(record_map[rid], score) for rid, score in fused[:k] if rid in record_map]

    def _gate_text_hits_by_cosine(
        self,
        *,
        query: str,
        hits: list[tuple[EpisodicRecord, float]],
        threshold: float,
    ) -> list[tuple[EpisodicRecord, float]]:
        """Drop BM25 hits whose dense cosine to the query is below
        `threshold`. Prevents the topical-bleed failure mode where
        a lexical match on a common token drags a semantically
        unrelated row into the hybrid top-K (harness-dffh).

        Batch-fetches embeddings for the hit ids in one query to
        keep this O(1) extra DB round-trip regardless of |hits|.
        Rows whose embedding_dim doesn't match the current embedder
        are dropped (same as the dense path), since we can't compute
        a meaningful cosine across dimensions."""
        if not hits:
            return []
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        ids = [rec.id for rec, _ in hits]
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id, embedding, embedding_dim FROM episodic WHERE id IN ({placeholders})",  # noqa: S608 — placeholders param-bind, not user data
            ids,
        ).fetchall()
        embed_map = {row[0]: (row[1], int(row[2])) for row in rows}
        out: list[tuple[EpisodicRecord, float]] = []
        for rec, bm in hits:
            payload = embed_map.get(rec.id)
            if payload is None:
                continue
            blob, dim = payload
            if dim != self.embedder.dimension:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            cos = float(np.dot(q_vec, vec))
            if cos >= threshold:
                out.append((rec, bm))
        return out

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
        source: str | None = None,
    ) -> datetime | None:
        """Most recent `created_at` among active rows matching the same
        scope as `count()`. `source` narrows further (e.g. 'scribe' for
        last-scribe-run watermark, 'consolidator' for last
        consolidation). Returns None when the filter matches no rows."""
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
            """SELECT id, title, body, principle, tier, created_at FROM episodic
               WHERE superseded_by IS NULL ORDER BY id"""
        )
        rows = cur.fetchall()
        if not rows:
            return 0, 0
        texts = [
            _build_embed_text(
                title=title,
                body=body,
                principle=principle,
                tier=tier,
                created_at_iso=created_at,
            )
            for _id, title, body, principle, tier, created_at in rows
        ]
        vectors = self.embedder.embed(texts)
        updated = 0
        for (record_id, _title, _body, _principle, _tier, _created_at), vec in zip(
            rows, vectors, strict=True
        ):
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
