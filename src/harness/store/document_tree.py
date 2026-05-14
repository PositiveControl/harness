"""Document tree store (harness-h5ly / Phase 1).

A structure-preserving SQLite store for hierarchical documents. The
deliberate departure from `EpisodicStore` is that the unit of storage
is *the tree*, not the chunk: a document is a tree of nodes, parents
link to children via `parent_id`, and embeddings live at section
granularity rather than at fine-grained chunks.

Why a separate store: chunk-flat retrieval (the Phase 0 baseline) loses
two things that matter for structured documents.
  1. Hierarchy. Once chunks are flat rows, "this rule sits inside that
     chapter" is reconstructable only via the principle string.
  2. Granularity. The right embed unit isn't always the smallest one.
     If a section has five paragraphs each making one part of the same
     argument, embedding each paragraph fragments the semantic field;
     embedding the section gives one richer vector.

The store doesn't dictate granularity — it lets callers ingest nodes at
whatever depth fits the corpus. The Phase 1 ATC ingest writes one node
per JO 7110.65 section (aggregating its chunks). Chapters and parent
sections are stored as structural-only nodes (no embedding) so the
hierarchy is queryable but they don't pollute the embedding index.

Search semantics (v1): hybrid (dense cosine + BM25) over embedded
nodes only. Returns `(TreeNode, score)` ordered. Auto-merge — promoting
parents when multiple sibling leaves match — is intentionally deferred
to a follow-up issue. Land the simpler thing first; measure; decide.

This module owns ONLY storage. The matching retriever lives in
`src/harness/retrieval/tree_retriever.py` so the two layers stay
swappable.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from harness.store._hybrid import reciprocal_rank_fusion, sanitize_fts_query

if TYPE_CHECKING:
    from harness.retrieval.embed import Embedder


SearchMode = Literal["hybrid", "dense", "text"]

NodeType = Literal["document", "chapter", "section_group", "section", "chunk"]
"""Node-type slugs. Open set — callers can introduce new types as long
as the ingest side and the retriever agree on which types carry
embeddings. `section_group` is the parent_section level for the ATC
corpus; rename in a follow-up if we grow non-ATC consumers."""


_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS tree_documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    source_uri   TEXT,
    created_at   TEXT    NOT NULL
);
"""

_CREATE_NODES = """
CREATE TABLE IF NOT EXISTS tree_nodes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id    INTEGER NOT NULL REFERENCES tree_documents(id),
    parent_id      INTEGER REFERENCES tree_nodes(id),
    -- Stable hierarchical identifier for matching against external
    -- fixtures (e.g. "2-4-3" for an ATC section). Unique per document.
    path           TEXT    NOT NULL,
    ordinal        INTEGER NOT NULL,
    depth          INTEGER NOT NULL,
    node_type      TEXT    NOT NULL,
    heading        TEXT    NOT NULL,
    body           TEXT    NOT NULL DEFAULT '',
    -- harness-q6zl: BM25-searchable rendering of the section anchor.
    -- Carries `§<path>`, the bare path, the document name, and the
    -- qualified `<document>:<path>` form so a BM25 query for any of
    -- those forms surfaces this row. Populated at ingest_node time
    -- and reindexed by the FTS5 sidecar via the same trigger that
    -- mirrors heading + body. Empty string is legal (back-compat
    -- for stores whose rows pre-date the column).
    anchors        TEXT    NOT NULL DEFAULT '',
    -- NULL embedding ⇒ structural-only node; not surfaced by search().
    embedding      BLOB,
    embedder_id    TEXT,
    embedding_dim  INTEGER,
    created_at     TEXT    NOT NULL,
    UNIQUE (document_id, path)
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS tree_nodes_parent_idx   ON tree_nodes (parent_id);
CREATE INDEX IF NOT EXISTS tree_nodes_depth_idx    ON tree_nodes (depth);
CREATE INDEX IF NOT EXISTS tree_nodes_doc_idx      ON tree_nodes (document_id);
CREATE INDEX IF NOT EXISTS tree_nodes_dim_idx      ON tree_nodes (embedding_dim);
"""

# FTS5 sidecar mirrors `tree_nodes`: heading + body + anchors. Porter
# stemming for tense-tolerant BM25 (same rationale as `episodic_fts`,
# harness-cpf). `tokenchars '.-:'` keeps multi-segment section paths
# (`91.131`, `2-4-3`, `CFR_14_Vol2:91.131`) as single tokens rather
# than splitting them at the punctuation — harness-q6zl: airton_c
# baseline failed because BM25 was splitting `91.131` into `91` and
# `131`, matching neither.
_FTS_TOKENIZE = "porter unicode61 remove_diacritics 1 tokenchars '.-:'"

_CREATE_FTS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS tree_nodes_fts USING fts5(
    heading, body, anchors,
    content='tree_nodes',
    content_rowid='id',
    tokenize="{_FTS_TOKENIZE}"
);

CREATE TRIGGER IF NOT EXISTS tree_nodes_fts_ai
AFTER INSERT ON tree_nodes BEGIN
    INSERT INTO tree_nodes_fts(rowid, heading, body, anchors)
    VALUES (new.id, new.heading, new.body, new.anchors);
END;

CREATE TRIGGER IF NOT EXISTS tree_nodes_fts_ad
AFTER DELETE ON tree_nodes BEGIN
    INSERT INTO tree_nodes_fts(tree_nodes_fts, rowid, heading, body, anchors)
    VALUES ('delete', old.id, old.heading, old.body, old.anchors);
END;
"""  # noqa: S608 — module-level constant; _FTS_TOKENIZE is never user-supplied


def _build_anchor_text(
    *,
    path: str,
    document_name: str,
    parent_section: str | None = None,
    chapter: str | None = None,
) -> str:
    """Compose the BM25-indexed anchor string for a tree node. Carries
    every form a caller might phrase as a search:

      - bare path:           91.131
      - §-prefixed path:     §91.131
      - parent_section:      91          (JO/AIM: `2-4`, CFR: `91`)
      - chapter:             91          (JO/AIM: `2`, CFR: `91`)
      - document only:       CFR_14_Vol2
      - qualified:           CFR_14_Vol2:91.131

    All forms share the same row, so a BM25 query on any of them
    surfaces this section. The tokenizer's tokenchars config
    (`.-:`) keeps dotted/hyphenated forms whole; § gets stripped by
    unicode61 but the bare-token forms still match (harness-q6zl /
    harness-8k94).

    parent_section + chapter are optional — markdown-shaped trees and
    older callers can leave them None. When present, they widen the
    BM25 surface so a query like `§5-3` or `Part 91` matches the
    sibling leaves under that parent without needing the auto-merge
    promotion path. Dedupe at emit time so `parent_section == path`
    (PCG-shape glossaries) doesn't double the same token."""
    parts: list[str] = [f"§{path}", path, document_name, f"{document_name}:{path}"]
    emitted = {f"§{path}", path}
    if parent_section and parent_section not in emitted:
        parts.append(parent_section)
        parts.append(f"§{parent_section}")
        emitted.add(parent_section)
    if chapter and chapter not in emitted:
        parts.append(chapter)
        parts.append(f"§{chapter}")
        emitted.add(chapter)
    return " ".join(parts)


# harness-5yzn: section-path-shaped tokens. Catches CFR (`91.131`,
# `91.205a`), JO/AIM hyphenated (`2-4-3`, `5-5-4`), and 2-segment
# parent-section paths (`2-4`, `91.3`). The leading `(?:§\s*)?` makes
# the §-prefix optional so both the synonym-expander output
# (`§91.131:`) and bare-anchor user queries (`91.131`) match. Three
# capture groups: an optional document prefix (`CFR_14_Vol2:`),
# whitespace tolerated after `§`, and the path itself.
_ANCHOR_QUERY_RE = re.compile(r"(?:([A-Za-z][A-Za-z0-9_.]*):)?§?\s*(\d+(?:[.-]\d+){1,3}[a-z]?)")


def _extract_query_anchors(query: str) -> list[tuple[str | None, str]]:
    """Pull `(document_name, path)` pairs out of an FTS5 query string
    (harness-5yzn). The `document_name` may be None when the query
    didn't qualify which corpus to look in.

    Used by `_search_text` to route an anchor-required pass alongside
    the standard OR-mode BM25 pass. Dedupes pairs preserving first-
    seen order so a query that names the same section twice (once
    bare, once via the synonym expander's `§<path>:` prefix) doesn't
    fire the same FTS pass twice."""
    seen: set[tuple[str | None, str]] = set()
    out: list[tuple[str | None, str]] = []
    for match in _ANCHOR_QUERY_RE.finditer(query):
        doc, path = match.group(1), match.group(2)
        key = (doc, path)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


@dataclass(frozen=True)
class TreeDocument:
    id: int
    name: str
    source_uri: str | None
    created_at: datetime


@dataclass(frozen=True)
class TreeNode:
    id: int
    document_id: int
    parent_id: int | None
    path: str
    ordinal: int
    depth: int
    node_type: str
    heading: str
    body: str
    created_at: datetime


def _build_embed_text(*, heading: str, body: str, path: str) -> str:
    """Compose the text fed to the embedder. The `[path: …]` tag
    mirrors the contextual-chunking header in `EpisodicStore` —
    section IDs reach the embedding directly so the dense ranker can
    match on hierarchical identifiers (helpful for queries that name
    the section, like 'JO 7110.65 §2-4-3')."""
    parts: list[str] = [f"[path: {path}]", heading]
    if body:
        parts.append(body)
    return "\n\n".join(parts)


class DocumentTreeStore:
    """SQLite-backed hierarchical-document store. Append-mostly: nodes
    are unique on `(document_id, path)` so ingest is idempotent. Mixed
    embedded + structural nodes coexist in one table; structural nodes
    carry NULL `embedding`/`embedder_id` and are excluded from
    `search()`.

    The store owns an `Embedder` because every embedded write must
    produce a vector — same contract as `EpisodicStore`. Caller-
    supplied vectors are not supported on the v1 API.
    """

    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Same connection contract as EpisodicStore: thread-share, WAL,
        # 5s busy timeout so the TUI worker thread doesn't deadlock on
        # a write from the main thread.
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_CREATE_DOCUMENTS)
        self._conn.executescript(_CREATE_NODES)
        self._conn.executescript(_CREATE_INDEXES)
        migrated = self._migrate_anchors_column()
        self._conn.executescript(_CREATE_FTS)
        if migrated:
            self._rebuild_fts_index()

    def _migrate_anchors_column(self) -> bool:
        """harness-q6zl: bring a pre-q6zl tree_nodes table up to the
        current schema. Adds the `anchors` column when missing,
        backfills it from existing rows, and rebuilds the FTS5 sidecar
        so its column list + tokenchars config match the new schema.

        Idempotent: on a fresh DB the column already exists from
        `_CREATE_NODES` and nothing more happens. On a pre-q6zl DB
        the migration runs once and finishes before the FTS5 schema
        executescript runs against the now-correct underlying table.

        The FTS sidecar gets a hard rebuild rather than an ALTER because
        FTS5 doesn't support column-list ALTER, AND the tokenizer
        change (tokenchars '.-:') means existing token rows wouldn't
        match queries under the new tokenizer anyway."""
        existing_cols = {
            row[1] for row in self._conn.execute("PRAGMA table_info(tree_nodes)").fetchall()
        }
        migrated = False
        if "anchors" in existing_cols:
            # New DB, or already migrated — but still drop the FTS5
            # table when it exists under the OLD column list so the
            # subsequent executescript can recreate it with the new
            # schema. We detect via a probe SELECT.
            fts_cols = {
                row[1] for row in self._conn.execute("PRAGMA table_info(tree_nodes_fts)").fetchall()
            }
            if fts_cols and "anchors" not in fts_cols:
                self._conn.execute("DROP TABLE IF EXISTS tree_nodes_fts")
                self._conn.execute("DROP TRIGGER IF EXISTS tree_nodes_fts_ai")
                self._conn.execute("DROP TRIGGER IF EXISTS tree_nodes_fts_ad")
                migrated = True
            return migrated

        # Pre-q6zl schema. Three steps:
        # 1. Add the column with the back-compat default ('').
        # 2. Backfill anchors from path + document_name for every row.
        # 3. Drop the FTS sidecar so the next executescript rebuilds it.
        self._conn.execute("ALTER TABLE tree_nodes ADD COLUMN anchors TEXT NOT NULL DEFAULT ''")
        rows = self._conn.execute(
            """SELECT n.id, n.path, n.parent_id, d.name
                 FROM tree_nodes n
                 JOIN tree_documents d ON d.id = n.document_id"""
        ).fetchall()
        for node_id, path, parent_id, doc_name in rows:
            # harness-8k94: backfill carries parent_section + chapter
            # tokens too. Resolved via the same parent-chain walk
            # ingest_node uses.
            parent_section_path, chapter_path = self._resolve_anchor_ancestors(
                int(parent_id) if parent_id is not None else None
            )
            anchors = _build_anchor_text(
                path=str(path),
                document_name=str(doc_name),
                parent_section=parent_section_path,
                chapter=chapter_path,
            )
            self._conn.execute(
                "UPDATE tree_nodes SET anchors = ? WHERE id = ?", (anchors, int(node_id))
            )
        # FTS sidecar must be rebuilt — old schema has no anchors
        # column, and the tokenizer change requires reindexing anyway.
        self._conn.execute("DROP TABLE IF EXISTS tree_nodes_fts")
        self._conn.execute("DROP TRIGGER IF EXISTS tree_nodes_fts_ai")
        self._conn.execute("DROP TRIGGER IF EXISTS tree_nodes_fts_ad")
        return True

    def _rebuild_fts_index(self) -> None:
        """harness-q6zl: after a migration drops and recreates the FTS5
        sidecar, its inverted index is empty even though tree_nodes
        still carries content. Use FTS5's `rebuild` command so the
        external-content table re-scans tree_nodes and rebuilds the
        index. Note: `SELECT COUNT(*) FROM tree_nodes_fts` on an
        external-content table returns the content-table count, not
        the inverted-index entry count, so it cannot be used as a
        sync check — only the caller's explicit migration flag
        tells us a rebuild is actually needed."""
        self._conn.execute("INSERT INTO tree_nodes_fts(tree_nodes_fts) VALUES('rebuild')")

    # ---------- document CRUD ----------

    def upsert_document(self, *, name: str, source_uri: str | None = None) -> TreeDocument:
        """Idempotent document create — re-runs return the existing row."""
        existing = self._conn.execute(
            "SELECT id, name, source_uri, created_at FROM tree_documents WHERE name = ?",
            (name,),
        ).fetchone()
        if existing is not None:
            return _row_to_document(existing)
        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            "INSERT INTO tree_documents (name, source_uri, created_at) VALUES (?, ?, ?)",
            (name, source_uri, now),
        )
        return TreeDocument(
            id=cur.lastrowid or 0,
            name=name,
            source_uri=source_uri,
            created_at=datetime.fromisoformat(now),
        )

    def get_document(self, document_id: int) -> TreeDocument:
        row = self._conn.execute(
            "SELECT id, name, source_uri, created_at FROM tree_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"tree document {document_id} not found")
        return _row_to_document(row)

    # ---------- node CRUD ----------

    def ingest_node(
        self,
        *,
        document_id: int,
        path: str,
        ordinal: int,
        depth: int,
        node_type: str,
        heading: str,
        body: str = "",
        parent_id: int | None = None,
        embed: bool = True,
    ) -> TreeNode:
        """Insert a node into the tree. Idempotent on `(document_id, path)`.

        `embed=False` writes a structural-only row (chapters,
        parent_section headers, etc.) — no embedding, no contribution
        to search, but the hierarchy is queryable. `embed=True` writes
        the row, embeds `_build_embed_text(...)`, and stores the vector.
        """
        existing = self._conn.execute(
            "SELECT id FROM tree_nodes WHERE document_id = ? AND path = ?",
            (document_id, path),
        ).fetchone()
        if existing is not None:
            return self.get_node(int(existing[0]))

        now = datetime.now(UTC).isoformat()
        embedding_bytes: bytes | None = None
        embedder_id: str | None = None
        embedding_dim: int | None = None
        if embed:
            vec = self.embedder.embed([_build_embed_text(heading=heading, body=body, path=path)])[
                0
            ].astype(np.float32)
            embedding_bytes = vec.tobytes()
            embedder_id = self.embedder.id
            embedding_dim = self.embedder.dimension

        # harness-q6zl + harness-8k94: compute the BM25-indexed anchor
        # string. Needs the document's name + the parent / grandparent
        # paths so queries naming `§5-3` or `Part 91` match without
        # waiting for auto-merge to fire. parent/grandparent come from
        # the existing parent_id chain; one extra SELECT per non-root
        # node is cheap because each ingest call writes one row.
        document = self.get_document(document_id)
        parent_section_path, chapter_path = self._resolve_anchor_ancestors(parent_id)
        anchors = _build_anchor_text(
            path=path,
            document_name=document.name,
            parent_section=parent_section_path,
            chapter=chapter_path,
        )

        cur = self._conn.execute(
            """INSERT INTO tree_nodes (
                document_id, parent_id, path, ordinal, depth, node_type,
                heading, body, anchors, embedding, embedder_id, embedding_dim,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document_id,
                parent_id,
                path,
                ordinal,
                depth,
                node_type,
                heading,
                body,
                anchors,
                embedding_bytes,
                embedder_id,
                embedding_dim,
                now,
            ),
        )
        return self.get_node(cur.lastrowid or 0)

    def get_node(self, node_id: int) -> TreeNode:
        row = self._conn.execute(
            """SELECT id, document_id, parent_id, path, ordinal, depth,
                      node_type, heading, body, created_at
               FROM tree_nodes WHERE id = ?""",
            (node_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"tree node {node_id} not found")
        return _row_to_node(row)

    def get_node_by_path(self, document_id: int, path: str) -> TreeNode | None:
        row = self._conn.execute(
            """SELECT id, document_id, parent_id, path, ordinal, depth,
                      node_type, heading, body, created_at
               FROM tree_nodes WHERE document_id = ? AND path = ?""",
            (document_id, path),
        ).fetchone()
        return _row_to_node(row) if row is not None else None

    def _resolve_anchor_ancestors(self, parent_id: int | None) -> tuple[str | None, str | None]:
        """harness-8k94: walk one or two steps up the parent chain to
        get the parent_section and chapter paths the anchors column
        wants. Returns `(parent_section, chapter)` — either may be
        None for top-level or single-level nodes.

        For a leaf at depth 3 (JO §2-4-3): parent_section = '2-4',
        chapter = '2'. For a leaf at depth 2 (PCG: letter + entry):
        parent_section = chapter = the letter. For a depth-1 node:
        both None (no useful ancestor).

        One SELECT per non-root node; results aren't cached because
        each ingest_node call writes one row and the cost is small."""
        if parent_id is None:
            return None, None
        parent_row = self._conn.execute(
            "SELECT path, parent_id FROM tree_nodes WHERE id = ?",
            (parent_id,),
        ).fetchone()
        if parent_row is None:
            return None, None
        parent_path = str(parent_row[0])
        grandparent_id = parent_row[1]
        if grandparent_id is None:
            # Two-level corpus (PCG: letter → entry). parent_section
            # IS the chapter at this depth.
            return parent_path, parent_path
        grandparent_row = self._conn.execute(
            "SELECT path FROM tree_nodes WHERE id = ?",
            (int(grandparent_id),),
        ).fetchone()
        chapter = str(grandparent_row[0]) if grandparent_row is not None else None
        return parent_path, chapter

    def count_embedded(self) -> int:
        """Number of nodes carrying an embedding — drives ingest progress
        reporting and lets tests assert that the right granularity got
        written without scanning every row."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM tree_nodes WHERE embedding IS NOT NULL"
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def count_mismatched_embeddings(self) -> int:
        """Count embedded nodes whose stored dimension doesn't match the
        current embedder. Mirrors the EpisodicStore / SemanticStore /
        TabularStore surface so `memory rebuild-embeddings` can fold
        this store in (harness-px7k). Structural-only nodes (embedding
        IS NULL) are excluded — they don't need rebuilding."""
        row = self._conn.execute(
            """SELECT COUNT(*) FROM tree_nodes
               WHERE embedding IS NOT NULL AND embedding_dim != ?""",
            (self.embedder.dimension,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def rebuild_embeddings(self) -> tuple[int, int]:
        """Re-embed every embedded node under the current embedder.
        Returns (rows_updated, rows_skipped). Skipped is always 0 —
        there's no superseded equivalent at the tree-node grain — but
        the tuple shape mirrors EpisodicStore / SemanticStore /
        TabularStore so the rebuild plumbing can call them uniformly.

        Useful after a `HARNESS_EMBEDDER_REPO` swap: existing embedded
        nodes are dim-locked to the old model and silently drop out of
        `search()` until rebuilt. Structural-only nodes (embedding IS
        NULL) are left alone — they never participated in retrieval
        and don't need an embedding."""
        rows = self._conn.execute(
            """SELECT id, heading, body, path FROM tree_nodes
               WHERE embedding IS NOT NULL ORDER BY id"""
        ).fetchall()
        if not rows:
            return 0, 0
        texts = [
            _build_embed_text(heading=heading, body=body, path=path)
            for _id, heading, body, path in rows
        ]
        vectors = self.embedder.embed(texts)
        updated = 0
        for (node_id, _heading, _body, _path), vec in zip(rows, vectors, strict=True):
            self._conn.execute(
                """UPDATE tree_nodes
                      SET embedding = ?, embedder_id = ?, embedding_dim = ?
                    WHERE id = ?""",
                (
                    vec.astype(np.float32).tobytes(),
                    self.embedder.id,
                    self.embedder.dimension,
                    node_id,
                ),
            )
            updated += 1
        return updated, 0

    def children(self, node_id: int) -> list[TreeNode]:
        rows = self._conn.execute(
            """SELECT id, document_id, parent_id, path, ordinal, depth,
                      node_type, heading, body, created_at
               FROM tree_nodes WHERE parent_id = ? ORDER BY ordinal, id""",
            (node_id,),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    # ---------- search ----------

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: SearchMode = "hybrid",
        min_score: float = 0.0,
    ) -> list[tuple[TreeNode, float]]:
        """Return up to `k` embedded nodes ranked by `mode`.

        Structural-only nodes (embedding IS NULL) are skipped — they
        don't participate in retrieval. Same `mode` semantics as
        `EpisodicStore.search`: hybrid (dense+BM25 via RRF), dense
        (cosine only), text (BM25 only). `min_score` applies to the
        dense component pre-fusion.

        Returns score values in `mode`-dependent units (RRF score for
        hybrid, cosine for dense, negated BM25 for text). Callers that
        compare across modes shouldn't.
        """
        if mode == "dense":
            return self._search_dense(query, k=k, min_score=min_score)
        if mode == "text":
            return self._search_text(query, k=k)
        return self._search_hybrid(query, k=k, min_score=min_score)

    def _search_dense(
        self, query: str, *, k: int, min_score: float
    ) -> list[tuple[TreeNode, float]]:
        # Embed the query first so a lazy embedder populates its real
        # dimension before the SQL filter reads it (same gotcha as
        # `EpisodicStore._search_dense`, harness-m35).
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        rows = self._conn.execute(
            """SELECT id, document_id, parent_id, path, ordinal, depth,
                      node_type, heading, body, created_at, embedding
               FROM tree_nodes
               WHERE embedding IS NOT NULL AND embedding_dim = ?""",
            (self.embedder.dimension,),
        ).fetchall()
        scored: list[tuple[TreeNode, float]] = []
        for row in rows:
            vec = np.frombuffer(row[10], dtype=np.float32)
            sim = float(np.dot(q_vec, vec))
            if sim >= min_score:
                scored.append((_row_to_node(row[:10]), sim))
        scored.sort(key=lambda t: (-t[1], t[0].id))
        return scored[:k]

    def _search_text(self, query: str, *, k: int) -> list[tuple[TreeNode, float]]:
        # harness-q6zl: preserve `.-:` in query tokens so multi-segment
        # section paths align with the tokenchars config on the FTS5
        # index (both sides see `91.131` as one token rather than
        # `91` + `131`).
        # harness-5yzn: route a parallel anchor-required pass when the
        # query carries explicit section-path tokens (CFR `91.131`, JO
        # `2-4-3`, etc.). The standard OR-mode BM25 pass dilutes anchor
        # signals under long synonym-expanded queries; anchor-matching
        # rows go to the FRONT of the result so a section-by-number
        # request lands its target even when 40+ lay-token clauses
        # compete for the same top-k slots.
        anchor_hits = self._search_anchor_required(query, k=k)
        standard_match = sanitize_fts_query(query, preserve_punctuation=".-:")
        if not standard_match and not anchor_hits:
            return []
        standard_rows = (
            self._conn.execute(
                """SELECT n.id, n.document_id, n.parent_id, n.path, n.ordinal,
                          n.depth, n.node_type, n.heading, n.body, n.created_at,
                          bm25(tree_nodes_fts, 1.0, 1.0, 3.0) AS bm25_score
                   FROM tree_nodes_fts
                   JOIN tree_nodes n ON n.id = tree_nodes_fts.rowid
                   WHERE tree_nodes_fts MATCH ?
                     AND n.embedding IS NOT NULL
                   ORDER BY bm25_score, n.id
                   LIMIT ?""",
                (standard_match, k),
            ).fetchall()
            if standard_match
            else []
        )
        # Merge: anchor-required rows lead, standard top-k fills. Dedup
        # by node id so a section that hit both passes only surfaces
        # once. Anchor scores get the most-negative-BM25 ceiling
        # (-1.0e9) so they outrank everything in the standard list when
        # callers compare scores; the absolute value is arbitrary, but
        # the relative ordering is the load-bearing signal.
        seen_ids: set[int] = set()
        merged: list[tuple[TreeNode, float]] = []
        for node in anchor_hits:
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)
            merged.append((node, 1.0e9))
        # BM25 returns lower=better; negate so callers see higher=better.
        for row in standard_rows:
            node = _row_to_node(row[:10])
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)
            merged.append((node, -float(row[10])))
        return merged[:k]

    def _search_anchor_required(self, query: str, *, k: int) -> list[TreeNode]:
        """harness-5yzn: when the query carries explicit section
        anchors (e.g., `§91.131:` from synonym expansion, or `91.131`
        from a user query), issue a separate FTS5 MATCH that scopes
        to the `anchors` column. Returns the matching nodes in
        BM25-rank order, deduped, up to `k`.

        Empty list when no anchor tokens appear in the query — most
        chat-shaped lay queries won't trip this. The detection regex
        is path-shape-only (digits + dots/hyphens) so prose keywords
        like `Class B` or `IFR` don't accidentally route here.
        """
        anchor_pairs = _extract_query_anchors(query)
        if not anchor_pairs:
            return []
        # Build an OR'd MATCH clause over the anchors column. Each
        # phrase is double-quoted so FTS5 treats the dotted/hyphenated
        # path as a literal token (the tokenizer's tokenchars config
        # already keeps these whole in the index).
        phrases: list[str] = []
        for doc, path in anchor_pairs:
            phrases.append(f'"{path}"')
            if doc is not None:
                phrases.append(f'"{doc}:{path}"')
        clause = " OR ".join(phrases)
        # Column-qualified MATCH: `anchors:` scopes the search to the
        # anchors column only. Avoids picking up cross-references in
        # body text that happen to mention the section number — that
        # noise is what the standard pass is for.
        rows = self._conn.execute(
            """SELECT n.id, n.document_id, n.parent_id, n.path, n.ordinal,
                      n.depth, n.node_type, n.heading, n.body, n.created_at,
                      bm25(tree_nodes_fts) AS bm25_score
               FROM tree_nodes_fts
               JOIN tree_nodes n ON n.id = tree_nodes_fts.rowid
               WHERE tree_nodes_fts MATCH ?
                 AND n.embedding IS NOT NULL
               ORDER BY bm25_score, n.id
               LIMIT ?""",
            (f"anchors:({clause})", k),
        ).fetchall()
        return [_row_to_node(r[:10]) for r in rows]

    def _search_hybrid(
        self, query: str, *, k: int, min_score: float
    ) -> list[tuple[TreeNode, float]]:
        candidate_k = max(k * 4, 20)
        dense_hits = self._search_dense(query, k=candidate_k, min_score=min_score)
        text_hits = self._search_text(query, k=candidate_k)
        if not dense_hits and not text_hits:
            return []
        # harness-5yzn: anchor-required hits (carried by _search_text
        # with the sentinel 1.0e9 score) bypass RRF and lead the fused
        # result. RRF uses rank position, not score, so a rank-0
        # anchor hit only contributes 1/(60+1) ≈ 0.016 — not enough
        # to beat a section that appears mid-pack in both dense AND
        # text. The bypass mirrors the semantics of "if the user
        # named the section, that's the section we surface."
        anchor_hits = [(n, s) for n, s in text_hits if s >= 1.0e9]
        anchor_ids = {n.id for n, _ in anchor_hits}
        node_map: dict[int, TreeNode] = {}
        for node, _ in dense_hits:
            node_map[node.id] = node
        for node, _ in text_hits:
            node_map.setdefault(node.id, node)
        rankings: list[list[int]] = [
            [n.id for n, _ in dense_hits if n.id not in anchor_ids],
            [n.id for n, _ in text_hits if n.id not in anchor_ids],
        ]
        fused = reciprocal_rank_fusion(rankings)
        out: list[tuple[TreeNode, float]] = list(anchor_hits)
        for nid, score in fused:
            if nid in node_map and nid not in anchor_ids:
                out.append((node_map[nid], score))
        return out[:k]


# ---------- row helpers ----------


def _row_to_document(row: tuple) -> TreeDocument:  # type: ignore[type-arg]
    return TreeDocument(
        id=int(row[0]),
        name=str(row[1]),
        source_uri=None if row[2] is None else str(row[2]),
        created_at=datetime.fromisoformat(str(row[3])),
    )


def _row_to_node(row: tuple) -> TreeNode:  # type: ignore[type-arg]
    return TreeNode(
        id=int(row[0]),
        document_id=int(row[1]),
        parent_id=None if row[2] is None else int(row[2]),
        path=str(row[3]),
        ordinal=int(row[4]),
        depth=int(row[5]),
        node_type=str(row[6]),
        heading=str(row[7]),
        body=str(row[8]),
        created_at=datetime.fromisoformat(str(row[9])),
    )


def build_document_tree_store_for_character(
    character_path: Path,
    embedder: Embedder,
    document_trees: tuple[Any, ...],
) -> DocumentTreeStore | None:
    """Build (and populate) a DocumentTreeStore for a character that
    ships `document_trees:` declarations in core.yaml (harness-px7k).

    Returns None when the character ships no document trees — callers
    use this to decide whether to wire the store at all.

    The store lives at `<character_path>/data/document_tree.sqlite`.
    Idempotent on re-launch: re-ingesting from the same source produces
    the same row count because `DocumentTreeStore.ingest_node` returns
    the existing row when `(document_id, path)` collides. Edits to the
    source file land on the next session start without manual rebuild
    — assuming additive edits; renumbers force fresh paths that look
    like new nodes (the structural-edit concern documented on
    harness-2zf4).

    `document_trees` is typed as `tuple[Any, ...]` to dodge an import
    cycle: `harness.character.DocumentTreeSpec` would force this
    module to import character.py at load time. Duck-typed access at
    runtime keeps the cycle clean — each spec needs `.name`,
    `.description`, `.source_path`, `.source_format`.

    JSONL specs dispatch to `iter_jsonl_nodes` with the per-corpus
    config carried on the spec (`jsonl_depth_fields`, etc.). The
    parser at `_load_document_trees` validates that `depth_fields` is
    present when format=jsonl, so by the time we reach here the spec
    is well-formed.
    """
    if not document_trees:
        return None
    db_path = character_path / "data" / "document_tree.sqlite"
    store = DocumentTreeStore(db_path=db_path, embedder=embedder)
    # Local import dodges a hard circular dep at module load (the
    # ingest module imports the store types).
    from harness.store.document_tree_ingest import (
        ingest_blueprints,
        iter_jsonl_nodes,
        iter_markdown_nodes,
    )

    for spec in document_trees:
        if spec.source_format == "markdown":
            ingest_blueprints(
                store,
                document_name=spec.name,
                source_uri=str(spec.source_path),
                blueprints=iter_markdown_nodes(spec.source_path),
            )
        elif spec.source_format == "jsonl":
            ingest_blueprints(
                store,
                document_name=spec.name,
                source_uri=str(spec.source_path),
                blueprints=iter_jsonl_nodes(
                    spec.source_path,
                    depth_fields=spec.jsonl_depth_fields,
                    heading_prefixes=spec.jsonl_heading_prefixes,
                    leaf_heading_field=spec.jsonl_leaf_heading_field,
                    body_field=spec.jsonl_body_field,
                    chunk_index_field=spec.jsonl_chunk_index_field,
                ),
            )
        else:
            # _load_document_trees validates the set; this is a guard
            # against future format additions that forget to wire here.
            raise ValueError(
                f"document_trees[{spec.name!r}]: unknown source_format {spec.source_format!r}"
            )
    return store
