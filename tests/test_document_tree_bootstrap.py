"""Tests for the per-character DocumentTreeStore bootstrap
(harness-px7k).

Covers:
  - `build_document_tree_store_for_character` returns None when the
    character ships no document_trees.
  - A markdown spec builds a populated store with the expected
    document, paths, and embed counts.
  - Idempotent re-runs over the same source don't duplicate nodes.
  - JSONL specs raise a clear NotImplementedError (deferred wiring).
  - `count_mismatched_embeddings` + `rebuild_embeddings` behave like
    their EpisodicStore / TabularStore peers.
  - `StoreBundle.tree` flows through `AssembleContextTool` so a tree-
    slot contract resolves end-to-end against a character-bootstrapped
    store.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.character import DocumentTreeSpec
from harness.retrieval.context_package import AccessPolicy
from harness.retrieval.contract import (
    ContractBundle,
    SlotSpec,
    StoreBundle,
    assemble_package,
)
from harness.store.document_tree import (
    DocumentTreeStore,
    build_document_tree_store_for_character,
)


@dataclass
class _HashEmbedder:
    id: str = "hash-test"
    dimension: int = 8

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            coords = np.array(
                [(digest[i * 4] - 128) / 128.0 for i in range(8)],
                dtype=np.float32,
            )
            norm = np.linalg.norm(coords) or 1.0
            rows.append(coords / norm)
        return np.stack(rows)


@dataclass
class _BiggerHashEmbedder:
    """Different ID + dim than `_HashEmbedder` so existing-row embeddings
    fall into the mismatched bucket. Used to exercise the rebuild path."""

    id: str = "hash-test-16"
    dimension: int = 16

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            coords = np.array(
                [(digest[i * 2] - 128) / 128.0 for i in range(16)],
                dtype=np.float32,
            )
            norm = np.linalg.norm(coords) or 1.0
            rows.append(coords / norm)
        return np.stack(rows)


def _write_markdown(path: Path) -> None:
    path.write_text(
        "# Refund Policy\n"
        "\n"
        "Overview of every refund pathway.\n"
        "\n"
        "## Damaged on arrival\n"
        "\n"
        "Photo evidence required.\n"
        "\n"
        "## Changed mind\n"
        "\n"
        "Thirty day window only.\n",
        encoding="utf-8",
    )


def _build_char_dir_with_tree(tmp_path: Path) -> tuple[Path, DocumentTreeSpec]:
    char_dir = tmp_path / "fake_char"
    (char_dir / "seed_documents").mkdir(parents=True)
    md_path = char_dir / "seed_documents" / "refund_policy.md"
    _write_markdown(md_path)
    spec = DocumentTreeSpec(
        name="refund_policy",
        description="Refund policy manual",
        source_path=md_path,
        source_format="markdown",
    )
    return char_dir, spec


# ---------- builder ----------


def test_builder_returns_none_when_no_document_trees() -> None:
    store = build_document_tree_store_for_character(
        character_path=Path("/nope"),
        embedder=_HashEmbedder(),
        document_trees=(),
    )
    assert store is None


def test_builder_populates_markdown_tree(tmp_path: Path) -> None:
    char_dir, spec = _build_char_dir_with_tree(tmp_path)
    store = build_document_tree_store_for_character(
        character_path=char_dir,
        embedder=_HashEmbedder(),
        document_trees=(spec,),
    )
    assert isinstance(store, DocumentTreeStore)
    # One document, three nodes total (all embed because every heading
    # has body) — the markdown adapter under the new embed-when-body
    # rule treats the top-level + the two child sections as embedded.
    doc = store.upsert_document(name="refund_policy")
    top = store.get_node_by_path(doc.id, "1")
    assert top is not None
    assert top.heading == "Refund Policy"
    child_1 = store.get_node_by_path(doc.id, "1-1")
    assert child_1 is not None
    assert child_1.heading == "Damaged on arrival"
    child_2 = store.get_node_by_path(doc.id, "1-2")
    assert child_2 is not None
    assert child_2.heading == "Changed mind"
    assert store.count_embedded() == 3


def test_builder_is_idempotent(tmp_path: Path) -> None:
    char_dir, spec = _build_char_dir_with_tree(tmp_path)
    first = build_document_tree_store_for_character(
        character_path=char_dir,
        embedder=_HashEmbedder(),
        document_trees=(spec,),
    )
    assert isinstance(first, DocumentTreeStore)
    first_count = first.count_embedded()
    # Same SQLite path on the second build — exercises the dedup path.
    second = build_document_tree_store_for_character(
        character_path=char_dir,
        embedder=_HashEmbedder(),
        document_trees=(spec,),
    )
    assert isinstance(second, DocumentTreeStore)
    assert second.count_embedded() == first_count


def test_bm25_search_matches_parent_section_and_chapter(tmp_path: Path) -> None:
    """harness-8k94: the anchors column carries parent_section and
    chapter tokens in addition to the leaf path. A BM25 query for a
    parent-section path (`5-3`) or a CFR part (`91`) surfaces leaves
    under that parent — useful when the auto-merge orchestrator hasn't
    yet collapsed the cluster, and load-bearing for queries that
    explicitly name a chapter.

    The structural-only parent itself is NOT embedded (its body is
    empty), so search() can't return §5-3 directly. The leaves under
    §5-3 carry `5-3` in their anchors column, so a query for `5-3`
    matches them."""
    from harness.store.document_tree import DocumentTreeStore

    embedder = _HashEmbedder()
    store = DocumentTreeStore(db_path=tmp_path / "extended.sqlite", embedder=embedder)
    doc = store.upsert_document(name="JO_7110.65")
    chapter_node = store.ingest_node(
        document_id=doc.id,
        parent_id=None,
        path="5",
        ordinal=1,
        depth=1,
        node_type="chapter",
        heading="Chapter 5",
        body="",
        embed=False,
    )
    parent_node = store.ingest_node(
        document_id=doc.id,
        parent_id=chapter_node.id,
        path="5-3",
        ordinal=1,
        depth=2,
        node_type="section_group",
        heading="§5-3",
        body="",
        embed=False,
    )
    for ord_idx, leaf_path in enumerate(["5-3-1", "5-3-2", "5-3-3"], start=1):
        store.ingest_node(
            document_id=doc.id,
            parent_id=parent_node.id,
            path=leaf_path,
            ordinal=ord_idx,
            depth=3,
            node_type="section",
            heading=f"IFR clearance {leaf_path}",
            body=f"Body for §{leaf_path}.",
            embed=True,
        )

    # BM25 for the parent-section path returns all three leaves under §5-3.
    parent_hits = store.search("5-3", k=5, mode="text")
    paths = {node.path for node, _ in parent_hits}
    assert {"5-3-1", "5-3-2", "5-3-3"}.issubset(paths), (
        f"expected all §5-3 children for parent-section query, got {paths}"
    )
    # BM25 for the chapter path returns the chapter's leaves too.
    chapter_hits = store.search("5", k=5, mode="text")
    chapter_paths = {node.path for node, _ in chapter_hits}
    assert {"5-3-1", "5-3-2", "5-3-3"}.issubset(chapter_paths), (
        f"expected chapter leaves for chapter query, got {chapter_paths}"
    )


def test_bm25_search_matches_bare_path_anchor(tmp_path: Path) -> None:
    """harness-q6zl: BM25 must surface a section when the query is its
    section number (e.g., '91.131'). Pre-q6zl, FTS5 indexed only
    heading + body and tokenized at . / - boundaries, so '91.131'
    split into '91' and '131' tokens and missed the actual section.
    The fix adds an `anchors` column carrying §<path> + <path> +
    <document> + <document>:<path>, and switches the tokenizer's
    tokenchars to keep multi-segment paths whole."""
    from harness.store.document_tree import DocumentTreeStore

    embedder = _HashEmbedder()
    store = DocumentTreeStore(db_path=tmp_path / "anchors.sqlite", embedder=embedder)
    doc = store.upsert_document(name="CFR_14_Vol2")
    # Body intentionally omits the section number — this is the
    # CFR pattern where sections rarely self-reference. Pre-q6zl this
    # case is a guaranteed BM25 miss; post-fix the anchors column
    # carries the path.
    store.ingest_node(
        document_id=doc.id,
        parent_id=None,
        path="91.131",
        ordinal=1,
        depth=1,
        node_type="section",
        heading="Operations in Class B airspace.",
        body="The operator must receive an ATC clearance from the ATC facility "
        "having jurisdiction for that airspace before operating an aircraft "
        "in that area.",
        embed=True,
    )
    # Add a few neighbor sections so the test demonstrates rank-not-just-presence.
    store.ingest_node(
        document_id=doc.id,
        parent_id=None,
        path="91.130",
        ordinal=2,
        depth=1,
        node_type="section",
        heading="Operations in Class C airspace.",
        body="Two-way radio communications required before entering Class C.",
        embed=True,
    )
    store.ingest_node(
        document_id=doc.id,
        parent_id=None,
        path="91.135",
        ordinal=3,
        depth=1,
        node_type="section",
        heading="Operations in Class A airspace.",
        body="An aircraft within Class A airspace must operate under IFR.",
        embed=True,
    )

    # BM25 query for the bare section path returns the matching node first.
    hits = store.search("91.131", k=3, mode="text")
    assert hits, "BM25 returned no hits for '91.131'"
    top_node, _ = hits[0]
    assert top_node.path == "91.131", (
        f"BM25 should rank §91.131 top for query '91.131', got §{top_node.path}"
    )

    # Same for the §-prefixed form.
    prefixed_hits = store.search("§91.131", k=3, mode="text")
    assert prefixed_hits, "BM25 returned no hits for '§91.131'"
    assert prefixed_hits[0][0].path == "91.131"

    # Document-qualified form also resolves.
    qualified_hits = store.search("CFR_14_Vol2:91.131", k=3, mode="text")
    assert qualified_hits, "BM25 returned no hits for 'CFR_14_Vol2:91.131'"
    assert qualified_hits[0][0].path == "91.131"


def test_anchors_migration_brings_old_store_up_to_date(tmp_path: Path) -> None:
    """harness-q6zl: pre-q6zl stores don't have the anchors column. The
    migration in __init__ should add it, backfill existing rows from
    path + document_name, and rebuild the FTS5 sidecar so BM25
    queries work against pre-existing data without a re-ingest."""
    import sqlite3

    from harness.store.document_tree import DocumentTreeStore

    # Step 1: write a pre-q6zl-shaped DB by hand (no anchors column,
    # old FTS5 schema). Just enough rows + indexes for the migration
    # to find something to backfill.
    db = tmp_path / "legacy.sqlite"
    raw = sqlite3.connect(db)
    raw.execute("""
        CREATE TABLE tree_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            source_uri TEXT,
            created_at TEXT NOT NULL
        )
    """)
    raw.execute("""
        CREATE TABLE tree_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL REFERENCES tree_documents(id),
            parent_id INTEGER REFERENCES tree_nodes(id),
            path TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            node_type TEXT NOT NULL,
            heading TEXT NOT NULL,
            body TEXT NOT NULL DEFAULT '',
            embedding BLOB,
            embedder_id TEXT,
            embedding_dim INTEGER,
            created_at TEXT NOT NULL,
            UNIQUE (document_id, path)
        )
    """)
    raw.execute(
        "INSERT INTO tree_documents (name, source_uri, created_at) "
        "VALUES ('JO_7110.65', NULL, '2026-05-14T12:00:00+00:00')"
    )
    doc_id = int(raw.execute("SELECT id FROM tree_documents").fetchone()[0])
    embedder = _HashEmbedder()
    # Pre-compute an embedding so dense search would still work.
    from harness.store.document_tree import _build_embed_text

    text = _build_embed_text(
        heading="PILOT ACKNOWLEDGMENT", body="Listen for readback.", path="2-4-3"
    )
    vec = embedder.embed([text])[0].astype(np.float32).tobytes()
    raw.execute(
        """INSERT INTO tree_nodes
           (document_id, parent_id, path, ordinal, depth, node_type, heading,
            body, embedding, embedder_id, embedding_dim, created_at)
           VALUES (?, NULL, '2-4-3', 1, 1, 'section', 'PILOT ACKNOWLEDGMENT',
                   'Listen for readback.', ?, ?, ?, '2026-05-14T12:00:00+00:00')""",
        (doc_id, vec, embedder.id, embedder.dimension),
    )
    raw.commit()
    raw.close()

    # Step 2: open with current code. The migration should run and
    # leave the store in a queryable state.
    store = DocumentTreeStore(db_path=db, embedder=embedder)

    # Anchors column exists and is backfilled for the existing row.
    anchors_value = store._conn.execute(
        "SELECT anchors FROM tree_nodes WHERE path = '2-4-3'"
    ).fetchone()[0]
    assert "2-4-3" in anchors_value
    assert "JO_7110.65" in anchors_value

    # BM25 search works post-migration without a re-ingest.
    hits = store.search("2-4-3", k=3, mode="text")
    assert hits
    assert hits[0][0].path == "2-4-3"


def test_anchor_aware_routing_promotes_path_match(tmp_path: Path) -> None:
    """harness-5yzn: when the query carries an explicit section anchor
    (a path-shaped token like `91.131`), the matching section should
    lead the hybrid result regardless of how many neighbor sections
    score well on the other tokens.

    Pre-5yzn, long synonym-expanded queries diluted the rare anchor
    token under common-token noise. The fix detects anchors in the
    query and routes them through a column-scoped FTS pass that
    bypasses RRF fusion."""
    from harness.store.document_tree import DocumentTreeStore

    embedder = _HashEmbedder()
    store = DocumentTreeStore(db_path=tmp_path / "anchor_routing.sqlite", embedder=embedder)
    doc = store.upsert_document(name="CFR_14_Vol2")
    # The target section. Body deliberately doesn't say "Class B" so
    # other Class B-shaped sections will outrank it on prose tokens.
    store.ingest_node(
        document_id=doc.id,
        parent_id=None,
        path="91.131",
        ordinal=1,
        depth=1,
        node_type="section",
        heading="Operations in Class B airspace.",
        body="The operator must receive an ATC clearance before operating.",
        embed=True,
    )
    # Distractor sections that share most of the lay-language tokens
    # in the query (Class B, VFR, airspace, etc.) and would normally
    # outrank §91.131 on raw BM25.
    for path, heading, body in [
        (
            "91.130",
            "Class C operations.",
            "Class B and Class C airspace VFR operations require radio.",
        ),
        ("91.135", "Class A operations.", "Class A airspace VFR Class B comparison airspace."),
        (
            "91.215",
            "Mode C transponder.",
            "Class B Class C airspace VFR Class B Class B airspace requirements.",
        ),
    ]:
        store.ingest_node(
            document_id=doc.id,
            parent_id=None,
            path=path,
            ordinal=int(path.split(".")[-1]),
            depth=1,
            node_type="section",
            heading=heading,
            body=body,
            embed=True,
        )

    # Query carries the §91.131 anchor explicitly (mimicking
    # synonym-expander output). The anchor-required pass picks up the
    # section even though the prose tokens favor the distractors.
    expanded_query = (
        "Class B VFR airspace operations entry requirements "
        "[related: §91.131: Class B VFR entry; permission Class B; clearance Class B]"
    )
    hits = store.search(expanded_query, k=4, mode="hybrid")
    assert hits, "search returned nothing"
    assert hits[0][0].path == "91.131", f"anchor-routed section should lead, got §{hits[0][0].path}"


def test_anchor_extraction_finds_section_paths() -> None:
    """harness-5yzn: _extract_query_anchors pulls path-shaped tokens
    out of arbitrary query strings. Covers the formats the synonym
    expander emits (`§<path>:`) and bare user-typed paths."""
    from harness.store.document_tree import _extract_query_anchors

    # Synonym-expander output.
    anchors = _extract_query_anchors("[related: §91.131: Class B entry]")
    assert (None, "91.131") in anchors

    # Bare anchor in a user query.
    anchors = _extract_query_anchors("What does §2-4-3 say about readback?")
    assert (None, "2-4-3") in anchors

    # Document-qualified anchor.
    anchors = _extract_query_anchors("CFR_14_Vol2:91.131 details")
    assert ("CFR_14_Vol2", "91.131") in anchors

    # Prose-only query — no anchors detected.
    anchors = _extract_query_anchors("How do I enter Class B as a VFR pilot?")
    assert anchors == []


def test_provenance_distinguishes_documents_with_colliding_paths(tmp_path: Path) -> None:
    """harness-mu22: when two documents in the same tree store share a
    path (e.g., JO §2-4-3 and AIM §2-4-3), provenance.record_id must
    distinguish them via the document name. The agent reading the
    package needs to cite the SOURCE, not just the path."""
    from harness.retrieval.contract import (
        ContractBundle,
        SlotSpec,
        StoreBundle,
        assemble_package,
    )
    from harness.store.document_tree import DocumentTreeStore

    embedder = _HashEmbedder()
    store = DocumentTreeStore(db_path=tmp_path / "multi.sqlite", embedder=embedder)

    # Two documents with the same path "2-4-3" but different content.
    doc_a = store.upsert_document(name="JO_7110.65")
    store.ingest_node(
        document_id=doc_a.id,
        parent_id=None,
        path="2-4-3",
        ordinal=1,
        depth=1,
        node_type="section",
        heading="PILOT ACKNOWLEDGMENT",
        body="Controllers ensure pilots acknowledge clearances.",
        embed=True,
    )
    doc_b = store.upsert_document(name="AIM")
    store.ingest_node(
        document_id=doc_b.id,
        parent_id=None,
        path="2-4-3",
        ordinal=1,
        depth=1,
        node_type="section",
        heading="HELICOPTER ROUTES",
        body="Helicopter VFR routes near busy terminals.",
        embed=True,
    )

    contract = ContractBundle(
        role="multi_source",
        intent="Find applicable sections",
        budget_tokens=500,
        slots=(
            SlotSpec(
                name="sections",
                store="tree",
                query_template="clearance acknowledgment",
                required=True,
                min_cardinality=1,
                max_hits=5,
            ),
        ),
    )
    package = assemble_package(
        contract,
        variables={},
        access=AccessPolicy(user_id="mark", role="multi_source"),
        stores=StoreBundle(tree=store),
    )
    assert package.hits, "search returned nothing"
    # Both colliding paths surface; provenance disambiguates by document.
    record_ids = {h.provenance.record_id for h in package.hits}
    assert "JO_7110.65:2-4-3" in record_ids or "AIM:2-4-3" in record_ids, (
        f"expected document-attributed record_id, got {record_ids}"
    )
    # The body for each hit carries [<document>] in the heading line.
    for hit in package.hits:
        document = hit.provenance.record_id.partition(":")[0]
        assert f"[{document}]" in hit.body, (
            f"body missing document tag [{document}]: {hit.body[:80]}"
        )


def test_make_tree_search_fn_returns_anchor_prefixed_principle(tmp_path: Path) -> None:
    """harness-c9fc: make_tree_search_fn wraps a DocumentTreeStore so
    the atc-retrieval eval can score against tree-shaped hits. The
    `principle` string is synthesized as `§<path>` so the existing
    _extract_anchors regex matches both the synthesized form and any
    bare path form a fixture might use."""
    from harness.evals.atc_retrieval import make_tree_search_fn

    char_dir, spec = _build_char_dir_with_tree(tmp_path)
    store = build_document_tree_store_for_character(
        character_path=char_dir,
        embedder=_HashEmbedder(),
        document_trees=(spec,),
    )
    assert isinstance(store, DocumentTreeStore)

    search_fn = make_tree_search_fn(store)
    hits = search_fn("photo evidence damaged", 3)
    assert hits, "search returned nothing"
    # Every hit carries a §-prefixed principle the eval can extract.
    for hit in hits:
        assert hit.principle.startswith("§"), f"principle missing § prefix: {hit.principle!r}"
        # The §-prefix is followed by a path-shaped slug like "1-1" or "1".
        assert hit.principle[1:].replace("-", "").isdigit() or hit.principle[1:].isdigit()


def test_make_tree_search_fn_applies_expander(tmp_path: Path) -> None:
    """The expander callback runs before the search. A query that
    misses without expansion can hit after expansion — same contract
    the episodic SearchFn observes for synonym-driven recovery."""
    from harness.evals.atc_retrieval import make_tree_search_fn

    char_dir, spec = _build_char_dir_with_tree(tmp_path)
    store = build_document_tree_store_for_character(
        character_path=char_dir,
        embedder=_HashEmbedder(),
        document_trees=(spec,),
    )
    assert isinstance(store, DocumentTreeStore)

    seen: list[str] = []

    def _expand(q: str) -> str:
        seen.append(q)
        return q + " evidence"

    search_fn = make_tree_search_fn(store, expand=_expand)
    search_fn("photo", 3)
    assert seen == ["photo"]


def test_builder_populates_jsonl_tree(tmp_path: Path) -> None:
    """harness-k38k: JSONL specs with per-corpus config (depth_fields,
    heading_prefixes, leaf_heading_field) flow through the builder
    and produce a populated tree. Mirrors the ATC ingest's shape:
    structural-only chapter + section_group + embedded leaf sections."""
    import json

    char_dir = tmp_path / "fake_char"
    (char_dir / "corpus").mkdir(parents=True)
    jsonl_path = char_dir / "corpus" / "atc.jsonl"
    rows = [
        {
            "chapter": "2",
            "parent_section": "2-4",
            "section": "2-4-3",
            "title": "VFR Aircraft",
            "chunk_index": 0,
            "body": "VFR rules apply.",
        },
        {
            "chapter": "2",
            "parent_section": "2-4",
            "section": "2-4-4",
            "title": "Special VFR",
            "chunk_index": 0,
            "body": "Special VFR rules.",
        },
    ]
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")

    spec = DocumentTreeSpec(
        name="jo_7110_65",
        description="ATC corpus",
        source_path=jsonl_path,
        source_format="jsonl",
        jsonl_depth_fields=("chapter", "parent_section", "section"),
        jsonl_heading_prefixes=("Chapter ", "§", ""),
        jsonl_leaf_heading_field="title",
    )
    store = build_document_tree_store_for_character(
        character_path=char_dir,
        embedder=_HashEmbedder(),
        document_trees=(spec,),
    )
    assert isinstance(store, DocumentTreeStore)
    # Two leaf sections embedded; the chapter and section_group are
    # structural-only (no body in JSONL → no embed).
    assert store.count_embedded() == 2

    doc = store.upsert_document(name="jo_7110_65")
    leaf = store.get_node_by_path(doc.id, "2-4-3")
    assert leaf is not None
    assert leaf.heading == "VFR Aircraft"
    assert "VFR rules apply." in leaf.body
    # Intermediate heading uses the prefix.
    chapter = store.get_node_by_path(doc.id, "2")
    assert chapter is not None
    assert chapter.heading == "Chapter 2"


# ---------- rebuild_embeddings ----------


def test_rebuild_embeddings_recovers_mismatched_dim(tmp_path: Path) -> None:
    """Build the store with one embedder, then re-open with a different-
    dimension embedder. Mismatch count == embedded-node count;
    rebuild_embeddings drops it to 0."""
    char_dir, spec = _build_char_dir_with_tree(tmp_path)
    original = build_document_tree_store_for_character(
        character_path=char_dir,
        embedder=_HashEmbedder(),
        document_trees=(spec,),
    )
    assert isinstance(original, DocumentTreeStore)
    embedded = original.count_embedded()
    assert embedded > 0

    # Open same DB with a different-dim embedder. All existing
    # embeddings are now dim-mismatched and dropped from search().
    swapped = DocumentTreeStore(
        db_path=char_dir / "data" / "document_tree.sqlite",
        embedder=_BiggerHashEmbedder(),
    )
    assert swapped.count_mismatched_embeddings() == embedded
    # search() returns nothing while dims don't match — the dense path
    # filters on embedding_dim explicitly.
    assert swapped.search("photo evidence", k=3, mode="dense") == []

    updated, skipped = swapped.rebuild_embeddings()
    assert updated == embedded
    assert skipped == 0
    assert swapped.count_mismatched_embeddings() == 0
    # Search works again under the new embedder.
    assert swapped.search("photo evidence", k=3, mode="dense"), (
        "search should return hits after rebuild"
    )


# ---------- StoreBundle.tree end-to-end ----------


def test_tree_store_flows_through_assemble_package(tmp_path: Path) -> None:
    """A tree-slot contract resolves against a character-bootstrapped
    DocumentTreeStore. Validates the entire chain: char spec →
    builder → StoreBundle.tree → orchestrator → package."""
    char_dir, spec = _build_char_dir_with_tree(tmp_path)
    store = build_document_tree_store_for_character(
        character_path=char_dir,
        embedder=_HashEmbedder(),
        document_trees=(spec,),
    )
    assert isinstance(store, DocumentTreeStore)

    contract = ContractBundle(
        role="returns_handler",
        intent="Cite the refund policy section that applies",
        budget_tokens=1000,
        slots=(
            SlotSpec(
                name="policy_section",
                store="tree",
                query_template="photo evidence damaged",
                required=True,
                min_cardinality=1,
                max_hits=2,
            ),
        ),
    )
    package = assemble_package(
        contract,
        variables={},
        access=AccessPolicy(user_id="C9148", role="returns_handler"),
        stores=StoreBundle(tree=store),
    )
    assert package.is_complete, (
        f"required tree slot should be filled but is missing: {package.missing_required_slots}"
    )
    assert package.hits, "tree-slot contract returned no hits"
    # Every hit names tree as its store and carries a `<document>:<path>`
    # provenance (harness-mu22).
    for hit in package.hits:
        assert hit.provenance.store == "tree"
        document, _, path = hit.provenance.record_id.partition(":")
        assert document == "refund_policy"
        assert "-" in path or path.isdigit()
