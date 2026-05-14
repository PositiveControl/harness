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
    # Every hit names tree as its store and carries a section path id.
    for hit in package.hits:
        assert hit.provenance.store == "tree"
        assert "-" in hit.provenance.record_id or hit.provenance.record_id.isdigit()
