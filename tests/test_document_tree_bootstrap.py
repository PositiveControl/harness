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
import pytest

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


def test_builder_raises_on_jsonl_spec(tmp_path: Path) -> None:
    """JSONL format is recognized by the spec parser but not yet wired
    through the character bootstrap. The builder raises a clear
    NotImplementedError so users see the deferred-wiring story rather
    than getting a confusing markdown-parse error."""
    char_dir = tmp_path / "fake_char"
    (char_dir / "corpus").mkdir(parents=True)
    jsonl_path = char_dir / "corpus" / "source.jsonl"
    jsonl_path.write_text("{}\n", encoding="utf-8")
    spec = DocumentTreeSpec(
        name="corpus",
        description="JSONL corpus",
        source_path=jsonl_path,
        source_format="jsonl",
    )
    with pytest.raises(NotImplementedError, match="jsonl source_format"):
        build_document_tree_store_for_character(
            character_path=char_dir,
            embedder=_HashEmbedder(),
            document_trees=(spec,),
        )


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
