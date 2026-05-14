"""Tests for harness.store.document_tree_ingest (harness-2zf4).

Real `DocumentTreeStore` over `tmp_path` SQLite. Deterministic
`_HashEmbedder` mirrors the pattern in `test_document_tree_store.py`."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.store.document_tree import DocumentTreeStore
from harness.store.document_tree_ingest import (
    NodeBlueprint,
    ingest_blueprints,
    iter_jsonl_nodes,
    iter_markdown_nodes,
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


def _new_store(tmp_path: Path) -> DocumentTreeStore:
    return DocumentTreeStore(db_path=tmp_path / "tree.sqlite", embedder=_HashEmbedder())


# ---------- markdown adapter ----------


def test_markdown_three_levels_emits_positional_paths(tmp_path: Path) -> None:
    md = tmp_path / "policy.md"
    md.write_text(
        "# Refund Policy\n"
        "\n"
        "Overview prose.\n"
        "\n"
        "## Damaged on arrival\n"
        "\n"
        "Photo evidence required.\n"
        "\n"
        "### With photo\n"
        "\n"
        "Auto-approve.\n"
        "\n"
        "### Without photo\n"
        "\n"
        "Escalate to human.\n"
        "\n"
        "## Changed mind\n"
        "\n"
        "Within 30 days only.\n",
        encoding="utf-8",
    )
    blueprints = list(iter_markdown_nodes(md))
    paths = [b.path for b in blueprints]
    assert paths == ["1", "1-1", "1-1-1", "1-1-2", "1-2"]

    by_path = {b.path: b for b in blueprints}
    assert by_path["1"].heading == "Refund Policy"
    assert by_path["1"].body == "Overview prose."
    assert by_path["1"].embed is True
    assert by_path["1-1"].parent_path == "1"
    assert by_path["1-1-2"].parent_path == "1-1"
    assert by_path["1-1-2"].depth == 3
    assert by_path["1-1-2"].body == "Escalate to human."


def test_markdown_no_body_is_structural(tmp_path: Path) -> None:
    md = tmp_path / "stub.md"
    md.write_text(
        "# Title\n\n## Empty container\n\n## With body\n\nLeaf prose.\n", encoding="utf-8"
    )
    blueprints = {b.path: b for b in iter_markdown_nodes(md)}
    # The container has no body, so embed=False; the leaf with body embeds.
    assert blueprints["1-1"].embed is False
    assert blueprints["1-1"].body == ""
    assert blueprints["1-2"].embed is True
    assert blueprints["1-2"].body == "Leaf prose."
    # The top-level heading has no body either.
    assert blueprints["1"].embed is False


def test_markdown_skipped_depth_pads_synthetic_ancestor(tmp_path: Path) -> None:
    # `#` jumps to `###` with no `##` between. The parser tolerates it
    # by treating the missing intermediate as implicit ordinal 1 — no
    # synthetic node is emitted (we don't have a heading to give it),
    # but the path stays well-formed.
    md = tmp_path / "skip.md"
    md.write_text("# Top\n\nTop body.\n\n### Deep\n\nDeep body.\n", encoding="utf-8")
    blueprints = list(iter_markdown_nodes(md))
    paths = [b.path for b in blueprints]
    assert paths == ["1", "1-1-1"]
    assert blueprints[1].depth == 3
    # parent_path points at the missing `1-1` — the driver will fail
    # cleanly on ingest, which is the right outcome (encourages
    # authors to add the missing heading rather than silently
    # generating synthetic structural nodes).
    assert blueprints[1].parent_path == "1-1"


def test_markdown_empty_file_yields_nothing(tmp_path: Path) -> None:
    md = tmp_path / "empty.md"
    md.write_text("", encoding="utf-8")
    assert list(iter_markdown_nodes(md)) == []


def test_markdown_prose_before_first_heading_is_dropped(tmp_path: Path) -> None:
    md = tmp_path / "frontmatter.md"
    md.write_text(
        "Some intro text that lives outside any heading.\n"
        "\n"
        "# Real start\n"
        "\n"
        "Body under the first heading.\n",
        encoding="utf-8",
    )
    blueprints = list(iter_markdown_nodes(md))
    assert len(blueprints) == 1
    assert blueprints[0].heading == "Real start"
    assert blueprints[0].body == "Body under the first heading."


def test_markdown_siblings_increment_ordinal(tmp_path: Path) -> None:
    md = tmp_path / "siblings.md"
    md.write_text(
        "# A\n\nbody a\n\n# B\n\nbody b\n\n# C\n\nbody c\n",
        encoding="utf-8",
    )
    blueprints = list(iter_markdown_nodes(md))
    assert [b.ordinal for b in blueprints] == [1, 2, 3]
    assert [b.path for b in blueprints] == ["1", "2", "3"]


# ---------- jsonl adapter ----------


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def test_jsonl_atc_shape(tmp_path: Path) -> None:
    src = tmp_path / "atc.jsonl"
    _write_jsonl(
        src,
        [
            {
                "chapter": "2",
                "parent_section": "2-4",
                "section": "2-4-3",
                "title": "VFR Aircraft",
                "chunk_index": 1,
                "body": "Second paragraph.",
            },
            {
                "chapter": "2",
                "parent_section": "2-4",
                "section": "2-4-3",
                "title": "VFR Aircraft",
                "chunk_index": 0,
                "body": "First paragraph.",
            },
            {
                "chapter": "2",
                "parent_section": "2-4",
                "section": "2-4-4",
                "title": "Special VFR",
                "chunk_index": 0,
                "body": "Special VFR body.",
            },
            {
                "chapter": "3",
                "parent_section": "3-1",
                "section": "3-1-1",
                "title": "Intro",
                "chunk_index": 0,
                "body": "Chapter 3 intro.",
            },
        ],
    )
    blueprints = list(
        iter_jsonl_nodes(
            src,
            depth_fields=("chapter", "parent_section", "section"),
            heading_prefixes=("Chapter ", "§", ""),
            leaf_heading_field="title",
        )
    )
    paths = [b.path for b in blueprints]
    # Order: chapter 2, group 2-4, sections 2-4-3 / 2-4-4, then chapter 3, group 3-1, section 3-1-1.
    assert paths == ["2", "2-4", "2-4-3", "2-4-4", "3", "3-1", "3-1-1"]

    by_path = {b.path: b for b in blueprints}
    # Intermediates structural-only.
    assert by_path["2"].embed is False
    assert by_path["2"].heading == "Chapter 2"
    assert by_path["2-4"].embed is False
    assert by_path["2-4"].heading == "§2-4"
    # Leaf embeds, uses title field, body ordered by chunk_index.
    assert by_path["2-4-3"].embed is True
    assert by_path["2-4-3"].heading == "VFR Aircraft"
    assert by_path["2-4-3"].body == "First paragraph.\n\nSecond paragraph."
    # Parent linkage.
    assert by_path["2-4-3"].parent_path == "2-4"
    assert by_path["2-4"].parent_path == "2"
    assert by_path["2"].parent_path is None


def test_jsonl_rejects_empty_depth_fields(tmp_path: Path) -> None:
    src = tmp_path / "empty.jsonl"
    src.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="depth_fields"):
        list(iter_jsonl_nodes(src, depth_fields=()))


def test_jsonl_skips_rows_missing_hierarchy_fields(tmp_path: Path) -> None:
    src = tmp_path / "partial.jsonl"
    _write_jsonl(
        src,
        [
            {"chapter": "2", "parent_section": "2-4", "section": "2-4-3", "body": "valid"},
            {"chapter": "2", "parent_section": "", "section": "2-4-4", "body": "missing parent"},
            {"chapter": "", "parent_section": "x", "section": "y", "body": "missing chapter"},
        ],
    )
    blueprints = list(
        iter_jsonl_nodes(
            src,
            depth_fields=("chapter", "parent_section", "section"),
        )
    )
    leaf_paths = [b.path for b in blueprints if b.depth == 3]
    assert leaf_paths == ["2-4-3"]


# ---------- driver ----------


def test_ingest_blueprints_writes_tree(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    md = tmp_path / "policy.md"
    md.write_text(
        "# Refund Policy\n\nOverview.\n\n## Damaged\n\nPhoto required.\n",
        encoding="utf-8",
    )
    counts = ingest_blueprints(
        store,
        document_name="refund_policy",
        source_uri=str(md),
        blueprints=iter_markdown_nodes(md),
    )
    assert counts == {"chapter": 1, "section_group": 1}

    doc = store.upsert_document(name="refund_policy")
    top = store.get_node_by_path(doc.id, "1")
    assert top is not None
    assert top.heading == "Refund Policy"

    child = store.get_node_by_path(doc.id, "1-1")
    assert child is not None
    assert child.parent_id == top.id


def test_ingest_blueprints_idempotent(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    md = tmp_path / "policy.md"
    md.write_text("# A\n\nbody\n\n## B\n\nmore\n", encoding="utf-8")

    ingest_blueprints(
        store,
        document_name="d",
        source_uri=None,
        blueprints=iter_markdown_nodes(md),
    )
    first_count = store.count_embedded()
    ingest_blueprints(
        store,
        document_name="d",
        source_uri=None,
        blueprints=iter_markdown_nodes(md),
    )
    assert store.count_embedded() == first_count


def test_ingest_blueprints_search_returns_embedded_leaf(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    md = tmp_path / "policy.md"
    md.write_text(
        "# Refund Policy\n\n"
        "## Damaged on arrival\n\nPhoto evidence approves.\n\n"
        "## Changed mind\n\nThirty day window only.\n",
        encoding="utf-8",
    )
    ingest_blueprints(
        store,
        document_name="refund_policy",
        source_uri=None,
        blueprints=iter_markdown_nodes(md),
    )
    hits = store.search("photo evidence", k=2, mode="text")
    assert hits, "search returned nothing for an exact-keyword query"
    top_paths = [h[0].path for h in hits]
    assert "1-1" in top_paths


def test_ingest_blueprints_rejects_out_of_order_blueprints(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    bad = [
        NodeBlueprint(
            parent_path="missing",
            path="1",
            ordinal=1,
            depth=2,
            node_type="section",
            heading="orphan",
            body="b",
            embed=True,
        )
    ]
    with pytest.raises(ValueError, match="hasn't been emitted yet"):
        ingest_blueprints(store, document_name="d", source_uri=None, blueprints=bad)


def test_ingest_blueprints_full_jsonl_roundtrip(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    src = tmp_path / "atc.jsonl"
    _write_jsonl(
        src,
        [
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
        ],
    )
    counts = ingest_blueprints(
        store,
        document_name="JO_7110.65",
        source_uri=str(src),
        blueprints=iter_jsonl_nodes(
            src,
            depth_fields=("chapter", "parent_section", "section"),
            heading_prefixes=("Chapter ", "§", ""),
            leaf_heading_field="title",
        ),
    )
    assert counts["chapter"] == 1
    assert counts["section_group"] == 1
    assert counts["section"] == 2
    # Only embedded leaves contribute to count_embedded.
    assert store.count_embedded() == 2
