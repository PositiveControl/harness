"""Integration test for the airton_c1 character post-contract-migration
(harness-k9vg / closes harness-i91c).

Loads `character/airton_c1/` end-to-end via load_character, exercises
the same plumbing the CLI uses to bootstrap a per-character
DocumentTreeStore from `core.yaml.document_trees`, and runs the JO
7110.65 contract through `assemble_package`. Asserts the contract
resolves end-to-end against tree-shaped stores — the worked example
for the data-retrieval-primitives migration of an existing character.

Pairs with `test_returns_handler_character.py` (the original
exemplar, harness-zae0): that one fans out to episodic + tabular +
tree under one contract; this one is the tree-only path airton_c1
uses today.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.character import DocumentTreeSpec, load_character
from harness.retrieval.context_package import AccessPolicy
from harness.retrieval.contract import StoreBundle, assemble_package, load_contract
from harness.store.document_tree import (
    DocumentTreeStore,
    build_document_tree_store_for_character,
)

REPO = Path(__file__).resolve().parents[1]
CHARACTER_PATH = REPO / "character" / "airton_c1"


@dataclass
class _HashEmbedder:
    """Deterministic test embedder; no model load. Same shape as the
    returns_handler test fixture."""

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


# ---------- character config wiring ----------


def test_airton_c1_ships_all_contract_wiring() -> None:
    """Single-glance sanity check that airton_c1 has every piece
    needed to drive the contract orchestrator at session start:
    the flag, the role name, the document_trees spec, and the
    matching contract YAML on disk."""
    character = load_character(CHARACTER_PATH)

    # Forced-call config (harness-jkmk).
    assert character.require_search_memory is False
    assert character.require_assemble_context is True
    assert character.default_contract_role == "airton_c1"

    # Tree spec (harness-j5cs).
    assert len(character.document_trees) == 1
    spec = character.document_trees[0]
    assert spec.name == "jo_7110_65"
    assert spec.source_format == "jsonl"
    if not spec.source_path.exists():
        # FAA corpus JSONL is not checked into the repo (heavy +
        # licensed). Wiring still verified above; skip the disk-
        # presence check when the corpus hasn't been materialized.
        pytest.skip(f"corpus not materialized: {spec.source_path.name}")
    assert spec.jsonl_depth_fields == ("chapter", "parent_section", "section")

    # Contract YAML matches the default_contract_role.
    contract_path = CHARACTER_PATH / "contracts" / f"{character.default_contract_role}.yaml"
    assert contract_path.exists(), (
        f"airton_c1 contract YAML missing at {contract_path}; "
        "the forced assemble_context call would fail at runtime"
    )
    contract = load_contract(contract_path)
    assert contract.role == "airton_c1"
    # The forced call passes {request_summary: <user_message>}; at least
    # one slot must reference it for the contract to be useful.
    template_vars = {
        var
        for slot in contract.slots
        for tpl in (slot.query_template, slot.sql_template)
        if tpl
        for var in _extract_template_vars(tpl)
    }
    assert "request_summary" in template_vars, (
        "contract slots must reference {request_summary} so the forced "
        "assemble_context call's variables map to something useful"
    )


def _extract_template_vars(template: str) -> set[str]:
    """Pull {var} names out of a query/sql template — same parse the
    assemble_context tool uses to spec its description."""
    import re

    return set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template))


# ---------- end-to-end contract resolution ----------


def _build_synthetic_atc_tree(
    tmp_path: Path, embedder: _HashEmbedder
) -> tuple[Path, DocumentTreeStore]:
    """Create a tiny ATC-shaped JSONL (3 sections across 2 chapters)
    and bootstrap a DocumentTreeStore against it using airton_c1's
    real spec config. Returns (fake_char_dir, store). Mirrors the
    fake_char_dir pattern used in test_returns_handler_character.py
    so the SQLite lands in tmp_path rather than clobbering the
    character's data dir."""
    fake_char_dir = tmp_path / "fake_airton_c1"
    corpus_dir = fake_char_dir / "corpus" / "chunks"
    corpus_dir.mkdir(parents=True)
    jsonl_path = corpus_dir / "jo_7110_65.jsonl"
    rows = [
        {
            "chapter": "2",
            "parent_section": "2-4",
            "section": "2-4-3",
            "title": "PILOT ACKNOWLEDGMENT/READ BACK",
            "chunk_index": 0,
            "body": (
                "Ensure pilots acknowledge all Air Traffic Control "
                "clearances and/or instructions. Listen for the readback "
                "of any hold short instruction and a correct call sign."
            ),
        },
        {
            "chapter": "5",
            "parent_section": "5-5",
            "section": "5-5-4",
            "title": "MINIMA",
            "chunk_index": 0,
            "body": (
                "Wake turbulence separation minima behind a heavy or "
                "B757. Apply 4 miles minimum behind a heavy aircraft."
            ),
        },
        {
            "chapter": "5",
            "parent_section": "5-2",
            "section": "5-2-17",
            "title": "ALTITUDE CONFIRMATION-NON-MODE C",
            "chunk_index": 0,
            "body": (
                "Request a pilot to confirm assigned altitude when an "
                "altitude readback discrepancy exists."
            ),
        },
    ]
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")

    real_spec = load_character(CHARACTER_PATH).document_trees[0]
    spec = DocumentTreeSpec(
        name=real_spec.name,
        description=real_spec.description,
        source_path=jsonl_path,
        source_format=real_spec.source_format,
        jsonl_depth_fields=real_spec.jsonl_depth_fields,
        jsonl_heading_prefixes=real_spec.jsonl_heading_prefixes,
        jsonl_leaf_heading_field=real_spec.jsonl_leaf_heading_field,
    )
    store = build_document_tree_store_for_character(
        character_path=fake_char_dir,
        embedder=embedder,
        document_trees=(spec,),
    )
    assert isinstance(store, DocumentTreeStore)
    return fake_char_dir, store


def test_airton_c1_contract_resolves_against_tree_store(tmp_path: Path) -> None:
    """The acceptance criterion for harness-k9vg: airton_c1's contract
    resolves end-to-end against a DocumentTreeStore bootstrapped from
    its own spec. All hits carry tree provenance, the record_id is a
    section path (e.g. '2-4-3'), and a typical controller question
    pulls the right section into the package."""
    embedder = _HashEmbedder()
    _, store = _build_synthetic_atc_tree(tmp_path, embedder)

    contract = load_contract(CHARACTER_PATH / "contracts" / "airton_c1.yaml")
    package = assemble_package(
        contract,
        variables={"request_summary": "pilot read back wrong altitude"},
        access=AccessPolicy(user_id="mark", role="airton_c1"),
        stores=StoreBundle(tree=store),
    )
    assert package.is_complete, f"missing: {package.missing_required_slots}"
    assert package.hits, "contract returned no hits"

    # All hits are tree-shaped (this is a tree-only contract).
    for hit in package.hits:
        assert hit.provenance.store == "tree"
        assert hit.provenance.method == "hybrid"
        # harness-mu22: record_id is `<document>:<path>`. For airton_c1
        # the document is `jo_7110_65`; the path is a section slug.
        document, _, path = hit.provenance.record_id.partition(":")
        assert document == "jo_7110_65", (
            f"tree provenance should attribute jo_7110_65, got {document!r}"
        )
        assert path.replace("-", "").isdigit(), (
            f"path component should be a section slug, got {path!r}"
        )


def test_airton_c1_contract_returns_tree_body_text(tmp_path: Path) -> None:
    """A query whose semantics overlap a synthetic section's body
    should surface that section's content in the package — proves the
    package isn't just shape-correct but actually carries useful text
    the model can read."""
    embedder = _HashEmbedder()
    _, store = _build_synthetic_atc_tree(tmp_path, embedder)

    contract = load_contract(CHARACTER_PATH / "contracts" / "airton_c1.yaml")
    package = assemble_package(
        contract,
        variables={"request_summary": "wake turbulence separation behind heavy aircraft"},
        access=AccessPolicy(user_id="mark", role="airton_c1"),
        stores=StoreBundle(tree=store),
    )
    assert package.is_complete
    # At least one hit's body should mention the topic the synthetic
    # §5-5-4 covers — confirms tree-search routed the query to the
    # right section, not just that some hit was returned.
    bodies = " ".join(h.body for h in package.hits)
    assert any(
        keyword in bodies.lower() for keyword in ("wake turbulence", "heavy aircraft", "separation")
    ), f"package body doesn't mention the query topic: {bodies[:200]}"
