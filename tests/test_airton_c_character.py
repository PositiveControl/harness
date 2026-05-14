"""Integration test for the airton_c character post-contract-migration
(harness-q4tj / closes harness-by1j).

airton_c is the third character on the data-retrieval-primitives
stack (after returns_handler and airton_c1) and the first one with a
multi-document tree (5 NAS corpora: JO, AIM, CFR Vol 1, CFR Vol 2,
PCG; PHAK stays in episodic per the harness-by1j hybrid decision).
Asserts the full wiring resolves a multi-document contract with
source-attributed provenance (harness-mu22), and that anchor-aware
routing (harness-5yzn) puts the right section at the top when the
query carries an explicit anchor.

Pairs with:
  - test_returns_handler_character.py (3-store fan-out)
  - test_airton_c1_character.py (tree-only, single document)
This one covers the multi-document tree case.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.character import DocumentTreeSpec, load_character
from harness.retrieval.context_package import AccessPolicy
from harness.retrieval.contract import StoreBundle, assemble_package, load_contract
from harness.store.document_tree import (
    DocumentTreeStore,
    build_document_tree_store_for_character,
)
from harness.store.episodic import EpisodicStore

REPO = Path(__file__).resolve().parents[1]
CHARACTER_PATH = REPO / "character" / "airton_c"


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


# ---------- character config wiring ----------


def test_airton_c_ships_all_contract_wiring() -> None:
    """Single-glance sanity that airton_c has every piece needed to
    drive the contract orchestrator at session start: the forced-call
    flags, the role name, all 5 document_trees specs, and the
    matching contract YAML on disk."""
    character = load_character(CHARACTER_PATH)

    # Forced-call config.
    assert character.require_search_memory is False
    assert character.require_assemble_context is True
    assert character.default_contract_role == "airton_c"

    # 5 hierarchical corpora as document_trees specs (PHAK stays in
    # episodic per the harness-by1j hybrid decision).
    spec_names = {spec.name for spec in character.document_trees}
    assert spec_names == {"JO_7110.65", "AIM", "CFR_14_Vol1", "CFR_14_Vol2", "PCG"}

    # Every spec is jsonl format with a leaf_heading_field of "title"
    # and a non-empty depth_fields tuple. PCG is 2-level; the others
    # are 3-level — both shapes pass through.
    for spec in character.document_trees:
        assert spec.source_format == "jsonl"
        assert spec.jsonl_leaf_heading_field == "title"
        assert len(spec.jsonl_depth_fields) >= 2
        assert spec.source_path.exists(), f"missing corpus: {spec.source_path}"

    # Contract YAML matches the default_contract_role and references
    # the {request_summary} variable the forced-call mechanism fills.
    contract_path = CHARACTER_PATH / "contracts" / f"{character.default_contract_role}.yaml"
    assert contract_path.exists()
    contract = load_contract(contract_path)
    assert contract.role == "airton_c"
    template_vars = {
        var
        for slot in contract.slots
        for tpl in (slot.query_template, slot.sql_template)
        if tpl
        for var in _extract_template_vars(tpl)
    }
    assert "request_summary" in template_vars


def _extract_template_vars(template: str) -> set[str]:
    import re

    return set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template))


# ---------- multi-document contract resolution ----------


def _build_synthetic_nas_tree(
    tmp_path: Path, embedder: _HashEmbedder
) -> tuple[Path, DocumentTreeStore]:
    """Create two synthetic NAS-shaped JSONL files (JO + CFR) with
    intentionally OVERLAPPING paths so the harness-mu22 source-
    attribution can be observed. Bootstraps a DocumentTreeStore with
    both documents loaded via airton_c's real spec config."""
    fake_char = tmp_path / "fake_airton_c"
    chunks_dir = fake_char / "corpus" / "chunks"
    chunks_dir.mkdir(parents=True)

    # JO 7110.65 sample row (hyphenated path).
    jo_path = chunks_dir / "jo_7110_65.jsonl"
    jo_rows = [
        {
            "chapter": "2",
            "parent_section": "2-4",
            "section": "2-4-3",
            "title": "PILOT ACKNOWLEDGMENT/READ BACK",
            "chunk_index": 0,
            "body": "Ensure pilots acknowledge all ATC clearances.",
        },
    ]
    with jo_path.open("w", encoding="utf-8") as fp:
        for row in jo_rows:
            fp.write(json.dumps(row) + "\n")

    # CFR Vol 2 sample row — dotted path, with one row whose section
    # number "91.131" sits in CFR's namespace. JO 7110.65 doesn't use
    # this path, but a parallel document could; the test proves the
    # provenance disambiguates by document.
    cfr_path = chunks_dir / "cfr_14_vol2.jsonl"
    cfr_rows = [
        {
            "chapter": "91",
            "parent_section": "91",
            "section": "91.131",
            "title": "Operations in Class B airspace.",
            "chunk_index": 0,
            "body": "The operator must receive an ATC clearance.",
        },
    ]
    with cfr_path.open("w", encoding="utf-8") as fp:
        for row in cfr_rows:
            fp.write(json.dumps(row) + "\n")

    # Build specs from the real airton_c declarations but pointed at
    # the synthetic JSONL files in tmp_path.
    real_specs = {spec.name: spec for spec in load_character(CHARACTER_PATH).document_trees}
    specs = (
        DocumentTreeSpec(
            name="JO_7110.65",
            description=real_specs["JO_7110.65"].description,
            source_path=jo_path,
            source_format="jsonl",
            jsonl_depth_fields=real_specs["JO_7110.65"].jsonl_depth_fields,
            jsonl_heading_prefixes=real_specs["JO_7110.65"].jsonl_heading_prefixes,
            jsonl_leaf_heading_field=real_specs["JO_7110.65"].jsonl_leaf_heading_field,
        ),
        DocumentTreeSpec(
            name="CFR_14_Vol2",
            description=real_specs["CFR_14_Vol2"].description,
            source_path=cfr_path,
            source_format="jsonl",
            jsonl_depth_fields=real_specs["CFR_14_Vol2"].jsonl_depth_fields,
            jsonl_heading_prefixes=real_specs["CFR_14_Vol2"].jsonl_heading_prefixes,
            jsonl_leaf_heading_field=real_specs["CFR_14_Vol2"].jsonl_leaf_heading_field,
        ),
    )
    store = build_document_tree_store_for_character(
        character_path=fake_char,
        embedder=embedder,
        document_trees=specs,
    )
    assert isinstance(store, DocumentTreeStore)
    return fake_char, store


def test_airton_c_contract_resolves_with_source_attribution(tmp_path: Path) -> None:
    """Multi-document tree contract resolves and every hit carries
    `<document>:<path>` in provenance (harness-mu22). Proves the
    full wiring works for airton_c's 5-corpus shape — both
    hyphenated JO-style and dotted CFR-style paths surface with
    distinct source attribution."""
    embedder = _HashEmbedder()
    _, store = _build_synthetic_nas_tree(tmp_path, embedder)
    contract = load_contract(CHARACTER_PATH / "contracts" / "airton_c.yaml")

    # The contract carries an optional `handbook_context` episodic
    # slot for PHAK (harness-by1j hybrid setup). The orchestrator
    # raises if the store is missing even for optional slots, so we
    # wire an empty episodic store — it produces zero hits, which is
    # what an empty handbook would.
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    package = assemble_package(
        contract,
        variables={"request_summary": "pilot readback clearance"},
        access=AccessPolicy(user_id="mark", role="airton_c"),
        stores=StoreBundle(tree=store, episodic=episodic),
    )
    assert package.is_complete, f"missing: {package.missing_required_slots}"
    tree_hits = [h for h in package.hits if h.provenance.store == "tree"]
    assert tree_hits, "no tree hits"
    for hit in tree_hits:
        document, separator, path = hit.provenance.record_id.partition(":")
        assert separator == ":", (
            f"provenance should be `<doc>:<path>`, got {hit.provenance.record_id!r}"
        )
        assert document in {"JO_7110.65", "CFR_14_Vol2"}, f"unexpected document {document!r}"
        assert path, "path should be non-empty"


def test_airton_c_body_carries_document_tag(tmp_path: Path) -> None:
    """The rendered hit body includes `[<document>]` so the agent
    reading the package sees source attribution inline (harness-mu22),
    not just in provenance metadata."""
    embedder = _HashEmbedder()
    _, store = _build_synthetic_nas_tree(tmp_path, embedder)
    contract = load_contract(CHARACTER_PATH / "contracts" / "airton_c.yaml")

    # The contract carries an optional `handbook_context` episodic
    # slot for PHAK (harness-by1j hybrid setup). The orchestrator
    # raises if the store is missing even for optional slots, so we
    # wire an empty episodic store — it produces zero hits, which is
    # what an empty handbook would.
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    package = assemble_package(
        contract,
        variables={"request_summary": "pilot readback clearance"},
        access=AccessPolicy(user_id="mark", role="airton_c"),
        stores=StoreBundle(tree=store, episodic=episodic),
    )
    tree_hits = [h for h in package.hits if h.provenance.store == "tree"]
    assert tree_hits
    for hit in tree_hits:
        document = hit.provenance.record_id.partition(":")[0]
        assert f"[{document}]" in hit.body, f"body missing [{document}] tag: {hit.body[:120]}"


def test_airton_c_auto_merge_promotes_parent_when_siblings_cluster(tmp_path: Path) -> None:
    """harness-0t7a: when 2+ sibling-leaf hits under the same parent
    appear in the slot's results, the contract orchestrator replaces
    the cluster with a single hit at the parent path. Exercises the
    auto_merge: true flag on airton_c's applicable_section slot
    against a synthetic JO §5-3 corpus where the fixture expects the
    parent path `5-3`."""
    embedder = _HashEmbedder()
    fake_char = tmp_path / "fake_merge"
    chunks_dir = fake_char / "corpus" / "chunks"
    chunks_dir.mkdir(parents=True)
    jo_path = chunks_dir / "jo_7110_65.jsonl"
    # Three sibling leaves under §5-3 — `auto_merge` fires when 2+
    # show up. Headings are intentionally similar so the embedder
    # ranks them together.
    jo_rows = [
        {
            "chapter": "5",
            "parent_section": "5-3",
            "section": "5-3-1",
            "title": "IFR clearance limit issuance",
            "chunk_index": 0,
            "body": "Issue an IFR clearance limit on initial contact.",
        },
        {
            "chapter": "5",
            "parent_section": "5-3",
            "section": "5-3-2",
            "title": "IFR clearance limit amendment",
            "chunk_index": 0,
            "body": "Amend an IFR clearance limit before reaching the fix.",
        },
        {
            "chapter": "5",
            "parent_section": "5-3",
            "section": "5-3-3",
            "title": "IFR clearance limit holding",
            "chunk_index": 0,
            "body": "When a hold is anticipated at the clearance limit.",
        },
    ]
    with jo_path.open("w", encoding="utf-8") as fp:
        for row in jo_rows:
            fp.write(json.dumps(row) + "\n")
    real_spec = {spec.name: spec for spec in load_character(CHARACTER_PATH).document_trees}[
        "JO_7110.65"
    ]
    spec = DocumentTreeSpec(
        name="JO_7110.65",
        description=real_spec.description,
        source_path=jo_path,
        source_format="jsonl",
        jsonl_depth_fields=real_spec.jsonl_depth_fields,
        jsonl_heading_prefixes=real_spec.jsonl_heading_prefixes,
        jsonl_leaf_heading_field=real_spec.jsonl_leaf_heading_field,
    )
    store = build_document_tree_store_for_character(
        character_path=fake_char,
        embedder=embedder,
        document_trees=(spec,),
    )
    assert isinstance(store, DocumentTreeStore)

    contract = load_contract(CHARACTER_PATH / "contracts" / "airton_c.yaml")
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    package = assemble_package(
        contract,
        variables={"request_summary": "IFR clearance limit"},
        access=AccessPolicy(user_id="mark", role="airton_c"),
        stores=StoreBundle(tree=store, episodic=episodic),
    )
    paths = [h.provenance.record_id.partition(":")[2] for h in package.hits]
    # The cluster of §5-3-1/§5-3-2/§5-3-3 collapses to the parent
    # §5-3. The fixture's expected_anchor `5-3` then matches.
    assert "5-3" in paths, f"expected promoted parent §5-3, got {paths}"


def test_airton_c_anchor_query_routes_to_named_section(tmp_path: Path) -> None:
    """harness-5yzn: when the contract's slot query includes an
    explicit section anchor (e.g. via query_synonyms.yaml expansion
    or a user-typed `§91.131`), the anchor-routed pass surfaces that
    section first regardless of how many neighbor sections match
    the prose tokens."""
    embedder = _HashEmbedder()
    _, store = _build_synthetic_nas_tree(tmp_path, embedder)

    # Direct search with an anchor token in the query — exercises the
    # anchor-routed path without needing the synonyms expander.
    hits = store.search("Class B airspace operations §91.131 entry rules", k=3, mode="hybrid")
    assert hits, "search returned nothing"
    top_node = hits[0][0]
    assert top_node.path == "91.131", (
        f"anchor-routed query should lead with §91.131, got §{top_node.path}"
    )
