"""Tests for harness.retrieval.contract (harness-s3f7 / Phase 3).

Real SQLite for the stores; deterministic hash-based fake embedder.
The orchestrator is pure; the end-to-end test demonstrates one worked
contract (returns-handler synthetic) pulling from all three shaped
stores."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.retrieval.context_package import AccessPolicy
from harness.retrieval.contract import (
    ContractBundle,
    SlotSpec,
    StoreBundle,
    assemble_package,
    load_contract,
)
from harness.store.document_tree import DocumentTreeStore
from harness.store.episodic import EpisodicStore
from harness.store.tabular import TableSchema, TabularStore


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


# ---------- SlotSpec validation ----------


def test_slot_spec_rejects_unknown_store() -> None:
    with pytest.raises(ValueError, match="store must be one of"):
        SlotSpec(name="x", store="graphdb", query_template="q")


def test_slot_spec_tabular_requires_sql_template_and_table_name() -> None:
    with pytest.raises(ValueError, match="needs `sql_template`"):
        SlotSpec(name="x", store="tabular", table_name="returns")
    with pytest.raises(ValueError, match="needs `table_name`"):
        SlotSpec(name="x", store="tabular", sql_template="SELECT 1")


def test_slot_spec_episodic_requires_query_template() -> None:
    with pytest.raises(ValueError, match="needs `query_template`"):
        SlotSpec(name="x", store="episodic")


def test_slot_spec_rejects_zero_max_hits() -> None:
    with pytest.raises(ValueError, match="max_hits must be >= 1"):
        SlotSpec(name="x", store="episodic", query_template="q", max_hits=0)


def test_slot_spec_rejects_negative_min_cardinality() -> None:
    with pytest.raises(ValueError, match="min_cardinality must be >= 0"):
        SlotSpec(name="x", store="episodic", query_template="q", min_cardinality=-1)


# ---------- load_contract ----------


def test_load_contract_parses_complete_yaml(tmp_path: Path) -> None:
    path = tmp_path / "returns.yaml"
    path.write_text(
        """
role: returns-handler
intent: Process a customer return request
budget_tokens: 1500
slots:
  - name: customer_history
    store: episodic
    query_template: "returns by customer {customer_id}"
    required: true
    max_hits: 3
  - name: matching_orders
    store: tabular
    table_name: returns
    sql_template: |
      SELECT order_id, item, amount_usd, status FROM returns
      WHERE customer_id = '{customer_id}'
    required: true
    min_cardinality: 1
"""
    )
    contract = load_contract(path)
    assert contract.role == "returns-handler"
    assert contract.budget_tokens == 1500
    assert len(contract.slots) == 2
    assert contract.slots[0].name == "customer_history"
    assert contract.slots[0].required is True
    assert contract.slots[1].store == "tabular"


def test_load_contract_rejects_missing_role(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "intent: x\nbudget_tokens: 100\nslots: [{name: a, store: episodic, query_template: q}]\n"
    )
    with pytest.raises(ValueError, match="missing or empty 'role'"):
        load_contract(path)


def test_load_contract_rejects_empty_slots(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("role: r\nintent: i\nbudget_tokens: 100\nslots: []\n")
    with pytest.raises(ValueError, match="`slots` must be a non-empty list"):
        load_contract(path)


def test_load_contract_rejects_non_positive_budget(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "role: r\n"
        "intent: i\n"
        "budget_tokens: 0\n"
        "slots: [{name: a, store: episodic, query_template: q}]\n"
    )
    with pytest.raises(ValueError, match="`budget_tokens` must be a positive int"):
        load_contract(path)


# ---------- assemble_package: variable substitution + dispatch ----------


def _empty_stores(tmp_path: Path) -> tuple[StoreBundle, EpisodicStore]:
    """Return (bundle, episodic-handle). The typed handle lets the
    tests `ingest()` without mypy union-attr noise; the bundle is
    what gets handed to `assemble_package`."""
    embedder = _HashEmbedder()
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    bundle = StoreBundle(
        episodic=episodic,
        tree=DocumentTreeStore(db_path=tmp_path / "tr.sqlite", embedder=embedder),
        tabular=TabularStore(db_path=tmp_path / "tb.sqlite", embedder=embedder),
    )
    return bundle, episodic


def test_assemble_package_renders_query_template_from_variables(tmp_path: Path) -> None:
    stores, episodic = _empty_stores(tmp_path)
    episodic.ingest(
        external_id="ch:C9148",
        title="Past return by C9148",
        body="Customer C9148 previously returned a jacket in March.",
        tier="seed",
    )
    contract = ContractBundle(
        role="returns-handler",
        intent="Process return",
        budget_tokens=500,
        slots=(
            SlotSpec(
                name="customer_history",
                store="episodic",
                query_template="returns by customer {customer_id}",
                required=True,
                max_hits=3,
            ),
        ),
    )
    pkg = assemble_package(
        contract,
        variables={"customer_id": "C9148"},
        access=AccessPolicy(user_id="C9148", role="returns-handler"),
        stores=stores,
    )
    assert pkg.is_complete
    assert len(pkg.hits) >= 1
    assert pkg.hits[0].provenance.store == "episodic"
    assert pkg.hits[0].slot_name == "customer_history"


def test_assemble_package_reports_missing_required_slot_when_empty(tmp_path: Path) -> None:
    stores, _ = _empty_stores(tmp_path)
    # No data ingested — episodic store returns no hits.
    contract = ContractBundle(
        role="returns-handler",
        intent="Process return",
        budget_tokens=500,
        slots=(
            SlotSpec(
                name="refund_policy",
                store="episodic",
                query_template="refund policy threshold",
                required=True,
                min_cardinality=1,
            ),
        ),
    )
    pkg = assemble_package(
        contract,
        variables={},
        access=AccessPolicy(user_id=None),
        stores=stores,
    )
    assert not pkg.is_complete
    assert pkg.missing_required_slots == ("refund_policy",)
    assert pkg.hits == ()


def test_assemble_package_does_not_flag_optional_slot_as_missing(tmp_path: Path) -> None:
    stores, _ = _empty_stores(tmp_path)
    contract = ContractBundle(
        role="returns-handler",
        intent="Process return",
        budget_tokens=500,
        slots=(
            SlotSpec(
                name="nice_to_have",
                store="episodic",
                query_template="anything",
                required=False,
            ),
        ),
    )
    pkg = assemble_package(contract, variables={}, access=AccessPolicy(user_id=None), stores=stores)
    assert pkg.is_complete  # optional empty slot doesn't break completeness


def test_assemble_package_raises_on_missing_store_for_slot(tmp_path: Path) -> None:
    embedder = _HashEmbedder()
    stores = StoreBundle(
        episodic=EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder),
        # No tree or tabular store provided.
    )
    contract = ContractBundle(
        role="x",
        intent="x",
        budget_tokens=100,
        slots=(
            SlotSpec(
                name="tree_slot",
                store="tree",
                query_template="anything",
                required=True,
            ),
        ),
    )
    with pytest.raises(ValueError, match="needs a tree store"):
        assemble_package(contract, variables={}, access=AccessPolicy(user_id=None), stores=stores)


def test_assemble_package_raises_on_template_variable_missing(tmp_path: Path) -> None:
    stores, _ = _empty_stores(tmp_path)
    contract = ContractBundle(
        role="x",
        intent="x",
        budget_tokens=100,
        slots=(
            SlotSpec(
                name="needs_var",
                store="episodic",
                query_template="customer {customer_id}",
                required=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="references variable 'customer_id'"):
        assemble_package(
            contract,
            variables={},  # missing customer_id
            access=AccessPolicy(user_id=None),
            stores=stores,
        )


# ---------- assemble_package: budget enforcement ----------


def test_assemble_package_packs_required_slots_even_when_over_budget(
    tmp_path: Path,
) -> None:
    """Required slots are always included. The point of the contract is
    that the agent gets the data it said it needed; silently dropping
    required content violates the contract."""
    stores, episodic = _empty_stores(tmp_path)
    for i in range(3):
        episodic.ingest(
            external_id=f"r:{i}",
            title=f"Required record {i}",
            body="x" * 200,  # ~50 tokens each
            tier="seed",
        )
    contract = ContractBundle(
        role="x",
        intent="x",
        budget_tokens=10,  # absurdly tight
        slots=(
            SlotSpec(
                name="req",
                store="episodic",
                query_template="record",
                required=True,
                max_hits=3,
            ),
        ),
    )
    pkg = assemble_package(contract, variables={}, access=AccessPolicy(user_id=None), stores=stores)
    # All 3 required hits included despite blowing the 10-token budget;
    # caller learns via tokens_used > budget.max_tokens.
    assert len(pkg.hits) >= 1
    assert pkg.tokens_used > pkg.budget.max_tokens


def test_assemble_package_overflows_optional_slots_when_budget_exhausted(
    tmp_path: Path,
) -> None:
    stores, episodic = _empty_stores(tmp_path)
    # One required hit that's small enough to fit, then several optional
    # hits each too big to fit alongside it.
    episodic.ingest(
        external_id="r:req",
        title="Small required",
        body="ok",
        tier="seed",
    )
    for i in range(3):
        episodic.ingest(
            external_id=f"r:opt-{i}",
            title=f"Optional {i}",
            body="x" * 400,  # ~100 tokens each, way over budget
            tier="seed",
        )
    contract = ContractBundle(
        role="x",
        intent="x",
        budget_tokens=30,  # tight: required fits, optionals overflow
        slots=(
            SlotSpec(
                name="req",
                store="episodic",
                query_template="small required",
                required=True,
                max_hits=1,
            ),
            SlotSpec(
                name="opt",
                store="episodic",
                query_template="optional",
                required=False,
                max_hits=3,
            ),
        ),
    )
    pkg = assemble_package(contract, variables={}, access=AccessPolicy(user_id=None), stores=stores)
    # Required slot got its hit (or at least 1).
    assert len(pkg.hits_for_slot("req")) >= 1
    # Optional hits overflowed.
    assert len(pkg.overflow_hits) >= 1


# ---------- assemble_package: tabular slot ----------


def test_assemble_package_tabular_slot_returns_one_hit_per_row(tmp_path: Path) -> None:
    embedder = _HashEmbedder()
    stores = StoreBundle(
        tabular=TabularStore(db_path=tmp_path / "tb.sqlite", embedder=embedder),
    )
    schema = TableSchema(
        name="returns",
        description="returns table",
        columns=(
            ("order_id", "TEXT", "order id"),
            ("customer_id", "TEXT", "customer code"),
            ("amount_usd", "REAL", "refund amount"),
        ),
    )
    stores.tabular.register_table(  # type: ignore[union-attr]
        schema=schema,
        rows=[
            ("10001", "C9148", "100.00"),
            ("10002", "C9148", "50.00"),
            ("10003", "C0000", "999.99"),  # not the customer we're filtering
        ],
    )
    contract = ContractBundle(
        role="returns-handler",
        intent="Find customer orders",
        budget_tokens=500,
        slots=(
            SlotSpec(
                name="customer_orders",
                store="tabular",
                table_name="returns",
                sql_template=(
                    "SELECT order_id, amount_usd FROM returns WHERE customer_id = '{customer_id}'"
                ),
                required=True,
                max_hits=10,
            ),
        ),
    )
    pkg = assemble_package(
        contract,
        variables={"customer_id": "C9148"},
        access=AccessPolicy(user_id=None),
        stores=stores,
    )
    assert pkg.is_complete
    hits = pkg.hits_for_slot("customer_orders")
    assert len(hits) == 2
    # Provenance reflects the SQL path, not vector retrieval.
    assert all(h.provenance.store == "tabular" for h in hits)
    assert all(h.provenance.method == "sql" for h in hits)
    # Body carries column=value pairs the model can read.
    assert "order_id=10001" in hits[0].body
    assert "amount_usd=100" in hits[0].body


# ---------- auto_merge validation (harness-0t7a) ----------


def test_slot_spec_auto_merge_rejects_non_tree_slots() -> None:
    """harness-0t7a: auto_merge only makes sense for tree slots. The
    SlotSpec validator catches mis-applied flags at construction time
    so fixture authors see the typo before chat boot."""
    with pytest.raises(ValueError, match=r"auto_merge applies to tree slots only"):
        SlotSpec(
            name="bad",
            store="episodic",
            query_template="{q}",
            auto_merge=True,
        )
    with pytest.raises(ValueError, match=r"auto_merge applies to tree slots only"):
        SlotSpec(
            name="bad",
            store="tabular",
            sql_template="SELECT * FROM t",
            table_name="t",
            auto_merge=True,
        )


def test_slot_spec_auto_merge_accepted_on_tree_slot() -> None:
    """Tree slots accept auto_merge=True without complaint. Default
    stays False so existing slots are unchanged."""
    spec = SlotSpec(
        name="ok",
        store="tree",
        query_template="{q}",
        auto_merge=True,
    )
    assert spec.auto_merge is True
    default = SlotSpec(
        name="default",
        store="tree",
        query_template="{q}",
    )
    assert default.auto_merge is False
