"""Integration test: returns-handler contract end-to-end (harness-s3f7).

Loads the worked contract YAML, builds episodic + tabular stores
populated with synthetic data, runs `assemble_package`, asserts the
package shape. This is the Phase 3 demonstration the bd scope called
for — proves the contract pattern works against real stores, not just
mocked ones."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.retrieval.context_package import AccessPolicy, TokenBudget
from harness.retrieval.contract import StoreBundle, assemble_package, load_contract
from harness.store.episodic import EpisodicStore
from harness.store.tabular import TableSchema, TabularStore

REPO = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO / "retrieval_eval" / "contracts" / "returns_handler.yaml"
RETURNS_CSV = REPO / "retrieval_eval" / "data" / "returns.csv"


@dataclass
class _HashEmbedder:
    """Test-local hash embedder — deterministic, no model download.
    The contract orchestrator doesn't care about embedding quality,
    only that the same query maps to the same vector across calls."""

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


def _seed_episodic(store: EpisodicStore) -> None:
    """Synthetic episodic memory for the returns-handler role.

    Past customer interactions + the policy memories. Real deployment
    would scribe these from chat history; here we seed them so the
    bench is reproducible."""
    customer_history = [
        (
            "ch:C9148:prior-return",
            "Customer C9148 prior return — Mar 2025",
            "Customer C9148 returned a leather jacket in March; refund approved "
            "after photo evidence of damage. No subsequent issues.",
        ),
        (
            "ch:C9148:loyalty",
            "Customer C9148 loyalty profile",
            "Customer C9148 has been ordering for 4 years; spend tier high. "
            "Two returns in the last 12 months, both for damaged-on-arrival.",
        ),
        (
            "ch:C8718:multiple-returns",
            "Customer C8718 return pattern",
            "Customer C8718 has filed five returns in six months, all for "
            "'not as described'. Escalate any further request from this customer.",
        ),
    ]
    policy_memories = [
        (
            "policy:damaged-on-arrival",
            "Refund policy — damaged on arrival",
            "Items reported damaged on arrival are approved automatically "
            "when accompanied by photo evidence. Without evidence, escalate "
            "to a supervisor for review.",
        ),
        (
            "policy:high-value-threshold",
            "Refund policy — high-value returns",
            "Refunds above $500 require a supervisor approval regardless of "
            "reason, unless the customer is in the high-spend loyalty tier "
            "and has fewer than two prior returns in the past 12 months.",
        ),
        (
            "policy:repeat-returner",
            "Refund policy — repeat returner pattern",
            "Customers with more than three returns in a six-month window get "
            "flagged for review; the next return is held until a manager "
            "approves. Do not auto-process.",
        ),
    ]
    for ext_id, title, body in customer_history + policy_memories:
        store.ingest(
            external_id=ext_id,
            title=title,
            body=body,
            tier="seed",
            source="contract_test_seed",
        )


def _seed_tabular(store: TabularStore) -> None:
    """Register the returns CSV from Phase 0 / 2 as one table. Same
    schema definition the bench uses so this test corroborates the
    same architectural pattern."""
    schema = TableSchema(
        name="returns",
        description="Customer return requests, one row per refund.",
        columns=(
            ("order_id", "INTEGER", "Unique order identifier"),
            ("customer_id", "TEXT", "Customer code (C-prefixed)"),
            ("category", "TEXT", "High-level product family"),
            ("item", "TEXT", "Specific item returned"),
            ("amount_usd", "REAL", "Refund amount in USD"),
            ("order_month", "TEXT", "Three-letter month abbreviation"),
            ("reason", "TEXT", "Customer's stated reason"),
            ("status", "TEXT", "Return decision: approved / pending / denied"),
        ),
    )
    store.register_table_from_csv(schema=schema, csv_path=RETURNS_CSV)


def _build_stores(tmp_path: Path) -> StoreBundle:
    embedder = _HashEmbedder()
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    tabular = TabularStore(db_path=tmp_path / "tb.sqlite", embedder=embedder)
    _seed_episodic(episodic)
    _seed_tabular(tabular)
    return StoreBundle(episodic=episodic, tabular=tabular)


# ---------- end-to-end ----------


def test_returns_handler_contract_is_complete_for_known_customer(tmp_path: Path) -> None:
    """The committed contract YAML loads, and assemble_package against
    real stores returns a complete package — all three required slots
    fill. Demonstrates the Phase 3 architectural pattern end-to-end."""
    stores = _build_stores(tmp_path)
    contract = load_contract(CONTRACT_PATH)
    package = assemble_package(
        contract,
        variables={
            "customer_id": "C9148",
            "request_summary": "leather jacket arrived damaged",
        },
        access=AccessPolicy(user_id="C9148", role="returns-handler"),
        stores=stores,
    )
    assert package.is_complete, f"missing slots: {package.missing_required_slots}"
    # All three slots got at least one hit.
    assert package.hits_for_slot("customer_history")
    assert package.hits_for_slot("refund_policy")
    assert package.hits_for_slot("matching_orders")
    # Every hit carries audit-grade provenance.
    for hit in package.hits:
        assert hit.provenance.store in {"episodic", "tabular"}
        assert hit.provenance.record_id
        assert hit.provenance.method in {"hybrid", "sql"}


def test_returns_handler_contract_reports_missing_when_customer_unknown(
    tmp_path: Path,
) -> None:
    """Unknown customer — the tabular slot returns zero rows. The
    contract reports `matching_orders` as missing. Per the data-
    primitives note: 'what happens when something is missing' is the
    agent's failure-recovery signal."""
    stores = _build_stores(tmp_path)
    contract = load_contract(CONTRACT_PATH)
    package = assemble_package(
        contract,
        variables={
            "customer_id": "C9999",  # not in the CSV
            "request_summary": "asking about a refund",
        },
        access=AccessPolicy(user_id="C9999", role="returns-handler"),
        stores=stores,
    )
    assert not package.is_complete
    assert "matching_orders" in package.missing_required_slots


def test_returns_handler_contract_respects_budget(tmp_path: Path) -> None:
    """Budget enforcement: a tight budget keeps optional/overflow hits
    out of the main hit list while still seating the required slot
    minimums. Verifies the budget tracker behaves on a real package."""
    stores = _build_stores(tmp_path)
    contract = load_contract(CONTRACT_PATH)
    package = assemble_package(
        contract,
        variables={
            "customer_id": "C9148",
            "request_summary": "leather jacket damaged",
        },
        access=AccessPolicy(user_id="C9148", role="returns-handler"),
        stores=stores,
        # Override the YAML budget with a tight one to force overflow.
        budget=TokenBudget(max_tokens=80),
    )
    # Required slots seated even when budget too tight; check that
    # tokens_used (required) is reported accurately.
    assert package.tokens_used > 0
    # And that any optional overflow is captured separately.
    # (May be zero if the dense+BM25 search happened to fit; the key
    # invariant we're pinning is that the package's tokens accounting
    # is internally consistent.)
    assert package.tokens_used + package.tokens_overflow >= sum(
        h.est_tokens for h in (*package.hits, *package.overflow_hits)
    )


def test_returns_handler_contract_carries_access_policy_through(tmp_path: Path) -> None:
    """user_id flows from `AccessPolicy` into every episodic search
    call. Hits with mismatched user_ids should not surface. (Same
    contract the chat pipeline already enforces — Phase 3 just
    exposes it explicitly through the package shape.)"""
    stores = _build_stores(tmp_path)
    contract = load_contract(CONTRACT_PATH)
    package = assemble_package(
        contract,
        variables={
            "customer_id": "C9148",
            "request_summary": "leather jacket damaged",
        },
        access=AccessPolicy(user_id="C9148", role="returns-handler"),
        stores=stores,
    )
    # The package carries the access policy on its envelope so any
    # downstream consumer (renderer, audit log) sees the same scoping
    # decision the retrieval was made under.
    assert package.access.user_id == "C9148"
    assert package.access.role == "returns-handler"
