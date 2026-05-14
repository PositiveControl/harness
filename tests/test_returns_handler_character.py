"""Integration test for the returns_handler character (harness-kgpi).

Loads `character/returns_handler/` end-to-end via the standard
`load_character` path, exercises the same plumbing the CLI uses to
bootstrap a per-character TabularStore from `core.yaml.tabular_tables`,
ingests the policy seed memories into a tmp episodic store, and runs
the returns-handler contract through `assemble_package`. Asserts that
all three required slots fill from the right stores — proves the
character + CLI wiring is wired correctly without needing the model.

Pairs with `test_contract_returns_handler.py` (which tested the
contract module against synthetic data); this test pins the
*character convention*, not just the orchestrator."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.character import load_character
from harness.retrieval.context_package import AccessPolicy
from harness.retrieval.contract import (
    StoreBundle,
    assemble_package,
    load_contract,
)
from harness.store.episodic import EpisodicStore
from harness.store.tabular import build_tabular_store_for_character

REPO = Path(__file__).resolve().parents[1]
CHARACTER_PATH = REPO / "character" / "returns_handler"


@dataclass
class _HashEmbedder:
    """Deterministic test embedder; no model load."""

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


def _seed_episodic_with_policies(store: EpisodicStore) -> None:
    """Ingest the character's policy seed memories into a tmp episodic
    store. Real chat does this via `harness memory ingest`; the test
    inlines the same content so it doesn't depend on prior CLI runs.
    Content matches the markdown frontmatter in
    character/returns_handler/seed_memories/."""
    char = load_character(CHARACTER_PATH)
    for seed in char.seed_memories:
        store.ingest(
            external_id=seed.id,
            title=seed.title,
            body=seed.body,
            principle=seed.principle,
            tags=seed.tags,
            tier="seed",
            source="character_seed",
        )


# ---------- character loading ----------


def test_character_loads_with_tabular_tables() -> None:
    """The minimal smoke: load_character reads tabular_tables from
    core.yaml, csv_path resolves to a real file, columns are
    well-formed."""
    char = load_character(CHARACTER_PATH)
    assert char.name == "returns_handler"
    assert len(char.tabular_tables) == 1
    table = char.tabular_tables[0]
    assert table.table_name == "returns"
    assert table.csv_path.exists()
    assert len(table.columns) == 8
    # First column tuple shape: (name, type, description).
    assert table.columns[0][0] == "order_id"


def test_character_seed_memories_load_all_four_policies() -> None:
    """Each of the four refund policies under seed_memories/ should
    parse via the frontmatter loader. None of them are character-
    specific to airton — the loader is character-agnostic."""
    char = load_character(CHARACTER_PATH)
    policy_ids = {seed.id for seed in char.seed_memories}
    expected = {
        "policy-damaged-on-arrival",
        "policy-high-value-threshold",
        "policy-repeat-returner",
        "policy-changed-mind",
    }
    assert expected.issubset(policy_ids)


# ---------- tabular store bootstrap ----------


def test_build_tabular_store_for_character_registers_the_returns_table(
    tmp_path: Path,
) -> None:
    """The CLI-side bootstrap: from character.tabular_tables, build
    a populated TabularStore. The test routes the DB to tmp_path so
    a real run doesn't clobber the character's data dir."""
    char = load_character(CHARACTER_PATH)
    embedder = _HashEmbedder()
    # Reroute the table's csv_path absolute write target to tmp via
    # constructing the store at a tmp DB but using the real CSV.
    # build_tabular_store_for_character uses character_path/data/tabular.sqlite,
    # so point character_path at tmp + symlink the CSV.
    fake_char_dir = tmp_path / "fake_char"
    (fake_char_dir / "data").mkdir(parents=True)
    (fake_char_dir / "data" / "returns.csv").symlink_to(char.tabular_tables[0].csv_path)
    # Rebuild a TabularTableSpec pointing at the fake-character CSV.
    from harness.character import TabularTableSpec

    spec = TabularTableSpec(
        table_name=char.tabular_tables[0].table_name,
        description=char.tabular_tables[0].description,
        csv_path=fake_char_dir / "data" / "returns.csv",
        columns=char.tabular_tables[0].columns,
    )
    store = build_tabular_store_for_character(
        character_path=fake_char_dir,
        embedder=embedder,
        tabular_tables=(spec,),
    )
    assert store is not None
    tables = store.tables()
    assert len(tables) == 1
    assert tables[0].name == "returns"
    # Confirm rows landed.
    result = store.query_sql("returns", "SELECT COUNT(*) FROM returns")
    assert int(result.rows[0][0]) == 200


def test_build_tabular_store_returns_none_when_no_tabular_tables() -> None:
    """A character without tabular_tables shouldn't get a store
    constructed for it. CLI uses this signal to decide whether to
    spin up the SQLite file."""
    store = build_tabular_store_for_character(
        character_path=Path("/nope"),
        embedder=_HashEmbedder(),
        tabular_tables=(),
    )
    assert store is None


# ---------- end-to-end contract via the character ----------


def test_returns_handler_contract_resolves_against_character_stores(
    tmp_path: Path,
) -> None:
    """The acceptance criterion from the bd issue: the contract YAML
    shipped with the character resolves end-to-end against stores
    bootstrapped from the same character. Customer history (episodic
    + seed policies), refund policy (episodic), matching orders
    (tabular SQL) all fill from the right places."""
    char = load_character(CHARACTER_PATH)
    embedder = _HashEmbedder()

    # Episodic store gets the policy seeds.
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    _seed_episodic_with_policies(episodic)
    # Add one customer-history memory so the customer_history slot
    # fills too (the policy seeds alone won't match the customer
    # query template).
    episodic.ingest(
        external_id="ch:C9148:prior-march",
        title="Customer C9148 prior return",
        body="Customer C9148 previously returned a leather jacket in March.",
        principle="Past customer interaction",
        tier="seed",
        source="character_seed",
    )

    # Tabular store — bootstrap via the same path the CLI uses.
    fake_char_dir = tmp_path / "fake_char"
    (fake_char_dir / "data").mkdir(parents=True)
    (fake_char_dir / "data" / "returns.csv").symlink_to(char.tabular_tables[0].csv_path)
    from harness.character import TabularTableSpec

    spec = TabularTableSpec(
        table_name=char.tabular_tables[0].table_name,
        description=char.tabular_tables[0].description,
        csv_path=fake_char_dir / "data" / "returns.csv",
        columns=char.tabular_tables[0].columns,
    )
    tabular = build_tabular_store_for_character(
        character_path=fake_char_dir,
        embedder=embedder,
        tabular_tables=(spec,),
    )
    assert tabular is not None

    contract = load_contract(CHARACTER_PATH / "contracts" / "returns_handler.yaml")
    package = assemble_package(
        contract,
        variables={
            "customer_id": "C9148",
            "request_summary": "leather jacket arrived damaged",
        },
        access=AccessPolicy(user_id="C9148", role="returns_handler"),
        stores=StoreBundle(episodic=episodic, tabular=tabular),
    )
    assert package.is_complete, f"missing slots: {package.missing_required_slots}"
    assert package.hits_for_slot("customer_history")
    assert package.hits_for_slot("refund_policy")
    matching = package.hits_for_slot("matching_orders")
    assert matching
    # Tabular slot pulls from SQL, not vector — provenance reflects it.
    assert all(h.provenance.store == "tabular" for h in matching)
    assert all(h.provenance.method == "sql" for h in matching)
    # And the row identified is the one we expect (C9148 had one
    # leather jacket return in the data: order:10063).
    assert any("10063" in h.body for h in matching)


def test_returns_handler_contract_reports_missing_for_unknown_customer(
    tmp_path: Path,
) -> None:
    """The 'what happens when something is missing' signal: with an
    unknown customer_id, the tabular slot returns zero rows and the
    package's missing_required_slots flags it. The agent's directive
    is to surface this rather than fabricate a decision."""
    char = load_character(CHARACTER_PATH)
    embedder = _HashEmbedder()
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    _seed_episodic_with_policies(episodic)
    episodic.ingest(
        external_id="ch:C9999:fake",
        title="Customer C9999 fake context",
        body="Stub history so customer_history doesn't itself flag missing.",
        principle="Fake test seed",
        tier="seed",
        source="character_seed",
    )

    fake_char_dir = tmp_path / "fake_char"
    (fake_char_dir / "data").mkdir(parents=True)
    (fake_char_dir / "data" / "returns.csv").symlink_to(char.tabular_tables[0].csv_path)
    from harness.character import TabularTableSpec

    spec = TabularTableSpec(
        table_name=char.tabular_tables[0].table_name,
        description=char.tabular_tables[0].description,
        csv_path=fake_char_dir / "data" / "returns.csv",
        columns=char.tabular_tables[0].columns,
    )
    tabular = build_tabular_store_for_character(
        character_path=fake_char_dir,
        embedder=embedder,
        tabular_tables=(spec,),
    )

    contract = load_contract(CHARACTER_PATH / "contracts" / "returns_handler.yaml")
    package = assemble_package(
        contract,
        variables={
            "customer_id": "C9999",  # not in the CSV
            "request_summary": "any return",
        },
        access=AccessPolicy(user_id="C9999", role="returns_handler"),
        stores=StoreBundle(episodic=episodic, tabular=tabular),
    )
    assert not package.is_complete
    assert "matching_orders" in package.missing_required_slots
