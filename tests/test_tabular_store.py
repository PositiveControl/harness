"""Tests for harness.store.tabular (harness-edt6 / Phase 2).

Real SQLite per the project convention; deterministic hash-based fake
embedder so dense-cosine ordering is stable without loading
sentence-transformers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.store.tabular import (
    TableSchema,
    TabularStore,
    _from_clause_identifiers,
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


def _returns_schema() -> TableSchema:
    return TableSchema(
        name="returns",
        description="Customer return requests with order, item, amount, status.",
        columns=(
            ("order_id", "INTEGER", "Unique order identifier"),
            ("customer_id", "TEXT", "Customer code"),
            ("category", "TEXT", "High-level product family"),
            ("item", "TEXT", "Specific item returned"),
            ("amount_usd", "REAL", "Refund amount in USD"),
            ("order_month", "TEXT", "Three-letter month of the original order"),
            ("reason", "TEXT", "Customer's stated reason for return"),
            ("status", "TEXT", "Return decision"),
        ),
    )


def _users_schema() -> TableSchema:
    return TableSchema(
        name="users",
        description="Customer accounts with name and signup date.",
        columns=(
            ("customer_id", "TEXT", "Unique customer code"),
            ("full_name", "TEXT", "Customer's full legal name"),
            ("signup_date", "TEXT", "Account creation date ISO 8601"),
        ),
    )


# ---------- schema rendering ----------


def test_schema_text_renders_columns_with_descriptions() -> None:
    schema = _returns_schema()
    out = schema.schema_text()
    assert "Table: returns" in out
    assert "Purpose: Customer return requests" in out
    assert "order_id" in out
    assert "Refund amount in USD" in out


def test_schema_column_names_preserves_order() -> None:
    schema = _returns_schema()
    assert schema.column_names() == (
        "order_id",
        "customer_id",
        "category",
        "item",
        "amount_usd",
        "order_month",
        "reason",
        "status",
    )


# ---------- registration ----------


def test_register_table_inserts_rows_and_metadata(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    schema = _returns_schema()
    rows = [
        ("10063", "C9148", "jacket", "leather", "182.12", "Mar", "damaged on arrival", "denied"),
    ]
    registered = store.register_table(schema=schema, rows=rows)
    assert registered.name == "returns"
    # Table itself was created and populated.
    result = store.query_sql("returns", "SELECT order_id, item FROM returns")
    assert result.rows == (("10063", "leather"),)


def test_register_table_raises_on_duplicate_without_replace(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    schema = _returns_schema()
    store.register_table(schema=schema, rows=[])
    with pytest.raises(ValueError, match="already registered"):
        store.register_table(schema=schema, rows=[])


def test_register_table_replace_drops_and_recreates(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    schema = _returns_schema()
    store.register_table(
        schema=schema,
        rows=[
            ("1", "c", "jacket", "leather", "10", "Jan", "x", "approved"),
        ],
    )
    store.register_table(
        schema=schema,
        rows=[
            ("2", "c", "shoes", "loafers", "50", "Feb", "y", "denied"),
        ],
        replace=True,
    )
    result = store.query_sql("returns", "SELECT order_id FROM returns")
    assert result.rows == (("2",),)  # original row dropped


def test_register_table_rejects_bad_identifier(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    bad = TableSchema(
        name="returns; DROP TABLE foo;--",
        description="malicious",
        columns=(("id", "INTEGER", "x"),),
    )
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        store.register_table(schema=bad, rows=[])


def test_register_table_from_csv_loads_header_correctly(tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "order_id,customer_id,category,item,amount_usd,order_month,reason,status\n"
        "10063,C9148,jacket,leather,182.12,Mar,damaged on arrival,denied\n"
    )
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table_from_csv(schema=_returns_schema(), csv_path=csv_path)
    result = store.query_sql("returns", "SELECT order_id, item FROM returns")
    assert result.rows == (("10063", "leather"),)


def test_register_csv_missing_header_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("order_id,item\n10063,leather\n")  # missing most columns
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    with pytest.raises(ValueError, match="header missing required columns"):
        store.register_table_from_csv(schema=_returns_schema(), csv_path=csv_path)


# ---------- tables() / get_table ----------


def test_tables_lists_registered_tables_in_insertion_order(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    store.register_table(schema=_users_schema(), rows=[])
    names = [t.name for t in store.tables()]
    assert names == ["returns", "users"]


def test_get_table_returns_none_when_unregistered(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    assert store.get_table("nope") is None


# ---------- find_tables (discovery) ----------


def test_find_tables_returns_score_ranked_hits(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    store.register_table(schema=_users_schema(), rows=[])
    hits = store.find_tables("which customers asked for refunds", k=2)
    assert len(hits) == 2
    # Hash embedder doesn't have semantic signal, but the ranking
    # contract still holds: scores descending, ties broken by name.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_find_tables_skips_dim_mismatched_rows(tmp_path: Path) -> None:
    """Embedder swap mid-store: rows with the wrong dim must be
    silently excluded (parallel to the EpisodicStore dim filter)."""
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    # Force a different embedder dim into the store.
    store.embedder = _HashEmbedder(id="wider", dimension=16)
    assert store.find_tables("anything", k=5) == []


def test_find_tables_empty_query_returns_no_hits(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    assert store.find_tables("  ", k=5) == []


# ---------- query_sql ----------


def test_query_sql_runs_select_and_returns_columns_plus_rows(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(
        schema=_returns_schema(),
        rows=[
            ("10063", "C9148", "jacket", "leather", "182.12", "Mar", "damaged", "denied"),
            (
                "10377",
                "C1234",
                "electronics",
                "monitor",
                "848.42",
                "Feb",
                "not as described",
                "approved",
            ),
        ],
    )
    result = store.query_sql(
        "returns",
        "SELECT order_id, amount_usd FROM returns ORDER BY CAST(amount_usd AS REAL) DESC LIMIT 1",
    )
    assert result.columns == ("order_id", "amount_usd")
    assert result.rows == (("10377", "848.42"),)


def test_query_sql_handles_numeric_comparison_against_text_column(tmp_path: Path) -> None:
    """All columns are stored as TEXT, but SQLite's NUMERIC affinity
    still lets `> 700` filter correctly when the column values look
    like numbers. Pinning this so the bench cases work."""
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(
        schema=_returns_schema(),
        rows=[
            ("10063", "C9148", "jacket", "leather", "182.12", "Mar", "damaged", "denied"),
            ("10377", "C1234", "electronics", "monitor", "848.42", "Feb", "x", "approved"),
        ],
    )
    result = store.query_sql(
        "returns",
        "SELECT order_id FROM returns WHERE CAST(amount_usd AS REAL) > 700",
    )
    assert result.rows == (("10377",),)


def test_query_sql_raises_on_unregistered_table(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    with pytest.raises(ValueError, match="not registered"):
        store.query_sql("returns", "SELECT * FROM returns")


def test_query_sql_blocks_disallowed_keywords(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    for sql in (
        "INSERT INTO returns VALUES (1, 2, 3, 4, 5, 6, 7, 8)",
        "DELETE FROM returns",
        "UPDATE returns SET status = 'approved'",
        "DROP TABLE returns",
        "ATTACH DATABASE 'evil.db' AS evil",
    ):
        with pytest.raises(ValueError, match="disallowed keyword"):
            store.query_sql("returns", sql)


def test_query_sql_blocks_cross_table_references(tmp_path: Path) -> None:
    """The v1 gate is single-table: SQL must only reference the table
    the caller named. JOINs onto another table are rejected even if
    that other table is also registered."""
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    store.register_table(schema=_users_schema(), rows=[])
    with pytest.raises(ValueError, match="unregistered or unauthorized"):
        store.query_sql("returns", "SELECT * FROM returns JOIN users USING (customer_id)")


def test_query_sql_blocks_from_clause_with_unknown_table(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    with pytest.raises(ValueError, match="unregistered or unauthorized"):
        store.query_sql("returns", "SELECT * FROM sqlite_master")


# ---------- helpers ----------


def test_from_clause_identifiers_picks_up_join_targets() -> None:
    refs = _from_clause_identifiers(
        "SELECT * FROM orders JOIN customers ON orders.cid = customers.id LEFT JOIN items"
    )
    assert refs == ["orders", "customers", "items"]


def test_from_clause_identifiers_is_case_insensitive() -> None:
    refs = _from_clause_identifiers("select * from RETURNS")
    assert refs == ["RETURNS"]


# ---------- embedder-swap robustness (harness-pzpy) ----------


def test_count_mismatched_embeddings_zero_when_dim_matches(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    store.register_table(schema=_users_schema(), rows=[])
    assert store.count_mismatched_embeddings() == 0


def test_count_mismatched_embeddings_surfaces_dim_drift(tmp_path: Path) -> None:
    """Register under one embedder dim, swap to a different dim, count
    should reflect every row as mismatched. This is the signal the CLI
    rebuild command uses to tell the user whether a rebuild is worth it."""
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    store.register_table(schema=_users_schema(), rows=[])
    # Swap to a wider embedder; both registered embeddings now have
    # the wrong dim.
    store.embedder = _HashEmbedder(id="wider", dimension=16)
    assert store.count_mismatched_embeddings() == 2


def test_rebuild_embeddings_restores_find_tables_after_swap(tmp_path: Path) -> None:
    """The motivating bug: register a table, swap embedders, find_tables
    silently filters the dim-mismatched row out. Rebuild restores
    the hit list."""
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    # Pre-swap: find_tables returns the registered table.
    pre = store.find_tables("anything", k=5)
    assert len(pre) == 1

    # Swap.
    store.embedder = _HashEmbedder(id="wider", dimension=16)
    mid = store.find_tables("anything", k=5)
    assert mid == []  # silently filtered

    # Rebuild restores.
    updated, skipped = store.rebuild_embeddings()
    assert updated == 1
    assert skipped == 0
    post = store.find_tables("anything", k=5)
    assert len(post) == 1


def test_rebuild_embeddings_is_idempotent(tmp_path: Path) -> None:
    """Running rebuild twice in a row produces the same count both
    times. Mirrors the EpisodicStore contract."""
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    store.register_table(schema=_returns_schema(), rows=[])
    store.register_table(schema=_users_schema(), rows=[])
    first_updated, _ = store.rebuild_embeddings()
    second_updated, _ = store.rebuild_embeddings()
    assert first_updated == second_updated == 2


def test_rebuild_embeddings_on_empty_store_returns_zero(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    assert store.rebuild_embeddings() == (0, 0)
