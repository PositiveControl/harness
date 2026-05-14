"""Tests for harness.tools.query_table (harness-edt6 / Phase 2)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.store.tabular import TableSchema, TabularStore
from harness.tools.query_table import QueryTableTool


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


def _seed_store(tmp_path: Path) -> TabularStore:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    schema = TableSchema(
        name="returns",
        description="Customer return requests, one row per refund.",
        columns=(
            ("order_id", "INTEGER", "Unique order id"),
            ("item", "TEXT", "Returned item"),
            ("amount_usd", "REAL", "Refund amount in USD"),
            ("status", "TEXT", "approved / pending / denied"),
        ),
    )
    store.register_table(
        schema=schema,
        rows=[
            ("10063", "leather", "182.12", "denied"),
            ("10377", "monitor", "848.42", "approved"),
            ("10708", "kettle", "103.30", "denied"),
        ],
    )
    return store


# ---------- spec ----------


def test_spec_interpolates_registered_schemas_into_description(tmp_path: Path) -> None:
    tool = QueryTableTool(store=_seed_store(tmp_path))
    description = tool.spec.description
    assert "Table: returns" in description
    assert "order_id" in description
    assert "amount_usd" in description
    assert "Refund amount in USD" in description


def test_spec_is_read_tier_so_no_confirmation_required(tmp_path: Path) -> None:
    tool = QueryTableTool(store=_seed_store(tmp_path))
    assert tool.spec.tier == "read"


def test_spec_lists_required_arguments(tmp_path: Path) -> None:
    tool = QueryTableTool(store=_seed_store(tmp_path))
    required = set(tool.spec.parameters["required"])
    assert required == {"table_name", "sql"}


def test_spec_message_when_no_tables_registered(tmp_path: Path) -> None:
    empty = TabularStore(db_path=tmp_path / "empty.sqlite", embedder=_HashEmbedder())
    tool = QueryTableTool(store=empty)
    assert "no tables registered" in tool.spec.description


# ---------- call() ----------


def test_call_returns_markdown_table_with_sql_echo(tmp_path: Path) -> None:
    tool = QueryTableTool(store=_seed_store(tmp_path))
    out = tool.call(
        table_name="returns",
        sql="SELECT order_id, item FROM returns WHERE status = 'denied'",
    )
    assert "SQL:" in out
    assert "| order_id | item |" in out
    assert "| 10063 | leather |" in out
    assert "| 10708 | kettle |" in out
    # `approved` row not present.
    assert "10377" not in out


def test_call_returns_no_rows_message_when_empty_result(tmp_path: Path) -> None:
    tool = QueryTableTool(store=_seed_store(tmp_path))
    out = tool.call(
        table_name="returns",
        sql="SELECT order_id FROM returns WHERE item = 'unicorn'",
    )
    assert "no rows matched" in out
    assert "SQL:" in out


def test_call_returns_error_string_on_validation_failure(tmp_path: Path) -> None:
    """Disallowed-keyword SQL must surface a graceful error string,
    not raise — the tool is invoked inside an agent loop and an
    uncaught exception would crash the turn."""
    tool = QueryTableTool(store=_seed_store(tmp_path))
    out = tool.call(table_name="returns", sql="DROP TABLE returns")
    assert "query_table error" in out


def test_call_returns_error_on_unknown_table(tmp_path: Path) -> None:
    tool = QueryTableTool(store=_seed_store(tmp_path))
    out = tool.call(table_name="ghosts", sql="SELECT 1 FROM ghosts")
    assert "query_table error" in out
    assert "ghosts" in out


def test_call_truncates_long_result_sets(tmp_path: Path) -> None:
    tool = QueryTableTool(store=_seed_store(tmp_path), max_rows=2)
    out = tool.call(table_name="returns", sql="SELECT order_id FROM returns")
    assert "3 rows, showing first 2" in out


def test_call_clips_overly_long_cell_values(tmp_path: Path) -> None:
    store = TabularStore(db_path=tmp_path / "tab.sqlite", embedder=_HashEmbedder())
    schema = TableSchema(
        name="notes",
        description="Long-form text notes",
        columns=(
            ("id", "TEXT", "Note id"),
            ("body", "TEXT", "Free-form note body"),
        ),
    )
    long_body = "x" * 500
    store.register_table(schema=schema, rows=[("n1", long_body)])
    tool = QueryTableTool(store=store, max_col_chars=30)
    out = tool.call(table_name="notes", sql="SELECT id, body FROM notes")
    assert "x" * 29 + "…" in out
    assert "x" * 500 not in out
