"""Tests for harness.tools.assemble_context (harness-xysp).

Same hash-embedder + real-SQLite pattern as the other store-touching
tool tests. Pins the contract→render boundary: the tool delegates to
`assemble_package`, so we don't re-test the orchestrator here. We DO
pin (a) the tool's spec materialization (interpolates available
contracts), (b) the rendered shape (grouped by slot, with provenance
+ missing-slot warning), and (c) graceful error strings."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.retrieval.contract import StoreBundle, load_contract
from harness.store.episodic import EpisodicStore
from harness.store.tabular import TableSchema, TabularStore
from harness.tools.assemble_context import (
    AssembleContextTool,
    _extract_template_variables,
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


def _write_contract(tmp_path: Path, body: str, *, name: str = "test_role.yaml") -> Path:
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    path = contracts_dir / name
    path.write_text(body)
    return path


_MINIMAL_CONTRACT = """\
role: test_role
intent: Test the assemble_context tool
budget_tokens: 500
slots:
  - name: customer_history
    store: episodic
    query_template: "history for {customer_id}"
    required: true
    max_hits: 3
"""


def _seed_stores(tmp_path: Path) -> tuple[StoreBundle, EpisodicStore]:
    embedder = _HashEmbedder()
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    bundle = StoreBundle(episodic=episodic)
    return bundle, episodic


# ---------- _extract_template_variables ----------


def test_extract_template_variables_dedupes_and_sorts(tmp_path: Path) -> None:
    path = _write_contract(
        tmp_path,
        """
role: r
intent: i
budget_tokens: 100
slots:
  - name: a
    store: episodic
    query_template: "{zebra} and {apple}"
  - name: b
    store: tabular
    table_name: t
    sql_template: "SELECT * FROM t WHERE x = '{apple}'"
""",
    )
    contract = load_contract(path)
    assert _extract_template_variables(contract) == ("apple", "zebra")


def test_extract_template_variables_returns_empty_when_no_vars(tmp_path: Path) -> None:
    path = _write_contract(
        tmp_path,
        """
role: r
intent: i
budget_tokens: 100
slots:
  - name: a
    store: episodic
    query_template: "constant query no vars here"
""",
    )
    contract = load_contract(path)
    assert _extract_template_variables(contract) == ()


# ---------- spec ----------


def test_spec_interpolates_available_contracts(tmp_path: Path) -> None:
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    description = tool.spec.description
    assert "role=test_role" in description
    assert "Test the assemble_context tool" in description
    assert "customer_id" in description


def test_spec_handles_empty_contracts_dir(tmp_path: Path) -> None:
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "no_contracts_here")
    assert "no contracts registered" in tool.spec.description


def test_spec_is_read_tier(tmp_path: Path) -> None:
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    assert tool.spec.tier == "read"


def test_spec_skips_malformed_contract_silently(tmp_path: Path) -> None:
    """Bad contract YAML shouldn't break tool init — it just doesn't
    surface in the available-contracts list."""
    _write_contract(tmp_path, _MINIMAL_CONTRACT, name="good.yaml")
    _write_contract(tmp_path, "not: a real contract", name="bad.yaml")
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    # Good contract loaded; bad one skipped.
    description = tool.spec.description
    assert "role=test_role" in description
    # The bad file's hypothetical role isn't in there.
    assert "bad" not in description.lower() or "Test the assemble" in description


# ---------- call() ----------


def test_call_renders_grouped_by_slot(tmp_path: Path) -> None:
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, episodic = _seed_stores(tmp_path)
    episodic.ingest(
        external_id="ch:C9148",
        title="Customer C9148 prior return",
        body="C9148 returned a jacket in March.",
        tier="seed",
    )
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts", user_id="C9148")
    out = tool.call(role="test_role", variables={"customer_id": "C9148"})
    assert "Context Package — Test the assemble_context tool" in out
    assert "Role: test_role" in out
    assert "customer_id=C9148" in out
    assert "## customer_history" in out
    # Provenance line carries store + method + record_id.
    assert "[episodic/hybrid" in out
    assert "ch:C9148" in out


def test_call_surfaces_missing_required_slot(tmp_path: Path) -> None:
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, _ = _seed_stores(tmp_path)  # no data ingested
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    out = tool.call(role="test_role", variables={"customer_id": "ghost"})
    assert "Missing required slots: customer_history" in out


def test_call_returns_error_string_on_unknown_role(tmp_path: Path) -> None:
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    out = tool.call(role="nope", variables={})
    assert "unknown role 'nope'" in out
    assert "test_role" in out  # lists what IS available


def test_call_returns_error_string_on_missing_template_variable(tmp_path: Path) -> None:
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    out = tool.call(role="test_role", variables={})  # missing customer_id
    assert "assemble_context error" in out
    assert "customer_id" in out


def test_call_prints_traceback_to_stderr_on_assemble_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """harness-ygvg follow-up: when assemble_package raises (caught as
    ValueError), the tool returns a clean one-liner to the model but
    MUST also dump the traceback to stderr so an intermittent failure
    like 'bad value(s) in fds_to_keep' (observed 2026-05-14) is
    diagnosable from the running session. Without this, the swallowed
    ValueError leaves no trail."""
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    # Force a ValueError out of assemble_package by omitting the
    # required `customer_id` template variable.
    out = tool.call(role="test_role", variables={})
    assert "assemble_context error" in out
    err = capsys.readouterr().err
    assert "Traceback" in err, "expected the swallowed ValueError to surface a traceback on stderr"
    assert "customer_id" in err


def test_call_appends_traceback_to_error_log_when_configured(tmp_path: Path) -> None:
    """harness-ygvg follow-up #2: TUI sessions swallow stderr to the
    alt-screen, so the stderr-only trail is invisible there. When
    `error_log_path` is wired, the caught traceback also appends to
    that file so a forensic record survives across both surfaces."""
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, _ = _seed_stores(tmp_path)
    log_path = tmp_path / "logs" / "assemble_context_errors.log"
    tool = AssembleContextTool(
        stores=stores,
        contracts_dir=tmp_path / "contracts",
        error_log_path=log_path,
    )
    tool.call(role="test_role", variables={})  # missing customer_id → ValueError
    assert log_path.exists(), "error log file should have been created lazily"
    body = log_path.read_text(encoding="utf-8")
    assert "Traceback" in body
    assert "role='test_role'" in body
    assert "customer_id" in body
    # Second failure appends rather than overwrites. Count the per-
    # call timestamp marker rather than 'Traceback' (chained
    # exceptions emit multiple Traceback lines per call).
    tool.call(role="test_role", variables={})
    body2 = log_path.read_text(encoding="utf-8")
    assert body2.count("--- ") == 2


def test_call_skips_error_log_when_path_unset(tmp_path: Path) -> None:
    """Default construction (no `error_log_path`) must not create a
    log file — keeps tests + ephemeral subagent calls from littering
    the working tree."""
    _write_contract(tmp_path, _MINIMAL_CONTRACT)
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    tool.call(role="test_role", variables={})
    # No path means no file. Walk the tmp_path looking for any *.log.
    assert not list(tmp_path.rglob("*.log"))


def test_call_with_no_variables_renders_none_placeholder(tmp_path: Path) -> None:
    """A contract whose templates don't have variables can be called
    with `variables=None` and the renderer says so explicitly."""
    _write_contract(
        tmp_path,
        """
role: novars
intent: No-args contract
budget_tokens: 200
slots:
  - name: scratch
    store: episodic
    query_template: "anything"
    required: false
""",
    )
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    out = tool.call(role="novars")
    assert "Variables: (none)" in out


def test_call_includes_tabular_slot_when_wired(tmp_path: Path) -> None:
    """End-to-end smoke: contract pulls from both episodic and tabular,
    rendered output shows both slot blocks with the right provenance
    methods (hybrid vs sql)."""
    embedder = _HashEmbedder()
    episodic = EpisodicStore(db_path=tmp_path / "ep.sqlite", embedder=embedder)
    tabular = TabularStore(db_path=tmp_path / "tb.sqlite", embedder=embedder)
    schema = TableSchema(
        name="returns",
        description="Customer returns",
        columns=(
            ("order_id", "INTEGER", "id"),
            ("customer_id", "TEXT", "customer"),
        ),
    )
    tabular.register_table(
        schema=schema,
        rows=[("10001", "C9148"), ("10002", "C9148")],
    )
    episodic.ingest(
        external_id="ch:C9148:past",
        title="C9148 past return",
        body="Returned a jacket last March.",
        tier="seed",
    )

    _write_contract(
        tmp_path,
        """
role: mixed
intent: Pull from both stores
budget_tokens: 500
slots:
  - name: customer_history
    store: episodic
    query_template: "history {customer_id}"
    required: true
    max_hits: 3
  - name: orders
    store: tabular
    table_name: returns
    sql_template: "SELECT order_id FROM returns WHERE customer_id = '{customer_id}'"
    required: true
    max_hits: 5
""",
    )
    tool = AssembleContextTool(
        stores=StoreBundle(episodic=episodic, tabular=tabular),
        contracts_dir=tmp_path / "contracts",
    )
    out = tool.call(role="mixed", variables={"customer_id": "C9148"})
    assert "## customer_history" in out
    assert "## orders" in out
    assert "[episodic/hybrid" in out
    assert "[tabular/sql" in out


def test_call_skips_empty_optional_slot_in_render(tmp_path: Path) -> None:
    """Empty optional slots don't get a section header — the rendered
    output stays tight. (Empty REQUIRED slots show up in the missing
    list at the top of the package, which is its own visibility.)"""
    _write_contract(
        tmp_path,
        """
role: opt_only
intent: Optional-only contract
budget_tokens: 200
slots:
  - name: maybe
    store: episodic
    query_template: "no data exists"
    required: false
""",
    )
    stores, _ = _seed_stores(tmp_path)
    tool = AssembleContextTool(stores=stores, contracts_dir=tmp_path / "contracts")
    out = tool.call(role="opt_only")
    assert "## maybe" not in out
