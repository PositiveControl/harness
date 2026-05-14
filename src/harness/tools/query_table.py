"""query_table tool (harness-edt6 / Phase 2).

Lets the agent run read-only SQL against a registered `TabularStore`.
The tool's description interpolates every registered table's schema
at construction time so the model can write correct SQL without a
separate `list_tables` round-trip — the schema is grounded context,
not data the agent has to fish for.

Read-only by gate, not by convention. The underlying store rejects
anything that isn't a SELECT and any FROM clause that names an
unregistered table, so the tool can carry `tier="read"` (no write-tier
confirmation flow). See `TabularStore.query_sql` for the security
contract.

Composition with the data-retrieval-primitives stack:
- Phase 0 / 1 retrievers (episodic, tree) handle prose + structured
  documents. They're good at semantic recall.
- This tool handles tabular data. The agent picks the right interface
  for the question's shape; the tool doesn't try to be everything.

The tool returns rows in a stable, model-readable format: a small
markdown table with column headers and up to `max_rows` rows. The
SQL that ran is echoed back so the user (and any audit log) can see
exactly which query produced which result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.tools.base import ToolSpec

if TYPE_CHECKING:
    from harness.store.tabular import TabularStore


_DEFAULT_MAX_ROWS = 50
_DEFAULT_MAX_COL_CHARS = 80


@dataclass
class QueryTableTool:
    """Read-only SQL executor over a `TabularStore`. The store is the
    single source of truth for which tables exist and what shape they
    have — the tool just renders the schema into its description so
    the model sees it whenever the spec is materialized."""

    store: TabularStore
    max_rows: int = _DEFAULT_MAX_ROWS
    max_col_chars: int = _DEFAULT_MAX_COL_CHARS

    @property
    def spec(self) -> ToolSpec:
        registered = self.store.tables()
        if not registered:
            schema_blurb = "(no tables registered — this tool will fail until one is loaded.)"
        else:
            sections = [t.schema_text for t in registered]
            schema_blurb = "\n\n".join(sections)
        description = (
            "Run a read-only SQL SELECT against one registered table and "
            "get the rows back. Use this for QUESTIONS WITH SHAPE: "
            "aggregations ('highest', 'cheapest'), numeric filters "
            "('above $700'), exact-equality lookups, and any query the "
            "user would ask of a spreadsheet. Vector / semantic search "
            "tools are wrong for these.\n\n"
            "Available tables and their schemas:\n\n"
            f"{schema_blurb}\n\n"
            "Write parametrized SQL (a single SELECT) referencing ONLY "
            "the table you name in `table_name`. Cross-table joins, "
            "INSERT / UPDATE / DELETE / DDL, and any statement against "
            "an unregistered table will be rejected."
        )
        return ToolSpec(
            name="query_table",
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the registered table to query.",
                    },
                    "sql": {
                        "type": "string",
                        "description": (
                            "A single SELECT statement against the named table. "
                            "Numeric columns are stored as TEXT — wrap arithmetic "
                            "comparisons in CAST(<col> AS REAL) when needed."
                        ),
                    },
                },
                "required": ["table_name", "sql"],
            },
            tier="read",
            display_name="Query table",
            high_noise=True,
        )

    def call(self, *, table_name: str, sql: str) -> str:
        try:
            result = self.store.query_sql(table_name, sql)
        except ValueError as exc:
            return f"query_table error: {exc}"

        if not result.rows:
            return f"(no rows matched)\nSQL: {result.sql.strip()}"

        # Render as a small markdown table. The agent re-reads its
        # own tool output downstream, and markdown tables compress
        # well in dense token budgets; full-row JSON would be larger.
        header = "| " + " | ".join(result.columns) + " |"
        sep = "|" + "|".join("---" for _ in result.columns) + "|"
        body_lines: list[str] = []
        for row in result.rows[: self.max_rows]:
            body_lines.append("| " + " | ".join(_clip(c, self.max_col_chars) for c in row) + " |")
        truncated = len(result.rows) > self.max_rows
        summary = (
            f"\n_({len(result.rows)} rows, showing first {self.max_rows})_" if truncated else ""
        )
        return f"SQL: {result.sql.strip()}\n\n" + "\n".join([header, sep, *body_lines]) + summary


def _clip(value: object, max_chars: int) -> str:
    """Render one cell value as a tight string, truncating long values
    so a single text-blob row doesn't blow the rendering budget.
    Empty / None survive as empty cells; truncation is marked with
    `…` so the model can detect it."""
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
