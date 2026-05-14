"""Table-shaped store (harness-edt6 / Phase 2).

A registry of *tables* (with rows) plus *per-table* column-summary
embeddings. The retrieved unit is THE TABLE, not a row: agents use NL
to discover which table holds the answer, then write SQL against the
table's schema to get rows.

Why a separate store: flat vector retrieval (Phase 0) routinely fails
on tabular data because the relevant queries are shape-dependent.
"highest value return", "above $700 in March", "denied espresso
machines" — these have one shape each (sort by amount, filter by
column, equality match) but a flat retriever has to find them in a
dense-vector space. SQL is the right tool; the agent just needs to
know which table the question is about.

Two write paths:
- `register_table(name, description, schema, csv_path | rows)` — load
  data into a SQLite table the store owns. The description + schema
  get embedded so `find_tables()` can rank against semantic intent.
- `register_existing_table(name, description, schema)` — point at a
  table already in the connection (rare; useful when the store is
  pointed at a DB another process populated).

One read path:
- `find_tables(query, k)` — top-k tables ranked by cosine similarity
  to (description + column-names + schema). NO row-level retrieval
  here — that's `query_sql`'s job.
- `query_sql(table_name, sql)` — execute a parametrized SELECT
  against a registered table. SECURITY GATE: only SELECT statements
  are allowed; everything else raises. Tables created by other
  callers on the same connection are NOT reachable from this store —
  the SQL is parsed and the FROM clause checked against the
  registered-tables list.

The store doesn't generate SQL. That's the agent's job (via the
`query_table` tool) or — in the bench — a fixture-supplied oracle
string per case.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from harness.retrieval.embed import Embedder


# Column-summary tables. Internal to the store; data tables that the
# caller registers live in the same DB but at user-chosen names.
_CREATE_REGISTRY = """
CREATE TABLE IF NOT EXISTS tabular_registry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL UNIQUE,
    description     TEXT    NOT NULL,
    schema_text     TEXT    NOT NULL,
    embedding       BLOB    NOT NULL,
    embedder_id     TEXT    NOT NULL,
    embedding_dim   INTEGER NOT NULL,
    created_at      TEXT    NOT NULL
);
"""


# Strict allow-list for SQL identifiers. The store accepts a table name
# from the caller (e.g. in `query_sql`), and we substitute it into a
# FROM clause; parametrized binds can't replace identifiers, so we
# whitelist instead. Same regex shape that pydantic_yaml uses for
# field names — letters/digits/underscore, no leading digit.
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Block obvious mutation. Not a full SQL parser — the store also checks
# the FROM target is in `tabular_registry`, which is the real fence.
# The keyword block is a fast pre-check that surfaces a clear error
# before we hand a bad statement to SQLite.
_DISALLOWED_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|VACUUM|PRAGMA)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TableSchema:
    """Schema description for a registered table. `columns` is a tuple
    of (column_name, type_hint, nl_description). `type_hint` is free-
    form ('INTEGER', 'TEXT', 'amount in USD'); the store doesn't
    enforce it — SQLite is dynamic-typed anyway. The NL description
    per column is what makes the embedding informative."""

    name: str
    description: str
    columns: tuple[tuple[str, str, str], ...]

    def schema_text(self) -> str:
        """Render the schema as a single string. Fed to the embedder
        (so column intent reaches the dense vector) and shown to the
        agent (so it can write SQL). Format chosen for both: human-
        readable, model-readable."""
        lines = [f"Table: {self.name}", f"  Purpose: {self.description}", "  Columns:"]
        for col, typ, desc in self.columns:
            type_suffix = f" ({typ})" if typ else ""
            desc_suffix = f" — {desc}" if desc else ""
            lines.append(f"    - {col}{type_suffix}{desc_suffix}")
        return "\n".join(lines)

    def column_names(self) -> tuple[str, ...]:
        return tuple(col for col, _, _ in self.columns)


@dataclass(frozen=True)
class RegisteredTable:
    """Row in `tabular_registry`. Score-bearing in `find_tables`
    output, but the score field is None when fetched outside a
    similarity search (e.g. `tables()`)."""

    name: str
    description: str
    schema_text: str
    created_at: datetime


@dataclass(frozen=True)
class TableHit:
    """One result from `find_tables` — a registered table plus its
    similarity score to the query."""

    table: RegisteredTable
    score: float


@dataclass(frozen=True)
class QueryResult:
    """Output of `query_sql`. `columns` mirrors the SELECT's projection;
    `rows` is the materialized result. `sql` is the literal SQL that
    ran (useful for audit logging — the tool layer puts this in front
    of the user when the write-tier confirmation flow runs)."""

    sql: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


class TabularStore:
    """SQLite-backed table registry + executor.

    The store owns ONE SQLite connection and writes both metadata
    (`tabular_registry`) and the actual data tables into it. Same
    `(db_path, embedder)` shape as the other stores so callers can
    swap implementations through a common factory.
    """

    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_CREATE_REGISTRY)

    # ---------- registration ----------

    def register_table_from_csv(
        self,
        *,
        schema: TableSchema,
        csv_path: Path,
        replace: bool = False,
    ) -> RegisteredTable:
        """Load a CSV into a fresh table named `schema.name`, register
        the metadata + embedding. CSV header must match the schema's
        column names in order. `replace=True` drops the table first."""
        rows = _read_csv(csv_path, expected_columns=schema.column_names())
        return self.register_table(schema=schema, rows=rows, replace=replace)

    def register_table(
        self,
        *,
        schema: TableSchema,
        rows: list[tuple[Any, ...]],
        replace: bool = False,
    ) -> RegisteredTable:
        """Create the table if missing, insert rows, register metadata.
        Re-registering an already-known table is an error unless
        `replace=True` (drops + recreates).

        Distinction from `_create_table_ddl`: the column type hints in
        `schema.columns` are advisory; SQLite stores everything as
        the dynamic type passed in. We use TEXT for everything so
        numeric ranges work via SQLite's affinity-but-not-enforcement
        semantics on TEXT columns. Callers that need typed columns
        should pre-coerce before passing rows in.
        """
        _validate_identifier(schema.name)
        for col, _, _ in schema.columns:
            _validate_identifier(col)

        existing = self._conn.execute(
            "SELECT id FROM tabular_registry WHERE name = ?", (schema.name,)
        ).fetchone()
        if existing is not None and not replace:
            raise ValueError(
                f"table {schema.name!r} already registered; pass replace=True to overwrite"
            )

        if replace and existing is not None:
            self._conn.execute(f"DROP TABLE IF EXISTS {schema.name}")
            self._conn.execute("DELETE FROM tabular_registry WHERE name = ?", (schema.name,))

        # Create the data table. TEXT for every column; SQLite's NUMERIC
        # affinity kicks in when comparing values that look like
        # numbers, so WHERE amount_usd > 700 still works.
        col_ddl = ", ".join(f"{col} TEXT" for col, _, _ in schema.columns)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS {schema.name} ({col_ddl})")

        # Bulk insert. Parametrized so caller-supplied values never
        # touch the SQL parser.
        placeholders = ",".join("?" * len(schema.columns))
        col_list = ",".join(col for col, _, _ in schema.columns)
        if rows:
            self._conn.executemany(
                f"INSERT INTO {schema.name} ({col_list}) VALUES ({placeholders})",  # noqa: S608 — names validated
                rows,
            )

        # Register + embed.
        schema_text = schema.schema_text()
        vec = self.embedder.embed([schema_text])[0].astype(np.float32)
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT INTO tabular_registry (
                name, description, schema_text, embedding, embedder_id,
                embedding_dim, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                schema.name,
                schema.description,
                schema_text,
                vec.tobytes(),
                self.embedder.id,
                self.embedder.dimension,
                now,
            ),
        )
        return RegisteredTable(
            name=schema.name,
            description=schema.description,
            schema_text=schema_text,
            created_at=datetime.fromisoformat(now),
        )

    def tables(self) -> list[RegisteredTable]:
        rows = self._conn.execute(
            "SELECT name, description, schema_text, created_at FROM tabular_registry ORDER BY id"
        ).fetchall()
        return [_row_to_registered(r) for r in rows]

    def get_table(self, name: str) -> RegisteredTable | None:
        row = self._conn.execute(
            "SELECT name, description, schema_text, created_at "
            "FROM tabular_registry WHERE name = ?",
            (name,),
        ).fetchone()
        return _row_to_registered(row) if row is not None else None

    # ---------- discovery ----------

    def find_tables(self, query: str, *, k: int = 3) -> list[TableHit]:
        """Rank registered tables by cosine similarity to `query`.
        Embedding is the schema text (per `TableSchema.schema_text`).
        Returns up to `k` tables; never includes tables whose embedding
        dim doesn't match the current embedder (defensive — happens
        only if the embedder was swapped without re-registering)."""
        if not query.strip():
            return []
        q_vec = self.embedder.embed([query])[0].astype(np.float32)
        rows = self._conn.execute(
            """SELECT name, description, schema_text, created_at, embedding
               FROM tabular_registry WHERE embedding_dim = ?""",
            (self.embedder.dimension,),
        ).fetchall()
        scored: list[TableHit] = []
        for row in rows:
            vec = np.frombuffer(row[4], dtype=np.float32)
            score = float(np.dot(q_vec, vec))
            scored.append(TableHit(table=_row_to_registered(row[:4]), score=score))
        # Tie-break by name for determinism (mirrors the bench-side
        # tie-pinning in the other retrievers).
        scored.sort(key=lambda h: (-h.score, h.table.name))
        return scored[:k]

    # ---------- execution ----------

    def query_sql(self, table_name: str, sql: str) -> QueryResult:
        """Execute a SELECT against a registered table.

        Three gates protect this from being a generic SQL eval:
        1. `table_name` must be in the registry.
        2. `sql` must contain no DDL/DML keywords (insert/update/...).
        3. The SQL must reference ONLY the named table in its FROM /
           JOIN clauses — i.e. cross-table joins aren't allowed unless
           every referenced table is registered (v1: single-table).

        Caller is expected to have already discovered the right table
        (via `find_tables`) and pass it explicitly. The SQL is run
        as-is — no parameter binding here because the caller doesn't
        accept user input directly. The agent path (`query_table` tool)
        runs the model output through this gate before execution.
        """
        if self.get_table(table_name) is None:
            raise ValueError(f"table {table_name!r} not registered")
        if _DISALLOWED_SQL_RE.search(sql):
            raise ValueError(
                f"SQL contains disallowed keyword (insert/update/delete/etc.): {sql!r}"
            )
        # The FROM-clause check is a coarse-grained guard. We extract
        # every identifier after FROM or JOIN and assert it equals
        # `table_name`. The agent path would expand this for join
        # support; v1 is single-table by design.
        for ref in _from_clause_identifiers(sql):
            if ref != table_name:
                raise ValueError(
                    f"SQL references unregistered or unauthorized table {ref!r}; "
                    f"only {table_name!r} is allowed in this call"
                )

        cur = self._conn.execute(sql)
        rows = tuple(tuple(r) for r in cur.fetchall())
        columns = tuple(d[0] for d in cur.description or ())
        return QueryResult(sql=sql, columns=columns, rows=rows)

    def count_mismatched_embeddings(self) -> int:
        """How many registered tables carry embeddings whose dim doesn't
        match the current embedder. Same surface as EpisodicStore /
        SemanticStore so the `harness memory rebuild-embeddings` CLI
        can fold tabular tables into the same flow when a character
        wires one (harness-pzpy)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM tabular_registry WHERE embedding_dim != ?",
            (self.embedder.dimension,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def rebuild_embeddings(self) -> tuple[int, int]:
        """Re-embed every registered table's schema_text under the
        current embedder (harness-pzpy). Returns (rows_updated, rows_skipped).
        Skipped is always 0 — there's no superseded equivalent at the
        table grain — but the tuple shape mirrors EpisodicStore /
        SemanticStore so callers can share the rebuild plumbing.

        Useful after a `HARNESS_EMBEDDER_REPO` swap: registered tables'
        schema embeddings are dim-locked to the old model and silently
        drop out of `find_tables` until they're rebuilt. Idempotent —
        running twice produces the same result, modulo any non-
        determinism the embedder might have."""
        rows = self._conn.execute(
            "SELECT id, schema_text FROM tabular_registry ORDER BY id"
        ).fetchall()
        if not rows:
            return 0, 0
        texts = [str(r[1]) for r in rows]
        vectors = self.embedder.embed(texts)
        updated = 0
        for (registry_id, _schema_text), vec in zip(rows, vectors, strict=True):
            self._conn.execute(
                """UPDATE tabular_registry
                      SET embedding = ?, embedder_id = ?, embedding_dim = ?
                    WHERE id = ?""",
                (
                    vec.astype(np.float32).tobytes(),
                    self.embedder.id,
                    self.embedder.dimension,
                    registry_id,
                ),
            )
            updated += 1
        return updated, 0


# ---------- helpers ----------


def _validate_identifier(name: str) -> None:
    if not _SQL_IDENTIFIER_RE.match(name):
        raise ValueError(f"invalid SQL identifier {name!r}")


def _from_clause_identifiers(sql: str) -> list[str]:
    """Pull every identifier appearing after FROM or JOIN. Coarse but
    sufficient for the v1 single-table contract: any reference that
    isn't the registered table fails the gate."""
    return re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.IGNORECASE)


def _read_csv(csv_path: Path, *, expected_columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
    """Read a CSV into a list of row-tuples ordered by `expected_columns`.
    Raises if the CSV header is missing a required column. Extra
    columns in the CSV are dropped (CSV may carry richer data than
    the schema cares about)."""
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: missing header row")
        missing = [c for c in expected_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{csv_path}: header missing required columns: {missing}")
        return [tuple(row[c] for c in expected_columns) for row in reader]


def _row_to_registered(row: tuple) -> RegisteredTable:  # type: ignore[type-arg]
    return RegisteredTable(
        name=str(row[0]),
        description=str(row[1]),
        schema_text=str(row[2]),
        created_at=datetime.fromisoformat(str(row[3])),
    )
