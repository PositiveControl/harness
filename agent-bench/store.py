"""Append-first, attributed, timestamped results store — SQLite + JSONL mirror.

Same philosophy as the harness stores: nothing is overwritten. Re-runs append,
so a results series over time is queryable. Per-run scores are stored as a JSON
blob so adding/removing a metric needs no schema migration.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    framework     TEXT NOT NULL,
    version       TEXT NOT NULL,
    run_idx       INTEGER NOT NULL,
    started_at    TEXT NOT NULL,
    spec_sha      TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    exit_ok       INTEGER NOT NULL,
    duration_s    REAL NOT NULL,
    scores_json   TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
"""


@dataclass
class RunRecord:
    framework: str
    version: str
    run_idx: int
    spec_sha: str
    model_id: str
    exit_ok: bool
    duration_s: float
    scores: dict[str, object]
    manifest: dict[str, object]
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ResultsStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._jsonl = db_path.with_suffix(".jsonl")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def append(self, rec: RunRecord) -> int:
        row = (
            rec.framework,
            rec.version,
            rec.run_idx,
            rec.started_at,
            rec.spec_sha,
            rec.model_id,
            int(rec.exit_ok),
            rec.duration_s,
            json.dumps(rec.scores),
            json.dumps(rec.manifest),
        )
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (framework, version, run_idx, started_at, spec_sha, "
                "model_id, exit_ok, duration_s, scores_json, manifest_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            run_id = int(cur.lastrowid or 0)
        with self._jsonl.open("a") as fh:
            fh.write(json.dumps({"id": run_id, **asdict(rec)}) + "\n")
        return run_id

    def all_rows(self) -> list[dict[str, object]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM runs ORDER BY framework, run_idx").fetchall()
        out: list[dict[str, object]] = []
        for r in rows:
            d = dict(r)
            d["scores"] = json.loads(d.pop("scores_json"))
            d["manifest"] = json.loads(d.pop("manifest_json"))
            out.append(d)
        return out
