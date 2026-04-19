from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS compaction_summary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    summary       TEXT    NOT NULL,
    up_to_turn_id INTEGER NOT NULL,
    covered_turns INTEGER NOT NULL,
    model_id      TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS compaction_summary_session_idx
    ON compaction_summary (session_id, id);
"""


@dataclass(frozen=True)
class CompactionRecord:
    id: int
    session_id: str
    summary: str
    up_to_turn_id: int
    covered_turns: int
    model_id: str
    created_at: datetime


class CompactionStore:
    """Append-only store of compaction summaries. Each row supersedes
    the previous summary for its session — we only ever read the latest
    row per session — but older rows are kept so a corrupt compaction
    can be rolled back by manually deleting the bad row.

    Lives in the same SQLite file as the transcript so the watermark
    (`up_to_turn_id`) references transcript IDs directly."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # `check_same_thread=False` mirrors Transcript / EpisodicStore /
        # SemanticStore: the Textual TUI fires user-triggered ops like
        # /compact on worker threads while the connection lives on the
        # main thread. SQLite's default thread check would reject the
        # cross-thread use; WAL + busy_timeout serialize the actual
        # writes, and only one writer runs at a time (is_busy gates
        # /compact out when a turn worker is active).
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)

    def latest_for_session(self, session_id: str) -> CompactionRecord | None:
        row = self._conn.execute(
            """SELECT id, session_id, summary, up_to_turn_id, covered_turns,
                      model_id, created_at
               FROM compaction_summary
               WHERE session_id = ?
               ORDER BY id DESC
               LIMIT 1""",
            (session_id,),
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def append(
        self,
        *,
        session_id: str,
        summary: str,
        up_to_turn_id: int,
        covered_turns: int,
        model_id: str,
    ) -> CompactionRecord:
        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            """INSERT INTO compaction_summary
               (session_id, summary, up_to_turn_id, covered_turns, model_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, summary, up_to_turn_id, covered_turns, model_id, now),
        )
        return CompactionRecord(
            id=cur.lastrowid or 0,
            session_id=session_id,
            summary=summary,
            up_to_turn_id=up_to_turn_id,
            covered_turns=covered_turns,
            model_id=model_id,
            created_at=datetime.fromisoformat(now),
        )

    def close(self) -> None:
        self._conn.close()


def _row_to_record(row: tuple) -> CompactionRecord:  # type: ignore[type-arg]
    return CompactionRecord(
        id=row[0],
        session_id=row[1],
        summary=row[2],
        up_to_turn_id=row[3],
        covered_turns=row[4],
        model_id=row[5],
        created_at=datetime.fromisoformat(row[6]),
    )
