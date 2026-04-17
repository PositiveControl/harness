from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcript (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session    TEXT    NOT NULL,
    channel    TEXT    NOT NULL,
    speaker    TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS transcript_session_idx
    ON transcript (session, id);

CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts
    USING fts5(content, content='transcript', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS transcript_ai AFTER INSERT ON transcript BEGIN
    INSERT INTO transcript_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


@dataclass(frozen=True)
class TranscriptMessage:
    id: int
    session: str
    channel: str
    speaker: str
    role: str
    content: str
    created_at: datetime


class Transcript:
    """Append-only transcript store. Single-file SQLite with WAL."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        # Wait up to 5s for other writers (chat + launchd scribe on same
        # file) before raising "database is locked". WAL already permits
        # concurrent readers; this covers the writer-writer edge.
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)

    def append(
        self,
        *,
        session: str,
        channel: str,
        speaker: str,
        role: str,
        content: str,
    ) -> TranscriptMessage:
        now = datetime.now(UTC).isoformat()
        cur = self._conn.execute(
            """INSERT INTO transcript (session, channel, speaker, role, content, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session, channel, speaker, role, content, now),
        )
        return TranscriptMessage(
            id=cur.lastrowid or 0,
            session=session,
            channel=channel,
            speaker=speaker,
            role=role,
            content=content,
            created_at=datetime.fromisoformat(now),
        )

    def tail(self, session: str, *, limit: int = 50) -> list[TranscriptMessage]:
        rows = self._conn.execute(
            """SELECT id, session, channel, speaker, role, content, created_at
               FROM transcript WHERE session = ? ORDER BY id DESC LIMIT ?""",
            (session, limit),
        ).fetchall()
        rows.reverse()
        return [_row_to_message(r) for r in rows]

    def fetch_after(self, session: str, *, after_id: int) -> list[TranscriptMessage]:
        """Fetch every message in `session` whose id > `after_id`, in
        insertion order. Used by the scribe to walk unprocessed turns
        past a watermark."""
        rows = self._conn.execute(
            """SELECT id, session, channel, speaker, role, content, created_at
               FROM transcript WHERE session = ? AND id > ? ORDER BY id""",
            (session, after_id),
        ).fetchall()
        return [_row_to_message(r) for r in rows]

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the underlying connection for out-of-band writes (the
        scribe keeps its watermark table in the same SQLite file)."""
        return self._conn

    def close(self) -> None:
        self._conn.close()


def _row_to_message(row: tuple) -> TranscriptMessage:  # type: ignore[type-arg]
    return TranscriptMessage(
        id=row[0],
        session=row[1],
        channel=row[2],
        speaker=row[3],
        role=row[4],
        content=row[5],
        created_at=datetime.fromisoformat(row[6]),
    )
