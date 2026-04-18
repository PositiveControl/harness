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


@dataclass(frozen=True)
class SessionStats:
    """Aggregate stats for one transcript session. Used by the
    introspect tool (harness-l62) so scope=session can report how
    long the session has been running and how many turns landed
    without needing any in-memory accumulator."""

    first_at: datetime
    last_at: datetime
    total_rows: int
    user_turns: int
    assistant_turns: int


class Transcript:
    """Append-only transcript store. Single-file SQLite with WAL."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # `check_same_thread=False`: the Textual TUI runs the worker
        # that writes turns on a background thread while the main
        # thread composes the UI. SQLite itself is thread-safe when
        # built with threadsafe=1 (the Python default) and the
        # per-connection Python lock + PRAGMA busy_timeout already
        # serialize writes; the check_same_thread guard is a Python-
        # side safety rail we opt out of intentionally. No cursors
        # are shared across threads — each .execute() here uses an
        # implicit fresh cursor.
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
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

    def session_stats(self, session: str) -> SessionStats | None:
        """Return aggregate stats for `session`, or None when the
        session has no rows yet. Single-query MIN/MAX/COUNT so the
        introspect tool doesn't have to pull every row just to count
        turns."""
        row = self._conn.execute(
            """SELECT MIN(created_at), MAX(created_at), COUNT(*),
                      SUM(CASE WHEN role='user' THEN 1 ELSE 0 END),
                      SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END)
               FROM transcript WHERE session = ?""",
            (session,),
        ).fetchone()
        if row is None or row[2] == 0:
            return None
        return SessionStats(
            first_at=datetime.fromisoformat(row[0]),
            last_at=datetime.fromisoformat(row[1]),
            total_rows=int(row[2]),
            user_turns=int(row[3] or 0),
            assistant_turns=int(row[4] or 0),
        )

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
