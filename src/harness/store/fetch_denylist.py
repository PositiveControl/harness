"""Persistent fetch_url denylist (harness-4dgm).

When fetch_url hits HTTP 401 or 403 we add the host to this store so
future fetches short-circuit before the network call. CDN/Cloudflare
bot blocks and policy 401s tend to be sticky for weeks; making the
model re-discover them every turn wastes time and tokens.

Storage shares the per-character `harness.sqlite` with episodic /
semantic / audit / transcript — one file per character keeps the
denylist scoped (atc's aviation allowlist context shouldn't leak
into airton's general-web sessions). Append-first via UPSERT: the
row remembers `count`, `first_seen_at`, and `last_seen_at` so
`harness denylist list` can show whether a host is freshly blocked
or has been hammered for weeks.

TTL: 30 days from `last_seen_at`. `is_blocked()` filters expired
rows out of the read path but leaves them on disk so manual
`harness denylist list --expired` can still inspect them.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_TTL_DAYS = 30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_denylist (
    host           TEXT    PRIMARY KEY,
    last_status    INTEGER NOT NULL,
    last_reason    TEXT    NOT NULL,
    last_url       TEXT    NOT NULL,
    first_seen_at  TEXT    NOT NULL,
    last_seen_at   TEXT    NOT NULL,
    count          INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS fetch_denylist_last_seen_idx
    ON fetch_denylist (last_seen_at);
"""


@dataclass(frozen=True)
class DenylistEntry:
    host: str
    last_status: int
    last_reason: str
    last_url: str
    first_seen_at: datetime
    last_seen_at: datetime
    count: int

    def is_active(self, *, now: datetime, ttl_days: int) -> bool:
        return self.last_seen_at >= now - timedelta(days=ttl_days)


class FetchDenylistStore:
    """SQLite-backed host denylist. Same file as the other per-character
    stores; safe to construct alongside AuditStore / EpisodicStore on the
    same db_path."""

    def __init__(self, db_path: Path, *, ttl_days: int = DEFAULT_TTL_DAYS):
        self.db_path = db_path
        self.ttl_days = ttl_days
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)

    def record(
        self,
        *,
        host: str,
        status: int,
        reason: str,
        url: str,
        now: datetime | None = None,
    ) -> DenylistEntry:
        """UPSERT a host. On first sighting `first_seen_at == last_seen_at`
        and `count = 1`; on repeat hits we bump `count`, refresh
        `last_seen_at` / `last_status` / `last_reason` / `last_url`, and
        leave `first_seen_at` alone so the operator can see how long the
        host has been blocked."""
        key = host.lower()
        ts = (now or datetime.now(UTC)).isoformat()
        self._conn.execute(
            """INSERT INTO fetch_denylist (
                host, last_status, last_reason, last_url,
                first_seen_at, last_seen_at, count
            ) VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(host) DO UPDATE SET
                last_status = excluded.last_status,
                last_reason = excluded.last_reason,
                last_url    = excluded.last_url,
                last_seen_at = excluded.last_seen_at,
                count       = fetch_denylist.count + 1""",
            (key, status, reason, url, ts, ts),
        )
        entry = self.get(key)
        assert entry is not None
        return entry

    def is_blocked(self, host: str, *, now: datetime | None = None) -> DenylistEntry | None:
        """Return the active entry for `host`, or None if absent or
        expired. Expired rows stay in the table — `list_all(include_expired=True)`
        can inspect them — but they don't gate fetches."""
        entry = self.get(host)
        if entry is None:
            return None
        check_at = now or datetime.now(UTC)
        if not entry.is_active(now=check_at, ttl_days=self.ttl_days):
            return None
        return entry

    def get(self, host: str) -> DenylistEntry | None:
        row = self._conn.execute(
            """SELECT host, last_status, last_reason, last_url,
                      first_seen_at, last_seen_at, count
               FROM fetch_denylist WHERE host = ?""",
            (host.lower(),),
        ).fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    def list_all(
        self,
        *,
        include_expired: bool = False,
        now: datetime | None = None,
    ) -> list[DenylistEntry]:
        """Return entries newest-first. Expired rows are hidden by
        default so the operator sees what's actively blocking fetches."""
        rows = self._conn.execute(
            """SELECT host, last_status, last_reason, last_url,
                      first_seen_at, last_seen_at, count
               FROM fetch_denylist
               ORDER BY last_seen_at DESC"""
        ).fetchall()
        entries = [_row_to_entry(r) for r in rows]
        if include_expired:
            return entries
        check_at = now or datetime.now(UTC)
        return [e for e in entries if e.is_active(now=check_at, ttl_days=self.ttl_days)]

    def clear(self, host: str | None = None) -> int:
        """Drop one host (when given) or every row (when None). Returns
        the number of rows removed."""
        if host is None:
            cur = self._conn.execute("DELETE FROM fetch_denylist")
        else:
            cur = self._conn.execute(
                "DELETE FROM fetch_denylist WHERE host = ?",
                (host.lower(),),
            )
        return cur.rowcount or 0

    def close(self) -> None:
        self._conn.close()


def _row_to_entry(row: tuple) -> DenylistEntry:  # type: ignore[type-arg]
    host, status, reason, url, first_seen_at, last_seen_at, count = row
    return DenylistEntry(
        host=host,
        last_status=int(status),
        last_reason=reason,
        last_url=url,
        first_seen_at=datetime.fromisoformat(first_seen_at),
        last_seen_at=datetime.fromisoformat(last_seen_at),
        count=int(count),
    )
