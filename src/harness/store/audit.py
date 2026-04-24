"""Per-turn audit log (harness-ywp.2).

One row per chat turn, recording what was retrieved, what the model
replied, what citations the tools grounded vs. what the reply cited,
what tools ran, and a retrieval confidence score. Load-bearing for
liability framing across the airton_c use-cases — every turn the
character produces advisory output, we want to be able to answer
'what did Airton know when it said that?' from durable audit rows.

Storage: shares the per-character `harness.sqlite` with episodic /
semantic / transcript. One file per character = one backup unit.

Denormalisation policy: `retrieval_top_score` is a dedicated REAL
column so the threshold calibration follow-up (harness-5c0) can
histogram scores without decoding JSON. Everything else that's a
set / list is stored as JSON text — consumers reconstruct via the
AuditRecord dataclass below, not via SQL."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from harness.tools.base import ToolHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_turns (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id               TEXT    NOT NULL,
    session               TEXT    NOT NULL,
    character             TEXT    NOT NULL,
    user_id               TEXT,
    user_message          TEXT    NOT NULL,
    model_reply           TEXT    NOT NULL,
    retrieval_top_score   REAL,
    retrieval_hits_json   TEXT,
    citations_grounded_json TEXT,
    citations_cited_json  TEXT,
    tools_ran_json        TEXT,
    created_at            TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS audit_turns_session_idx
    ON audit_turns (session, id);
CREATE INDEX IF NOT EXISTS audit_turns_created_at_idx
    ON audit_turns (created_at);
"""


@dataclass(frozen=True)
class AuditRecord:
    id: int
    turn_id: str
    session: str
    character: str
    user_id: str | None
    user_message: str
    model_reply: str
    retrieval_top_score: float | None
    retrieval_hits: tuple[ToolHit, ...]
    citations_grounded: frozenset[str]
    citations_cited: frozenset[str]
    tools_ran: frozenset[str]
    created_at: datetime


def _serialise_hits(hits: tuple[ToolHit, ...]) -> str | None:
    """JSON-encode a tuple of ToolHit for the retrieval_hits_json
    column. Empty tuple → None so the column stays NULL (easier to
    distinguish 'retrieval didn't run' from 'retrieval returned
    nothing' — the latter would be an empty list, not null)."""
    if not hits:
        return None
    return json.dumps(
        [
            {
                "source": h.source,
                "external_id": h.external_id,
                "title": h.title,
                "score": h.score,
                "principle": h.principle,
            }
            for h in hits
        ]
    )


def _deserialise_hits(raw: str | None) -> tuple[ToolHit, ...]:
    if raw is None:
        return ()
    data = json.loads(raw)
    return tuple(
        ToolHit(
            source=item["source"],
            external_id=item.get("external_id"),
            title=item["title"],
            score=float(item["score"]),
            principle=item.get("principle"),
        )
        for item in data
    )


def _serialise_set(items: frozenset[str]) -> str | None:
    if not items:
        return None
    # Sort for stable JSON (helps diffing audit output in reviews).
    return json.dumps(sorted(items))


def _deserialise_set(raw: str | None) -> frozenset[str]:
    if raw is None:
        return frozenset()
    return frozenset(json.loads(raw))


class AuditStore:
    """Append-only per-turn audit log. Single-file SQLite with WAL
    (shares the character's harness.sqlite with other stores)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)

    def record(
        self,
        *,
        session: str,
        character: str,
        user_id: str | None,
        user_message: str,
        model_reply: str,
        retrieval_hits: tuple[ToolHit, ...] = (),
        citations_grounded: frozenset[str] = frozenset(),
        citations_cited: frozenset[str] = frozenset(),
        tools_ran: frozenset[str] = frozenset(),
        turn_id: str | None = None,
    ) -> AuditRecord:
        """Insert one audit row. Returns the AuditRecord with its
        assigned id + timestamp.

        `retrieval_top_score` is derived from the hits tuple (max
        score, or None when empty) — callers don't compute it
        separately."""
        tid = turn_id or uuid.uuid4().hex[:12]
        top_score = max((h.score for h in retrieval_hits), default=None)
        now = datetime.now(UTC)
        now_iso = now.isoformat()

        cur = self._conn.execute(
            """INSERT INTO audit_turns (
                turn_id, session, character, user_id,
                user_message, model_reply,
                retrieval_top_score, retrieval_hits_json,
                citations_grounded_json, citations_cited_json,
                tools_ran_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tid,
                session,
                character,
                user_id,
                user_message,
                model_reply,
                top_score,
                _serialise_hits(retrieval_hits),
                _serialise_set(citations_grounded),
                _serialise_set(citations_cited),
                _serialise_set(tools_ran),
                now_iso,
            ),
        )
        return AuditRecord(
            id=cur.lastrowid or 0,
            turn_id=tid,
            session=session,
            character=character,
            user_id=user_id,
            user_message=user_message,
            model_reply=model_reply,
            retrieval_top_score=top_score,
            retrieval_hits=retrieval_hits,
            citations_grounded=citations_grounded,
            citations_cited=citations_cited,
            tools_ran=tools_ran,
            created_at=now,
        )

    def get(self, record_id: int) -> AuditRecord:
        row = self._conn.execute(
            """SELECT id, turn_id, session, character, user_id,
                      user_message, model_reply, retrieval_top_score,
                      retrieval_hits_json, citations_grounded_json,
                      citations_cited_json, tools_ran_json, created_at
               FROM audit_turns WHERE id = ?""",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no audit record #{record_id}")
        return self._row_to_record(row)

    def list(
        self, *, session: str | None = None, limit: int = 50
    ) -> list[AuditRecord]:
        """Return the most recent audit rows, newest first. Filters
        by session when given; otherwise across all sessions."""
        if session is not None:
            rows = self._conn.execute(
                """SELECT id, turn_id, session, character, user_id,
                          user_message, model_reply, retrieval_top_score,
                          retrieval_hits_json, citations_grounded_json,
                          citations_cited_json, tools_ran_json, created_at
                   FROM audit_turns WHERE session = ?
                   ORDER BY id DESC LIMIT ?""",
                (session, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT id, turn_id, session, character, user_id,
                          user_message, model_reply, retrieval_top_score,
                          retrieval_hits_json, citations_grounded_json,
                          citations_cited_json, tools_ran_json, created_at
                   FROM audit_turns
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_record(row: tuple) -> AuditRecord:  # type: ignore[type-arg]
        (
            rid,
            turn_id,
            session,
            character,
            user_id,
            user_message,
            model_reply,
            top_score,
            hits_json,
            cit_grounded_json,
            cit_cited_json,
            tools_json,
            created_at,
        ) = row
        return AuditRecord(
            id=rid,
            turn_id=turn_id,
            session=session,
            character=character,
            user_id=user_id,
            user_message=user_message,
            model_reply=model_reply,
            retrieval_top_score=top_score,
            retrieval_hits=_deserialise_hits(hits_json),
            citations_grounded=_deserialise_set(cit_grounded_json),
            citations_cited=_deserialise_set(cit_cited_json),
            tools_ran=_deserialise_set(tools_json),
            created_at=datetime.fromisoformat(created_at),
        )
