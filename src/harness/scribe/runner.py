from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from harness.scribe.extractor import ScribeResult, extract_candidates
from harness.scribe.locks import session_lock

if TYPE_CHECKING:
    from pathlib import Path

    from harness.character import Character
    from harness.model.adapter import ModelAdapter
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore
    from harness.store.transcript import Transcript

_WATERMARK_SCHEMA = """
CREATE TABLE IF NOT EXISTS scribe_watermark (
    session_id             TEXT PRIMARY KEY,
    last_processed_turn_id INTEGER NOT NULL,
    updated_at             TEXT    NOT NULL
);
"""


@dataclass
class ScribeRunSummary:
    session_id: str
    turns_processed: int = 0
    episodic_written: int = 0
    semantic_written: int = 0
    windows: int = 0
    parse_errors: list[str] = field(default_factory=list)


def _ensure_watermark_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_WATERMARK_SCHEMA)


def _get_watermark(conn: sqlite3.Connection, session_id: str) -> int:
    _ensure_watermark_table(conn)
    row = conn.execute(
        "SELECT last_processed_turn_id FROM scribe_watermark WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _set_watermark(conn: sqlite3.Connection, session_id: str, turn_id: int) -> None:
    _ensure_watermark_table(conn)
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """INSERT INTO scribe_watermark (session_id, last_processed_turn_id, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
             last_processed_turn_id = excluded.last_processed_turn_id,
             updated_at = excluded.updated_at""",
        (session_id, turn_id, now),
    )


def _persist_candidates(
    result: ScribeResult,
    *,
    episodic_store: EpisodicStore,
    semantic_store: SemanticStore,
    session_id: str,
    source_label: str,
    user_id: str | None,
) -> tuple[int, int]:
    episodic_written = 0
    for cand in result.episodic:
        episodic_store.ingest(
            external_id=None,  # working-tier writes don't collide with seeds
            title=cand.title,
            body=cand.body,
            principle=cand.principle,
            tags=cand.tags,
            tier="working",
            source=source_label,
            session_id=session_id,
            user_id=user_id,
        )
        episodic_written += 1

    semantic_written = 0
    for fact in result.semantic:
        semantic_store.add(
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            confidence=fact.confidence,
            source=source_label,
            session_id=session_id,
            user_id=user_id,
            tier="working",
        )
        semantic_written += 1

    return episodic_written, semantic_written


def run_scribe(
    adapter: ModelAdapter,
    character: Character,
    transcript: Transcript,
    episodic_store: EpisodicStore,
    semantic_store: SemanticStore,
    *,
    session_id: str,
    user_id: str | None = None,
    window_size: int = 20,
    lock_dir: Path | None = None,
    blocking_lock: bool = True,
) -> ScribeRunSummary:
    """Walk unprocessed transcript turns for one session, extract
    candidates in windows, and persist them. Advances a watermark so
    reruns are incremental.

    `user_id` tags every candidate with a user scope — the session's
    participant whose relationship memory this data belongs to. Pass
    None to write shared (character-level) memory; usually you want the
    speaker.

    Windows are non-overlapping for simplicity. If the model produces
    unparseable output for a window, that window's error is logged in
    the summary and processing continues — we do NOT re-try, because
    malformed output tends to repeat.

    The run is wrapped in a filesystem advisory lock keyed by
    session_id, so two concurrent scribes (chat + nightly launchd job)
    can't double-process the same window. Default lock_dir is
    <db_dir>/locks; tests can pin their own. Pass blocking_lock=False to
    raise ScribeLockBusy instead of waiting."""
    summary = ScribeRunSummary(session_id=session_id)

    lock_path = lock_dir if lock_dir is not None else transcript.db_path.parent / "locks"
    with session_lock(session_id, lock_path, blocking=blocking_lock):
        watermark = _get_watermark(transcript.connection, session_id)
        turns = transcript.fetch_after(session_id, after_id=watermark)
        if not turns:
            return summary

        for start in range(0, len(turns), window_size):
            window = turns[start : start + window_size]
            if not window:
                continue
            summary.windows += 1
            source_label = f"scribe:session={session_id}:turns={window[0].id}-{window[-1].id}"

            result = extract_candidates(adapter, character, turns=window)
            if result.parse_error is not None:
                summary.parse_errors.append(f"window {start}: {result.parse_error}")

            wrote_ep, wrote_sem = _persist_candidates(
                result,
                episodic_store=episodic_store,
                semantic_store=semantic_store,
                session_id=session_id,
                source_label=source_label,
                user_id=user_id,
            )
            summary.episodic_written += wrote_ep
            summary.semantic_written += wrote_sem
            summary.turns_processed += len(window)
            _set_watermark(transcript.connection, session_id, window[-1].id)

    return summary
