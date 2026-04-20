"""Classic-REPL slash-handler + ContextMeter tests (harness-c1r).

Scope:
- `/clear` — ContextMeter.clear pins a cutoff; load_history then
  returns only rows appended after the cutoff, and a prior
  compaction summary no longer leaks back into the next turn.
- `handle_clear_slash` — prints a confirmation banner and calls
  through to ctx_meter.clear.

Stores are real SQLite files under tmp_path per the project test
convention — no mocking of the transcript layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from harness.character import load_character
from harness.cli_repl import ContextMeter, handle_clear_slash
from harness.compaction import CompactionStore
from harness.model.echo import EchoAdapter
from harness.store.transcript import Transcript

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON = REPO_ROOT / "character" / "airton"


def _seed_transcript(transcript: Transcript, session: str, *, pairs: int) -> int:
    """Append `pairs` user+assistant exchanges; return the last row id."""
    for i in range(pairs):
        transcript.append(
            session=session,
            channel="cli",
            speaker="mark",
            role="user",
            content=f"u{i}",
        )
        transcript.append(
            session=session,
            channel="cli",
            speaker="airton",
            role="assistant",
            content=f"a{i}",
        )
    tail = transcript.tail(session, limit=1)
    return tail[-1].id if tail else 0


def _ctx_meter(
    tmp_path: Path,
    *,
    with_compaction: bool = False,
    session: str = "test",
) -> tuple[ContextMeter, Transcript, CompactionStore | None]:
    db = tmp_path / "harness.sqlite"
    transcript = Transcript(db)
    compaction = CompactionStore(db) if with_compaction else None
    meter = ContextMeter(
        adapter=EchoAdapter(),
        character=load_character(AIRTON),
        transcript=transcript,
        compaction_store=compaction,
        session=session,
        console=Console(file=open("/dev/null", "w")),  # noqa: SIM115 — lifetime = test
    )
    return meter, transcript, compaction


def test_clear_pins_cutoff_to_current_tip(tmp_path: Path) -> None:
    """After clear(), load_history must return only rows appended
    after the moment of the clear. Rows already in the transcript
    stay in the DB but never surface to the model again."""
    meter, transcript, _ = _ctx_meter(tmp_path)
    pre_id = _seed_transcript(transcript, meter.session, pairs=3)

    meter.clear()

    assert meter.clear_after_id == pre_id
    summary, history = meter.load_history()
    assert summary is None
    assert history == []


def test_clear_lets_new_turns_rebuild_history(tmp_path: Path) -> None:
    """Post-clear, a new turn's appended rows must be the only thing
    load_history returns. Verifies the cutoff moves-with-new-rows
    semantics, not a permanent blackout."""
    meter, transcript, _ = _ctx_meter(tmp_path)
    _seed_transcript(transcript, meter.session, pairs=2)
    meter.clear()

    transcript.append(
        session=meter.session,
        channel="cli",
        speaker="mark",
        role="user",
        content="after-clear prompt",
    )
    transcript.append(
        session=meter.session,
        channel="cli",
        speaker="airton",
        role="assistant",
        content="after-clear reply",
    )

    _, history = meter.load_history()
    assert [m.role for m in history] == ["user", "assistant"]
    assert [m.content for m in history] == ["after-clear prompt", "after-clear reply"]


def test_clear_suppresses_compaction_summary(tmp_path: Path) -> None:
    """A compaction summary is normally prepended to history so the
    model remembers folded turns. /clear must override that too —
    otherwise the 'fresh start' wouldn't actually be fresh."""
    meter, transcript, compaction = _ctx_meter(tmp_path, with_compaction=True)
    assert compaction is not None  # for mypy
    last_id = _seed_transcript(transcript, meter.session, pairs=3)
    compaction.append(
        session_id=meter.session,
        summary="summary-of-folded-turns",
        up_to_turn_id=last_id,
        covered_turns=6,
        model_id="echo",
    )

    # Sanity: without clear, load_history returns the summary.
    summary, _ = meter.load_history()
    assert summary is not None
    assert "summary-of-folded-turns" in summary.content

    meter.clear()

    summary_after, history_after = meter.load_history()
    assert summary_after is None
    assert history_after == []


def test_clear_is_a_noop_on_empty_transcript(tmp_path: Path) -> None:
    """A brand-new session with nothing in the transcript is a valid
    /clear target — set the cutoff to 0 so future rows (ids start at
    1) still surface."""
    meter, transcript, _ = _ctx_meter(tmp_path)
    meter.clear()
    assert meter.clear_after_id == 0

    transcript.append(
        session=meter.session,
        channel="cli",
        speaker="mark",
        role="user",
        content="first prompt",
    )
    _, history = meter.load_history()
    assert [m.content for m in history] == ["first prompt"]


def test_handle_clear_slash_calls_meter_and_prints_banner(tmp_path: Path) -> None:
    """The REPL's /clear handler is a thin shim: delegate to the
    ContextMeter and emit a visual separator so the user sees the
    reset happened. Verified with a spy on clear() + a captured
    Console."""
    from io import StringIO

    buf = StringIO()
    console = Console(file=buf, force_terminal=False)
    meter = MagicMock(spec=ContextMeter)

    handle_clear_slash(meter, console)

    meter.clear.assert_called_once()
    assert "context cleared" in buf.getvalue()


@pytest.mark.parametrize("cmd", ["/clear", "/CLEAR", "/Clear"])
def test_clear_command_set_is_case_insensitive(cmd: str) -> None:
    """Classic REPL lowercases user_input before matching, so every
    case variant of /clear must be in the command set."""
    from harness.cli_classic import _CLEAR_COMMANDS

    assert cmd.lower() in _CLEAR_COMMANDS
