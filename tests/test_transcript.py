from __future__ import annotations

from pathlib import Path

from harness.store.transcript import Transcript


def test_append_and_tail_round_trip(tmp_path: Path) -> None:
    ts = Transcript(tmp_path / "harness.sqlite")
    try:
        ts.append(
            session="s1", channel="cli", speaker="mark", role="user", content="Morning, Airton."
        )
        ts.append(
            session="s1",
            channel="cli",
            speaker="airton",
            role="assistant",
            content="Morning. Queue's quiet.",
        )
        ts.append(
            session="s2", channel="cli", speaker="mark", role="user", content="Unrelated to s1."
        )

        tail = ts.tail("s1")
        assert [m.content for m in tail] == [
            "Morning, Airton.",
            "Morning. Queue's quiet.",
        ]
        assert tail[0].created_at.tzinfo is not None
    finally:
        ts.close()


def test_tail_respects_limit(tmp_path: Path) -> None:
    ts = Transcript(tmp_path / "harness.sqlite")
    try:
        for i in range(10):
            ts.append(session="s", channel="cli", speaker="mark", role="user", content=f"msg {i}")
        tail = ts.tail("s", limit=3)
        assert [m.content for m in tail] == ["msg 7", "msg 8", "msg 9"]
    finally:
        ts.close()


def test_session_stats_aggregates(tmp_path: Path) -> None:
    """harness-l62: session_stats returns the aggregate query the
    introspect tool needs — min/max timestamp, total rows, user and
    assistant counts. Returns None on empty sessions."""
    ts = Transcript(tmp_path / "harness.sqlite")
    try:
        assert ts.session_stats("empty") is None

        ts.append(session="s", channel="cli", speaker="mark", role="user", content="hi")
        ts.append(session="s", channel="cli", speaker="airton", role="assistant", content="hey")
        ts.append(session="s", channel="cli", speaker="mark", role="user", content="one more")
        ts.append(
            session="s", channel="cli", speaker="read_file", role="tool", content="file bytes"
        )

        stats = ts.session_stats("s")
        assert stats is not None
        assert stats.total_rows == 4
        assert stats.user_turns == 2
        assert stats.assistant_turns == 1
        assert stats.first_at <= stats.last_at
    finally:
        ts.close()


def test_session_stats_excludes_other_sessions(tmp_path: Path) -> None:
    """Stats for 's1' must not leak rows from 's2'."""
    ts = Transcript(tmp_path / "harness.sqlite")
    try:
        ts.append(session="s1", channel="cli", speaker="mark", role="user", content="one")
        ts.append(session="s2", channel="cli", speaker="mark", role="user", content="two")
        ts.append(session="s2", channel="cli", speaker="mark", role="user", content="three")

        s1 = ts.session_stats("s1")
        s2 = ts.session_stats("s2")
        assert s1 is not None
        assert s2 is not None
        assert s1.total_rows == 1
        assert s2.total_rows == 2
    finally:
        ts.close()
