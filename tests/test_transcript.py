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
