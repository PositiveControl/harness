from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from harness.store.episodic import EpisodicStore
from harness.tools.transcript_ingest import TranscriptIngestTool


class _FakeEmbedder:
    id = "fake:test"
    dimension = 8

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[0.1] * 8 for _ in texts], dtype=np.float32)


def _tool(db_path: Path) -> TranscriptIngestTool:
    store = EpisodicStore(db_path, embedder=_FakeEmbedder())  # type: ignore[arg-type]
    return TranscriptIngestTool(store=store)


def _turns(n: int = 3) -> list[dict[str, Any]]:
    return [
        {"speaker": "TWR", "text": f"Cleared for takeoff turn {i}", "timestamp": f"T+00:0{i}:00"}
        for i in range(n)
    ]


def test_happy_path_inserts_rows_with_expected_fields(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    out = tool.call(
        turns=_turns(3),
        session_id="sim-42",
        user_id="trainee-alice",
        source_tag="maxsim",
    )
    assert "ingested 3 turns" in out
    assert "sim-42" in out
    assert "trainee=trainee-alice" in out

    rows = list(tool.store.all())
    assert len(rows) == 3
    for r in rows:
        assert r.session_id == "sim-42"
        assert r.user_id == "trainee-alice"
        assert r.tier == "working"
        assert r.source == "transcript:maxsim"
        # Title carries speaker, timestamp, and a snippet of text for
        # BM25 weight.
        assert r.title.startswith("TWR @ T+00:0")
        assert "Cleared for takeoff" in r.title


def test_idempotent_on_re_ingest(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    tool.call(turns=_turns(3), session_id="sim-1", user_id="t", source_tag="manual")
    out = tool.call(turns=_turns(3), session_id="sim-1", user_id="t", source_tag="manual")
    assert "ingested 0 turns" in out
    assert "3 already present" in out
    assert len(list(tool.store.all())) == 3


def test_distinct_sessions_do_not_collide(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    tool.call(turns=_turns(2), session_id="sim-A", user_id="t", source_tag="maxsim")
    tool.call(turns=_turns(2), session_id="sim-B", user_id="t", source_tag="maxsim")
    assert len(list(tool.store.all())) == 4


def test_empty_turns_rejected(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match="non-empty"):
        tool.call(turns=[], session_id="s", user_id="u", source_tag="x")


def test_blank_session_id_rejected(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match="session_id"):
        tool.call(turns=_turns(1), session_id="   ", user_id="u", source_tag="x")


def test_blank_user_id_rejected(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match="user_id"):
        tool.call(turns=_turns(1), session_id="s", user_id="", source_tag="x")


def test_blank_source_tag_rejected(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match="source_tag"):
        tool.call(turns=_turns(1), session_id="s", user_id="u", source_tag="")


def test_turn_missing_field_rejected(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    bad = [{"speaker": "TWR", "text": "hi"}]  # no timestamp
    with pytest.raises(ValueError, match="turn 0 missing required field"):
        tool.call(turns=bad, session_id="s", user_id="u", source_tag="x")


def test_spec_is_write_tier(tmp_path: Path) -> None:
    spec = _tool(tmp_path / "x.sqlite").spec
    assert spec.tier == "write"
    assert spec.name == "transcript_ingest"
    assert set(spec.parameters["required"]) == {
        "turns",
        "session_id",
        "user_id",
        "source_tag",
    }


def test_long_text_title_is_truncated(tmp_path: Path) -> None:
    tool = _tool(tmp_path / "mem.sqlite")
    long_turn = [
        {"speaker": "ATIS", "text": "A" * 500, "timestamp": "T+00:00:00"},
    ]
    tool.call(turns=long_turn, session_id="s", user_id="u", source_tag="manual")
    (row,) = list(tool.store.all())
    # Title snippet capped at 100 chars + ellipsis; body preserves full text.
    assert "…" in row.title
    assert len(row.body) == 500
