from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore
from harness.tools.remember import RememberEventTool, RememberFactTool


class _FakeEmbedder:
    id = "fake:test"
    dimension = 8

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[0.1] * 8 for _ in texts], dtype=np.float32)


def _fact_tool(db_path: Path, user_id: str | None = "mark") -> RememberFactTool:
    embedder = _FakeEmbedder()
    store = SemanticStore(db_path, embedder=embedder)  # type: ignore[arg-type]
    return RememberFactTool(store=store, user_id=user_id, session_id="local")


def _event_tool(db_path: Path, user_id: str | None = "mark") -> RememberEventTool:
    embedder = _FakeEmbedder()
    store = EpisodicStore(db_path, embedder=embedder)  # type: ignore[arg-type]
    return RememberEventTool(store=store, user_id=user_id, session_id="local")


# ---------- RememberFactTool ----------


def test_fact_insert_and_retrieve(tmp_path: Path) -> None:
    tool = _fact_tool(tmp_path / "mem.sqlite")
    out = tool.call(subject="mark", predicate="prefers", object="terse replies")
    assert "recorded fact" in out
    assert "mark" in out
    # Verify it actually landed with the right attribution.
    rows = list(tool.store.all())
    assert len(rows) == 1
    f = rows[0]
    assert f.subject == "mark"
    assert f.predicate == "prefers"
    assert f.object == "terse replies"
    assert f.source == "tool:remember_fact"
    assert f.user_id == "mark"
    assert f.tier == "working"


def test_fact_confidence_respected(tmp_path: Path) -> None:
    tool = _fact_tool(tmp_path / "mem.sqlite")
    tool.call(subject="a", predicate="b", object="c", confidence=0.55)
    f = next(iter(tool.store.all()))
    assert f.confidence == 0.55


def test_fact_confidence_bounds(tmp_path: Path) -> None:
    tool = _fact_tool(tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        tool.call(subject="a", predicate="b", object="c", confidence=1.5)


def test_fact_empty_fields_rejected(tmp_path: Path) -> None:
    tool = _fact_tool(tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match="non-empty"):
        tool.call(subject="", predicate="b", object="c")


def test_fact_shared_when_user_none(tmp_path: Path) -> None:
    tool = _fact_tool(tmp_path / "mem.sqlite", user_id=None)
    out = tool.call(subject="project", predicate="name", object="harness")
    assert "shared" in out
    f = next(iter(tool.store.all()))
    assert f.user_id is None


def test_fact_spec_is_write_tier(tmp_path: Path) -> None:
    assert _fact_tool(tmp_path / "x.sqlite").spec.tier == "write"


# ---------- RememberEventTool ----------


def test_event_insert_and_retrieve(tmp_path: Path) -> None:
    tool = _event_tool(tmp_path / "mem.sqlite")
    out = tool.call(
        title="Mark's first session",
        body="We shipped the tool-use phase today. First real use of edit_file.",
        principle="Ship small, inspect often.",
        tags=["milestone", "phase-3"],
    )
    assert "recorded memory" in out
    rows = list(tool.store.all())
    assert len(rows) == 1
    r = rows[0]
    assert r.title == "Mark's first session"
    assert "shipped the tool-use" in r.body
    assert r.principle == "Ship small, inspect often."
    assert "milestone" in r.tags
    assert r.source == "tool:remember_event"
    assert r.user_id == "mark"
    assert r.tier == "working"


def test_event_empty_title_rejected(tmp_path: Path) -> None:
    tool = _event_tool(tmp_path / "mem.sqlite")
    with pytest.raises(ValueError, match="non-empty"):
        tool.call(title="", body="something")


def test_event_shared_when_user_none(tmp_path: Path) -> None:
    tool = _event_tool(tmp_path / "mem.sqlite", user_id=None)
    out = tool.call(title="project-wide note", body="applies to everyone")
    assert "shared" in out
    r = next(iter(tool.store.all()))
    assert r.user_id is None


def test_event_spec_is_write_tier(tmp_path: Path) -> None:
    assert _event_tool(tmp_path / "x.sqlite").spec.tier == "write"
