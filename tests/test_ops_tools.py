from __future__ import annotations

from pathlib import Path

import numpy as np

from harness.character import load_character
from harness.model.echo import EchoAdapter
from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore
from harness.store.transcript import Transcript
from harness.tools.ops import ConsolidateMemoryTool, ScribeSessionTool


class _FakeEmbedder:
    id = "fake:test"
    dimension = 8

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[0.1] * 8 for _ in texts], dtype=np.float32)


def _build_ops(tmp_path: Path) -> tuple[ScribeSessionTool, ConsolidateMemoryTool, Transcript]:
    db = tmp_path / "harness.sqlite"
    embedder = _FakeEmbedder()
    transcript = Transcript(db)
    episodic = EpisodicStore(db, embedder=embedder)  # type: ignore[arg-type]
    semantic = SemanticStore(db, embedder=embedder)  # type: ignore[arg-type]
    # echo adapter suffices — scribe's parse path simply produces no
    # candidates when the adapter returns non-JSON, which is exactly what
    # we want for a unit-level test: we exercise the plumbing, not the LLM.
    adapter = EchoAdapter()
    character = load_character(Path("character/airton"))
    scribe = ScribeSessionTool(
        adapter=adapter,
        character=character,
        transcript=transcript,
        episodic_store=episodic,
        semantic_store=semantic,
        default_user_id="mark",
        lock_dir=tmp_path / "locks",
    )
    consolidate = ConsolidateMemoryTool(
        episodic_store=episodic,
        semantic_store=semantic,
    )
    return scribe, consolidate, transcript


def test_scribe_reports_summary(tmp_path: Path) -> None:
    scribe, _, transcript = _build_ops(tmp_path)
    # Seed a minimal transcript so the scribe has something to walk.
    transcript.append(
        session="local",
        channel="cli",
        speaker="mark",
        role="user",
        content="Project `harness` starts today.",
    )
    transcript.append(
        session="local",
        channel="cli",
        speaker="airton",
        role="assistant",
        content="noted.",
    )
    out = scribe.call(session="local")
    assert "scribed session=local" in out
    assert "episodic" in out
    assert "semantic" in out
    assert "windows" in out


def test_scribe_default_session(tmp_path: Path) -> None:
    scribe, _, _ = _build_ops(tmp_path)
    out = scribe.call()  # no session arg
    assert "session=local" in out


def test_consolidate_reports_zero_on_empty(tmp_path: Path) -> None:
    _, consolidate, _ = _build_ops(tmp_path)
    out = consolidate.call()
    assert "considered" in out
    assert "merged" in out
    assert "superseded" in out


def test_scribe_is_write_tier(tmp_path: Path) -> None:
    scribe, _, _ = _build_ops(tmp_path)
    assert scribe.spec.tier == "write"


def test_consolidate_is_write_tier(tmp_path: Path) -> None:
    _, consolidate, _ = _build_ops(tmp_path)
    assert consolidate.spec.tier == "write"
