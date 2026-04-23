"""Tests for _maybe_harvest_bd_memories (harness-9yd).

Session-start bd-memory mirror must:
- no-op when either the bd adapter or the episodic store is missing;
- no-op when the user disabled it via --no-harvest-memories;
- log a one-line summary when new memories land, stay silent when
  nothing's new (steady-state start);
- swallow harvester exceptions without propagating (session startup
  should never break because bd misbehaved).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from harness.cli import _maybe_harvest_bd_memories
from harness.store.episodic import EpisodicStore


@dataclass
class _Embedder:
    id: str = "fake"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([np.ones(4, dtype=np.float32) / 2.0 for _ in texts])


@dataclass
class _StubAdapter:
    """Mirrors the slice of BeadsAdapter the memory harvester reaches
    for: a single `memories_json()` entrypoint."""

    memories: dict[str, str] = field(default_factory=dict)
    raise_on_call: Exception | None = None

    def memories_json(self, query: str = "") -> dict[str, str]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return dict(self.memories)


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "h.sqlite", embedder=_Embedder())


def test_noops_when_ab_adapter_missing(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    _maybe_harvest_bd_memories(None, episodic)
    assert episodic.all() == []
    # No log line for the no-op case — a missing adapter is a normal
    # state for non-ab characters.
    assert "harvest" not in capsys.readouterr().out.lower()


def test_noops_when_episodic_missing(capsys: pytest.CaptureFixture[str]) -> None:
    adapter = _StubAdapter(memories={"k": "v"})
    _maybe_harvest_bd_memories(adapter, None)  # type: ignore[arg-type]
    assert "harvest" not in capsys.readouterr().out.lower()


def test_noops_when_disabled_by_flag(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _StubAdapter(memories={"k": "v"})
    _maybe_harvest_bd_memories(adapter, episodic, enabled=False)  # type: ignore[arg-type]
    assert episodic.all() == []
    assert "harvest" not in capsys.readouterr().out.lower()


def test_harvests_and_logs_when_new_memories_exist(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _StubAdapter(
        memories={
            "mark-is-the-creator-and-primary-user": "Mark is the creator and primary user.",
            "mark-first-computer": "Mark's first computer was a Macintosh Performa 405.",
        }
    )
    _maybe_harvest_bd_memories(adapter, episodic)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "harvested 2 new bd memorie" in out
    assert len(episodic.all()) == 2


def test_silent_when_no_new_memories(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Steady-state: nothing new to mirror → no console line."""
    adapter = _StubAdapter(memories={"k": "v"})
    _maybe_harvest_bd_memories(adapter, episodic)  # type: ignore[arg-type]
    capsys.readouterr()  # drain first-run log
    _maybe_harvest_bd_memories(adapter, episodic)  # type: ignore[arg-type]
    assert "harvest" not in capsys.readouterr().out.lower()


def test_swallows_harvester_exceptions(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flaky bd subprocess must not break session startup."""
    adapter = _StubAdapter(raise_on_call=RuntimeError("bd is down"))
    # Must not raise.
    _maybe_harvest_bd_memories(adapter, episodic)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "bd memory harvest skipped" in out
    assert "bd is down" in out
