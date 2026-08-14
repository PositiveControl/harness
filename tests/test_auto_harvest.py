"""Tests for _maybe_harvest_skills (harness-j5b).

Session-start auto-harvest must:
- no-op when either the bd adapter or the episodic store is missing;
- no-op when the user disabled it via --no-harvest-skills;
- log a one-line summary when new beads land;
- swallow harvester exceptions without propagating (session startup
  should never break because bd misbehaved).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from harness.cli import _maybe_harvest_skills, _print_session_end_retro
from harness.store._bd_types import BeadsIssue
from harness.store.episodic import EpisodicStore


@dataclass
class _Embedder:
    id: str = "fake"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([np.ones(4, dtype=np.float32) / 2.0 for _ in texts])


@dataclass
class _StubAdapter:
    """Mirrors the subset of BeadsAdapter that the harvester hits."""

    issues: list[BeadsIssue] = field(default_factory=list)
    raise_on_list: Exception | None = None

    def list_issues(
        self,
        *,
        scope: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        issue_type: str | None = None,
        limit: int | None = None,
        assignee: str | None = None,
    ) -> list[BeadsIssue]:
        if self.raise_on_list is not None:
            raise self.raise_on_list
        return [i for i in self.issues if status is None or status == "all" or i.status == status]


def _issue(id_: str, title: str = "t") -> BeadsIssue:
    return BeadsIssue(
        id=id_,
        title=title,
        status="closed",
        priority=2,
        issue_type="task",
        labels=("thought:decision",),
        raw={"description": f"body for {id_}"},
        assignee="airton_b",
    )


@pytest.fixture
def episodic(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "h.sqlite", embedder=_Embedder())


def test_noops_when_ab_adapter_missing(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    _maybe_harvest_skills(None, episodic)
    assert episodic.all() == []
    # No log line for the no-op case — a missing adapter is a normal
    # state for non-ab characters.
    assert "harvest" not in capsys.readouterr().out.lower()


def test_noops_when_episodic_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _StubAdapter(issues=[_issue("harness-1")])
    _maybe_harvest_skills(adapter, None)  # type: ignore[arg-type]
    assert "harvest" not in capsys.readouterr().out.lower()


def test_noops_when_disabled_by_flag(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _StubAdapter(issues=[_issue("harness-1")])
    _maybe_harvest_skills(adapter, episodic, enabled=False)  # type: ignore[arg-type]
    assert episodic.all() == []
    assert "harvest" not in capsys.readouterr().out.lower()


def test_harvests_and_logs_when_new_beads_exist(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = _StubAdapter(issues=[_issue("harness-1"), _issue("harness-2")])
    _maybe_harvest_skills(adapter, episodic)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "harvested 2 new skill" in out
    assert len(episodic.all()) == 2


def test_silent_when_no_new_beads(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Steady-state session start: nothing new to say → don't clutter
    the console with 'harvested 0'."""
    adapter = _StubAdapter(issues=[_issue("harness-1")])
    _maybe_harvest_skills(adapter, episodic)  # type: ignore[arg-type]
    capsys.readouterr()  # drain first-run log
    _maybe_harvest_skills(adapter, episodic)  # type: ignore[arg-type]
    assert "harvest" not in capsys.readouterr().out.lower()


def test_swallows_harvester_exceptions(
    episodic: EpisodicStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flaky bd subprocess must not break session startup. The
    helper logs a yellow warning and returns; the REPL carries on."""
    adapter = _StubAdapter(raise_on_list=RuntimeError("bd is down"))
    # Must not raise.
    _maybe_harvest_skills(adapter, episodic)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "skill harvest skipped" in out
    assert "bd is down" in out


# ---------- _print_session_end_retro ----------
#
# harness-z4k1.1 step 3: the retro print was 12% covered — only the
# "no ab adapter" early return ran. It is a shutdown path, so its
# contract is that it NEVER raises: a broken retro must not turn a
# clean /exit into a traceback. Pinned here before the helper moves to
# cli_bd.py.


def _retro_owner() -> object:
    """The module that currently defines the helper — cli.py before the
    step-3 extraction, cli_bd.py after. Patching there keeps these tests
    valid across the move."""
    import sys

    return sys.modules[_print_session_end_retro.__module__]


class _StubRetroTool:
    """Stands in for RetroTool; records construction and mode."""

    instances: ClassVar[list[_StubRetroTool]] = []

    def __init__(self, adapter: object) -> None:
        self.adapter = adapter
        self.calls: list[str] = []
        _StubRetroTool.instances.append(self)

    def call(self, *, mode: str) -> str:
        self.calls.append(mode)
        return "3 closed · 1 open · focus: harness-z4k1"


class _ExplodingRetroTool:
    def __init__(self, adapter: object) -> None:
        pass

    def call(self, *, mode: str) -> str:
        raise RuntimeError("bd went away mid-shutdown")


def test_session_end_retro_prints_the_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _StubRetroTool.instances = []
    monkeypatch.setattr(_retro_owner(), "RetroTool", _StubRetroTool)
    adapter = _StubAdapter()

    _print_session_end_retro(adapter)  # type: ignore[arg-type]

    assert "focus: harness-z4k1" in capsys.readouterr().out
    assert len(_StubRetroTool.instances) == 1
    assert _StubRetroTool.instances[0].adapter is adapter
    assert _StubRetroTool.instances[0].calls == ["summary"]


def test_session_end_retro_is_silent_without_an_ab_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-ab sessions get no retro at all — and RetroTool is never
    even constructed."""

    def _never(adapter: object) -> object:
        raise AssertionError("RetroTool must not be built for a non-ab session")

    monkeypatch.setattr(_retro_owner(), "RetroTool", _never)

    _print_session_end_retro(None)

    assert capsys.readouterr().out == ""


def test_session_end_retro_swallows_a_failing_retro(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A retro that raises on the way out must not break /exit. No
    traceback, and no half-rendered line either."""
    monkeypatch.setattr(_retro_owner(), "RetroTool", _ExplodingRetroTool)

    _print_session_end_retro(_StubAdapter())  # type: ignore[arg-type]

    assert capsys.readouterr().out == ""
