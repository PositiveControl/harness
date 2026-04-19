"""Tests for scripts/sweep_assign.py — harness-60y."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from harness.store.bd_adapter import BeadsIssue

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_assign.py"
_SPEC = importlib.util.spec_from_file_location("sweep_assign", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
sweep_assign = importlib.util.module_from_spec(_SPEC)
sys.modules["sweep_assign"] = sweep_assign
_SPEC.loader.exec_module(sweep_assign)


def _issue(issue_id: str, *, assignee: str | None = None) -> BeadsIssue:
    return BeadsIssue(
        id=issue_id,
        title=f"t-{issue_id}",
        status="open",
        priority=2,
        issue_type="task",
        labels=(),
        raw={"id": issue_id, "assignee": assignee},
        assignee=assignee,
    )


class FakeAdapter:
    def __init__(self, issues: list[BeadsIssue]) -> None:
        self._issues = issues
        self.update_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []

    def list_issues(self, *, status: str | None = None) -> list[BeadsIssue]:
        self.list_calls.append({"status": status})
        return list(self._issues)

    def update(self, issue_id: str, **fields: str) -> None:
        self.update_calls.append({"issue_id": issue_id, "fields": fields})


def test_classify_splits_on_assignee() -> None:
    plans = sweep_assign.classify(
        [
            _issue("h-1", assignee=None),
            _issue("h-2", assignee=""),
            _issue("h-3", assignee="mark"),
            _issue("h-4", assignee="airton_b"),
        ],
        target="mark",
    )
    assert {p.issue_id: p.action for p in plans} == {
        "h-1": "assign",
        "h-2": "assign",
        "h-3": "skip",
        "h-4": "skip",
    }


def test_dry_run_performs_no_updates(capsys: pytest.CaptureFixture[str]) -> None:
    adapter = FakeAdapter([_issue("h-1"), _issue("h-2", assignee="airton_b")])
    plans = sweep_assign.run_sweep(adapter, target="mark", apply=False)
    out = capsys.readouterr().out
    assert adapter.update_calls == []
    assert [p.action for p in plans] == ["assign", "skip"]
    assert "mode=dry-run" in out
    assert "would-assign: h-1 -> mark" in out
    assert "skip: h-2 already assigned to airton_b" in out
    assert "done (1 assign, 1 skip)" in out


def test_apply_calls_update_per_unassigned(capsys: pytest.CaptureFixture[str]) -> None:
    adapter = FakeAdapter([_issue("h-1"), _issue("h-2", assignee="airton_b"), _issue("h-3")])
    sweep_assign.run_sweep(adapter, target="mark", apply=True)
    out = capsys.readouterr().out
    assert adapter.update_calls == [
        {"issue_id": "h-1", "fields": {"assignee": "mark"}},
        {"issue_id": "h-3", "fields": {"assignee": "mark"}},
    ]
    assert "mode=apply" in out
    assert "assign: h-1 -> mark" in out
    assert "skip: h-2 already assigned to airton_b" in out


def test_empty_list_and_status_passthrough(capsys: pytest.CaptureFixture[str]) -> None:
    adapter = FakeAdapter([])
    plans = sweep_assign.run_sweep(adapter, target="mark", apply=True, status=None)
    assert plans == []
    assert adapter.update_calls == []
    assert adapter.list_calls == [{"status": None}]
    assert "scanned 0 bead(s)" in capsys.readouterr().out
