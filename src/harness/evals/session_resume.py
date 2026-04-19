"""Session-resume eval — replays scripted fixtures through
build_resume_summary and scores each against contains / not_contains
assertions.

Builds an in-memory adapter per fixture (no real bd subprocess, no
temp DB). Fast + deterministic. Promote to a real bd repo per
scenario later if the fake-adapter coverage gap ever bites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from harness.store.bd_adapter import BeadsIssue
from harness.tools.ab_ops import AB_ASSIGNEE, build_resume_summary


@dataclass(frozen=True)
class SessionResumeCase:
    id: str
    description: str
    summary: str
    missing_contains: tuple[str, ...]
    unexpected_contains: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing_contains and not self.unexpected_contains


@dataclass(frozen=True)
class SessionResumeResult:
    cases: tuple[SessionResumeCase, ...]

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    def failures(self) -> tuple[SessionResumeCase, ...]:
        return tuple(c for c in self.cases if not c.passed)


@dataclass
class _EvalAdapter:
    """In-memory adapter subset sufficient for build_resume_summary.
    Mirrors the _Adapter Protocol's read surface; write methods are
    unused for this eval and intentionally omitted."""

    focus_issue: BeadsIssue | None
    in_progress_issues: list[BeadsIssue] = field(default_factory=list)
    memories_text: str = ""
    drift_issues: list[BeadsIssue] = field(default_factory=list)

    def get_focus(self, assignee: str) -> BeadsIssue | None:
        return self.focus_issue

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
        if status == "in_progress":
            return list(self.in_progress_issues)
        if status == "open":
            # build_resume_summary pulls open ab-beads for its drift
            # sub-query. Use the drift fixture rows directly.
            return list(self.drift_issues)
        return []

    def memories(self, query: str = "") -> str:
        return self.memories_text


def _build_issue(entry: dict[str, Any], *, updated_days_ago: int | None = None) -> BeadsIssue:
    labels = (f"scope:{entry['scope']}",) if entry.get("scope") else ()
    raw: dict[str, Any] = {
        "id": entry["id"],
        "title": entry["title"],
        "priority": int(entry.get("priority", 2)),
    }
    if updated_days_ago is not None:
        ts = (datetime.now(UTC) - timedelta(days=updated_days_ago)).isoformat()
        raw["updated_at"] = ts
    return BeadsIssue(
        id=str(entry["id"]),
        title=str(entry["title"]),
        status=str(entry.get("status", "open")),
        priority=int(entry.get("priority", 2)),
        issue_type=str(entry.get("issue_type", "task")),
        labels=labels,
        raw=raw,
    )


def _build_adapter(fixture: dict[str, Any]) -> _EvalAdapter:
    focus_entry = fixture.get("focus")
    focus = _build_issue(focus_entry) if focus_entry else None
    in_progress = [_build_issue(e) for e in fixture.get("in_progress", [])]
    memories = "\n".join(fixture.get("memories", []))
    drift_entries = fixture.get("drift", [])
    drift = [
        _build_issue(e, updated_days_ago=int(e.get("updated_days_ago", 30))) for e in drift_entries
    ]
    return _EvalAdapter(
        focus_issue=focus,
        in_progress_issues=in_progress,
        memories_text=memories,
        drift_issues=drift,
    )


def load_fixture(path: Path) -> tuple[dict[str, Any], ...]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"session-resume eval fixture {path} is not a YAML list")
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{idx}] is not a mapping")
        if "id" not in entry:
            raise ValueError(f"{path}[{idx}] missing 'id'")
        out.append(entry)
    return tuple(out)


def run_session_resume_eval(fixtures: tuple[dict[str, Any], ...]) -> SessionResumeResult:
    """Score each fixture: build adapter, render summary, check the
    contains / not_contains sets. Pure — caller supplies parsed
    fixtures; no filesystem."""
    cases: list[SessionResumeCase] = []
    for fx in fixtures:
        adapter = _build_adapter(fx)
        summary = build_resume_summary(adapter)  # type: ignore[arg-type]  # structural typing
        expected_contains = tuple(str(s) for s in fx.get("expected_contains", []))
        expected_not_contains = tuple(str(s) for s in fx.get("expected_not_contains", []))
        missing = tuple(s for s in expected_contains if s not in summary)
        unexpected = tuple(s for s in expected_not_contains if s in summary)
        cases.append(
            SessionResumeCase(
                id=str(fx["id"]),
                description=str(fx.get("description", "")).strip(),
                summary=summary,
                missing_contains=missing,
                unexpected_contains=unexpected,
            )
        )
    return SessionResumeResult(cases=tuple(cases))


def default_fixture_path(character_path: Path) -> Path:
    """Conventional location under character/<name>/ for the fixture."""
    return character_path / "session_resume_eval.yaml"


__all__ = [
    "AB_ASSIGNEE",
    "SessionResumeCase",
    "SessionResumeResult",
    "default_fixture_path",
    "load_fixture",
    "run_session_resume_eval",
]
