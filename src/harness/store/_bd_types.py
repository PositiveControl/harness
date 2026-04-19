"""Shared types + constants for BeadsAdapter (harness-vhoj).

Holds `BeadsIssue`, `_issue_from_json`, and the ALLOWED_* constants
so crud / focus / graph / memory modules can import without
reaching back into the main adapter module (which would circularize).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.store._bd_runner import AB_CUSTOM_TYPES

ALLOWED_SCOPES = ("professional", "personal")
# Built-in bd types plus the ab-specific additions. `bug`, `feature`,
# `chore`, `epic` are intentionally excluded from ab's vocabulary —
# keeps the ops surface clean and nudges the ops loop away from
# engineering-flavored types.
ALLOWED_TYPES = ("task", "decision", *AB_CUSTOM_TYPES)


@dataclass(frozen=True)
class BeadsIssue:
    """Lightweight view of a bd issue. Only the fields ab cares about;
    the full JSON is available on `raw` for callers that need more."""

    id: str
    title: str
    status: str
    priority: int
    issue_type: str
    labels: tuple[str, ...]
    raw: dict[str, Any]
    assignee: str | None = None

    @property
    def scope(self) -> str | None:
        for label in self.labels:
            if label.startswith("scope:"):
                return label.split(":", 1)[1]
        return None


def _issue_from_json(data: dict[str, Any]) -> BeadsIssue:
    raw_assignee = data.get("assignee")
    assignee = str(raw_assignee) if raw_assignee else None
    return BeadsIssue(
        id=str(data["id"]),
        title=str(data.get("title", "")),
        status=str(data.get("status", "")),
        priority=int(data.get("priority", 0)),
        issue_type=str(data.get("issue_type", "")),
        labels=tuple(data.get("labels", []) or []),
        raw=data,
        assignee=assignee,
    )
