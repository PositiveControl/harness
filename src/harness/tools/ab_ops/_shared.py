"""Shared types, constants, and rendering helpers for the ab_ops
tools package (harness-u73g).

Holds the `_Adapter` Protocol every tool depends on, tier /
classification constants, scope-description boilerplate, and pure
render helpers used by multiple domain modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from harness.store.bd_adapter import ALLOWED_SCOPES, BeadsIssue


@runtime_checkable
class _Adapter(Protocol):
    """Narrow view of BeadsAdapter that ab's tools actually call. Kept
    behind a Protocol so FakeAdapters in tests don't have to inherit
    the concrete class — structural typing handles duck-shaped stubs.
    """

    def ready(self, *, scope: str | None = ..., limit: int | None = ...) -> list[BeadsIssue]: ...

    def list_issues(
        self,
        *,
        scope: str | None = ...,
        status: str | None = ...,
        priority: str | None = ...,
        issue_type: str | None = ...,
        limit: int | None = ...,
        assignee: str | None = ...,
    ) -> list[BeadsIssue]: ...

    def search(
        self,
        query: str,
        *,
        status: str | None = ...,
        limit: int | None = ...,
    ) -> list[BeadsIssue]: ...

    def show(self, issue_id: str) -> BeadsIssue: ...

    def stale(self) -> list[BeadsIssue]: ...

    def get_focus(self, assignee: str) -> BeadsIssue | None: ...

    def persist_to_focus(self, summary: str, *, assignee: str) -> str: ...

    def create(
        self,
        *,
        title: str,
        scope: str,
        issue_type: str = ...,
        description: str = ...,
        priority: int = ...,
        parent: str | None = ...,
        deps: Sequence[str] = ...,
        extra_labels: Sequence[str] = ...,
        assignee: str | None = ...,
    ) -> str: ...

    def close(self, issue_id: str, *, reason: str | None = ...) -> None: ...

    def reopen(self, issue_id: str, *, reason: str | None = ...) -> None: ...

    def delete(self, issue_id: str, *, cascade: bool = ...) -> None: ...

    def update(self, issue_id: str, **fields: Any) -> None: ...

    def dep_add(self, issue: str, depends_on: str) -> None: ...

    def dep_rm(self, issue: str, depends_on: str) -> None: ...

    def remember(self, insight: str) -> None: ...

    def memories(self, query: str = ...) -> str: ...

    def forget(self, key: str) -> None: ...

    def label_add(self, issue_id: str, label: str) -> None: ...

    def label_rm(self, issue_id: str, label: str) -> None: ...

    def label_list(self, issue_id: str) -> str: ...

    def comment_add(self, issue_id: str, text: str) -> None: ...

    def comments_list(self, issue_id: str) -> str: ...

    def find_duplicates(
        self,
        *,
        threshold: float | None = ...,
        limit: int | None = ...,
        status: str | None = ...,
    ) -> list[dict[str, Any]]: ...


CAPTURE_REQUIRED_FIELDS = ("scope", "outcome", "next_action")

# Tier labels are fixed; display labels come from register_map via the
# rewriter. Order matters — render follows this order.
TIERS = ("shall", "should", "shmaybe", "watching")

# Assignee name for ab's own thought-graph beads. Plan / status use
# this to surface the current focus without polluting the broader tool
# interface with an extra 'who owns this' arg.
AB_ASSIGNEE = "airton_b"

# Defer-count threshold for stall escalation.
STALL_DEFERS = 3
STALL_LABEL = "stall-escalated"
_DEFER_COUNT_PREFIX = "defer-count:"

# Ab-internal drift threshold in days. bd's `stale` is tuned for user
# work (~14d default); ab's thought-graph rots faster. Settings knob
# lives at Settings.ab_drift_days (harness-6y5).
AB_DRIFT_DAYS = 7

# Shared description for every scope parameter (harness-ilru). Small
# models ignore JSON-schema enums when the description is vague and
# will pass 'tomorrow' / 'this week' after a time-phrased user turn,
# conflating scope (categorical) with when (temporal). Spelling out
# the contract and naming concrete anti-examples is what stops it.
_SCOPE_PARAM_DESCRIPTION = (
    "Scope is CATEGORICAL, not temporal — one of {professional, personal}. "
    "OMIT this field on time phrasings like 'tomorrow', 'today', 'this week', "
    "'next month' — scope does not filter by time. Omit also when the user "
    "hasn't specified a category."
)

_UPDATE_FIELD_FLAGS: dict[str, str] = {
    "title": "title",
    "description": "description",
    "notes": "notes",
    "assignee": "assignee",
    "priority": "priority",
    "status": "status",
}


@dataclass(frozen=True)
class _TieredLine:
    issue: BeadsIssue
    tier: str
    reason: str


def classify_issue(issue: BeadsIssue) -> tuple[str, str]:
    """Classify a BeadsIssue into one of the four tiers with a short
    reason suitable for rendering on the line. Pure: priority and
    dependent-count are the only signals v1 uses. Reason is a terse
    string the caller splices into the rendered plan line.

    Priority convention (bd): 0 = critical, 4 = backlog. ab maps:
      0 / 1          → shall
      2 + blocks     → shall
      2              → should
      3              → shmaybe
      4              → watching
    """
    blocks_others = int(issue.raw.get("dependent_count") or 0) > 0
    if issue.priority <= 1:
        return "shall", f"P{issue.priority}"
    if issue.priority == 2 and blocks_others:
        n = int(issue.raw["dependent_count"])
        return "shall", f"blocks {n}"
    if issue.priority == 2:
        return "should", "default P2"
    if issue.priority == 3:
        return "shmaybe", "low-priority"
    return "watching", "backlog"


def _classify_dict(issue: BeadsIssue) -> dict[str, str]:
    tier, reason = classify_issue(issue)
    return {"tier": tier, "reason": reason}


def _validate_scope(scope: str | None) -> str | None:
    """Reject invalid scope values with a clear error string for the
    model. Tool schemas declare `enum` but small-model tool emitters
    sometimes ignore it; returning an explicit error lets the model
    course-correct instead of silently filtering to an empty set."""
    if scope is None:
        return None
    if scope in ALLOWED_SCOPES:
        return None
    return (
        f"invalid scope {scope!r}. Valid scopes: {', '.join(ALLOWED_SCOPES)}. "
        "Retry the call with a valid scope, or omit `scope` to see all."
    )


def _render_plan(lines: Iterable[_TieredLine], today: Any) -> str:
    """Render a tiered plan string. Empty tiers are listed explicitly
    ("(none)") so absence reads as deliberate, not as a rendering bug.

    `today` is a datetime.date; typed as Any to avoid a narrow dep."""
    buckets: dict[str, list[_TieredLine]] = {t: [] for t in TIERS}
    for line in lines:
        buckets[line.tier].append(line)
    out = [f"Today — {today.isoformat()}"]
    labels = {
        "shall": "Shall",
        "should": "Should",
        "shmaybe": "Shmaybe",
        "watching": "Watching",
    }
    counter = 1
    for tier in TIERS:
        label = labels[tier]
        items = buckets[tier]
        out.append(f"  {label}:")
        if not items:
            out.append("    (none)")
            continue
        for line in items:
            scope_tag = line.issue.scope or "?"
            out.append(
                f"    {counter}. [{scope_tag}/{line.issue.id}] {line.issue.title} — {line.reason}"
            )
            counter += 1
    return "\n".join(out)


def _render_status(issue: BeadsIssue, blockers: list[BeadsIssue]) -> str:
    """Render /status output: state, next, blockers, blocks."""
    scope_tag = issue.scope or "?"
    header = (
        f"[{scope_tag}/{issue.id}] {issue.title} — "
        f"state: {issue.status}, priority: P{issue.priority}"
    )
    out = [header]
    dependent_count = int(issue.raw.get("dependent_count") or 0)
    blocker_count = int(issue.raw.get("dependency_count") or 0)
    if blocker_count:
        out.append(f"  Blockers ({blocker_count}):")
        for b in blockers:
            out.append(f"    - [{b.status}] {b.id} — {b.title}")
    else:
        out.append("  Blockers: none")
    if dependent_count:
        out.append(f"  Blocks: {dependent_count} item(s)")
    else:
        out.append("  Blocks: none")
    return "\n".join(out)


def _render_focus_banner(issue: BeadsIssue) -> str:
    scope_tag = issue.scope or "?"
    return f"Focus: [{scope_tag}/{issue.id}] {issue.title} (P{issue.priority})"


def _render_status_with_focus(issue: BeadsIssue, *, focus_id: str | None) -> str:
    """Shared rendering for StatusTool. Extracts blockers from
    issue.raw and appends a focus-switch hint when the subject bead
    isn't the current focus — advisory only, does not mutate."""
    blockers_raw = issue.raw.get("dependencies") or []
    blockers: list[BeadsIssue] = []
    if isinstance(blockers_raw, list):
        for dep in blockers_raw:
            if isinstance(dep, dict) and "id" in dep:
                blockers.append(
                    BeadsIssue(
                        id=str(dep["id"]),
                        title=str(dep.get("title", "")),
                        status=str(dep.get("status", "")),
                        priority=int(dep.get("priority") or 0),
                        issue_type=str(dep.get("issue_type", "")),
                        labels=tuple(dep.get("labels") or []),
                        raw=dep,
                    )
                )
    rendered = _render_status(issue, blockers)
    if focus_id is not None and focus_id != issue.id:
        rendered += f"\n  (Not current focus — ab is focused on {focus_id}.)"
    return rendered


def _parse_iso_utc(ts: str) -> datetime | None:
    """Parse an ISO-8601 'Z' suffix timestamp into a timezone-aware
    UTC datetime. Returns None on anything unparseable so callers can
    skip rather than crash on an odd bd output."""
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _render_issue_list(issues: list[BeadsIssue], *, empty_label: str) -> str:
    """Shared compact renderer for search / list results. One line per
    issue with scope, id, priority, status, title. Empty label lets
    callers distinguish 'no matches' from 'no open items'.

    Exact-title duplicates are collapsed onto the highest-priority
    sibling with an '(also: <id>, …)' annotation so the user sees the
    duplication instead of reading two independent rows and assuming
    they're distinct work (harness-mmz1). Fuzzy dedup is
    `find_duplicates`' job — this helper stays mechanical."""
    if not issues:
        return empty_label

    primary_by_title: dict[str, BeadsIssue] = {}
    sibling_ids_by_title: dict[str, list[str]] = {}
    order: list[str] = []
    for issue in issues:
        key = issue.title
        if key not in primary_by_title:
            primary_by_title[key] = issue
            sibling_ids_by_title[key] = []
            order.append(key)
            continue
        current = primary_by_title[key]
        if issue.priority < current.priority:
            sibling_ids_by_title[key].append(current.id)
            primary_by_title[key] = issue
        else:
            sibling_ids_by_title[key].append(issue.id)

    lines: list[str] = []
    for key in order:
        issue = primary_by_title[key]
        scope_tag = issue.scope or "?"
        suffix = ""
        siblings = sibling_ids_by_title[key]
        if siblings:
            suffix = f" (also: {', '.join(siblings)})"
        lines.append(
            f"  - [{scope_tag}/{issue.id}] P{issue.priority} {issue.status}: {issue.title}{suffix}"
        )
    return "\n".join(lines)
