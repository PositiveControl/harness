"""Tests for src/harness/tools/ab_ops.py — harness-inj.5.

BeadsAdapter is stubbed with a lightweight fake so tests don't need
bd or a running Dolt server. Each tool's happy path is covered plus:

- plan: tier classifier rules; render shape
- capture: missing-field hint; successful create with scope label
- status: transitive blockers rollup
- drift: scope filter
- close / defer: dispatch to bd adapter with expected args
- retro: summary vs record modes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from harness.store.bd_adapter import BeadsIssue
from harness.tools.ab_ops import (
    CaptureTool,
    CloseTool,
    DeferTool,
    DeleteTool,
    DriftTool,
    PlanTool,
    ReopenTool,
    ReprioritizeTool,
    RetroTool,
    StatusTool,
    UpdateTool,
    _render_plan,
    _TieredLine,
    classify_issue,
    make_ops_tools,
)


def _issue(
    issue_id: str = "harness-x",
    *,
    title: str = "thing",
    status: str = "open",
    priority: int = 2,
    issue_type: str = "task",
    scope: str | None = "professional",
    dependent_count: int = 0,
    dependency_count: int = 0,
    dependencies: list[dict[str, Any]] | None = None,
) -> BeadsIssue:
    labels: tuple[str, ...] = ()
    if scope is not None:
        labels = (f"scope:{scope}",)
    raw: dict[str, Any] = {
        "id": issue_id,
        "title": title,
        "status": status,
        "priority": priority,
        "issue_type": issue_type,
        "dependent_count": dependent_count,
        "dependency_count": dependency_count,
    }
    if dependencies is not None:
        raw["dependencies"] = dependencies
    return BeadsIssue(
        id=issue_id,
        title=title,
        status=status,
        priority=priority,
        issue_type=issue_type,
        labels=labels,
        raw=raw,
    )


@dataclass
class FakeAdapter:
    """In-memory stand-in for BeadsAdapter. Records every call so tests
    can assert dispatch shape; returns canned data for reads."""

    ready_issues: list[BeadsIssue] = field(default_factory=list)
    list_issues_data: list[BeadsIssue] = field(default_factory=list)
    show_issues: dict[str, BeadsIssue] = field(default_factory=dict)
    stale_issues: list[BeadsIssue] = field(default_factory=list)
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    close_calls: list[dict[str, Any]] = field(default_factory=list)
    reopen_calls: list[dict[str, Any]] = field(default_factory=list)
    delete_calls: list[dict[str, Any]] = field(default_factory=list)
    update_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    remember_calls: list[str] = field(default_factory=list)
    next_create_id: str = "harness-new"

    def ready(
        self,
        *,
        scope: str | None = None,
        limit: int | None = None,
    ) -> list[BeadsIssue]:
        if scope is None:
            return list(self.ready_issues)
        label = f"scope:{scope}"
        return [i for i in self.ready_issues if label in i.labels]

    def list_issues(
        self,
        *,
        scope: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[BeadsIssue]:
        return list(self.list_issues_data)

    def show(self, issue_id: str) -> BeadsIssue:
        return self.show_issues[issue_id]

    def stale(self) -> list[BeadsIssue]:
        return list(self.stale_issues)

    def create(self, **kwargs: Any) -> str:
        self.create_calls.append(kwargs)
        return self.next_create_id

    def close(self, issue_id: str, *, reason: str | None = None) -> None:
        self.close_calls.append({"id": issue_id, "reason": reason})

    def reopen(self, issue_id: str, *, reason: str | None = None) -> None:
        self.reopen_calls.append({"id": issue_id, "reason": reason})

    def delete(self, issue_id: str, *, cascade: bool = False) -> None:
        self.delete_calls.append({"id": issue_id, "cascade": cascade})

    def update(self, issue_id: str, **fields: Any) -> None:
        self.update_calls.append((issue_id, dict(fields)))

    def remember(self, insight: str) -> None:
        self.remember_calls.append(insight)


def test_classify_priority_zero_is_shall() -> None:
    issue = _issue(priority=0)
    assert classify_issue(issue) == ("shall", "P0")


def test_classify_priority_two_with_blockers_is_shall() -> None:
    issue = _issue(priority=2, dependent_count=3)
    tier, reason = classify_issue(issue)
    assert tier == "shall"
    assert reason == "blocks 3"


def test_classify_priority_two_no_blockers_is_should() -> None:
    issue = _issue(priority=2, dependent_count=0)
    assert classify_issue(issue) == ("should", "default P2")


def test_classify_priority_three_is_shmaybe() -> None:
    assert classify_issue(_issue(priority=3)) == ("shmaybe", "low-priority")


def test_classify_priority_four_is_watching() -> None:
    assert classify_issue(_issue(priority=4)) == ("watching", "backlog")


def test_render_plan_emits_tier_order_and_empty_buckets() -> None:
    issue = _issue(issue_id="harness-a", title="ship it", priority=1)
    line = _TieredLine(issue=issue, tier="shall", reason="P1")
    out = _render_plan([line], date(2026, 4, 18))
    assert "Today — 2026-04-18" in out
    # Every tier appears even when empty — absence reads as deliberate.
    for label in ("Shall:", "Should:", "Shmaybe:", "Watching:"):
        assert label in out
    # Populated tier renders with id, scope, reason.
    assert "[professional/harness-a] ship it — P1" in out


def test_plan_tool_uses_ready_and_renders() -> None:
    adapter = FakeAdapter(
        ready_issues=[
            _issue(issue_id="harness-a", title="ship a", priority=1),
            _issue(issue_id="harness-b", title="ship b", priority=2),
            _issue(issue_id="harness-c", title="ship c", priority=3),
        ]
    )
    out = PlanTool(adapter).call()
    assert "Shall:" in out
    assert "ship a" in out
    assert "ship b" in out
    assert "P1" in out
    assert "default P2" in out


def test_plan_rejects_invalid_scope() -> None:
    """Defense in depth — schema enum alone doesn't stop small models
    from emitting bogus scopes like 'today' or 'schmaybe'. The tool
    must return an explicit error so the model can course-correct
    instead of silently filtering to empty and hallucinating state."""
    adapter = FakeAdapter(
        ready_issues=[_issue(issue_id="harness-a", priority=1, scope="professional")]
    )
    out = PlanTool(adapter).call(scope="today")
    assert "invalid scope" in out
    assert "professional" in out
    assert "personal" in out


def test_drift_rejects_invalid_scope() -> None:
    out = DriftTool(FakeAdapter()).call(scope="schmaybe")
    assert "invalid scope" in out


def test_reprioritize_rejects_invalid_scope() -> None:
    out = ReprioritizeTool(FakeAdapter()).call(scope="nope")
    assert "invalid scope" in out


def test_plan_tool_respects_scope_filter() -> None:
    adapter = FakeAdapter(
        ready_issues=[
            _issue(issue_id="harness-a", priority=2, scope="professional"),
            _issue(issue_id="harness-b", priority=2, scope="personal"),
        ]
    )
    out = PlanTool(adapter).call(scope="personal")
    assert "harness-b" in out
    assert "harness-a" not in out


def test_capture_reports_missing_fields() -> None:
    tool = CaptureTool(FakeAdapter())
    out = tool.call(raw="I should renew my passport")
    assert "missing field" in out
    # All three required fields surface; user-facing nudge names the first.
    for f in ("scope", "outcome", "next_action"):
        assert f in out


def test_capture_creates_with_scope_label() -> None:
    adapter = FakeAdapter(next_create_id="harness-zzz")
    tool = CaptureTool(adapter)
    out = tool.call(
        raw="renew passport",
        scope="personal",
        outcome="valid passport in hand by June 1",
        next_action="fill DS-82 online",
        deadline="2026-06-01",
        estimate="S",
    )
    assert "Captured harness-zzz" in out
    assert len(adapter.create_calls) == 1
    payload = adapter.create_calls[0]
    assert payload["scope"] == "personal"
    assert payload["issue_type"] == "task"
    assert "est:S" in payload["extra_labels"]
    description = payload["description"]
    assert "outcome: valid passport" in description
    assert "next: fill DS-82 online" in description
    assert "deadline: 2026-06-01" in description


def test_capture_title_truncated_when_long() -> None:
    adapter = FakeAdapter()
    tool = CaptureTool(adapter)
    very_long = "x" * 200
    tool.call(
        raw=very_long,
        scope="personal",
        outcome="done",
        next_action="do",
    )
    title = adapter.create_calls[0]["title"]
    assert len(title) <= 80
    assert title.endswith("…")


def test_status_tool_renders_blockers() -> None:
    deps = [
        {
            "id": "harness-dep-a",
            "title": "blocker one",
            "status": "open",
            "priority": 2,
            "issue_type": "task",
        },
    ]
    target = _issue(
        issue_id="harness-t",
        title="the target",
        priority=2,
        dependent_count=2,
        dependency_count=1,
        dependencies=deps,
    )
    adapter = FakeAdapter(show_issues={"harness-t": target})
    out = StatusTool(adapter).call(id="harness-t")
    assert "harness-t" in out
    assert "state: open" in out
    assert "Blockers (1)" in out
    assert "harness-dep-a" in out
    assert "Blocks: 2 item(s)" in out


def test_status_tool_no_blockers_branch() -> None:
    target = _issue(issue_id="harness-lone", dependent_count=0, dependency_count=0)
    adapter = FakeAdapter(show_issues={"harness-lone": target})
    out = StatusTool(adapter).call(id="harness-lone")
    assert "Blockers: none" in out
    assert "Blocks: none" in out


def test_drift_tool_empty_path() -> None:
    adapter = FakeAdapter(stale_issues=[])
    out = DriftTool(adapter).call()
    assert "no drift" in out


def test_drift_tool_scope_filter() -> None:
    adapter = FakeAdapter(
        stale_issues=[
            _issue(issue_id="harness-a", scope="professional", title="pro"),
            _issue(issue_id="harness-b", scope="personal", title="pers"),
        ]
    )
    out = DriftTool(adapter).call(scope="personal")
    assert "harness-b" in out
    assert "harness-a" not in out


def test_reprioritize_rerenders_current_state() -> None:
    adapter = FakeAdapter(
        ready_issues=[_issue(issue_id="harness-a", priority=2, dependent_count=5)]
    )
    out = ReprioritizeTool(adapter).call()
    # Priority-2-with-blockers promotes to Shall with a count reason.
    assert "Shall:" in out
    assert "blocks 5" in out


def test_close_tool_dispatch() -> None:
    adapter = FakeAdapter()
    CloseTool(adapter).call(id="harness-x", reason="shipped")
    assert adapter.close_calls == [{"id": "harness-x", "reason": "shipped"}]


def test_defer_tool_numeric_priority() -> None:
    adapter = FakeAdapter()
    out = DeferTool(adapter).call(id="harness-x", priority="3", reason="too big")
    assert adapter.update_calls == [("harness-x", {"priority": "3"})]
    assert adapter.remember_calls == ["deferred harness-x → P3: too big"]
    assert "P3" in out


def test_defer_tool_down_reads_current_priority() -> None:
    adapter = FakeAdapter(show_issues={"harness-y": _issue(issue_id="harness-y", priority=1)})
    DeferTool(adapter).call(id="harness-y", priority="down")
    # down shifts from P1 → P2 (capped at P4).
    assert adapter.update_calls == [("harness-y", {"priority": "2"})]


def test_defer_tool_rejects_out_of_range() -> None:
    out = DeferTool(FakeAdapter()).call(id="harness-x", priority="7")
    assert "must be 0-4" in out


def test_retro_summary_mode_renders_state() -> None:
    adapter = FakeAdapter(
        ready_issues=[_issue(issue_id="harness-a", priority=1, title="due today")],
        list_issues_data=[
            _issue(issue_id="harness-done", status="closed", priority=2),
        ],
    )
    out = RetroTool(adapter).call(mode="summary")
    assert "Retro —" in out
    assert "Open Shall/Should today: 1" in out
    assert "harness-a" in out
    assert "Total closed in history: 1" in out


def test_retro_record_mode_persists_insight() -> None:
    adapter = FakeAdapter()
    out = RetroTool(adapter).call(mode="record", insight="friday afternoons low-exec")
    assert adapter.remember_calls == ["retro: friday afternoons low-exec"]
    assert "Recorded" in out


def test_retro_record_mode_requires_insight() -> None:
    out = RetroTool(FakeAdapter()).call(mode="record")
    assert "requires insight" in out


def test_reopen_tool_dispatch() -> None:
    adapter = FakeAdapter()
    out = ReopenTool(adapter).call(id="harness-x", reason="wasn't really done")
    assert adapter.reopen_calls == [{"id": "harness-x", "reason": "wasn't really done"}]
    assert "Reopened" in out


def test_reopen_tool_without_reason() -> None:
    adapter = FakeAdapter()
    ReopenTool(adapter).call(id="harness-x")
    assert adapter.reopen_calls == [{"id": "harness-x", "reason": None}]


def test_delete_tool_dispatch_default() -> None:
    adapter = FakeAdapter()
    out = DeleteTool(adapter).call(id="harness-x")
    assert adapter.delete_calls == [{"id": "harness-x", "cascade": False}]
    assert "Deleted" in out
    assert "cascade" not in out


def test_delete_tool_dispatch_cascade() -> None:
    adapter = FakeAdapter()
    out = DeleteTool(adapter).call(id="harness-x", cascade=True)
    assert adapter.delete_calls == [{"id": "harness-x", "cascade": True}]
    assert "cascade" in out


def test_update_tool_dispatches_single_field() -> None:
    adapter = FakeAdapter()
    out = UpdateTool(adapter).call(id="harness-x", title="new name")
    assert adapter.update_calls == [("harness-x", {"title": "new name"})]
    assert "title" in out


def test_update_tool_dispatches_multiple_fields() -> None:
    adapter = FakeAdapter()
    UpdateTool(adapter).call(
        id="harness-x",
        title="t",
        description="d",
        priority=1,
        status="in_progress",
    )
    # priority is coerced to string for the bd CLI interface.
    assert adapter.update_calls == [
        (
            "harness-x",
            {
                "title": "t",
                "description": "d",
                "priority": "1",
                "status": "in_progress",
            },
        )
    ]


def test_update_tool_rejects_out_of_range_priority() -> None:
    adapter = FakeAdapter()
    out = UpdateTool(adapter).call(id="harness-x", priority=9)
    assert "must be 0-4" in out
    # No dispatch on validation failure.
    assert adapter.update_calls == []


def test_update_tool_requires_at_least_one_field() -> None:
    adapter = FakeAdapter()
    out = UpdateTool(adapter).call(id="harness-x")
    assert "no fields provided" in out
    assert adapter.update_calls == []


def test_make_ops_tools_returns_eleven_distinct_names() -> None:
    tools = make_ops_tools(FakeAdapter())
    names = [t.spec.name for t in tools]
    assert len(names) == 11
    assert len(set(names)) == 11
    # Expected surface matches the v1 spec plus tranche-1 additions.
    assert set(names) == {
        "plan",
        "capture",
        "status",
        "drift",
        "reprioritize",
        "close",
        "defer",
        "retro",
        "reopen",
        "delete",
        "update",
    }
