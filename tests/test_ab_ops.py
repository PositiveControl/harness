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
    CommentsTool,
    DeferTool,
    DeleteTool,
    DepTool,
    DriftTool,
    FindDuplicatesTool,
    ForgetTool,
    LabelTool,
    ListTool,
    MemoriesTool,
    PlanTool,
    RememberTool,
    ReopenTool,
    ReprioritizeTool,
    RetroTool,
    SearchTool,
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
    search_hits: list[BeadsIssue] = field(default_factory=list)
    show_issues: dict[str, BeadsIssue] = field(default_factory=dict)
    stale_issues: list[BeadsIssue] = field(default_factory=list)
    memories_output: str = ""
    label_list_output: str = ""
    comments_list_output: str = ""
    duplicate_pairs: list[dict[str, Any]] = field(default_factory=list)
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    close_calls: list[dict[str, Any]] = field(default_factory=list)
    reopen_calls: list[dict[str, Any]] = field(default_factory=list)
    delete_calls: list[dict[str, Any]] = field(default_factory=list)
    update_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    search_calls: list[dict[str, Any]] = field(default_factory=list)
    list_calls: list[dict[str, Any]] = field(default_factory=list)
    dep_add_calls: list[tuple[str, str]] = field(default_factory=list)
    dep_rm_calls: list[tuple[str, str]] = field(default_factory=list)
    forget_calls: list[str] = field(default_factory=list)
    memories_calls: list[str] = field(default_factory=list)
    remember_calls: list[str] = field(default_factory=list)
    label_add_calls: list[tuple[str, str]] = field(default_factory=list)
    label_rm_calls: list[tuple[str, str]] = field(default_factory=list)
    label_list_calls: list[str] = field(default_factory=list)
    comment_add_calls: list[tuple[str, str]] = field(default_factory=list)
    comments_list_calls: list[str] = field(default_factory=list)
    find_duplicates_calls: list[dict[str, Any]] = field(default_factory=list)
    focus_issue: BeadsIssue | None = None
    get_focus_calls: list[str] = field(default_factory=list)
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
        priority: str | None = None,
        issue_type: str | None = None,
        limit: int | None = None,
        assignee: str | None = None,
    ) -> list[BeadsIssue]:
        self.list_calls.append(
            {
                "scope": scope,
                "status": status,
                "priority": priority,
                "issue_type": issue_type,
                "limit": limit,
                "assignee": assignee,
            }
        )
        return list(self.list_issues_data)

    def search(
        self,
        query: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[BeadsIssue]:
        self.search_calls.append({"query": query, "status": status, "limit": limit})
        return list(self.search_hits)

    def show(self, issue_id: str) -> BeadsIssue:
        return self.show_issues[issue_id]

    def stale(self) -> list[BeadsIssue]:
        return list(self.stale_issues)

    def get_focus(self, assignee: str) -> BeadsIssue | None:
        self.get_focus_calls.append(assignee)
        return self.focus_issue

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

    def dep_add(self, issue: str, depends_on: str) -> None:
        self.dep_add_calls.append((issue, depends_on))

    def dep_rm(self, issue: str, depends_on: str) -> None:
        self.dep_rm_calls.append((issue, depends_on))

    def remember(self, insight: str) -> None:
        self.remember_calls.append(insight)

    def memories(self, query: str = "") -> str:
        self.memories_calls.append(query)
        return self.memories_output

    def forget(self, key: str) -> None:
        self.forget_calls.append(key)

    def label_add(self, issue_id: str, label: str) -> None:
        self.label_add_calls.append((issue_id, label))

    def label_rm(self, issue_id: str, label: str) -> None:
        self.label_rm_calls.append((issue_id, label))

    def label_list(self, issue_id: str) -> str:
        self.label_list_calls.append(issue_id)
        return self.label_list_output

    def comment_add(self, issue_id: str, text: str) -> None:
        self.comment_add_calls.append((issue_id, text))

    def comments_list(self, issue_id: str) -> str:
        self.comments_list_calls.append(issue_id)
        return self.comments_list_output

    def find_duplicates(
        self,
        *,
        threshold: float | None = None,
        limit: int | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self.find_duplicates_calls.append(
            {"threshold": threshold, "limit": limit, "status": status}
        )
        return list(self.duplicate_pairs)


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


def test_capture_passes_ab_assignee_when_ab_owned() -> None:
    """ab_owned=True routes the create through assignee=airton_b so
    the adapter's per-turn budget mechanism can count it. False (the
    default) leaves assignee unset — user-owned captures stay
    unassigned by default."""
    adapter = FakeAdapter(next_create_id="harness-ab")
    CaptureTool(adapter).call(
        raw="track this hypothesis",
        scope="personal",
        outcome="confirmed or refuted",
        next_action="check tests/test_foo.py",
        ab_owned=True,
    )
    assert adapter.create_calls[0]["assignee"] == "airton_b"


def test_capture_no_assignee_when_not_ab_owned() -> None:
    adapter = FakeAdapter(next_create_id="harness-usr")
    CaptureTool(adapter).call(
        raw="buy milk",
        scope="personal",
        outcome="milk in fridge",
        next_action="grocery run",
    )
    assert adapter.create_calls[0]["assignee"] is None


def test_capture_surfaces_turn_cap_error() -> None:
    """CaptureTool catches TurnCapExceededError through the generic
    BeadsAdapterError branch and returns the informative message —
    model sees 'capture failed: turn-cap reached …'."""
    from harness.store.bd_adapter import TurnCapExceededError

    class CappedAdapter(FakeAdapter):
        def create(self, **kwargs: Any) -> str:
            raise TurnCapExceededError("turn-cap reached (3 ab-owned beads this turn).")

    out = CaptureTool(CappedAdapter()).call(
        raw="new thought",
        scope="personal",
        outcome="resolved",
        next_action="think",
        ab_owned=True,
    )
    assert "capture failed" in out
    assert "turn-cap reached" in out


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


def test_plan_tool_prepends_focus_banner_when_focus_set() -> None:
    focus = _issue(
        issue_id="harness-focus",
        scope="professional",
        title="current work",
        priority=1,
    )
    ready = [_issue(issue_id="harness-a", priority=2, scope="professional")]
    adapter = FakeAdapter(ready_issues=ready, focus_issue=focus)

    out = PlanTool(adapter).call()

    assert out.startswith("Focus:")
    assert "harness-focus" in out
    assert "current work" in out
    assert "Shall:" in out or "Should:" in out  # plan still renders below


def test_plan_tool_omits_focus_banner_when_no_focus() -> None:
    ready = [_issue(issue_id="harness-a", priority=2, scope="professional")]
    adapter = FakeAdapter(ready_issues=ready, focus_issue=None)

    out = PlanTool(adapter).call()

    assert not out.startswith("Focus:")
    assert "harness-a" in out


def test_status_tool_without_id_shows_focus() -> None:
    focus = _issue(
        issue_id="harness-focus",
        scope="personal",
        title="mid-task",
        status="in_progress",
    )
    adapter = FakeAdapter(focus_issue=focus, show_issues={"harness-focus": focus})

    out = StatusTool(adapter).call()

    assert "harness-focus" in out
    assert "mid-task" in out


def test_status_tool_without_id_no_focus_returns_hint() -> None:
    adapter = FakeAdapter(focus_issue=None)

    out = StatusTool(adapter).call()

    assert "(no focus)" in out


def test_status_tool_appends_hint_when_id_differs_from_focus() -> None:
    focus = _issue(issue_id="harness-focus", title="focused", status="in_progress")
    target = _issue(issue_id="harness-other", title="other")
    adapter = FakeAdapter(
        focus_issue=focus,
        show_issues={"harness-other": target},
    )

    out = StatusTool(adapter).call(id="harness-other")

    assert "harness-other" in out
    assert "Not current focus" in out
    assert "harness-focus" in out


def test_status_tool_no_hint_when_id_matches_focus() -> None:
    focus = _issue(issue_id="harness-focus", title="focused", status="in_progress")
    adapter = FakeAdapter(
        focus_issue=focus,
        show_issues={"harness-focus": focus},
    )

    out = StatusTool(adapter).call(id="harness-focus")

    assert "Not current focus" not in out


def test_status_tool_id_not_in_required_schema() -> None:
    """B2 makes the id parameter optional — confirm the schema matches
    so small-model tool emitters don't reject the omission."""
    adapter = FakeAdapter()
    spec = StatusTool(adapter).spec
    assert spec.parameters["required"] == []


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


def _ab_issue_with_updated_at(issue_id: str, updated_at: str) -> BeadsIssue:
    """Helper: build an ab-owned open bead with a controllable
    updated_at timestamp in the raw payload."""
    base = _issue(issue_id=issue_id, scope="personal", title=f"ab {issue_id}")
    raw = dict(base.raw)
    raw["updated_at"] = updated_at
    return BeadsIssue(
        id=base.id,
        title=base.title,
        status="open",
        priority=base.priority,
        issue_type=base.issue_type,
        labels=base.labels,
        raw=raw,
    )


def test_drift_flags_ab_bead_past_threshold() -> None:
    """Ab-owned bead last-updated 10 days ago appears in drift output
    even when bd's own stale list is empty."""
    from datetime import UTC, datetime, timedelta

    old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    ab_old = _ab_issue_with_updated_at("harness-old", old_ts)
    adapter = FakeAdapter(stale_issues=[], list_issues_data=[ab_old])

    out = DriftTool(adapter).call()

    assert "harness-old" in out


def test_drift_skips_ab_bead_below_threshold() -> None:
    """Ab-owned bead updated 2 days ago is fresh and stays out of
    drift output."""
    from datetime import UTC, datetime, timedelta

    recent_ts = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    ab_fresh = _ab_issue_with_updated_at("harness-fresh", recent_ts)
    adapter = FakeAdapter(stale_issues=[], list_issues_data=[ab_fresh])

    out = DriftTool(adapter).call()

    assert "harness-fresh" not in out


def test_drift_threshold_override_changes_cutoff() -> None:
    """Explicit ab_drift_days on the tool changes the threshold. At
    14 days the 10-day-old bead is no longer stale; at 7 days it is."""
    from datetime import UTC, datetime, timedelta

    ts_10_days_ago = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    ab_old = _ab_issue_with_updated_at("harness-10", ts_10_days_ago)
    adapter = FakeAdapter(stale_issues=[], list_issues_data=[ab_old])

    tight = DriftTool(adapter, ab_drift_days=7).call()
    lax = DriftTool(adapter, ab_drift_days=14).call()

    assert "harness-10" in tight
    assert "harness-10" not in lax


def test_drift_merges_bd_stale_and_ab_stale_deduped() -> None:
    """If bd.stale already flagged a bead AND the ab filter also
    considers it stale, the output lists it once, not twice."""
    from datetime import UTC, datetime, timedelta

    old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    shared = _ab_issue_with_updated_at("harness-shared", old_ts)
    adapter = FakeAdapter(stale_issues=[shared], list_issues_data=[shared])

    out = DriftTool(adapter).call()
    # Count rendered bullet lines referencing the bead id — one per
    # appearance. Dedupe means exactly one bullet.
    bullet_lines = [line for line in out.splitlines() if "harness-shared]" in line]
    assert len(bullet_lines) == 1


def test_drift_ab_stale_queries_airton_b_assignee() -> None:
    """The client-side ab drift filter must list issues with
    assignee=airton_b so bd applies the server-side filter —
    otherwise the client-side age check would scan every open bead."""
    adapter = FakeAdapter(stale_issues=[], list_issues_data=[])
    DriftTool(adapter).call()
    last_call = adapter.list_calls[-1]
    assert last_call["status"] == "open"
    assert last_call["assignee"] == "airton_b"


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
    adapter = FakeAdapter(show_issues={"harness-x": _issue(issue_id="harness-x", priority=2)})
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


def test_defer_first_adds_count_label() -> None:
    """First defer on a bead that has no defer-count label just adds
    `defer-count:1`; no label_rm needed."""
    adapter = FakeAdapter(show_issues={"harness-x": _issue(issue_id="harness-x", priority=2)})
    DeferTool(adapter).call(id="harness-x", priority="3")
    assert adapter.label_add_calls == [("harness-x", "defer-count:1")]
    assert adapter.label_rm_calls == []


def test_defer_increments_existing_count_label() -> None:
    """Second defer swaps defer-count:1 → defer-count:2 via rm + add."""
    issue = _issue(issue_id="harness-x", priority=2)
    # Replace labels to include a pre-existing defer counter.
    issue_with_count = BeadsIssue(
        id=issue.id,
        title=issue.title,
        status=issue.status,
        priority=issue.priority,
        issue_type=issue.issue_type,
        labels=("defer-count:1",),
        raw=issue.raw,
    )
    adapter = FakeAdapter(show_issues={"harness-x": issue_with_count})
    DeferTool(adapter).call(id="harness-x", priority="3")
    assert adapter.label_rm_calls == [("harness-x", "defer-count:1")]
    assert adapter.label_add_calls == [("harness-x", "defer-count:2")]


def test_defer_escalates_on_third_defer() -> None:
    """Third defer (count 2 → 3) triggers stall escalation: creates
    thought:question child parent-linked, adds stall-escalated label
    to parent."""
    issue = _issue(
        issue_id="harness-parent",
        priority=2,
        scope="personal",
        title="original work",
    )
    with_count_2 = BeadsIssue(
        id=issue.id,
        title=issue.title,
        status=issue.status,
        priority=issue.priority,
        issue_type=issue.issue_type,
        labels=("scope:personal", "defer-count:2"),
        raw=issue.raw,
    )
    adapter = FakeAdapter(
        show_issues={"harness-parent": with_count_2},
        next_create_id="harness-question",
    )

    out = DeferTool(adapter).call(id="harness-parent", priority="4")

    assert "Stall-escalated" in out
    assert "harness-question" in out
    # The escalation create carries the expected shape.
    assert len(adapter.create_calls) == 1
    payload = adapter.create_calls[0]
    assert payload["scope"] == "personal"
    assert payload["parent"] == "harness-parent"
    assert "thought:question" in payload["extra_labels"]
    assert payload["assignee"] == "airton_b"
    # Parent gets the stall-escalated label so a 4th defer won't
    # spawn another escalation.
    assert ("harness-parent", "stall-escalated") in adapter.label_add_calls


def test_defer_skips_escalation_when_already_escalated() -> None:
    """If the bead already carries the stall-escalated label, a
    subsequent defer must not spawn another escalation child."""
    issue = _issue(issue_id="harness-parent", priority=3, scope="personal")
    already = BeadsIssue(
        id=issue.id,
        title=issue.title,
        status=issue.status,
        priority=issue.priority,
        issue_type=issue.issue_type,
        labels=("scope:personal", "defer-count:3", "stall-escalated"),
        raw=issue.raw,
    )
    adapter = FakeAdapter(show_issues={"harness-parent": already})

    out = DeferTool(adapter).call(id="harness-parent", priority="4")

    assert "Stall-escalated" not in out
    assert adapter.create_calls == []


def test_defer_first_two_defers_do_not_escalate() -> None:
    """Defers 1 and 2 just bump the counter; no escalation child
    before the threshold."""
    issue = _issue(issue_id="harness-x", priority=2, scope="personal")
    adapter = FakeAdapter(show_issues={"harness-x": issue})

    DeferTool(adapter).call(id="harness-x", priority="3")
    assert adapter.create_calls == []

    # Simulate second defer by replacing the stored issue with
    # defer-count:1 labels to match what bd would show after the first.
    with_1 = BeadsIssue(
        id=issue.id,
        title=issue.title,
        status=issue.status,
        priority=issue.priority,
        issue_type=issue.issue_type,
        labels=("scope:personal", "defer-count:1"),
        raw=issue.raw,
    )
    adapter.show_issues["harness-x"] = with_1
    DeferTool(adapter).call(id="harness-x", priority="3")
    assert adapter.create_calls == []


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


def test_search_tool_renders_hits() -> None:
    adapter = FakeAdapter(
        search_hits=[
            _issue(issue_id="harness-a", title="auth bug", priority=1),
        ]
    )
    out = SearchTool(adapter).call(query="auth", status="all", limit=5)
    assert adapter.search_calls == [{"query": "auth", "status": "all", "limit": 5}]
    assert "auth" in out
    assert "harness-a" in out
    assert "P1" in out


def test_search_tool_empty_hits() -> None:
    out = SearchTool(FakeAdapter()).call(query="nope")
    assert "no matches" in out


def test_search_tool_rejects_empty_query() -> None:
    adapter = FakeAdapter()
    out = SearchTool(adapter).call(query="   ")
    assert "non-empty" in out
    assert adapter.search_calls == []


def test_list_tool_filters_passed_through() -> None:
    adapter = FakeAdapter(
        list_issues_data=[
            _issue(issue_id="harness-a", priority=2, title="thing"),
        ]
    )
    out = ListTool(adapter).call(status="open", priority=2, issue_type="task")
    # priority is coerced to string when forwarded to the adapter (bd
    # takes it as a string flag).
    assert adapter.list_calls == [
        {
            "scope": None,
            "status": "open",
            "priority": "2",
            "issue_type": "task",
            "limit": None,
            "assignee": None,
        }
    ]
    assert "harness-a" in out


def test_list_tool_empty_branch() -> None:
    out = ListTool(FakeAdapter()).call()
    assert "no matching items" in out


def test_list_tool_rejects_out_of_range_priority() -> None:
    adapter = FakeAdapter()
    out = ListTool(adapter).call(priority=9)
    assert "must be 0-4" in out
    assert adapter.list_calls == []


def test_list_tool_rejects_invalid_scope() -> None:
    out = ListTool(FakeAdapter()).call(scope="social")
    assert "invalid scope" in out


def test_memories_tool_returns_bd_output() -> None:
    adapter = FakeAdapter(memories_output="  dolt-phantoms: be careful about ...\n")
    out = MemoriesTool(adapter).call(query="dolt")
    assert adapter.memories_calls == ["dolt"]
    assert "dolt-phantoms" in out


def test_memories_tool_empty_branch() -> None:
    adapter = FakeAdapter(memories_output="")
    out = MemoriesTool(adapter).call()
    assert "no memories" in out


def test_forget_tool_dispatch() -> None:
    adapter = FakeAdapter()
    out = ForgetTool(adapter).call(key="dolt-phantoms")
    assert adapter.forget_calls == ["dolt-phantoms"]
    assert "Forgot" in out


def test_remember_tool_dispatch() -> None:
    adapter = FakeAdapter()
    out = RememberTool(adapter).call(insight="airton_b is gay")
    assert adapter.remember_calls == ["airton_b is gay"]
    assert "Remembered" in out


def test_remember_tool_rejects_empty_insight() -> None:
    adapter = FakeAdapter()
    out = RememberTool(adapter).call(insight="   ")
    assert "non-empty" in out
    assert adapter.remember_calls == []


def test_dep_tool_add() -> None:
    adapter = FakeAdapter()
    out = DepTool(adapter).call(op="add", issue="harness-a", depends_on="harness-b")
    assert adapter.dep_add_calls == [("harness-a", "harness-b")]
    assert adapter.dep_rm_calls == []
    assert "Linked" in out


def test_dep_tool_remove() -> None:
    adapter = FakeAdapter()
    out = DepTool(adapter).call(op="remove", issue="harness-a", depends_on="harness-b")
    assert adapter.dep_rm_calls == [("harness-a", "harness-b")]
    assert adapter.dep_add_calls == []
    assert "Unlinked" in out


def test_dep_tool_rejects_unknown_op() -> None:
    adapter = FakeAdapter()
    out = DepTool(adapter).call(op="toggle", issue="harness-a", depends_on="harness-b")
    assert "unknown op" in out
    assert adapter.dep_add_calls == []
    assert adapter.dep_rm_calls == []


def test_label_tool_add() -> None:
    adapter = FakeAdapter()
    out = LabelTool(adapter).call(op="add", id="harness-x", label="tech-debt")
    assert adapter.label_add_calls == [("harness-x", "tech-debt")]
    assert "Added label" in out


def test_label_tool_remove() -> None:
    adapter = FakeAdapter()
    out = LabelTool(adapter).call(op="remove", id="harness-x", label="tech-debt")
    assert adapter.label_rm_calls == [("harness-x", "tech-debt")]
    assert "Removed label" in out


def test_label_tool_list() -> None:
    adapter = FakeAdapter(label_list_output="scope:professional\ntech-debt\n")
    out = LabelTool(adapter).call(op="list", id="harness-x")
    assert adapter.label_list_calls == ["harness-x"]
    assert "tech-debt" in out


def test_label_tool_list_empty_branch() -> None:
    adapter = FakeAdapter(label_list_output="")
    out = LabelTool(adapter).call(op="list", id="harness-x")
    assert "no labels" in out


def test_label_tool_add_requires_label() -> None:
    adapter = FakeAdapter()
    out = LabelTool(adapter).call(op="add", id="harness-x")
    assert "requires a label" in out
    assert adapter.label_add_calls == []


def test_label_tool_rejects_unknown_op() -> None:
    out = LabelTool(FakeAdapter()).call(op="toggle", id="harness-x", label="x")
    assert "unknown op" in out


def test_comments_tool_add() -> None:
    adapter = FakeAdapter()
    out = CommentsTool(adapter).call(op="add", id="harness-x", text="looking into this")
    assert adapter.comment_add_calls == [("harness-x", "looking into this")]
    assert "Added comment" in out


def test_comments_tool_list() -> None:
    adapter = FakeAdapter(comments_list_output="2026-04-18: looking into this\n")
    out = CommentsTool(adapter).call(op="list", id="harness-x")
    assert adapter.comments_list_calls == ["harness-x"]
    assert "looking into this" in out


def test_comments_tool_list_empty_branch() -> None:
    adapter = FakeAdapter(comments_list_output="")
    out = CommentsTool(adapter).call(op="list", id="harness-x")
    assert "no comments" in out


def test_comments_tool_add_requires_text() -> None:
    adapter = FakeAdapter()
    out = CommentsTool(adapter).call(op="add", id="harness-x", text="   ")
    assert "requires non-empty text" in out
    assert adapter.comment_add_calls == []


def test_find_duplicates_tool_renders_pairs() -> None:
    adapter = FakeAdapter(
        duplicate_pairs=[
            {
                "a_id": "harness-a",
                "b_id": "harness-b",
                "a_title": "auth refactor",
                "b_title": "refactor the auth path",
                "similarity": 0.78,
            }
        ]
    )
    out = FindDuplicatesTool(adapter).call(threshold=0.4, limit=10, status="open")
    assert adapter.find_duplicates_calls == [{"threshold": 0.4, "limit": 10, "status": "open"}]
    assert "harness-a" in out
    assert "harness-b" in out
    assert "0.78" in out


def test_find_duplicates_tool_empty_branch() -> None:
    out = FindDuplicatesTool(FakeAdapter()).call()
    assert "no duplicate candidates" in out


def test_find_duplicates_tool_rejects_out_of_range_threshold() -> None:
    out = FindDuplicatesTool(FakeAdapter()).call(threshold=1.5)
    assert "0.0-1.0" in out


def test_make_ops_tools_returns_twenty_distinct_names() -> None:
    tools = make_ops_tools(FakeAdapter())
    names = [t.spec.name for t in tools]
    assert len(names) == 20
    assert len(set(names)) == 20
    # Expected surface covers every tranche (v1 + tranche-1 + tranche-2
    # + tranche-3) plus remember (harness-0dj fix).
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
        "search",
        "list",
        "memories",
        "remember",
        "forget",
        "dep",
        "label",
        "comments",
        "find_duplicates",
    }
