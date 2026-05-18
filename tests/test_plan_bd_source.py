"""Tests for the bd → Plan read adapter — harness-8jhq."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from harness.plan import Precondition, build_plan_from_bd
from harness.store._bd_types import BeadsIssue


@dataclass
class _StubBdAdapter:
    """Duck-typed BeadsAdapter for the bd-source tests. Mirrors the
    pattern in test_runtime_drift_task.py — we only stub the two
    methods the adapter actually calls (list_issues + show)."""

    issues_by_status: dict[str, list[BeadsIssue]] = field(default_factory=dict)
    show_returns: dict[str, BeadsIssue] = field(default_factory=dict)
    list_calls: list[dict[str, Any]] = field(default_factory=list)
    show_calls: list[str] = field(default_factory=list)

    def list_issues(
        self,
        *,
        status: str | None = None,
        assignee: str | None = None,
        **_: Any,
    ) -> list[BeadsIssue]:
        self.list_calls.append({"status": status, "assignee": assignee})
        return list(self.issues_by_status.get(status or "", []))

    def show(self, issue_id: str) -> BeadsIssue:
        self.show_calls.append(issue_id)
        if issue_id in self.show_returns:
            return self.show_returns[issue_id]
        return BeadsIssue(
            id=issue_id,
            title=issue_id,
            status="open",
            priority=2,
            issue_type="task",
            labels=(),
            raw={"id": issue_id},
            assignee=None,
        )


def _bead(
    id: str,
    *,
    status: str = "open",
    title: str | None = None,
    assignee: str = "mark",
    dependencies: list[Any] | None = None,
) -> BeadsIssue:
    """Make a BeadsIssue with the `raw.dependencies` shape `show`
    returns (entries like {id, dependency_type}).

    Note: dependency types other than 'blocks' are filtered out by the
    extractor — tests can include 'blocks-strongly' or similar to
    verify the filter."""
    raw: dict[str, Any] = {"id": id, "title": title or id, "status": status, "assignee": assignee}
    if dependencies is not None:
        raw["dependencies"] = dependencies
    return BeadsIssue(
        id=id,
        title=title or id,
        status=status,
        priority=2,
        issue_type="task",
        labels=(),
        raw=raw,
        assignee=assignee,
    )


_FROZEN_CLOCK = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)


def _clock() -> datetime:
    return _FROZEN_CLOCK


# --- status mapping --------------------------------------------------------


def test_in_progress_bead_becomes_active_subgoal() -> None:
    adapter = _StubBdAdapter(
        issues_by_status={"in_progress": [_bead("harness-a", status="in_progress")]},
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    sg = plan.subgoal("bd:harness-a")
    assert sg.status == "active"


def test_open_bead_becomes_pending_subgoal() -> None:
    adapter = _StubBdAdapter(
        issues_by_status={"open": [_bead("harness-b", status="open")]},
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    sg = plan.subgoal("bd:harness-b")
    assert sg.status == "pending"


def test_blocked_bead_becomes_pending_subgoal() -> None:
    """A blocked bead is still a goal we want to track — it's just
    waiting on a dependency. Maps to pending."""
    adapter = _StubBdAdapter(
        issues_by_status={"blocked": [_bead("harness-c", status="blocked")]},
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    sg = plan.subgoal("bd:harness-c")
    assert sg.status == "pending"


def test_closed_bead_excluded_by_default() -> None:
    """closed beads aren't current goals — exclude unless caller opts in."""
    adapter = _StubBdAdapter(
        issues_by_status={"closed": [_bead("harness-d", status="closed")]},
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    assert "bd:harness-d" not in plan.subgoals


def test_closed_bead_included_as_achieved_when_opt_in() -> None:
    adapter = _StubBdAdapter(
        issues_by_status={"closed": [_bead("harness-d", status="closed")]},
    )
    plan = build_plan_from_bd(adapter, assignee="mark", include_closed=True, clock=_clock)
    sg = plan.subgoal("bd:harness-d")
    assert sg.status == "achieved"


# --- structure -------------------------------------------------------------


def test_plan_has_root_subgoal_with_each_bead_as_child() -> None:
    adapter = _StubBdAdapter(
        issues_by_status={
            "in_progress": [_bead("harness-a", status="in_progress")],
            "open": [_bead("harness-b", status="open")],
        },
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    root_id = plan.root_subgoal_id
    assert plan.subgoal(root_id).status == "active"
    children = plan.children_of(root_id)
    child_ids = {c.id for c in children}
    assert child_ids == {"bd:harness-a", "bd:harness-b"}


def test_plan_id_defaults_to_bd_assignee() -> None:
    """Reproducibility: same assignee → same plan id."""
    adapter = _StubBdAdapter()
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    assert plan.id == "bd:mark"
    assert plan.root_subgoal_id == "bd:mark:root"


def test_explicit_plan_id_overrides_default() -> None:
    adapter = _StubBdAdapter()
    plan = build_plan_from_bd(adapter, assignee="mark", plan_id="main", clock=_clock)
    assert plan.id == "main"
    assert plan.root_subgoal_id == "main:root"


def test_same_bd_state_produces_identical_plan() -> None:
    """Determinism: two consecutive calls with identical adapter state
    produce the same Plan (modulo identity)."""
    adapter1 = _StubBdAdapter(
        issues_by_status={
            "in_progress": [_bead("harness-b", status="in_progress")],
            "open": [_bead("harness-a", status="open"), _bead("harness-c", status="open")],
        },
    )
    adapter2 = _StubBdAdapter(
        issues_by_status={
            # Different YAML ordering — sorted iteration normalizes.
            "open": [_bead("harness-c", status="open"), _bead("harness-a", status="open")],
            "in_progress": [_bead("harness-b", status="in_progress")],
        },
    )
    p1 = build_plan_from_bd(adapter1, assignee="mark", clock=_clock)
    p2 = build_plan_from_bd(adapter2, assignee="mark", clock=_clock)
    assert p1.to_dict() == p2.to_dict()


# --- dependencies → preconditions -----------------------------------------


def test_bd_dependencies_become_bd_closed_preconditions() -> None:
    """A bead that depends on (is blocked by) harness-x carries a
    bd_closed precondition for harness-x."""
    adapter = _StubBdAdapter(
        issues_by_status={"in_progress": [_bead("harness-a", status="in_progress")]},
        show_returns={
            "harness-a": _bead(
                "harness-a",
                status="in_progress",
                dependencies=[
                    {"id": "harness-x", "dependency_type": "blocks"},
                    {"id": "harness-y", "dependency_type": "blocks"},
                ],
            ),
        },
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    sg = plan.subgoal("bd:harness-a")
    assert sg.preconditions == (
        Precondition(kind="bd_closed", payload={"bead": "harness-x"}),
        Precondition(kind="bd_closed", payload={"bead": "harness-y"}),
    )


def test_non_blocks_dependency_types_filtered_out() -> None:
    """Only `dependency_type='blocks'` becomes a bd_closed precondition.
    A 'related' or future dep type doesn't translate to a hard
    precondition."""
    adapter = _StubBdAdapter(
        issues_by_status={"in_progress": [_bead("harness-a", status="in_progress")]},
        show_returns={
            "harness-a": _bead(
                "harness-a",
                status="in_progress",
                dependencies=[
                    {"id": "harness-x", "dependency_type": "blocks"},
                    {"id": "harness-y", "dependency_type": "related"},
                ],
            ),
        },
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    sg = plan.subgoal("bd:harness-a")
    assert sg.preconditions == (Precondition(kind="bd_closed", payload={"bead": "harness-x"}),)


def test_preconditions_are_sorted_by_bead_id() -> None:
    """Determinism: preconditions are sorted alphabetically so re-runs
    produce identical Plan dicts."""
    adapter = _StubBdAdapter(
        issues_by_status={"in_progress": [_bead("harness-a", status="in_progress")]},
        show_returns={
            "harness-a": _bead(
                "harness-a",
                status="in_progress",
                dependencies=[
                    {"id": "harness-z", "dependency_type": "blocks"},
                    {"id": "harness-a-dep", "dependency_type": "blocks"},
                    {"id": "harness-m", "dependency_type": "blocks"},
                ],
            ),
        },
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    sg = plan.subgoal("bd:harness-a")
    bead_ids = [p.payload["bead"] for p in sg.preconditions]
    assert bead_ids == sorted(bead_ids)


def test_missing_dependencies_field_yields_no_preconditions() -> None:
    adapter = _StubBdAdapter(
        issues_by_status={"in_progress": [_bead("harness-a", status="in_progress")]},
        # show returns a bead with no `dependencies` field in raw.
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    sg = plan.subgoal("bd:harness-a")
    assert sg.preconditions == ()


def test_malformed_dependency_entries_skipped() -> None:
    """A non-dict entry (corrupt bd export) shouldn't crash the
    extractor — just drop the entry."""
    adapter = _StubBdAdapter(
        issues_by_status={"in_progress": [_bead("harness-a", status="in_progress")]},
        show_returns={
            "harness-a": _bead(
                "harness-a",
                status="in_progress",
                dependencies=[
                    "not-a-dict",
                    {"id": "harness-x", "dependency_type": "blocks"},
                    {"dependency_type": "blocks"},  # missing id
                    {"id": "", "dependency_type": "blocks"},  # empty id
                ],
            ),
        },
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    sg = plan.subgoal("bd:harness-a")
    assert sg.preconditions == (Precondition(kind="bd_closed", payload={"bead": "harness-x"}),)


# --- deduplication ---------------------------------------------------------


def test_bead_appearing_in_multiple_status_lists_keeps_first_seen() -> None:
    """If a stub adapter returns the same bead under two statuses
    (unrealistic in practice but the code shouldn't double-add),
    only one Subgoal results."""
    adapter = _StubBdAdapter(
        issues_by_status={
            "in_progress": [_bead("harness-a", status="in_progress")],
            "open": [_bead("harness-a", status="open")],
        },
    )
    plan = build_plan_from_bd(adapter, assignee="mark", clock=_clock)
    # Only one Subgoal, status = active (in_progress wins because that
    # status is fetched first).
    assert plan.subgoal("bd:harness-a").status == "active"
    children = plan.children_of(plan.root_subgoal_id)
    assert len([c for c in children if c.id == "bd:harness-a"]) == 1


def test_assignee_filter_passes_through_to_adapter() -> None:
    """Sanity: the adapter is queried for the configured assignee."""
    adapter = _StubBdAdapter()
    build_plan_from_bd(adapter, assignee="airton_b", clock=_clock)
    assignees_queried = {call["assignee"] for call in adapter.list_calls}
    assert assignees_queried == {"airton_b"}
