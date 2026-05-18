"""Tests for the Plan → bd writeback — harness-wxq4."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from harness.plan import (
    BeadsIssueDraft,
    PlanWriteback,
    Status,
    Subgoal,
    WritebackOpResult,
    apply_writeback,
    diff_plans,
    new_plan,
    new_subgoal,
)


@dataclass
class _StubBdAdapter:
    """Duck-typed BeadsAdapter recording every write op. Tests inject
    `raise_on_close` / `raise_on_create` etc. to exercise the per-op
    failure paths.

    `create_returns_id`: id to return from create() — drives the
    created_id captured into the WritebackOpResult."""

    close_calls: list[tuple[str, str | None]] = field(default_factory=list)
    reopen_calls: list[tuple[str, str | None]] = field(default_factory=list)
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    comment_calls: list[tuple[str, str]] = field(default_factory=list)
    raise_on_close: Exception | None = None
    raise_on_reopen: Exception | None = None
    raise_on_create: Exception | None = None
    raise_on_comment: Exception | None = None
    create_returns_id: str = "harness-new"

    def close(self, issue_id: str, *, reason: str | None = None) -> None:
        self.close_calls.append((issue_id, reason))
        if self.raise_on_close is not None:
            raise self.raise_on_close

    def reopen(self, issue_id: str, *, reason: str | None = None) -> None:
        self.reopen_calls.append((issue_id, reason))
        if self.raise_on_reopen is not None:
            raise self.raise_on_reopen

    def create(self, **kwargs: Any) -> str:
        self.create_calls.append(kwargs)
        if self.raise_on_create is not None:
            raise self.raise_on_create
        return self.create_returns_id

    def comment_add(self, issue_id: str, text: str) -> None:
        self.comment_calls.append((issue_id, text))
        if self.raise_on_comment is not None:
            raise self.raise_on_comment


# --- diff_plans -----------------------------------------------------------


def _plan_with(*subgoals: Subgoal, plan_id: str = "plan-1") -> Any:
    p = new_plan("p", plan_id=plan_id)
    for sg in subgoals:
        # Ensure parent_id is the root if not set.
        if sg.parent_id is None:
            sg = replace(sg, parent_id=p.root_subgoal_id)
        p = p.with_subgoal(sg)
    return p


def _bd_subgoal(bd_id: str, status: Status = "pending") -> Subgoal:
    return Subgoal(id=f"bd:{bd_id}", title=bd_id, status=status, parent_id=None)


def test_active_to_achieved_bd_sourced_emits_close() -> None:
    old = _plan_with(_bd_subgoal("harness-x", status="active"))
    new = _plan_with(_bd_subgoal("harness-x", status="achieved"))
    wb = diff_plans(old, new)
    assert wb.to_close == ("harness-x",)
    assert wb.to_reopen == ()
    assert wb.to_create == ()


def test_achieved_to_active_bd_sourced_emits_reopen() -> None:
    """Plan-revision undid an achievement (rare — bd bead reopened
    externally). Diff emits a reopen so the writeback can re-sync."""
    old = _plan_with(_bd_subgoal("harness-x", status="achieved"))
    new = _plan_with(_bd_subgoal("harness-x", status="active"))
    wb = diff_plans(old, new)
    assert wb.to_reopen == ("harness-x",)
    assert wb.to_close == ()


def test_pending_to_active_emits_nothing() -> None:
    """bd doesn't model 'active' as distinct from 'open' — this is
    a Plan-only transition; no bd op."""
    old = _plan_with(_bd_subgoal("harness-x", status="pending"))
    new = _plan_with(_bd_subgoal("harness-x", status="active"))
    wb = diff_plans(old, new)
    assert wb.is_empty


def test_active_to_pending_emits_nothing() -> None:
    """Reversible Plan-side demotion; bd state unchanged."""
    old = _plan_with(_bd_subgoal("harness-x", status="active"))
    new = _plan_with(_bd_subgoal("harness-x", status="pending"))
    wb = diff_plans(old, new)
    assert wb.is_empty


def test_no_status_change_emits_nothing() -> None:
    old = _plan_with(_bd_subgoal("harness-x", status="active"))
    new = _plan_with(_bd_subgoal("harness-x", status="active"))
    wb = diff_plans(old, new)
    assert wb.is_empty


def test_non_bd_subgoal_transitions_are_ignored() -> None:
    """A Subgoal without the 'bd:' prefix has no corresponding bead
    yet — close/reopen don't make sense. The create path handles new
    non-bd Subgoals."""
    sg_old = new_subgoal("a", status="active")
    sg_new = replace(sg_old, status="achieved")
    p_old = _plan_with(sg_old)
    p_new = _plan_with(sg_new)
    wb = diff_plans(p_old, p_new)
    assert wb.is_empty


def test_new_non_bd_subgoal_becomes_create() -> None:
    """A brand-new Subgoal (in `new` but not `old`) without a bd
    prefix becomes a to_create entry. The writeback fills in the
    bd id at apply time."""
    old = _plan_with()  # just the root
    new_sg = new_subgoal("ship the doc", subgoal_id="sg-fresh")
    new = _plan_with(new_sg)
    wb = diff_plans(old, new)
    assert len(wb.to_create) == 1
    draft = wb.to_create[0]
    assert draft.subgoal_id == "sg-fresh"
    assert draft.title == "ship the doc"


def test_new_bd_prefixed_subgoal_is_not_recreated() -> None:
    """A bd-prefixed Subgoal in `new` but not `old` already has a
    bd backing — creating it again would dupe."""
    old = _plan_with()
    new = _plan_with(_bd_subgoal("harness-x", status="pending"))
    wb = diff_plans(old, new)
    assert wb.to_create == ()


def test_removed_subgoals_emit_nothing() -> None:
    """Plan revisions don't auto-delete beads (the operator does that
    via bd directly). A Subgoal in `old` but not `new` is a no-op."""
    old = _plan_with(_bd_subgoal("harness-x", status="active"))
    new = _plan_with()
    wb = diff_plans(old, new)
    assert wb.is_empty


def test_diff_is_deterministic() -> None:
    """Sort order matches across calls so two diffs of the same
    (old, new) pair compare equal — important for an audit log."""
    old = _plan_with(
        _bd_subgoal("harness-b", status="active"),
        _bd_subgoal("harness-a", status="active"),
        _bd_subgoal("harness-c", status="active"),
    )
    new = _plan_with(
        _bd_subgoal("harness-b", status="achieved"),
        _bd_subgoal("harness-a", status="achieved"),
        _bd_subgoal("harness-c", status="achieved"),
    )
    wb1 = diff_plans(old, new)
    wb2 = diff_plans(old, new)
    assert wb1 == wb2
    assert wb1.to_close == ("harness-a", "harness-b", "harness-c")


# --- apply_writeback ------------------------------------------------------


def test_apply_close_calls_adapter() -> None:
    adapter = _StubBdAdapter()
    wb = PlanWriteback(to_close=("harness-x",))
    result = apply_writeback(adapter, wb)
    assert adapter.close_calls == [("harness-x", "closed via plan writeback")]
    assert result.all_succeeded


def test_apply_reopen_calls_adapter() -> None:
    adapter = _StubBdAdapter()
    wb = PlanWriteback(to_reopen=("harness-x",))
    result = apply_writeback(adapter, wb)
    assert adapter.reopen_calls == [("harness-x", "reopened via plan writeback")]
    assert result.all_succeeded


def test_apply_create_captures_new_bd_id() -> None:
    adapter = _StubBdAdapter(create_returns_id="harness-new")
    wb = PlanWriteback(to_create=(BeadsIssueDraft(subgoal_id="sg-fresh", title="ship the doc"),))
    result = apply_writeback(adapter, wb)
    assert result.all_succeeded
    assert result.created_ids() == {"sg-fresh": "harness-new"}
    assert len(adapter.create_calls) == 1
    assert adapter.create_calls[0]["title"] == "ship the doc"


def test_apply_create_passes_scope_and_assignee() -> None:
    adapter = _StubBdAdapter()
    wb = PlanWriteback(to_create=(BeadsIssueDraft(subgoal_id="sg-fresh", title="t"),))
    apply_writeback(adapter, wb, scope="personal", assignee="airton_b")
    call = adapter.create_calls[0]
    assert call["scope"] == "personal"
    assert call["assignee"] == "airton_b"


def test_apply_comment_calls_adapter() -> None:
    adapter = _StubBdAdapter()
    wb = PlanWriteback(comments=(("harness-x", "noted"),))
    result = apply_writeback(adapter, wb)
    assert adapter.comment_calls == [("harness-x", "noted")]
    assert result.all_succeeded


def test_apply_close_idempotent_on_already_closed_error() -> None:
    """bd's 'already closed' error is swallowed and the op recorded
    as success — mid-tick crash + retry doesn't see spurious failures."""
    adapter = _StubBdAdapter(raise_on_close=ValueError("issue is already closed"))
    wb = PlanWriteback(to_close=("harness-x",))
    result = apply_writeback(adapter, wb)
    assert result.all_succeeded
    assert result.ops[0].kind == "close"
    assert result.ops[0].success is True


def test_apply_reopen_idempotent_on_already_open_error() -> None:
    adapter = _StubBdAdapter(raise_on_reopen=ValueError("issue is already open"))
    wb = PlanWriteback(to_reopen=("harness-x",))
    result = apply_writeback(adapter, wb)
    assert result.all_succeeded


def test_apply_one_failing_op_does_not_abort_others() -> None:
    """Partial-failure: a close that hits a real (non-idempotent)
    error is captured into result.errors; the rest of the ops still
    run."""

    @dataclass
    class _SelectiveAdapter:
        close_calls: list[tuple[str, str | None]] = field(default_factory=list)
        comment_calls: list[tuple[str, str]] = field(default_factory=list)

        def close(self, issue_id: str, *, reason: str | None = None) -> None:
            self.close_calls.append((issue_id, reason))
            if issue_id == "harness-bad":
                raise RuntimeError("bd subprocess crashed")

        def comment_add(self, issue_id: str, text: str) -> None:
            self.comment_calls.append((issue_id, text))

    adapter = _SelectiveAdapter()
    wb = PlanWriteback(
        to_close=("harness-bad", "harness-good"),
        comments=(("harness-other", "note"),),
    )
    result = apply_writeback(adapter, wb)
    # Both closes attempted; the comment runs after the failed close.
    assert len(adapter.close_calls) == 2
    assert adapter.comment_calls == [("harness-other", "note")]
    assert not result.all_succeeded
    errors = result.errors
    assert len(errors) == 1
    assert errors[0].target == "harness-bad"
    assert "RuntimeError" in (errors[0].error or "")


def test_apply_empty_writeback_returns_empty_result() -> None:
    adapter = _StubBdAdapter()
    result = apply_writeback(adapter, PlanWriteback())
    assert result.ops == ()
    assert result.all_succeeded
    assert adapter.close_calls == []


def test_apply_create_failure_captured_into_op_result() -> None:
    adapter = _StubBdAdapter(raise_on_create=ValueError("scope not allowed"))
    wb = PlanWriteback(to_create=(BeadsIssueDraft(subgoal_id="sg-x", title="t"),))
    result = apply_writeback(adapter, wb)
    assert not result.all_succeeded
    assert result.ops[0].kind == "create"
    assert result.ops[0].success is False
    assert result.ops[0].created_id is None


def test_writeback_op_result_aggregates() -> None:
    """Sanity check the result.all_succeeded / .errors / .created_ids
    helpers used by downstream consumers."""
    ops = (
        WritebackOpResult(kind="close", target="a", success=True),
        WritebackOpResult(kind="close", target="b", success=False, error="x"),
        WritebackOpResult(kind="create", target="sg-1", success=True, created_id="harness-new"),
    )
    from harness.plan.writeback import WritebackResult

    result = WritebackResult(ops=ops)
    assert result.all_succeeded is False
    assert len(result.errors) == 1
    assert result.errors[0].target == "b"
    assert result.created_ids() == {"sg-1": "harness-new"}


def test_plan_writeback_is_empty_helper() -> None:
    assert PlanWriteback().is_empty is True
    assert PlanWriteback(to_close=("x",)).is_empty is False
    assert PlanWriteback(comments=(("x", "y"),)).is_empty is False
