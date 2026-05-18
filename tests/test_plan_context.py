"""Tests for the Plan → context-block renderer + orchestrator hook —
harness-uzan."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from harness.model.adapter import ChatMessage
from harness.orchestrator.tool_loop import run_tool_loop
from harness.plan import (
    Plan,
    Precondition,
    new_plan,
    new_subgoal,
    render_plan_block,
)
from harness.plan.context import DEFAULT_CHAR_BUDGET
from harness.tools.base import ModelReply, ToolRegistry, ToolSpec


@dataclass
class _ScriptedToolCapableAdapter:
    """Minimal adapter satisfying _ToolCapableAdapter for orchestrator
    tests. Returns ModelReply with empty tool_calls so the loop exits
    after one round."""

    id: str = "scripted"
    context_window: int = 8192
    seen_messages: list[list[ChatMessage]] = field(default_factory=list)
    reply_text: str = "ok"

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        self.seen_messages.append(list(messages))
        return ModelReply(content=self.reply_text, tool_calls=())


# --- render_plan_block ----------------------------------------------------


def test_renders_title_and_active_subgoal() -> None:
    p = new_plan("ship harness 1.0", plan_id="plan-1")
    p = p.with_subgoal(new_subgoal("write the doc", parent_id=p.root_subgoal_id, status="active"))
    block = render_plan_block(p)
    assert "# Active plan: ship harness 1.0" in block
    assert "- write the doc" in block


def test_renders_preconditions_under_active_subgoal() -> None:
    p = new_plan("p", plan_id="plan-1")
    p = p.with_subgoal(
        new_subgoal(
            "needs y closed",
            parent_id=p.root_subgoal_id,
            status="active",
            preconditions=(
                Precondition(kind="bd_closed", payload={"bead": "harness-y"}),
                Precondition(kind="bd_open", payload={"bead": "harness-z"}),
                Precondition(kind="timestamp_past", payload={"when": "2026-06-01T00:00:00+00:00"}),
            ),
        )
    )
    block = render_plan_block(p)
    assert "waits on: harness-y (closed)" in block
    assert "waits on: harness-z (open)" in block
    assert "waits until: 2026-06-01T00:00:00+00:00" in block


def test_omits_root_subgoal_from_listing() -> None:
    """The synthesized root is implementation detail; the model
    shouldn't see it as a goal."""
    p = new_plan("ship", plan_id="plan-1")
    block = render_plan_block(p)
    # Only the title + 'no active or pending' marker should appear.
    root = p.subgoal(p.root_subgoal_id)
    assert root.title not in block.split("# Active plan:", 1)[1].split("\n", 1)[1]


def test_lists_pending_subgoals_in_upcoming_section() -> None:
    p = new_plan("p", plan_id="plan-1")
    p = p.with_subgoal(new_subgoal("draft", parent_id=p.root_subgoal_id, status="active"))
    p = p.with_subgoal(new_subgoal("review", parent_id=p.root_subgoal_id, status="pending"))
    p = p.with_subgoal(new_subgoal("publish", parent_id=p.root_subgoal_id, status="pending"))
    block = render_plan_block(p)
    assert "Upcoming (2 pending):" in block
    assert "- review" in block
    assert "- publish" in block


def test_omits_achieved_and_abandoned_subgoals() -> None:
    """Done work doesn't enter the model's current-turn block."""
    p = new_plan("p", plan_id="plan-1")
    p = p.with_subgoal(new_subgoal("done", parent_id=p.root_subgoal_id, status="achieved"))
    p = p.with_subgoal(new_subgoal("dropped", parent_id=p.root_subgoal_id, status="abandoned"))
    block = render_plan_block(p)
    assert "done" not in block
    assert "dropped" not in block
    assert "no active or pending subgoals" in block


def test_max_subgoals_caps_active_list_with_more_marker() -> None:
    p = new_plan("p", plan_id="plan-1")
    for i in range(12):
        p = p.with_subgoal(new_subgoal(f"task-{i}", parent_id=p.root_subgoal_id, status="active"))
    block = render_plan_block(p, max_subgoals=3)
    # First three appear; the rest collapse into '+N more active'.
    assert "- task-0" in block
    assert "- task-2" in block
    assert "(+9 more active)" in block
    # task-3..task-11 not individually listed.
    assert "- task-5" not in block


def test_max_subgoals_caps_pending_list_with_more_marker() -> None:
    p = new_plan("p", plan_id="plan-1")
    for i in range(5):
        p = p.with_subgoal(
            new_subgoal(f"pending-{i}", parent_id=p.root_subgoal_id, status="pending")
        )
    block = render_plan_block(p, max_subgoals=2)
    assert "Upcoming (5 pending):" in block
    assert "- pending-0" in block
    assert "- pending-1" in block
    assert "(+3 more pending)" in block
    assert "- pending-3" not in block


def test_char_budget_enforced_with_ellipsis() -> None:
    """Even a sprawling plan stays under the budget — the renderer is
    the last guard against blowing the system-prompt token cap."""
    p = new_plan("p" * 50, plan_id="plan-1")
    for i in range(30):
        p = p.with_subgoal(
            new_subgoal(f"long subgoal name {i} " * 5, parent_id=p.root_subgoal_id, status="active")
        )
    block = render_plan_block(p, char_budget=300)
    assert len(block) <= 300
    assert block.endswith("...")


def test_under_budget_block_unchanged() -> None:
    """Small plan: the block stops exactly where it stops — no
    spurious ellipsis."""
    p = new_plan("short", plan_id="plan-1")
    p = p.with_subgoal(new_subgoal("one", parent_id=p.root_subgoal_id, status="active"))
    block = render_plan_block(p)
    assert not block.endswith("...")


def test_default_char_budget_is_documented_constant() -> None:
    """If a future PR tightens the budget this test will catch it,
    so callers downstream know the contract changed."""
    assert DEFAULT_CHAR_BUDGET == 800


def test_unknown_precondition_kind_is_omitted_silently() -> None:
    """A future precondition kind the renderer doesn't know yet
    shouldn't dump a raw 'kind=xyz' line into the system prompt."""
    p = new_plan("p", plan_id="plan-1")
    p = p.with_subgoal(
        new_subgoal(
            "needs future thing",
            parent_id=p.root_subgoal_id,
            status="active",
            preconditions=(Precondition(kind="future_kind", payload={"x": 1}),),
        )
    )
    block = render_plan_block(p)
    assert "future_kind" not in block
    # The subgoal itself still shows up.
    assert "- needs future thing" in block


# --- orchestrator integration -------------------------------------------


def _empty_plan() -> Plan:
    return new_plan("test", plan_id="plan-test")


def test_run_tool_loop_without_plan_does_not_inject_block() -> None:
    """Back-compat: existing callers that don't pass `plan=` see the
    same message stream they always have."""
    adapter = _ScriptedToolCapableAdapter()
    registry = ToolRegistry()
    user_msg = ChatMessage(role="user", content="hi")
    run_tool_loop(
        adapter,
        [user_msg],
        registry,
        max_rounds=1,
    )
    seen = adapter.seen_messages[0]
    assert all("Active plan" not in (m.content or "") for m in seen)


def test_run_tool_loop_with_plan_prepends_system_block() -> None:
    """When plan is passed, a system message carrying the rendered
    block appears at the start of the message stream."""
    adapter = _ScriptedToolCapableAdapter()
    registry = ToolRegistry()
    p = _empty_plan()
    p = p.with_subgoal(new_subgoal("write the doc", parent_id=p.root_subgoal_id, status="active"))
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        registry,
        max_rounds=1,
        plan=p,
    )
    seen = adapter.seen_messages[0]
    # First message should be the plan block.
    assert seen[0].role == "system"
    assert "# Active plan: test" in (seen[0].content or "")
    assert "- write the doc" in (seen[0].content or "")
    # User message follows.
    assert seen[1].role == "user"
    assert seen[1].content == "hi"


def test_run_tool_loop_empty_plan_block_is_still_injected() -> None:
    """A plan with only a root subgoal renders the title + 'no active'
    marker. We still inject it so the model knows a plan is anchored
    here at all (rather than implying 'no plan at all')."""
    adapter = _ScriptedToolCapableAdapter()
    registry = ToolRegistry()
    p = _empty_plan()
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        registry,
        max_rounds=1,
        plan=p,
    )
    seen = adapter.seen_messages[0]
    assert seen[0].role == "system"
    assert "Active plan: test" in (seen[0].content or "")
    assert "no active or pending subgoals" in (seen[0].content or "")


def test_run_tool_loop_with_plan_preserves_pre_existing_system_prompt() -> None:
    """Caller's own system prompt isn't dropped — the plan block
    prepends as a separate system message, leaving the original
    intact."""
    adapter = _ScriptedToolCapableAdapter()
    registry = ToolRegistry()
    p = _empty_plan()
    pre = ChatMessage(role="system", content="You are Airton.")
    run_tool_loop(
        adapter,
        [pre, ChatMessage(role="user", content="hi")],
        registry,
        max_rounds=1,
        plan=p,
    )
    seen = adapter.seen_messages[0]
    # Plan block, then the character system prompt, then user.
    assert seen[0].role == "system"
    assert "Active plan" in (seen[0].content or "")
    assert seen[1].role == "system"
    assert "You are Airton" in (seen[1].content or "")
    assert seen[2].role == "user"


def test_run_tool_loop_with_plan_active_subgoals_visible() -> None:
    """A populated plan: the model sees the active subgoals + the
    preconditions in the same shape `render_plan_block` produces."""
    adapter = _ScriptedToolCapableAdapter()
    registry = ToolRegistry()
    p = new_plan("ship harness 1.0", plan_id="plan-1")
    p = p.with_subgoal(
        new_subgoal(
            "ship the docs",
            parent_id=p.root_subgoal_id,
            status="active",
            preconditions=(Precondition(kind="bd_closed", payload={"bead": "harness-q"}),),
        )
    )
    p = p.with_subgoal(
        replace(
            new_subgoal("draft the spec", parent_id=p.root_subgoal_id, status="pending"),
            preconditions=(),
        )
    )
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        registry,
        max_rounds=1,
        plan=p,
    )
    seen = adapter.seen_messages[0]
    block_text = seen[0].content or ""
    assert "- ship the docs" in block_text
    assert "waits on: harness-q (closed)" in block_text
    assert "Upcoming (1 pending):" in block_text
    assert "- draft the spec" in block_text
