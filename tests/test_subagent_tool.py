"""Tests for SpawnSubagentTool (harness-qxr, sota §1.A).

Scripted adapters drive the child loop so the tests don't need a real
model. Key invariants exercised:

- Returns the child's final reply as a string.
- Filters the child registry to the requested (read-tier) tools only.
- Rejects write-tier tools, unknown tools, self-recursion, empty lists.
- Max depth 1 — nested spawn from inside a running subagent errors.
- Shared hooks pipeline — fabrication catchers fire inside the child.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.model.adapter import ChatMessage
from harness.orchestrator.hooks import default_hook_pipeline
from harness.tools.base import ModelReply, ToolCall, ToolRegistry, ToolSpec
from harness.tools.list_dir import ListDirTool
from harness.tools.read_file import ReadFileTool
from harness.tools.subagent import (
    _SUBAGENT_DEPTH,
    DEFAULT_SUBAGENT_SYSTEM_PROMPT,
    SpawnSubagentTool,
    current_subagent_depth,
)
from harness.tools.write_file import WriteFileTool


@dataclass
class _ScriptedAdapter:
    """Returns pre-queued ModelReply objects in order. The child loop
    uses the same adapter as the parent, so tests queue up the replies
    the subagent should see."""

    replies: list[ModelReply]
    calls_seen: list[list[ChatMessage]] = field(default_factory=list)

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        self.calls_seen.append(list(messages))
        return self.replies.pop(0) if self.replies else ModelReply(content="done")


def _registry_with(*tools_: object) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools_:
        reg.register(t)  # type: ignore[arg-type]
    return reg


def _make_tool(workspace: Path) -> SpawnSubagentTool:
    """Parent registry exposes read_file + list_dir + write_file so tests
    can assert filtering. Adapter is overridden per test."""
    parent = _registry_with(
        ReadFileTool(root=workspace),
        ListDirTool(root=workspace),
        WriteFileTool(root=workspace),
    )
    return SpawnSubagentTool(
        adapter=_ScriptedAdapter(replies=[]),
        registry=parent,
        hooks=default_hook_pipeline(),
    )


# ---------- spec surface ----------


def test_spec_is_read_tier(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    assert tool.spec.tier == "read"
    assert tool.spec.name == "spawn_subagent"


def test_spec_requires_task_and_tools(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    required = tool.spec.parameters.get("required", [])
    assert "task" in required
    assert "tools" in required


# ---------- happy path ----------


def test_subagent_returns_final_reply(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\n")
    parent = _registry_with(ReadFileTool(root=tmp_path), ListDirTool(root=tmp_path))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="alpha\n — from notes.txt")])
    tool = SpawnSubagentTool(
        adapter=adapter,
        registry=parent,
        hooks=default_hook_pipeline(),
    )
    out = tool.call(task="what's in notes.txt?", tools=["read_file"])
    assert out == "alpha\n — from notes.txt"


def test_subagent_child_registry_filters_to_requested_tools(tmp_path: Path) -> None:
    """When the subagent gets a reply with a tool call, the tool call
    must be resolvable only against the requested subset — the child
    has no knowledge of the parent's other tools."""
    (tmp_path / "a.txt").write_text("A\n")
    parent = _registry_with(ReadFileTool(root=tmp_path), ListDirTool(root=tmp_path))
    # Scripted child: round 0 emits a read_file call; round 1 wraps up.
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "a.txt"}),),
            ),
            ModelReply(content="contents: A"),
        ]
    )
    tool = SpawnSubagentTool(
        adapter=adapter,
        registry=parent,
        hooks=default_hook_pipeline(),
    )
    out = tool.call(task="read a.txt", tools=["read_file"])
    assert out == "contents: A"
    # The child only ever saw its own preamble — no parent history.
    # Preamble shape is [synthesis-continue nudge (harness-b7yd),
    # subagent default system, user task].
    first_call = adapter.calls_seen[0]
    assert first_call[0].role == "system"
    assert "Tool-use rule" in (first_call[0].content or "")
    assert first_call[1].role == "system"
    assert first_call[2].role == "user"
    assert first_call[2].content == "read a.txt"


# ---------- validation: rejections ----------


def test_subagent_rejects_write_tier_tool(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    out = tool.call(task="overwrite config", tools=["write_file"])
    assert "write-tier" in out
    assert "write_file" in out


def test_subagent_rejects_unknown_tool(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    out = tool.call(task="do something", tools=["does_not_exist"])
    assert "unknown tools" in out
    assert "does_not_exist" in out


def test_subagent_rejects_self_recursion_in_tool_list(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    out = tool.call(task="recurse", tools=["spawn_subagent", "read_file"])
    assert "subagents cannot spawn further subagents" in out


def test_subagent_rejects_empty_tool_list(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    out = tool.call(task="anything", tools=[])
    assert "tools list is empty" in out


def test_subagent_rejects_zero_max_rounds(tmp_path: Path) -> None:
    tool = _make_tool(tmp_path)
    out = tool.call(task="anything", tools=["read_file"], max_rounds=0)
    assert "max_rounds must be >= 1" in out


# ---------- depth guard ----------


def test_subagent_depth_counter_reset_after_call(tmp_path: Path) -> None:
    parent = _registry_with(ReadFileTool(root=tmp_path))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="ok")])
    tool = SpawnSubagentTool(adapter=adapter, registry=parent, hooks=default_hook_pipeline())
    tool.call(task="noop", tools=["read_file"])
    assert current_subagent_depth() == 0


def test_subagent_max_depth_exceeded_returns_error(tmp_path: Path) -> None:
    """Simulate being already inside a subagent by pre-setting the
    ContextVar, then call spawn. It must refuse."""
    tool = _make_tool(tmp_path)
    token = _SUBAGENT_DEPTH.set(1)
    try:
        out = tool.call(task="nested", tools=["read_file"])
    finally:
        _SUBAGENT_DEPTH.reset(token)
    assert "max depth 1 exceeded" in out


# ---------- budget exhaustion ----------


def test_subagent_budget_exhaustion_reports_cleanly(tmp_path: Path) -> None:
    """Child that keeps re-calling tools without a final reply until
    max_rounds runs out returns the exhaustion marker, not empty."""
    (tmp_path / "a.txt").write_text("A\n")
    parent = _registry_with(ReadFileTool(root=tmp_path))
    # Every round emits a tool call — the loop never exits via a
    # final text reply — so we hit max_rounds=2 and exit via the
    # "loop exhausted" branch with empty content.
    looping_reply = ModelReply(
        content="",
        tool_calls=(ToolCall(name="read_file", arguments={"path": "a.txt"}),),
    )
    adapter = _ScriptedAdapter(replies=[looping_reply, looping_reply])
    tool = SpawnSubagentTool(adapter=adapter, registry=parent, hooks=default_hook_pipeline())
    out = tool.call(task="read endlessly", tools=["read_file"], max_rounds=2)
    assert "budget exhausted" in out or "loop returned empty content" in out


# ---------- hooks share across boundary ----------


def test_subagent_hooks_catch_fabrication_in_child(tmp_path: Path) -> None:
    """Fabrication-shaped first reply without tool_calls should trip
    the bail pipeline inside the child loop. The hooks pipeline is
    shared with the parent, so the canned fallback fires after retries
    exhaust instead of leaking the fabricated reply to the parent.

    Note: post-harness-rlza, bail retries are bounded by
    `_BAIL_RETRIES_PER_TURN` (3), not `max_rounds`. Adapter needs at
    least 1 + _BAIL_RETRIES_PER_TURN fab replies so iter 3 still
    returns fab (rather than the adapter's empty-queue default) and
    fabrication_fallback substitutes the canned refusal."""
    parent = _registry_with(ReadFileTool(root=tmp_path))
    # Round 0: fabricated meta-confirm (trips the meta_confirm bail hook).
    # Retries keep producing the same shape, so the fabrication_fallback
    # finalize hook substitutes the canned refusal.
    fab_reply = ModelReply(content="Would you like me to proceed?")
    adapter = _ScriptedAdapter(replies=[fab_reply] * 5)
    tool = SpawnSubagentTool(adapter=adapter, registry=parent, hooks=default_hook_pipeline())
    out = tool.call(task="do something", tools=["read_file"], max_rounds=3)
    assert "I couldn't answer that without calling a tool" in out


# ---------- system prompt override ----------


def test_subagent_uses_default_system_prompt(tmp_path: Path) -> None:
    parent = _registry_with(ReadFileTool(root=tmp_path))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="ok")])
    tool = SpawnSubagentTool(adapter=adapter, registry=parent, hooks=default_hook_pipeline())
    tool.call(task="noop", tools=["read_file"])
    # Subagent prompt sits at index 1 — index 0 is the orchestrator's
    # synthesis-continue nudge (harness-b7yd).
    assert adapter.calls_seen[0][1].content == DEFAULT_SUBAGENT_SYSTEM_PROMPT


def test_subagent_honors_custom_system_prompt(tmp_path: Path) -> None:
    parent = _registry_with(ReadFileTool(root=tmp_path))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="ok")])
    tool = SpawnSubagentTool(adapter=adapter, registry=parent, hooks=default_hook_pipeline())
    tool.call(
        task="noop",
        tools=["read_file"],
        system_prompt="custom brief for this task",
    )
    # Same shape — custom prompt at index 1 behind the synthesis nudge.
    assert adapter.calls_seen[0][1].content == "custom brief for this task"


# ---------- profile membership ----------


def test_subagent_in_research_coding_diagnostic_profiles() -> None:
    from harness.tools.profiles import TOOL_PROFILES

    assert "spawn_subagent" in TOOL_PROFILES["research"]
    assert "spawn_subagent" in TOOL_PROFILES["coding"]
    assert "spawn_subagent" in TOOL_PROFILES["diagnostic"]


def test_subagent_not_in_core_or_minimal_profiles() -> None:
    from harness.tools.profiles import TOOL_PROFILES

    assert "spawn_subagent" not in TOOL_PROFILES["minimal"]
    assert "spawn_subagent" not in TOOL_PROFILES["core"]
    assert "spawn_subagent" not in TOOL_PROFILES["memory"]
