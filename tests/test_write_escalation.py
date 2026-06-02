"""Argument-dependent write authorization (harness-qcukc).

`stream_edit` / `python_stream` are statically read-tier but overwrite
files when called with `in_place=True`. The orchestrator must gate
confirmation on `ToolSpec.effective_tier(arguments)`, never on the
static `tier` alone — otherwise an in-place rewrite slips through the
read-tier path with no approval, including the router prelude, which
auto-executes read-tier calls outright.

Covers:
- `ToolSpec.effective_tier` / `can_write` for escalating and fixed tools.
- The main tool loop: in_place=True is confirm-gated; in_place=False is not.
- The router prelude: in_place=True is never auto-executed; a genuine
  read call still is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from harness.model.adapter import ChatMessage
from harness.orchestrator import run_tool_loop
from harness.router.intent import RouterIntent
from harness.tools import (
    ModelReply,
    PythonStreamTool,
    ReadFileTool,
    StreamEditTool,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    WriteFileTool,
)


@dataclass
class _ScriptedAdapter:
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


@dataclass
class _FixedRouter:
    """Router that always returns the same pre-baked intent."""

    intent: RouterIntent | None

    def classify(self, user_message: str, tool_specs: Sequence[ToolSpec]) -> RouterIntent | None:
        return self.intent


# ---------- unit: effective_tier / can_write ----------


def test_stream_edit_escalates_on_in_place(tmp_path: Path) -> None:
    spec = StreamEditTool(root=tmp_path).spec
    assert spec.tier == "read"
    assert spec.effective_tier({"in_place": True}) == "write"
    assert spec.effective_tier({"in_place": False}) == "read"
    assert spec.effective_tier({}) == "read"
    assert spec.can_write is True


def test_python_stream_escalates_on_in_place(tmp_path: Path) -> None:
    spec = PythonStreamTool(root=tmp_path).spec
    assert spec.tier == "read"
    assert spec.effective_tier({"in_place": True}) == "write"
    assert spec.effective_tier({"in_place": False}) == "read"
    assert spec.can_write is True


def test_fixed_read_tool_never_escalates(tmp_path: Path) -> None:
    spec = ReadFileTool(root=tmp_path).spec
    assert spec.can_write is False
    assert spec.effective_tier({"in_place": True}) == "read"


def test_fixed_write_tool_is_always_write(tmp_path: Path) -> None:
    spec = WriteFileTool(root=tmp_path).spec
    assert spec.tier == "write"
    assert spec.can_write is True
    # Static write tier wins regardless of arguments.
    assert spec.effective_tier({}) == "write"


# ---------- main loop: confirmation gating ----------


def _stream_edit_registry(tmp_path: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(StreamEditTool(root=tmp_path))
    return registry


def _in_place_call() -> ToolCall:
    return ToolCall(
        name="stream_edit",
        arguments={"tool": "sed", "args": ["s/foo/bar/g"], "paths": ["f.txt"], "in_place": True},
    )


def test_in_place_declined_does_not_write(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("foo\n")
    registry = _stream_edit_registry(tmp_path)
    confirmed: list[ToolCall] = []

    def decline(call: ToolCall) -> bool:
        confirmed.append(call)
        return False

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="", tool_calls=(_in_place_call(),)),
            ModelReply(content="skipped"),
        ]
    )
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="rewrite f.txt")],
        registry,
        confirm=decline,
    )
    # in_place=True must hit the write-approval path...
    assert len(confirmed) == 1
    # ...and declining must leave the file untouched.
    assert (tmp_path / "f.txt").read_text() == "foo\n"


def test_in_place_approved_writes(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("foo\n")
    registry = _stream_edit_registry(tmp_path)
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="", tool_calls=(_in_place_call(),)),
            ModelReply(content="done"),
        ]
    )
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="rewrite f.txt")],
        registry,
        confirm=lambda _call: True,
    )
    assert (tmp_path / "f.txt").read_text() == "bar\n"


def test_read_capture_does_not_prompt_confirm(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("foo\n")
    registry = _stream_edit_registry(tmp_path)
    confirmed: list[ToolCall] = []

    def confirm(call: ToolCall) -> bool:
        confirmed.append(call)
        return True

    capture_call = ToolCall(
        name="stream_edit",
        arguments={"tool": "sed", "args": ["s/foo/bar/g"], "paths": ["f.txt"]},
    )
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="", tool_calls=(capture_call,)),
            ModelReply(content="here is the transformed text"),
        ]
    )
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="show f.txt with foo->bar")],
        registry,
        confirm=confirm,
    )
    # Read-tier capture keeps its low-friction path: no confirm prompt...
    assert confirmed == []
    # ...and the file on disk is untouched.
    assert (tmp_path / "f.txt").read_text() == "foo\n"


# ---------- router prelude ----------


def test_router_does_not_auto_execute_in_place_write(tmp_path: Path) -> None:
    """The router auto-executes read-tier calls with no confirmation.
    An in_place=True intent escalates to write-tier and must fall
    through to the main loop instead of being silently executed."""
    (tmp_path / "f.txt").write_text("foo\n")
    registry = _stream_edit_registry(tmp_path)
    router = _FixedRouter(
        intent=RouterIntent(
            tool_name="stream_edit",
            arguments={
                "tool": "sed",
                "args": ["s/foo/bar/g"],
                "paths": ["f.txt"],
                "in_place": True,
            },
        )
    )
    # Main model emits no tool call — so if the file changes, only the
    # router prelude could have done it.
    adapter = _ScriptedAdapter(replies=[ModelReply(content="ok")])
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="rewrite f.txt in place")],
        registry,
        router=router,
        confirm=lambda _call: True,
    )
    assert (tmp_path / "f.txt").read_text() == "foo\n"


def test_router_auto_executes_read_capture(tmp_path: Path) -> None:
    """Genuine read-tier router intents still auto-execute — the
    escalation guard must not break the normal prelude path."""
    (tmp_path / "f.txt").write_text("foo\n")
    registry = _stream_edit_registry(tmp_path)
    router = _FixedRouter(
        intent=RouterIntent(
            tool_name="stream_edit",
            arguments={"tool": "sed", "args": ["s/foo/bar/g"], "paths": ["f.txt"]},
        )
    )
    adapter = _ScriptedAdapter(replies=[ModelReply(content="transformed output above")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="show f.txt with foo->bar")],
        registry,
        router=router,
    )
    # The router executed the capture and the main model wrapped up.
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "bar" in tool_msgs[0].content
    # Capture mode never touches the file.
    assert (tmp_path / "f.txt").read_text() == "foo\n"
