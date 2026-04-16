from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.model.adapter import ChatMessage
from harness.model.mlx import _parse_qwen_tool_calls
from harness.orchestrator import ToolLoopEvent, run_tool_loop
from harness.tools import (
    ModelReply,
    ReadFileTool,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)


@dataclass
class _ScriptedAdapter:
    """Returns pre-queued ModelReply objects in order."""

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


# ---------- run_tool_loop ----------


def test_loop_returns_immediately_when_no_tool_calls() -> None:
    adapter = _ScriptedAdapter(replies=[ModelReply(content="just text")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        ToolRegistry(),
    )
    assert result.content == "just text"
    assert result.rounds == 1


def test_loop_executes_read_tier_without_confirm(tmp_path: Path) -> None:
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
            ),
            ModelReply(content="the file said 'contents'"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read hi.txt")],
        registry,
    )
    assert result.content == "the file said 'contents'"
    assert result.rounds == 2

    # The tool result was fed back as a tool-role message
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "contents"


def test_loop_requires_confirm_for_write_tier(tmp_path: Path) -> None:
    from harness.tools import WriteFileTool

    registry = ToolRegistry()
    registry.register(WriteFileTool(root=tmp_path))

    calls_confirmed: list[ToolCall] = []

    def decline(call: ToolCall) -> bool:
        calls_confirmed.append(call)
        return False

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="write_file",
                        arguments={"path": "new.txt", "content": "data"},
                    ),
                ),
            ),
            ModelReply(content="ok, skipped"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="write something")],
        registry,
        confirm=decline,
    )

    assert len(calls_confirmed) == 1
    # File should NOT exist because we declined
    assert not (tmp_path / "new.txt").exists()
    # The declined result was fed back to the model
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "user declined" in tool_msgs[0].content


def test_loop_executes_write_tier_when_confirmed(tmp_path: Path) -> None:
    from harness.tools import WriteFileTool

    registry = ToolRegistry()
    registry.register(WriteFileTool(root=tmp_path))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="write_file",
                        arguments={"path": "new.txt", "content": "data"},
                    ),
                ),
            ),
            ModelReply(content="done"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="write something")],
        registry,
        confirm=lambda _call: True,
    )

    assert (tmp_path / "new.txt").read_text() == "data"
    assert result.content == "done"


def test_loop_observer_receives_events(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hi")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "f.txt"}),),
            ),
            ModelReply(content="final"),
        ]
    )

    observed: list[ToolLoopEvent] = []
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read f")],
        registry,
        observe=lambda e: observed.append(e),
    )

    kinds = [e.kind for e in observed]
    assert "tool_call_start" in kinds
    assert "tool_call_end" in kinds
    assert "round_complete" in kinds
    assert len(observed) == len(result.events)


def test_loop_respects_max_rounds() -> None:
    # Model always wants to call read_file, never finishes
    from harness.tools import ReadFileTool

    tmp_registry = ToolRegistry()
    # Use a bogus path so the read fails; the model will just keep retrying
    # in this scripted setup.
    reply = ModelReply(
        content="",
        tool_calls=(ToolCall(name="read_file", arguments={"path": "ghost"}),),
    )
    adapter = _ScriptedAdapter(replies=[reply, reply, reply, reply, reply])

    # Register with a tmp root so the tool itself works (ghost file doesn't exist,
    # but the tool-registry call catches that as a failed result).
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp_registry.register(ReadFileTool(root=Path(td)))
        result = run_tool_loop(
            adapter,
            [ChatMessage(role="user", content="spin")],
            tmp_registry,
            max_rounds=3,
        )
        assert result.rounds == 3


def test_loop_handles_unknown_tool_gracefully() -> None:
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="nonexistent", arguments={}),),
            ),
            ModelReply(content="ok"),
        ]
    )
    registry = ToolRegistry()
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="go")],
        registry,
    )
    # Unknown-tool result is fed back to the model; loop continues
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "unknown tool" in tool_msgs[0].content


# ---------- Qwen tool-call parser ----------


def test_parse_qwen_tool_calls_single() -> None:
    raw = (
        "I will read that file.\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "foo.txt"}}</tool_call>'
    )
    content, calls = _parse_qwen_tool_calls(raw)
    assert content == "I will read that file."
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "foo.txt"}


def test_parse_qwen_tool_calls_multiple() -> None:
    raw = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>\n'
        '<tool_call>{"name": "read_file", "arguments": {"path": "b"}}</tool_call>'
    )
    content, calls = _parse_qwen_tool_calls(raw)
    assert content == ""
    assert len(calls) == 2


def test_parse_qwen_tool_calls_no_calls() -> None:
    content, calls = _parse_qwen_tool_calls("Just plain text.")
    assert content == "Just plain text."
    assert calls == []


def test_parse_qwen_tool_calls_drops_malformed() -> None:
    raw = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "ok"}}</tool_call>\n'
        "<tool_call>{broken json}</tool_call>"
    )
    _content, calls = _parse_qwen_tool_calls(raw)
    # Only the valid one survives
    assert len(calls) == 1
    assert calls[0].name == "read_file"


def test_parse_qwen_tool_calls_handles_stringified_arguments() -> None:
    raw = '<tool_call>{"name": "shell", "arguments": "{\\"cmd\\": \\"ls\\"}"}</tool_call>'
    _content, calls = _parse_qwen_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].arguments == {"cmd": "ls"}


def test_parse_drops_nonstring_names() -> None:
    raw = '<tool_call>{"name": 42, "arguments": {}}</tool_call>'
    _content, calls = _parse_qwen_tool_calls(raw)
    assert calls == []


# Explicit import to confirm we can pass pytest from the tests folder
def test_tools_module_importable() -> None:
    import harness.tools  # noqa: F401 — import-for-side-effect check

    assert True
