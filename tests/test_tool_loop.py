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
    StreamChunk,
    StreamComplete,
    StreamText,
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


@dataclass
class _StreamingScriptedAdapter:
    """Scripted adapter exposing stream_with_tools. Each scripted entry
    is either a list of raw text deltas (to be yielded as StreamText)
    paired with a final ModelReply, or a ready-to-yield tuple. The
    adapter also keeps complete_with_tools for fallback-path tests."""

    replies: list[tuple[list[str], ModelReply]]

    def stream_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> Iterable[StreamChunk]:
        if not self.replies:
            yield StreamComplete(reply=ModelReply(content="done"))
            return
        deltas, reply = self.replies.pop(0)
        for delta in deltas:
            yield StreamText(text=delta)
        yield StreamComplete(reply=reply)

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        for chunk in self.stream_with_tools(
            messages, tools=tools, max_tokens=max_tokens, temperature=temperature
        ):
            if isinstance(chunk, StreamComplete):
                return chunk.reply
        raise RuntimeError("no StreamComplete")


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
    # round_start fires once per inference round (here: 2 — tool round + final)
    assert kinds.count("round_start") == 2
    assert kinds.index("round_start") < kinds.index("tool_call_start")
    # model_call_start / model_call_end bracket every adapter invocation; the
    # CLI spinner relies on this to tick while the model is running and stop
    # before any confirmation prompt or tool-call print.
    assert kinds.count("model_call_start") == 2
    assert kinds.count("model_call_end") == 2
    first_mcs = kinds.index("model_call_start")
    first_mce = kinds.index("model_call_end")
    assert first_mcs < first_mce < kinds.index("tool_call_start")
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


def test_loop_emits_token_delta_events_when_streaming(tmp_path: Path) -> None:
    """A streaming-capable adapter drives the loop through
    stream_with_tools, and each visible delta surfaces as a token_delta
    event the CLI can render live."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    adapter = _StreamingScriptedAdapter(
        replies=[
            (
                ["I'll read ", "the file."],
                ModelReply(
                    content="I'll read the file.",
                    tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
                ),
            ),
            (["the file says ", "contents."], ModelReply(content="the file says contents.")),
        ]
    )

    observed: list[ToolLoopEvent] = []
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read hi.txt")],
        registry,
        observe=lambda e: observed.append(e),
    )
    assert result.content == "the file says contents."
    deltas = [e.delta for e in observed if e.kind == "token_delta"]
    assert deltas == ["I'll read ", "the file.", "the file says ", "contents."]
    # Round 0 deltas come before its tool_call_start; final round deltas come
    # before round_complete.
    kinds = [e.kind for e in observed]
    assert kinds.index("token_delta") < kinds.index("tool_call_start")


def test_loop_falls_back_to_complete_without_stream(tmp_path: Path) -> None:
    """Adapters without stream_with_tools still work via the blocking
    complete_with_tools path — no token_delta events, but the loop
    completes normally."""
    (tmp_path / "hi.txt").write_text("x")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
            ),
            ModelReply(content="done"),
        ]
    )
    observed: list[ToolLoopEvent] = []
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="go")],
        registry,
        observe=lambda e: observed.append(e),
    )
    assert result.content == "done"
    assert not any(e.kind == "token_delta" for e in observed)


def test_loop_recovers_from_truncated_reply(tmp_path: Path) -> None:
    """When the adapter signals truncation, the loop should retry with
    a bigger token budget instead of terminating with the partial reply."""
    (tmp_path / "f.txt").write_text("hi")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    adapter = _ScriptedAdapter(
        replies=[
            # First call comes back truncated with no tool calls — would
            # otherwise be treated as the final reply.
            ModelReply(content="Let me read", was_truncated=True),
            # Second call (after loop bumps max_tokens) emits the real call.
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "f.txt"}),),
            ),
            ModelReply(content="file says hi"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read f")],
        registry,
    )
    assert result.content == "file says hi"
    # Loop must have actually called the tool, not bailed out
    assert any(m.role == "tool" for m in result.messages)


def test_loop_recovers_from_unparseable_tool_call() -> None:
    """When the adapter signals a malformed tool-call block, inject a
    repair-instruction nudge and retry."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="here you go", had_unparseable_call=True),
            ModelReply(content="ok done"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        ToolRegistry(),
    )
    assert result.content == "ok done"
    # The loop must have appended a user-role nudge mentioning the malformed block
    nudges = [m for m in result.messages if m.role == "user" and "malformed" in m.content]
    assert len(nudges) == 1


def test_loop_recovers_from_teaser_bail() -> None:
    """When the model emits a 'let me check…' teaser without tool calls,
    nudge it to either call or finish."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="Now let me check the tests:"),
            ModelReply(content="all good"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="audit")],
        ToolRegistry(),
    )
    assert result.content == "all good"
    nudges = [m for m in result.messages if m.role == "user" and "announced more work" in m.content]
    assert len(nudges) == 1


def test_loop_does_not_nudge_genuine_final_reply() -> None:
    """A normal terminating reply must not trigger any recovery."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="The answer is 42.")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="ask")],
        ToolRegistry(),
    )
    assert result.content == "The answer is 42."
    assert result.rounds == 1
    # No nudges injected
    assert all(m.role != "user" or m.content == "ask" for m in result.messages)


def test_loop_caps_bail_retries() -> None:
    """If the model keeps bailing, give up after the per-turn cap (2)
    rather than consuming the full max_rounds budget."""
    teaser = ModelReply(content="Let me check:")
    adapter = _ScriptedAdapter(replies=[teaser, teaser, teaser, teaser, teaser])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="go")],
        ToolRegistry(),
        max_rounds=8,
    )
    # 1 initial + 2 retries = 3 rounds, then terminate with last teaser
    assert result.rounds == 3
    assert result.content == "Let me check:"


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
    # Failure is signalled with tool_call_failed, not tool_call_end
    kinds = [e.kind for e in result.events]
    assert "tool_call_failed" in kinds
    assert "tool_call_end" not in kinds


# ---------- Qwen tool-call parser ----------


def test_tag_masker_passes_plain_text() -> None:
    from harness.model.mlx import _TagMasker

    m = _TagMasker()
    out = m.feed("hello world, this is some text")
    # With the rolling 15-char tail, the last 15 chars stay buffered; flush
    # them at stream end.
    assert out + m.flush() == "hello world, this is some text"


def test_tag_masker_hides_json_tool_call() -> None:
    from harness.model.mlx import _TagMasker

    m = _TagMasker()
    raw = 'before <tool_call>{"name": "read", "arguments": {}}</tool_call> after'
    out = "".join(m.feed(c) for c in raw) + m.flush()
    assert "<tool_call>" not in out
    assert '{"name"' not in out
    assert out.strip() == "before  after"


def test_tag_masker_hides_xml_function_call() -> None:
    from harness.model.mlx import _TagMasker

    m = _TagMasker()
    raw = "intro <function=read_file><parameter=path>x</parameter></function> outro"
    out = "".join(m.feed(c) for c in raw) + m.flush()
    assert "<function=" not in out
    assert "<parameter=" not in out
    assert out.strip() == "intro  outro"


def test_tag_masker_hides_tag_split_across_chunks() -> None:
    """Delta boundaries must never leak the start of a tag. Here the
    opening `<tool_call>` straddles two deltas."""
    from harness.model.mlx import _TagMasker

    m = _TagMasker()
    chunks = ["hello ", "<to", "ol_call>", '{"name":"x"}', "</tool_call>", " world"]
    out = "".join(m.feed(c) for c in chunks) + m.flush()
    assert "<tool_call>" not in out
    assert "ol_call" not in out
    assert out == "hello  world"


def test_tag_masker_drops_buffer_on_unclosed_tag() -> None:
    """If the stream ends inside a tool-call span (truncation) the
    buffered hidden text is dropped — we can't safely show half a JSON
    object."""
    from harness.model.mlx import _TagMasker

    m = _TagMasker()
    raw = 'visible <tool_call>{"name": "read"'
    out = "".join(m.feed(c) for c in raw) + m.flush()
    assert out.strip() == "visible"


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


def test_parse_qwen3_coder_xml_format() -> None:
    """Qwen3-Coder emits XML function-call syntax instead of JSON."""
    raw = (
        "I will read the file.\n"
        "<tool_call>\n"
        "<function=read_file>\n"
        "<parameter=path>\npyproject.toml\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    content, calls = _parse_qwen_tool_calls(raw)
    assert content == "I will read the file."
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "pyproject.toml"}


def test_parse_qwen3_coder_multi_parameter() -> None:
    raw = (
        "<tool_call>\n"
        "<function=shell>\n"
        "<parameter=cmd>\nls -F\n</parameter>\n"
        "<parameter=timeout>\n30\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    _content, calls = _parse_qwen_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "shell"
    assert calls[0].arguments == {"cmd": "ls -F", "timeout": "30"}


def test_parse_qwen3_coder_multiple_calls() -> None:
    block = (
        "<tool_call>\n<function=read_file>\n<parameter=path>\n{p}\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    raw = f"{block.format(p='a')}\n{block.format(p='b')}"
    content, calls = _parse_qwen_tool_calls(raw)
    assert content == ""
    assert [c.arguments["path"] for c in calls] == ["a", "b"]


# Explicit import to confirm we can pass pytest from the tests folder
def test_tools_module_importable() -> None:
    import harness.tools  # noqa: F401 — import-for-side-effect check

    assert True
