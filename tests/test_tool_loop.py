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


def test_loop_observer_receives_events_for_every_round(tmp_path: Path) -> None:
    """harness-cx2: when the model chains multiple tool calls across
    rounds, each invocation must reach the observer. Earlier bug report
    (TUI only showed the first 🔧 line per turn) implied a 'first call
    only' guard; this locks in that the orchestrator emits one
    start+end pair per tool call, regardless of round."""
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "a.txt"}),),
            ),
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "b.txt"}),),
            ),
            ModelReply(content="done"),
        ]
    )

    observed: list[ToolLoopEvent] = []
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read both")],
        registry,
        observe=lambda e: observed.append(e),
    )

    starts = [e for e in observed if e.kind == "tool_call_start"]
    ends = [e for e in observed if e.kind == "tool_call_end"]
    assert len(starts) == 2, [e.kind for e in observed]
    assert len(ends) == 2
    # Distinct arguments visible on each event (not just the first call
    # echoed twice).
    start_paths = [e.call.arguments["path"] for e in starts if e.call is not None]
    assert start_paths == ["a.txt", "b.txt"]
    # round_index advances between the two calls.
    assert [e.round_index for e in starts] == [0, 1]


def test_loop_dedupes_identical_call_across_rounds(tmp_path: Path) -> None:
    """harness-pun: when the model re-emits an identical call across
    two rounds, the second invocation must be short-circuited — no
    second execution, a tool_call_deduped event instead of start+end,
    and a nudge tool-role message feeding 'finalize, don't re-call'
    back to the model."""
    (tmp_path / "a.txt").write_text("A")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    call_count = 0

    @dataclass
    class _CountingReadFile:
        inner: ReadFileTool

        @property
        def spec(self) -> ToolSpec:
            return self.inner.spec

        def call(self, *, path: str) -> str:
            nonlocal call_count
            call_count += 1
            return self.inner.call(path=path)

    registry = ToolRegistry()
    registry.register(_CountingReadFile(inner=ReadFileTool(root=tmp_path)))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "a.txt"}),),
            ),
            ModelReply(
                content="here is a summary",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "a.txt"}),),
            ),
            ModelReply(content="final answer"),
        ]
    )

    observed: list[ToolLoopEvent] = []
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read a")],
        registry,
        observe=lambda e: observed.append(e),
    )

    assert call_count == 1, "second identical call should not have executed"
    kinds = [e.kind for e in observed]
    assert kinds.count("tool_call_start") == 1
    assert kinds.count("tool_call_end") == 1
    assert kinds.count("tool_call_deduped") == 1
    # The nudge message is appended as a tool-role turn so the next
    # round's model context includes 'stop, finalize'.
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert "duplicate call" in tool_msgs[1].content
    assert result.content == "final answer"


def test_loop_dedupes_identical_call_in_same_round(tmp_path: Path) -> None:
    """Two identical tool_calls in a single ModelReply: first executes,
    second is deduped. Guards against the pathological case where a
    model emits a list with repeated entries."""
    (tmp_path / "a.txt").write_text("A")

    call_count = 0

    @dataclass
    class _CountingReadFile:
        inner: ReadFileTool

        @property
        def spec(self) -> ToolSpec:
            return self.inner.spec

        def call(self, *, path: str) -> str:
            nonlocal call_count
            call_count += 1
            return self.inner.call(path=path)

    registry = ToolRegistry()
    registry.register(_CountingReadFile(inner=ReadFileTool(root=tmp_path)))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(name="read_file", arguments={"path": "a.txt"}),
                    ToolCall(name="read_file", arguments={"path": "a.txt"}),
                ),
            ),
            ModelReply(content="done"),
        ]
    )

    observed: list[ToolLoopEvent] = []
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read a twice")],
        registry,
        observe=lambda e: observed.append(e),
    )

    assert call_count == 1
    kinds = [e.kind for e in observed]
    assert kinds.count("tool_call_start") == 1
    assert kinds.count("tool_call_deduped") == 1


def test_loop_does_not_dedupe_different_args(tmp_path: Path) -> None:
    """Same tool with different arguments is not a duplicate — both
    must execute. Guards against an over-aggressive dedup that breaks
    legitimate multi-file reads."""
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")

    call_count = 0

    @dataclass
    class _CountingReadFile:
        inner: ReadFileTool

        @property
        def spec(self) -> ToolSpec:
            return self.inner.spec

        def call(self, *, path: str) -> str:
            nonlocal call_count
            call_count += 1
            return self.inner.call(path=path)

    registry = ToolRegistry()
    registry.register(_CountingReadFile(inner=ReadFileTool(root=tmp_path)))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(name="read_file", arguments={"path": "a.txt"}),
                    ToolCall(name="read_file", arguments={"path": "b.txt"}),
                ),
            ),
            ModelReply(content="both read"),
        ]
    )

    observed: list[ToolLoopEvent] = []
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read both")],
        registry,
        observe=lambda e: observed.append(e),
    )

    assert call_count == 2
    kinds = [e.kind for e in observed]
    assert kinds.count("tool_call_start") == 2
    assert kinds.count("tool_call_deduped") == 0


def test_loop_dedupes_argument_order_insensitive(tmp_path: Path) -> None:
    """Argument-key order must not defeat the dedup. {'a': 1, 'b': 2}
    and {'b': 2, 'a': 1} are the same call. A naive equality check
    that compared repr() would pass this; a dict-set check would miss
    it. Canonical json.dumps(sort_keys=True) handles it."""
    (tmp_path / "a.txt").write_text("A")

    call_count = 0

    @dataclass
    class _CountingReadFile:
        inner: ReadFileTool

        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="read_file",
                description="read a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_bytes": {"type": "integer"},
                    },
                    "required": ["path"],
                },
                tier="read",
                display_name="Read",
            )

        def call(self, *, path: str, max_bytes: int = 200_000) -> str:
            nonlocal call_count
            call_count += 1
            return self.inner.call(path=path)

    registry = ToolRegistry()
    registry.register(_CountingReadFile(inner=ReadFileTool(root=tmp_path)))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="read_file",
                        arguments={"path": "a.txt", "max_bytes": 100},
                    ),
                    ToolCall(
                        name="read_file",
                        arguments={"max_bytes": 100, "path": "a.txt"},
                    ),
                ),
            ),
            ModelReply(content="done"),
        ]
    )

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read a")],
        registry,
    )
    assert call_count == 1


def test_loop_observer_receives_events_for_each_call_in_one_round(tmp_path: Path) -> None:
    """Same invariant as above but within a single round: if the model
    emits two tool_calls in one ModelReply, both must fire their own
    start+end pair."""
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "b.txt").write_text("B")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(name="read_file", arguments={"path": "a.txt"}),
                    ToolCall(name="read_file", arguments={"path": "b.txt"}),
                ),
            ),
            ModelReply(content="both read"),
        ]
    )

    observed: list[ToolLoopEvent] = []
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read both at once")],
        registry,
        observe=lambda e: observed.append(e),
    )

    starts = [e for e in observed if e.kind == "tool_call_start"]
    assert len(starts) == 2
    assert [e.call.arguments["path"] for e in starts if e.call is not None] == [
        "a.txt",
        "b.txt",
    ]


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


def test_loop_recovers_from_false_success_claim() -> None:
    """Regression (harness-3fn): asked to 'add scratch to .gitignore',
    the model replied 'The scratch directory is now included in the
    .gitignore file' without calling any tool. We now detect that
    completion-claim-without-action and re-prompt."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content=(
                    "The scratch directory has been added to the .gitignore "
                    "file, so it will be excluded from version control."
                )
            ),
            ModelReply(content="I cannot modify files with no write tool available."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="add scratch to .gitignore")],
        ToolRegistry(),
    )
    nudges = [
        m
        for m in result.messages
        if m.role == "user" and "claims that a file was changed" in m.content
    ]
    assert len(nudges) == 1
    assert "cannot modify" in result.content.lower()


def test_loop_allows_success_claim_after_tool_actually_ran() -> None:
    """Wrap-up round: the model summarizes after a successful tool call.
    The completion-claim regex matches but should NOT trigger a nudge
    because a tool already ran this turn (legitimate summary)."""
    # Round 1: model calls edit_file, gets result. Round 2: model wraps
    # up with a past-tense summary that would match _FALSE_SUCCESS_RE.
    registry = ToolRegistry()

    class _AlwaysOk:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="noop",
                description="d",
                parameters={"type": "object", "properties": {}},
                tier="read",
            )

        def call(self) -> str:
            return "ok"

    registry.register(_AlwaysOk())
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="", tool_calls=(ToolCall(name="noop", arguments={}),)),
            ModelReply(content="The file has been updated successfully."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="do it")],
        registry,
    )
    # No nudge — the claim follows a real tool execution this turn.
    nudges = [
        m
        for m in result.messages
        if m.role == "user" and "claims that a file was changed" in m.content
    ]
    assert not nudges
    assert "updated successfully" in result.content


def test_loop_catches_chat_level_meta_confirm() -> None:
    """Regression (harness-edj): asked 'add scratch to .gitignore',
    the model replied 'Would you like me to add scratch to the
    .gitignore file?' without calling a tool. Small models default
    to this when they misread the user's request as a proposal.
    The orchestrator must nudge them to call the tool now."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content=(
                    "To add the `scratch` directory to the `.gitignore` file, "
                    "we need to make sure the user confirms the action. "
                    "Would you like me to add `scratch` to the `.gitignore` file?"
                )
            ),
            ModelReply(content="Okay, adding it now."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="add scratch to .gitignore")],
        ToolRegistry(),
    )
    nudges = [
        m
        for m in result.messages
        if m.role == "user" and "Do NOT ask the user to confirm" in m.content
    ]
    assert len(nudges) == 1


def test_loop_catches_would_you_like_to_add() -> None:
    """Regression (harness-edj, second round): original regex required
    'would you like me to' / 'us to'. 7B Qwen actually emits 'would you
    like to add scratch to .gitignore?' — no intervening 'me'/'us'."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content=(
                    "Let's confirm the user's approval. Would you like to "
                    "add `scratch` to the `.gitignore` file? Please confirm "
                    "your approval."
                )
            ),
            ModelReply(content="Okay, calling the tool."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="add scratch to the gitignore")],
        ToolRegistry(),
    )
    nudges = [
        m
        for m in result.messages
        if m.role == "user" and "Do NOT ask the user to confirm" in m.content
    ]
    assert len(nudges) == 1


def test_loop_strips_meta_confirm_when_tool_call_present(tmp_path: Path) -> None:
    """Regression (harness-edj, third round): 7B sometimes emits BOTH a
    chat-level meta-confirm narrative AND a tool call in the same reply.
    The tool call is valid; the narrative is noise. We blank the content
    so the wrap-up round doesn't see the model's own hallucinated
    confirmation dialog in history."""
    (tmp_path / "a.txt").write_text("content")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content=(
                    "To read the file, we need to make sure the user "
                    "confirms the action. Would you like me to proceed?"
                ),
                tool_calls=(ToolCall(name="read_file", arguments={"path": "a.txt"}),),
            ),
            ModelReply(content="the file says 'content'"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read a.txt")],
        registry,
    )
    # Tool ran.
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    # The assistant turn that carried the tool call had its noisy
    # content blanked out (we replaced the ModelReply).
    assistant_turns = [m for m in result.messages if m.role == "assistant"]
    # The stripped turn is the one with tool_calls; its content is "".
    tool_turn = next(m for m in assistant_turns if m.tool_calls)
    assert tool_turn.content == ""
    # Final reply unaffected.
    assert "content" in result.content


def test_loop_catches_should_i_proceed() -> None:
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="Should I proceed with updating the file?"),
            ModelReply(content="Done."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="update the readme")],
        ToolRegistry(),
    )
    nudges = [
        m
        for m in result.messages
        if m.role == "user" and "Do NOT ask the user to confirm" in m.content
    ]
    assert len(nudges) == 1


def test_loop_does_not_nudge_legitimate_questions() -> None:
    """A genuine clarifying question (the user gave an ambiguous request,
    not an instruction to mutate state) should not trigger the meta-confirm
    nudge. The regex is narrow enough that 'what should I know about X?' /
    'which file should I look at?' don't match — only ask-for-go-ahead
    patterns do."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="Which file did you mean?")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="run the tests")],
        ToolRegistry(),
    )
    # No bail-nudge injected for the genuine clarifier.
    nudges = [
        m
        for m in result.messages
        if m.role == "user" and "Do NOT ask the user to confirm" in m.content
    ]
    assert not nudges


def test_wrap_up_rounds_use_tighter_max_tokens() -> None:
    """Post-tool rounds are brief summaries — they shouldn't inherit the
    initial round's full token budget. Cap prevents small models from
    burning 10s+ of spinner time on filler after the real work is done."""

    @dataclass
    class _CapturingAdapter:
        calls: list[int] = field(default_factory=list)

        def complete_with_tools(
            self,
            messages: Iterable[ChatMessage],
            *,
            tools: list[ToolSpec] | None = None,
            max_tokens: int = 1024,
            temperature: float = 0.5,
        ) -> ModelReply:
            self.calls.append(max_tokens)
            # First call: emit a tool call. Second call: plain text (exits loop).
            if len(self.calls) == 1:
                return ModelReply(
                    content="",
                    tool_calls=(ToolCall(name="nullop", arguments={}),),
                )
            return ModelReply(content="done")

    # Minimal registry with a no-op read-tier tool.
    class _Noop:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="nullop",
                description="d",
                parameters={"type": "object", "properties": {}},
                tier="read",
            )

        def call(self) -> str:
            return ""

    registry = ToolRegistry()
    registry.register(_Noop())
    adapter = _CapturingAdapter()

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="do a thing")],
        registry,
        max_tokens=1024,
        wrap_up_max_tokens=128,
    )
    # First round: no tool has run yet → full 1024-token budget.
    # Second round: wrap-up → capped at 128.
    assert adapter.calls == [1024, 128]


def test_wrap_up_cap_widens_on_truncated_recovery() -> None:
    """harness-jly: when a wrap-up summary truncates, the retry must
    use a wider budget. Before this fix, current_max_tokens doubled
    but the wrap_up_max_tokens floor stayed put, so every retry
    re-truncated at the original cap and the user saw the same
    partial summary streamed up to bail_retries+1 times."""

    @dataclass
    class _CapturingAdapter:
        calls: list[int] = field(default_factory=list)

        def complete_with_tools(
            self,
            messages: Iterable[ChatMessage],
            *,
            tools: list[ToolSpec] | None = None,
            max_tokens: int = 1024,
            temperature: float = 0.5,
        ) -> ModelReply:
            self.calls.append(max_tokens)
            # Round 0 (pre-tool): emit a tool call.
            if len(self.calls) == 1:
                return ModelReply(
                    content="",
                    tool_calls=(ToolCall(name="nullop", arguments={}),),
                )
            # Round 1 (wrap-up): truncated summary — triggers recovery.
            if len(self.calls) == 2:
                return ModelReply(content="partial summary", was_truncated=True)
            # Round 2 (wrap-up retry): complete summary.
            return ModelReply(content="complete summary")

    class _Noop:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="nullop",
                description="d",
                parameters={"type": "object", "properties": {}},
                tier="read",
            )

        def call(self) -> str:
            return ""

    registry = ToolRegistry()
    registry.register(_Noop())
    adapter = _CapturingAdapter()

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="summarize the big thing")],
        registry,
        max_tokens=1024,
        wrap_up_max_tokens=128,
    )
    # Round 0: 1024 (no tool run yet).
    # Round 1: 128 (wrap-up, initial cap).
    # Round 2: 256 (wrap-up cap doubled after truncated recovery). If
    #   the cap hadn't widened, this would still be 128 — and a model
    #   that needs > 128 tokens would re-truncate indefinitely.
    assert adapter.calls == [1024, 128, 256]
    assert result.content == "complete summary"
    # harness-6rl: a truncated_retry event must fire between rounds 1
    # and 2 so renderers can drop the partial-reply stream buffer
    # before the retry re-streams the full reply. Without the event,
    # the CLI / TUI show the partial and the full back-to-back and it
    # looks like the model answered twice.
    retry_events = [e for e in result.events if e.kind == "truncated_retry"]
    assert len(retry_events) == 1
    assert retry_events[0].round_index == 1


def test_loop_catches_bare_tool_intent() -> None:
    """Regression follow-up (harness-q27): 7B says 'I will search the
    web for ...' and then fabricates a numbered list without ever
    emitting a tool_call. Broadened _TOOL_INTENT_RE catches 'I will
    <verb>' (not just 'I'll <verb>') and the resulting nudge includes
    a concrete <tool_call> template for the model to copy."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content=("I will search the web for places to eat BBQ in Ahwatukee.")),
            ModelReply(content="I cannot figure out the tool syntax — giving up."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="search for BBQ")],
        ToolRegistry(),
    )
    nudges = [
        m
        for m in result.messages
        if m.role == "user" and "Stated intent is not action" in m.content
    ]
    assert len(nudges) == 1
    # Nudge gives a concrete template the model can copy.
    assert '"name": "search_web"' in nudges[0].content


def test_loop_catches_fabricated_search_results() -> None:
    """Regression (harness-q27): asked 'search the web for X', the
    7B replied 'Here are the results: 1. Title: … URL:
    https://www.example.com/…' — completely fabricated, no tool call.
    Orchestrator must nudge it to actually invoke search_web."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content=(
                    "Here are the results: "
                    "1. Title: Ahwatukee BBQ Dinner — "
                    "URL: https://www.example.com/ahwatukee-bbq-dinner — "
                    "SNIPPET: Ahwatukee BBQ Dinner is a popular local event."
                )
            ),
            ModelReply(content="I cannot search right now."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="search the web for ahwatukee bbq")],
        ToolRegistry(),
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert len(nudges) == 1


def test_loop_catches_fabricated_quoted_snippet_list() -> None:
    """Regression (harness-j1d): asked 'search the web for BBQ', the
    Qwen 7B 4-bit replied with a numbered list of prose snippets ending
    in `."` and emitted no URLs. The earlier _FABRICATED_SEARCH_RE keyed
    on 'here are the results' / placeholder URLs and missed this shape,
    so the loop returned the fabrication as the final reply."""
    fabrication = (
        "1. From local favorites to new openings, we've got you covered "
        'with our list of top BBQ spots."\n'
        "2. From classic barbecues to gourmet options, our guide has "
        'everything you need to know."\n'
        "3. From traditional ribs to pulled pork, find the perfect spot "
        'for your next BBQ meal."'
    )
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content=fabrication),
            ModelReply(content="I cannot search right now."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="search the web for ahwatukee bbq")],
        ToolRegistry(),
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert len(nudges) == 1


def test_loop_allows_real_search_result_format() -> None:
    """Guard for harness-j1d: a numbered entry that contains a real URL
    (the shape search_web actually emits) must NOT be diagnosed as
    fabrication. The tempered match in _FABRICATED_SEARCH_RE rejects
    entries with `https://` inside."""
    real_shape = (
        "1. Weber BBQ — https://weberbbq.com\n"
        '   "A local spot with great ribs."\n'
        "2. Joe's Pit — https://joespit.example-real.com\n"
        '   "Pulled pork and brisket."'
    )
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content=real_shape),
            ModelReply(content="retry"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="anything")],
        ToolRegistry(),
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert not nudges


def test_loop_catches_placeholder_domains() -> None:
    """Any reply containing example.com / your-site.com etc. without a
    tool call is almost certainly fabricated."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="The answer is at https://your-site.com/path for reference."),
            ModelReply(content="retry"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="where is that doc?")],
        ToolRegistry(),
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert len(nudges) == 1


def test_loop_allows_search_result_summary_after_real_call() -> None:
    """When a tool actually ran this turn, the model summarizing 'here
    are the results' in its wrap-up is legitimate, not fabrication."""
    registry = ToolRegistry()

    class _Noop:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="noop",
                description="d",
                parameters={"type": "object", "properties": {}},
                tier="read",
            )

        def call(self) -> str:
            return "real tool output"

    registry.register(_Noop())
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="", tool_calls=(ToolCall(name="noop", arguments={}),)),
            ModelReply(content="Here are the results: ..."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="look it up")],
        registry,
    )
    # A real tool ran, so the 'here are the results' wrap-up is fine.
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert not nudges


def test_loop_catches_fabricated_ab_capture_receipt() -> None:
    """Regression (harness-lbh): airton_b imitates CaptureTool's receipt
    shape ('Captured: trim bushes (personal, Should, this weekend).') without
    emitting a tool_call. Real capture output is 'Captured <id> — [scope]
    <type> …'; the fabrication drops the id and restructures the parens.
    _FALSE_SUCCESS_RE / _META_CONFIRM_RE / _TOOL_INTENT_RE all miss it."""
    fabrication = (
        "Captured: trim bushes (personal, Should, this weekend).\n"
        "Every captured item is echoed back. This one is added to your "
        "personal tasks with a clear deadline."
    )
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content=fabrication),
            ModelReply(content="I cannot capture right now."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="I need to trim the bushes in the backyard")],
        ToolRegistry(),
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert len(nudges) == 1


def test_loop_catches_fabricated_ab_plan_output() -> None:
    """Regression (harness-lbh): asked 'plan', airton_b fabricates a full
    tiered plan with fake scope abbreviations (prof/pers — real scopes are
    professional/personal) and fake ids, with no tool_call emitted.
    Real _render_plan output starts 'Today — <ISO>' and indents tier labels
    two spaces; the fabrication reshapes the header but is unmistakably
    imitating plan structure."""
    fabrication = (
        "Overload. Shall-tier holds 4 items, cutting 3 to Should with reasons.\n"
        "Shall:\n"
        "  1. [prof/web-gateway] Auth doc — blocks two tasks, stakeholders waiting\n"
        "  2. [prof/retrieval-refactor] Contract tests — Mon deadline\n"
        "Should (cut from Shall — reason each):\n"
        "  3. [prof/router-eval] 30m on drift — caught early 10x cheaper\n"
        "Watching:\n"
        "  - [pers/passport] 51 days out, no action yet — fine\n"
    )
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content=fabrication),
            ModelReply(content="I cannot plan right now."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="plan")],
        ToolRegistry(),
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert len(nudges) == 1


def test_loop_allows_real_plan_output_after_tool_call() -> None:
    """Guard: if `plan` really ran this turn, the model echoing back
    'Shall: 1. [professional/harness-abc] …' as its wrap-up is legitimate,
    not fabrication."""
    registry = ToolRegistry()

    class _FakePlan:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="plan",
                description="d",
                parameters={"type": "object", "properties": {}},
                tier="read",
            )

        def call(self) -> str:
            return "Today — 2026-04-18\n  Shall:\n    (none)"

    registry.register(_FakePlan())
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content="", tool_calls=(ToolCall(name="plan", arguments={}),)),
            ModelReply(
                content=("Shall: nothing today. Should:\n  1. [professional/harness-abc] …")
            ),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="plan")],
        registry,
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert not nudges


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


# ---------- Router integration (harness-ut3) ----------


@dataclass
class _ScriptedRouter:
    """Returns a queued RouterIntent on each classify() call."""

    intents: list[object]  # RouterIntent | None
    classify_calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def classify(self, user_message: str, tool_specs: list[ToolSpec]) -> object:
        self.classify_calls.append((user_message, tuple(s.name for s in tool_specs)))
        return self.intents.pop(0) if self.intents else None


def _read_tool(name: str, output: str = "tool ran") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        tier="read",
    )


@dataclass
class _ReadTool:
    """Concrete read-tier tool for router integration tests."""

    name: str
    output: str = "tool ran"

    @property
    def spec(self) -> ToolSpec:
        return _read_tool(self.name)

    def call(self, *, query: str) -> str:
        _ = query  # kwarg is part of the tool schema; content ignored in tests
        return self.output


def test_router_injects_tool_call_and_skips_fabrication() -> None:
    """Happy path — router picks search_web with valid args, orchestrator
    runs the tool itself, main model only does a wrap-up round. Adapter
    was never asked to emit <tool_call> so it couldn't fabricate."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web", output="1. Weber BBQ — https://weberbbq.com"))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="Weber BBQ is a good option near you.")])
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="search_web", arguments={"query": "bbq"})]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="search the web for bbq")],
        registry,
        router=router,  # type: ignore[arg-type]  # structural match
    )
    assert result.content == "Weber BBQ is a good option near you."
    # One synthetic assistant + one tool result were injected before round 0.
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "search_web"
    # Main model ran exactly once — no fabrication retries.
    assert len(adapter.calls_seen) == 1
    # router_intent event fired.
    intents = [e for e in result.events if e.kind == "router_intent"]
    assert len(intents) == 1
    assert intents[0].call is not None
    assert intents[0].call.name == "search_web"


def test_router_none_return_falls_through() -> None:
    """Router gave up (unparseable output) → normal loop runs."""
    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="hello back")])
    router = _ScriptedRouter(intents=[None])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hey")],
        registry,
        router=router,  # type: ignore[arg-type]
    )
    assert result.content == "hello back"
    assert not any(e.kind == "router_intent" for e in result.events)


def test_router_null_tool_falls_through() -> None:
    """Router said `{"tool": null}` → no tool to run, normal loop."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="casual reply")])
    router = _ScriptedRouter(intents=[RouterIntent(tool_name=None, arguments={})])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hey")],
        registry,
        router=router,  # type: ignore[arg-type]
    )
    assert result.content == "casual reply"
    assert not any(e.kind == "router_intent" for e in result.events)


def test_router_unknown_tool_falls_through() -> None:
    """Router hallucinated a tool name that isn't registered. Don't
    execute; fall through and let the main model handle it."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web"))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="fallback reply")])
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="does_not_exist", arguments={"query": "x"})]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hey")],
        registry,
        router=router,  # type: ignore[arg-type]
    )
    assert result.content == "fallback reply"
    assert not any(e.kind == "router_intent" for e in result.events)


def test_router_missing_required_arg_falls_through() -> None:
    """Required `query` arg not present → skip routing, fall through."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web"))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="fallback")])
    router = _ScriptedRouter(intents=[RouterIntent(tool_name="search_web", arguments={})])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="search")],
        registry,
        router=router,  # type: ignore[arg-type]
    )
    assert result.content == "fallback"
    assert not any(e.kind == "router_intent" for e in result.events)


def test_router_skips_write_tier_tools() -> None:
    """Write-tier tools need the main model's richer context + their
    own confirmation UX. Router is read-tier only for now."""
    from harness.router.intent import RouterIntent

    class _WriteTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="edit_file",
                description="edit",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                tier="write",
            )

        def call(self, *, path: str) -> str:
            _ = path
            return "edited"

    registry = ToolRegistry()
    registry.register(_WriteTool())
    adapter = _ScriptedAdapter(replies=[ModelReply(content="fallback")])
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="edit_file", arguments={"path": "x.py"})]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="edit x.py")],
        registry,
        router=router,  # type: ignore[arg-type]
    )
    assert result.content == "fallback"
    assert not any(e.kind == "router_intent" for e in result.events)


def test_router_passes_only_last_user_message() -> None:
    """Classification key is the most recent user turn, not the whole
    thread — the system prompt and history are the main model's job."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web"))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="done")])
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="search_web", arguments={"query": "pho"})]
    )
    run_tool_loop(
        adapter,
        [
            ChatMessage(role="system", content="you are airton"),
            ChatMessage(role="user", content="earlier turn"),
            ChatMessage(role="assistant", content="earlier reply"),
            ChatMessage(role="user", content="now search for pho"),
        ],
        registry,
        router=router,  # type: ignore[arg-type]
    )
    assert router.classify_calls[0][0] == "now search for pho"


def test_router_prelude_call_seeds_dedup_set() -> None:
    """harness-pun: if the router ran a tool in the prelude, the main
    model must not be able to re-invoke the same (name, args) — that
    would just re-run a read the orchestrator already answered. The
    dedup set is seeded with the router's call, so a main-model
    re-emission gets short-circuited on the first round."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web", output="weak hits"))
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="search_web", arguments={"query": "pho"})]
    )
    # Main model re-emits the same call (identical args). Second reply
    # wraps up with text.
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="search_web", arguments={"query": "pho"}),),
            ),
            ModelReply(content="final"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="find pho places")],
        registry,
        router=router,  # type: ignore[arg-type]
    )

    kinds = [e.kind for e in result.events]
    # Router prelude ran the call once (tool_call_start + tool_call_end).
    # Main model's attempt is short-circuited (tool_call_deduped).
    assert kinds.count("tool_call_start") == 1
    assert kinds.count("tool_call_deduped") == 1


# Explicit import to confirm we can pass pytest from the tests folder
def test_tools_module_importable() -> None:
    import harness.tools  # noqa: F401 — import-for-side-effect check

    assert True
