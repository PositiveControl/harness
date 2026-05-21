from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.model.adapter import ChatMessage
from harness.model.mlx import _parse_qwen_tool_calls
from harness.orchestrator import (
    ToolLoopEvent,
    format_truncated_retry_suffix,
    run_tool_loop,
)
from harness.orchestrator.hooks import default_hook_pipeline
from harness.tools import (
    ModelReply,
    ReadFileTool,
    StreamChunk,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

# Pipeline carrying ab's domain catcher (harness-qvwq). The default
# pipeline omits opt-in catchers so non-corpus characters don't run
# them; tests that exercise AbFabricationHook construct the ab roster
# explicitly and pass it through `hooks=`.
_AB_PIPELINE = default_hook_pipeline(catchers=("ab_fabrication",))


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
    # The duplicate's tool-role turn re-issues the prior result
    # (harness-v5w), prefixed with the 'duplicate' annotation so the
    # next round knows not to retry.
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert "DUPLICATE CALL" in tool_msgs[1].content
    # Prior output preserved verbatim after the prefix.
    assert tool_msgs[1].content.endswith(tool_msgs[0].content)
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


def test_in_round_dedup_does_not_emit_nudge_in_tool_thread(tmp_path: Path) -> None:
    """harness-69f2: when a single model reply emits the same call
    twice, the FIRST execution must succeed cleanly — no
    'duplicate call' nudge in the tool-role thread. Previously the
    in-round duplicate hit DuplicateCallHook and the model saw its
    first attempt rejected, which it then paraphrased as 'I couldn't
    fetch that.'"""
    (tmp_path / "a.txt").write_text("hello")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

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
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read")],
        registry,
    )
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    # Exactly ONE tool-role message — the in-round duplicate is silent.
    assert len(tool_msgs) == 1
    # The model sees the real file content, not a duplicate-nudge.
    assert "hello" in tool_msgs[0].content
    assert "duplicate" not in tool_msgs[0].content.lower()


def test_cross_round_duplicate_of_failed_call_reissues_failure() -> None:
    """harness-v5w: a duplicate of a FAILED call must re-issue the
    failure, not a fixed success=True nudge. Previously the model saw
    the duplicate flagged success=True and hallucinated 'captured/done'
    on top of a real bd failure."""
    registry = ToolRegistry()

    @dataclass
    class _FailingTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="capture",
                description="capture a thing",
                tier="read",
                parameters={
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            )

        def call(self, *, title: str) -> ToolResult:
            del title
            return ToolResult(
                tool_name="capture",
                output="bd command failed: invalid flag --add-label",
                success=False,
                error="bd_command_failed",
            )

    registry.register(_FailingTool())

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="capture", arguments={"title": "x"}),),
            ),
            # Round 2: model retries with identical args after seeing the failure.
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="capture", arguments={"title": "x"}),),
            ),
            ModelReply(content="I couldn't capture that — bd failed."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="capture x")],
        registry,
    )
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    # First tool message: the real failure.
    assert "bd command failed" in tool_msgs[0].content
    # Second tool message: duplicate annotation + the SAME failure body.
    # Crucially, the model can't paraphrase this as 'captured' because
    # it can read 'bd command failed' in the duplicate's body.
    assert "DUPLICATE CALL" in tool_msgs[1].content
    assert "bd command failed" in tool_msgs[1].content


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
    # harness-738f: the event carries the effective per-round budget
    # before and after the doubling so the renderer can show the
    # progression inline (`128 → 256`).
    assert retry_events[0].budget_before == 128
    assert retry_events[0].budget_after == 256


def test_truncated_retry_caps_at_per_turn_budget() -> None:
    """harness-ndsu: consecutive truncated_retry events are capped at
    `_TRUNCATED_RETRIES_PER_TURN` (default 2). On the third truncated
    outcome the orchestrator converts to a bail Nudge instead of
    doubling the budget further — prevents the stuck-generating-prose
    pattern from burning ~15 min of inference while the budget walks
    to _MAX_TOKENS_CEILING.

    Setup: adapter always returns truncated. Verify:
      - exactly 2 truncated_retry events fire (not 3+)
      - on the third truncated outcome, a bail_retry event fires
        with catcher='truncated_retry_cap' (the synthetic nudge)
      - budget doesn't exceed 4x original (2 doublings)"""

    @dataclass
    class _AlwaysTruncated:
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
            return ModelReply(content="partial", was_truncated=True)

    adapter = _AlwaysTruncated()
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="generate something long")],
        ToolRegistry(),
        max_tokens=1024,
        wrap_up_max_tokens=1024,
    )
    truncated_events = [e for e in result.events if e.kind == "truncated_retry"]
    bail_events = [e for e in result.events if e.kind == "bail_retry"]
    # Two truncated retries fire (cap), then the third Truncated
    # outcome rewrites to Nudge → bail_retry event with the cap catcher.
    assert len(truncated_events) == 2, (
        f"expected 2 truncated_retry events (cap), got {len(truncated_events)}: "
        f"{[(e.budget_before, e.budget_after) for e in truncated_events]}"
    )
    cap_bail = [e for e in bail_events if e.catcher == "truncated_retry_cap"]
    assert len(cap_bail) >= 1
    # Budget capped at 4x (2 doublings from 1024): max budget seen = 4096.
    assert max(adapter.calls) == 4096, (
        f"budget should cap at 4096 (2 doublings); saw {adapter.calls}"
    )


def test_truncated_retry_emits_ceiling_marker_when_clamped() -> None:
    """harness-738f: when on_truncated() clamps at _MAX_TOKENS_CEILING,
    the event reports equal before/after so renderers can swap the
    arrow for a `; ceiling` annotation. Drives the round at exactly
    the ceiling so a doubling attempt has nowhere to grow."""

    from harness.orchestrator.tool_loop import _MAX_TOKENS_CEILING

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
            if len(self.calls) == 1:
                # First round: truncated. Recovery widens but clamps
                # at the ceiling.
                return ModelReply(content="partial", was_truncated=True)
            return ModelReply(content="ok")

    adapter = _CapturingAdapter()
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        ToolRegistry(),
        max_tokens=_MAX_TOKENS_CEILING,
    )
    retry_events = [e for e in result.events if e.kind == "truncated_retry"]
    assert len(retry_events) == 1
    assert retry_events[0].budget_before == _MAX_TOKENS_CEILING
    assert retry_events[0].budget_after == _MAX_TOKENS_CEILING
    assert result.content == "ok"


def test_format_truncated_retry_suffix() -> None:
    """harness-738f: the shared helper renders the same parenthetical
    for both renderers (CLI Rich + TUI Textual). Three paths:
    normal doubling, ceiling clamp, missing data."""
    assert format_truncated_retry_suffix(1024, 2048) == " (1024 → 2048)"
    assert format_truncated_retry_suffix(32768, 32768) == " (32768; ceiling)"
    assert format_truncated_retry_suffix(None, 2048) == ""
    assert format_truncated_retry_suffix(1024, None) == ""
    assert format_truncated_retry_suffix(None, None) == ""


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


def test_loop_catches_numbered_url_list_without_tool_call() -> None:
    """harness-q7kn (revising harness-j1d): a numbered list of URLs
    emitted WITHOUT a content-producing tool call is fabrication,
    even when the URLs look real. The model can't verify URLs without
    fetching; presenting them as if it knows them is hallucination.

    The j1d test originally protected this shape from being caught,
    on the premise that real-looking URLs might be legitimately
    recalled. q7kn's repro showed otherwise — the model fabricates
    plausible-but-wrong URLs (LNKN9999_LNKN9999 etc.) in this exact
    shape. The catcher now fires."""
    fab_shape = (
        "1. Weber BBQ — https://weberbbq.com\n"
        '   "A local spot with great ribs."\n'
        "2. Joe's Pit — https://joespit.example-real.com\n"
        '   "Pulled pork and brisket."'
    )
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content=fab_shape),
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
    assert nudges, "numbered-URL-list without a tool call must trip the catcher"


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
        hooks=_AB_PIPELINE,
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
        hooks=_AB_PIPELINE,
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert len(nudges) == 1


def test_loop_catches_fabrication_after_errored_tool_call() -> None:
    """Regression (harness-a0y): when a tool is called this turn but
    returns success=False (e.g. 'invalid scope'), the existing gate
    'tools_ran_this_turn' flips True and disarms every fabrication
    catcher — the model is treated as 'in wrap-up' when in reality
    it has no successful tool output to summarize. User-visible repro:
    asked 'tell me about our current tasks', model called plan with
    scope='current tasks' (invalid), tool errored, model then emitted
    two fabricated tiered plans as 'wrap-up'. The fix gates on
    'at least one successful tool this turn' instead of 'any tool
    executed'."""
    registry = ToolRegistry()

    class _AlwaysFailPlan:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="plan",
                description="d",
                parameters={
                    "type": "object",
                    "properties": {"scope": {"type": "string"}},
                },
                tier="read",
            )

        def call(self, **_: object) -> str:
            raise ValueError("invalid scope 'current tasks'")

    registry.register(_AlwaysFailPlan())
    fabrication = (
        "For professional tasks: Today — 2026-04-18 Shall: (none) "
        "Should: 1. [professional/airton_b-kyl] deliver cabin computer UI POC "
        "Shmaybe: (none) Watching: (none)\n"
        "For personal tasks: Today — 2026-04-18 Shall: (none) "
        "Should: 1. [personal/birthday] August 29 2026 — date-locked "
        "Shmaybe: (none) Watching: (none)"
    )
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="plan", arguments={"scope": "current tasks"}),),
            ),
            ModelReply(content=fabrication),
            ModelReply(content="I cannot plan right now."),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="tell me about our current tasks")],
        registry,
        hooks=_AB_PIPELINE,
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert len(nudges) == 1, (
        "fabrication after all-errored tool calls must be caught, not returned as wrap-up"
    )


def test_loop_catches_full_scope_plan_fabrication() -> None:
    """Regression (harness-jj9): _looks_like_ab_fabrication missed the
    user's observed shape because it used FULL scope names
    ('[professional/airton_b-kyl]' not 'prof') and rendered tier labels
    mid-line rather than at line-start. Catcher must also fire on:
    (a) a 'Today — YYYY-MM-DD' date-header prefix (real _render_plan
    shape, fabricated verbatim), (b) four tier labels in one reply
    regardless of line-start anchoring."""
    from harness.orchestrator.tool_loop import _looks_like_ab_fabrication

    # Full-scope + inline tier labels — exact shape from harness-a0y repro.
    inline_fab = (
        "For professional tasks: Today — 2026-04-18 Shall: (none) "
        "Should: 1. [professional/airton_b-kyl] deliver cabin computer UI POC "
        "Shmaybe: (none) Watching: (none)"
    )
    assert _looks_like_ab_fabrication(inline_fab), (
        "full-scope plan fabrication with inline tier labels must be caught"
    )

    # Date-header prefix alone is a strong tell (real tool output starts
    # with this exact shape; prose almost never does).
    date_header_only = "Today — 2026-04-18\n  Shall: (none)"
    assert _looks_like_ab_fabrication(date_header_only), (
        "'Today — YYYY-MM-DD' header must be caught as plan fabrication"
    )


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


def test_loop_catches_period_prefixed_capture_fabrication() -> None:
    """harness-ce2x: user saw 'Captured. Scope: personal. Outcome: …'
    — fabrication used a period after 'Captured' where the old regex
    required a colon. Broaden to match either punctuation at line
    start."""
    from harness.orchestrator.tool_loop import _looks_like_ab_fabrication

    period_fab = (
        'Captured. Scope: personal. Outcome: "dog poop in back yard cleaned". '
        'Next action: "gather supplies and head to backyard". '
        "Deadline: 2026-04-19."
    )
    assert _looks_like_ab_fabrication(period_fab), (
        "period-prefixed 'Captured.' fabrication must be caught"
    )


def test_loop_catches_fabricated_bead_id() -> None:
    """harness-ce2x: fabrication emits 'Bead id: harness-abc123..' in
    free text. Real CaptureTool output never uses the 'Bead id:'
    label — the id is the second bare token after 'Captured'. Any
    'Bead id: <prefix>-...' is fabrication."""
    from harness.orchestrator.tool_loop import _looks_like_ab_fabrication

    assert _looks_like_ab_fabrication("…outcome set. Bead id: harness-abc123..")
    # Variant without 'id:' label — the shorter 'Bead: harness-xxx'
    # form is also imaginary schema.
    assert _looks_like_ab_fabrication("work queued. Bead: harness-xyz789")
    # Case-insensitive.
    assert _looks_like_ab_fabrication("BEAD ID = harness-zzz")


def test_loop_catches_bare_claim_sentence() -> None:
    """harness-ce2x: bare 'Updated.' / 'Captured.' as a standalone
    sentence in a turn with no tool call. Observed:
      `Missed "…backyard." Updated. Rerun /plan.`
    — past-tense claim with no tool layer to back it. Catcher must
    fire even when the claim word is mid-reply, not line-start."""
    from harness.orchestrator.tool_loop import _looks_like_ab_fabrication

    inline_updated = (
        'Missed "picking up the Ada\'s dog poop in the backyard." Updated. Rerun /plan.'
    )
    assert _looks_like_ab_fabrication(inline_updated), (
        "bare 'Updated.' sentence inside a larger reply must be caught"
    )

    # Several other claim words should all fire.
    for verb in ("Captured", "Created", "Deleted", "Removed", "Added", "Saved", "Noted"):
        assert _looks_like_ab_fabrication(f"Ok. {verb}. Done."), (
            f"bare '{verb}.' sentence must be caught"
        )


def test_loop_bare_claim_does_not_match_real_wrap_up() -> None:
    """Negative case: real tool output `Captured harness-abc — [personal]
    task` (id follows, no period after the word) must NOT trip the
    bare-claim catcher. The catcher gates on tools_ran_this_turn=False
    upstream, but the regex itself should still reject real shape so
    the signal stays specific."""
    from harness.orchestrator.tool_loop import (
        _BARE_CLAIM_RE,
        _FABRICATED_AB_CAPTURE_RE,
    )

    real = "Captured harness-abc — [personal] task under harness-parent. Next: go."
    assert not _BARE_CLAIM_RE.search(real), "real Captured output has id after word, no period"
    assert not _FABRICATED_AB_CAPTURE_RE.search(real), (
        "real 'Captured <id> —' must not match the fabrication regex"
    )
    # Natural prose that mentions 'updated' as a verb should also pass.
    prose = "I updated the plan based on your feedback and it looks better now."
    assert not _BARE_CLAIM_RE.search(prose), (
        "mid-sentence 'updated' (lowercase, no trailing period) must not match"
    )


def test_loop_catches_remembered_fabrication() -> None:
    """harness-z734: user asked ab to remember a fact. Round 0
    fabricated ('Updated.' — caught by bare-claim). Round 1 fabricated
    'Remembered: dad Steve's birthday is Oct 8th' and slipped through
    because no 'Remembered'-shaped catcher existed. Add one."""
    from harness.orchestrator.tool_loop import _looks_like_ab_fabrication

    # Colon form — what RememberTool really emits, and what the model
    # imitates. Gating on tools_ran_this_turn=False keeps this safe.
    assert _looks_like_ab_fabrication("Remembered: dad Steve's birthday is Oct 8th")
    # Line-start after other prose.
    multi = (
        "Missed your request to capture the task. Let's try again.\n"
        "Remembered: dad Steve's birthday is Oct 8th."
    )
    assert _looks_like_ab_fabrication(multi)
    # Period form — bare-claim catcher covers this.
    assert _looks_like_ab_fabrication("Ok. Remembered. Done.")


def test_loop_remembered_not_flagged_mid_sentence() -> None:
    """Negative guard: prose that mentions 'remembered' mid-sentence
    without the receipt shape must not trip. Real usage: 'I remembered
    to look that up' should pass."""
    from harness.orchestrator.tool_loop import _FABRICATED_REMEMBER_RE

    assert not _FABRICATED_REMEMBER_RE.search(
        "I remembered to look that up — should we check the notes?"
    )


def test_loop_second_round_remembered_fabrication_caught() -> None:
    """harness-z734 end-to-end: round 0 fabricates 'Updated.' (caught
    by bare-claim, bail_retry fires), round 1 fabricates 'Remembered:
    ...' and must ALSO be caught. Before the fix, round 1 slipped
    through as the final answer and the user saw the fabricated
    receipt."""
    round0 = ModelReply(
        content=("Missed remembering dad Steve's birthday is Oct 8th. Updated. Rerun /plan.")
    )
    round1 = ModelReply(
        content=(
            "Missed your request to capture the task. Let's try again.\n"
            "Remembered: dad Steve's birthday is Oct 8th."
        )
    )
    round2 = ModelReply(content="ok — cannot do that without a tool")
    adapter = _ScriptedAdapter(replies=[round0, round1, round2])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="remember dad steve's birthday is Oct 8th")],
        ToolRegistry(),
        max_rounds=6,
        hooks=_AB_PIPELINE,
    )
    bail_events = [e for e in result.events if e.kind == "bail_retry"]
    assert len(bail_events) == 2, (
        f"expected 2 bail_retry events (one per fabricated round), got {len(bail_events)}"
    )
    assert result.content == "ok — cannot do that without a tool"


def test_loop_catches_capture_fabrication_end_to_end() -> None:
    """harness-ce2x end-to-end: the exact reply the user saw —
    `Captured. Scope: personal. Outcome: … Bead id: harness-abc123..`
    — must trigger the fabrication nudge when no tool ran this turn.
    Regression for the shape that slipped past the colon-only
    regex."""
    fabrication = (
        'Captured. Scope: personal. Outcome: "dog poop in back yard '
        'cleaned". Next action: "gather supplies and head to backyard". '
        "Deadline: 2026-04-19. Bead id: harness-abc123.."
    )
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(content=fabrication),
            ModelReply(content="actually I can't produce that without a tool"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="/plan clean up dog poop")],
        ToolRegistry(),
        max_rounds=4,
        hooks=_AB_PIPELINE,
    )
    nudges = [
        m for m in result.messages if m.role == "user" and "fabricated tool output" in m.content
    ]
    assert nudges, "fabricated capture receipt must trigger the fabrication nudge"


def test_loop_caps_bail_retries() -> None:
    """If the model keeps bailing, give up after the per-turn cap
    (`_BAIL_RETRIES_PER_TURN`) rather than consuming the full
    max_rounds budget. Each retry emits a bail_retry event
    (harness-24xj) so renderers can drop the in-flight stream
    buffer, and the final exhausted reply is replaced with a canned
    fallback rather than surfacing the still-fabricated teaser as
    the answer."""
    from harness.orchestrator.tool_loop import (
        _BAIL_RETRIES_PER_TURN,
        _EXHAUSTED_FABRICATION_FALLBACK,
    )

    teaser = ModelReply(content="Let me check:")
    # Enough replies to outlast any reasonable retry budget.
    adapter = _ScriptedAdapter(replies=[teaser] * (_BAIL_RETRIES_PER_TURN + 4))
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="go")],
        ToolRegistry(),
        max_rounds=8,
    )
    # 1 initial + N retries = N+1 rounds, then terminate.
    assert result.rounds == _BAIL_RETRIES_PER_TURN + 1
    # Exhaustion fallback — NOT the fabricated teaser (harness-24xj).
    assert result.content == _EXHAUSTED_FABRICATION_FALLBACK
    # One bail_retry per retry: renderers use it to drop the partial
    # stream buffer so retries replace rather than stack.
    bail_events = [e for e in result.events if e.kind == "bail_retry"]
    assert len(bail_events) == _BAIL_RETRIES_PER_TURN
    assert [e.round_index for e in bail_events] == list(range(_BAIL_RETRIES_PER_TURN))


def test_loop_bail_retry_event_fires_for_fabrication() -> None:
    """harness-24xj: a fabrication-shaped 0-tool-call reply (e.g. the
    ab-plan date header) must emit bail_retry BEFORE the nudge is
    queued, so CLI / TUI handlers can drop the in-flight stream buffer
    before the retry starts streaming. If the event fires after, the
    fabricated draft stays on screen above the retry."""
    fabrication = ModelReply(content="Today — 2026-04-18. Nothing scheduled.")
    recovery = ModelReply(content="actually I can't answer that")
    adapter = _ScriptedAdapter(replies=[fabrication, recovery])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="what is the date?")],
        ToolRegistry(),
        max_rounds=4,
        hooks=_AB_PIPELINE,
    )
    kinds = [e.kind for e in result.events]
    # bail_retry must appear exactly once (first round fabricated,
    # second round recovered cleanly).
    assert kinds.count("bail_retry") == 1
    # And must fire during round 0 (the round that produced the
    # fabrication), not later.
    bail = next(e for e in result.events if e.kind == "bail_retry")
    assert bail.round_index == 0
    # Recovery reply is the final content — no fallback substitution
    # because diag returned None on round 1.
    assert result.content == "actually I can't answer that"


def test_loop_exhausted_fabrication_replaced_with_fallback() -> None:
    """harness-24xj: when bail-retries run out and the reply is still
    fabrication-shaped, return the canned fallback instead of the
    hallucinated content. Before the fix we returned the fabricated
    plan verbatim and the user saw 'Today — 2026-04-18 Shall: 1
    [prof/web-gateway] ...' as airton_b's final answer."""
    from harness.orchestrator.tool_loop import (
        _BAIL_RETRIES_PER_TURN,
        _EXHAUSTED_FABRICATION_FALLBACK,
    )

    fab = ModelReply(content="Today — 2026-04-18. Nothing scheduled today.")
    # All attempts fabricate; retries exhaust.
    adapter = _ScriptedAdapter(replies=[fab] * (_BAIL_RETRIES_PER_TURN + 2))
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="what is the date?")],
        ToolRegistry(),
        max_rounds=8,
        hooks=_AB_PIPELINE,
    )
    assert result.content == _EXHAUSTED_FABRICATION_FALLBACK
    # Substitution doesn't swallow the original — events trace shows
    # the model ran 1 + N retries times.
    assert result.rounds == _BAIL_RETRIES_PER_TURN + 1


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


# ---------- registry unknown-tool error shapes (harness-yczi) -------


def test_unknown_tool_without_catalog_uses_legacy_shape() -> None:
    """Registry built without a catalog gives the generic miss error —
    points the model at tool_search, no load_tool recovery hint."""
    registry = ToolRegistry()
    result = registry.call("nonexistent", {})

    assert not result.success
    assert result.error == "unknown_tool"
    assert "unknown tool: 'nonexistent'" in result.output
    assert "load_tool" not in result.output
    assert "tool_search" in result.output


def test_unknown_tool_with_catalog_miss_points_at_tool_search() -> None:
    """Catalog wired but the name isn't in it either: same shape as
    no-catalog — a genuine miss, the model should fall back to
    tool_search."""
    from datetime import UTC, datetime

    from harness.tools.catalog import ToolCatalog, seed_builtins_into

    catalog = ToolCatalog()
    seed_builtins_into(catalog, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))

    registry = ToolRegistry(catalog=catalog)
    result = registry.call("totally_made_up_tool_xyz", {})

    assert not result.success
    assert result.error == "unknown_tool"
    assert "unknown tool: 'totally_made_up_tool_xyz'" in result.output
    assert "Not in the catalog" in result.output
    assert "tool_search" in result.output
    # The hint must not falsely promise that load_tool will help — the
    # name genuinely doesn't exist.
    assert "Call load_tool" not in result.output


def test_unknown_tool_in_catalog_hints_at_load_tool() -> None:
    """Catalog-known name not registered this session: recover the
    model with a load_tool(name=X) hint instead of routing it back
    through tool_search (the doom-loop failure mode from harness-yczi
    session repro)."""
    from datetime import UTC, datetime

    from harness.tools.catalog import ToolCatalog, seed_builtins_into

    catalog = ToolCatalog()
    seed_builtins_into(catalog, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))
    # `search_web` is in the builtin catalog seed but not registered
    # on a bare registry — exactly the session-repro shape.
    assert catalog.get("search_web") is not None

    registry = ToolRegistry(catalog=catalog)
    result = registry.call("search_web", {"query": "anything"})

    assert not result.success
    assert result.error == "unknown_tool"
    assert "unknown tool: 'search_web'" in result.output
    assert "exists in the catalog" in result.output
    assert "load_tool(name='search_web')" in result.output


def test_set_catalog_upgrades_error_shape_for_existing_registry() -> None:
    """A registry constructed without a catalog can be retrofitted via
    set_catalog(); subsequent unknown-tool errors gain the recovery
    hint without rebuilding the registry."""
    from datetime import UTC, datetime

    from harness.tools.catalog import ToolCatalog, seed_builtins_into

    registry = ToolRegistry()
    legacy = registry.call("search_web", {}).output
    assert "load_tool" not in legacy

    catalog = ToolCatalog()
    seed_builtins_into(catalog, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))
    registry.set_catalog(catalog)

    upgraded = registry.call("search_web", {}).output
    assert "load_tool(name='search_web')" in upgraded


def test_loop_executes_call_after_load_tool_clears_unknown_tool(tmp_path: Path) -> None:
    """harness-cck4 end-to-end: model emits a call before the tool is
    activated (gets the yczi catalog-aware unknown_tool error), then
    activates the tool via a side-effect 'late_register' tool, then
    re-emits the original call. Without the cck4 carve-out
    DuplicateCallHook replays the stale unknown_tool error and the
    tool never runs. With the fix, the second emission EXECUTES."""
    (tmp_path / "hi.txt").write_text("real contents")

    registry = ToolRegistry()

    # late_register: when the model calls this tool, it registers
    # read_file into the same registry as a side effect. Stands in for
    # the real load_tool path without needing the full catalog + builder
    # machinery.
    @dataclass
    class _LateRegister:
        target_registry: ToolRegistry
        spec: ToolSpec = field(
            default_factory=lambda: ToolSpec(
                name="late_register",
                description="Register read_file into the registry as a side effect.",
                parameters={"type": "object", "properties": {}, "required": []},
                tier="read",
            )
        )

        def call(self) -> str:
            self.target_registry.register(ReadFileTool(root=tmp_path))
            return "read_file registered"

    registry.register(_LateRegister(target_registry=registry))

    read_call = ToolCall(name="read_file", arguments={"path": "hi.txt"})
    adapter = _ScriptedAdapter(
        replies=[
            # R1: read_file before activation → unknown_tool error.
            ModelReply(content="", tool_calls=(read_call,)),
            # R2: side-effect register the tool.
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="late_register", arguments={}),),
            ),
            # R3: re-emit read_file with the same args. Without the
            # cck4 fix this would be deduped against the R1 result;
            # with the fix it actually executes.
            ModelReply(content="", tool_calls=(read_call,)),
            ModelReply(content="the file contents were real"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read hi.txt")],
        registry,
    )

    tool_msgs = [m.content for m in result.messages if m.role == "tool"]
    # Three tool-role messages: unknown_tool error, late_register ack,
    # real read_file contents (NOT a duplicate-call replay).
    assert len(tool_msgs) == 3
    assert "unknown tool: 'read_file'" in tool_msgs[0]
    assert "registered" in tool_msgs[1]
    # The third tool message is the real read result, NOT the
    # duplicate-call replay prefix from the stale unknown_tool error.
    assert tool_msgs[2] == "real contents"
    assert "DUPLICATE CALL" not in tool_msgs[2]
    assert result.content == "the file contents were real"


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


def test_tag_masker_hides_qwen3_coder_nested_tool_call() -> None:
    """harness-od8w regression: Qwen3-Coder emits the NESTED format
    `<tool_call><function=…>…</function></tool_call>`. The inner
    `</function>` used to flip visible mode back on, leaking the
    outer `</tool_call>` into the user's transcript (visible in
    Mark's 2026-05-20 GTA2 session). Open/close pairing fixes this:
    a `<tool_call>` entry only exits on `</tool_call>`, ignoring any
    inner closes while hidden."""
    from harness.model.mlx import _TagMasker

    m = _TagMasker()
    raw = (
        "before "
        "<tool_call>"
        "<function=read_file><parameter=path>foo.txt</parameter></function>"
        "</tool_call>"
        " after"
    )
    out = "".join(m.feed(c) for c in raw) + m.flush()
    # The whole nested block is hidden; neither outer tag nor inner
    # function/parameter markup leaks. No orphan `</tool_call>` survives.
    assert "<tool_call>" not in out
    assert "</tool_call>" not in out
    assert "<function=" not in out
    assert "</function>" not in out
    assert "<parameter=" not in out
    assert "foo.txt" not in out
    assert out.strip() == "before  after"


def test_tag_masker_qwen3_close_tag_across_delta_boundary() -> None:
    """harness-od8w: the close tag for an outer `<tool_call>` arrives
    across a stream delta boundary AFTER an inner `</function>`. The
    pairing fix must keep us hidden until `</tool_call>` arrives, even
    when it straddles deltas."""
    from harness.model.mlx import _TagMasker

    m = _TagMasker()
    chunks = [
        "intro ",
        "<tool_call>",
        "<function=run><parameter=x>1</parameter>",
        "</function>",  # inner close — MUST NOT flip visible back on
        "</tool",  # outer close straddles two deltas
        "_call>",
        " outro",
    ]
    out = "".join(m.feed(c) for c in chunks) + m.flush()
    assert "<tool_call>" not in out
    assert "</tool_call>" not in out
    assert "</function>" not in out
    assert out == "intro  outro"


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


# ---------- Scope-gate short-circuit (harness-8dop) ----------


def test_scope_redirect_short_circuits_when_scope_out_and_template_set() -> None:
    """Router classifies the turn `scope=out` AND the persona supplied a
    redirect template → orchestrator returns the template directly,
    rounds=0, no main-model call."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    adapter = _ScriptedAdapter(
        replies=[ModelReply(content="MAIN MODEL FABRICATED — should not be reached")]
    )
    router = _ScriptedRouter(intents=[RouterIntent(tool_name=None, arguments={}, scope="out")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="What VFR cloud clearance do I need?")],
        registry,
        router=router,  # type: ignore[arg-type]
        scope_redirect_template="Outside JO 7110.65 — ask airton_c.",
    )
    assert result.content == "Outside JO 7110.65 — ask airton_c."
    assert result.rounds == 0
    # Main model was NEVER called — adapter scripted reply is untouched.
    assert adapter.calls_seen == []
    # scope_redirected event is what observers (CLI / TUI) read to
    # render an inline marker instead of a normal turn.
    redirects = [e for e in result.events if e.kind == "scope_redirected"]
    assert len(redirects) == 1


def test_scope_in_does_not_short_circuit() -> None:
    """`in` is the normal flow — router intent (if any) routes a tool,
    main model handles wrap-up. Template is irrelevant unless scope=out."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="real answer")])
    router = _ScriptedRouter(intents=[RouterIntent(tool_name=None, arguments={}, scope="in")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="What's wake-turbulence separation?")],
        registry,
        router=router,  # type: ignore[arg-type]
        scope_redirect_template="should not appear",
    )
    assert result.content == "real answer"
    assert not any(e.kind == "scope_redirected" for e in result.events)


def test_scope_unsure_does_not_short_circuit() -> None:
    """`unsure` is the conservative default — fall through to the normal
    flow so a router misclassification can't false-block a legitimate
    in-scope question."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="real answer")])
    router = _ScriptedRouter(intents=[RouterIntent(tool_name=None, arguments={}, scope="unsure")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="ambiguous question")],
        registry,
        router=router,  # type: ignore[arg-type]
        scope_redirect_template="should not appear",
    )
    assert result.content == "real answer"
    assert not any(e.kind == "scope_redirected" for e in result.events)


def test_scope_out_without_template_does_not_short_circuit() -> None:
    """Without a redirect template the gate is disabled — `scope=out`
    falls through. This is how every persona except airton_c1 stays
    unaffected by the scope gate."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="main model answer")])
    router = _ScriptedRouter(intents=[RouterIntent(tool_name=None, arguments={}, scope="out")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="anything")],
        registry,
        router=router,  # type: ignore[arg-type]
        # scope_redirect_template not set
    )
    assert result.content == "main model answer"
    assert not any(e.kind == "scope_redirected" for e in result.events)


def test_scope_out_classifier_still_runs_tool_routing() -> None:
    """Edge case: a router emitting both a tool_name AND scope=out is
    still gated as out (template wins). The redirect template is the
    authored answer; the tool that would have run is irrelevant."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web"))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="should not appear")])
    router = _ScriptedRouter(
        intents=[
            RouterIntent(
                tool_name="search_web",
                arguments={"query": "x"},
                scope="out",
            )
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="VFR cloud clearance question")],
        registry,
        router=router,  # type: ignore[arg-type]
        scope_redirect_template="redirect text",
    )
    assert result.content == "redirect text"
    # Tool was NOT executed — scope check ran before _router_prelude.
    assert not any(m.role == "tool" for m in result.messages)
    assert not any(e.kind == "tool_call_start" for e in result.events)


# ---------- Lexical scope-gate fallback (harness-8dop option 4) ----------


def test_scope_lexicon_short_circuits_when_zero_hits() -> None:
    """A user message with no overlap against the character's scope_lexicon
    short-circuits to the redirect template — even before the router
    runs. Motivating prompts: 'how many fruit bats can fit into a cave'
    and 'wedding ring on which finger'. Both have zero aviation tokens."""
    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="MAIN MODEL SHOULD NOT BE REACHED")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="how many fruit bats can fit into a cave?")],
        registry,
        scope_redirect_template="Outside JO 7110.65 — ask airton_c.",
        scope_lexicon=("aircraft", "pilot", "controller", "runway"),
    )
    assert result.content == "Outside JO 7110.65 — ask airton_c."
    assert result.rounds == 0
    assert adapter.calls_seen == []
    assert any(e.kind == "scope_redirected" for e in result.events)


def test_scope_lexicon_falls_through_when_any_hit() -> None:
    """Any lexicon hit — even a glancing one — lets the prompt through
    to the normal flow. The lexical fallback is the 'obvious-out'
    pre-filter, not a precise scope classifier."""
    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="real answer")])
    result = run_tool_loop(
        adapter,
        [
            ChatMessage(
                role="user",
                content="What's the controller side of wake turbulence separation?",
            )
        ],
        registry,
        scope_redirect_template="should not appear",
        scope_lexicon=("aircraft", "controller", "runway"),
    )
    assert result.content == "real answer"
    assert not any(e.kind == "scope_redirected" for e in result.events)


def test_scope_lexicon_disabled_when_empty_tuple() -> None:
    """An empty scope_lexicon disables the fallback — non-bounded
    characters (airton, ab, airton_c) leave this empty and don't
    short-circuit on any prompt."""
    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="real answer")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="how many fruit bats fit in a cave?")],
        registry,
        scope_redirect_template="should not appear",
        scope_lexicon=(),
    )
    assert result.content == "real answer"
    assert not any(e.kind == "scope_redirected" for e in result.events)


def test_scope_lexicon_silent_without_template() -> None:
    """A lexicon without a redirect template is a misconfiguration; the
    gate stays disabled rather than dropping the user's question on the
    floor with nothing to say."""
    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="real answer")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="off-topic prompt")],
        registry,
        scope_lexicon=("aircraft", "runway"),
    )
    assert result.content == "real answer"
    assert not any(e.kind == "scope_redirected" for e in result.events)


def test_scope_lexicon_runs_before_router_and_preludes() -> None:
    """Pre-prelude gate skips forced search_memory AND router classify.
    Verified by passing a router that would normally classify `in` — the
    lexical gate fires first, so the router never sees the turn."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="should not appear")])
    router = _ScriptedRouter(intents=[RouterIntent(tool_name=None, scope="in")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="entirely unrelated question")],
        registry,
        router=router,  # type: ignore[arg-type]
        scope_redirect_template="Outside JO 7110.65 — ask airton_c.",
        scope_lexicon=("aircraft", "runway", "controller"),
    )
    assert result.content == "Outside JO 7110.65 — ask airton_c."
    assert result.rounds == 0
    # router.classify() was never called — its intent queue is intact.
    assert router.classify_calls == []


def test_scope_lexicon_matches_section_sigil() -> None:
    """The `§` sigil is a non-word character — word boundaries don't
    fire on either side. `_has_lexicon_hit` strips `\\b` anchors for
    non-word chars so `§` actually matches in a prompt."""
    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="real answer")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="quote me §5-5-4 please")],
        registry,
        scope_redirect_template="should not appear",
        scope_lexicon=("§",),
    )
    # `§` hit → fall through to the model, NOT short-circuit.
    assert result.content == "real answer"
    assert not any(e.kind == "scope_redirected" for e in result.events)


def test_scope_lexicon_helper_word_boundary_basic() -> None:
    """Direct test on the helper: word-boundary matching is real
    boundary, not substring. 'air' in 'airway' must NOT hit when the
    lexicon token is just 'air'."""
    from harness.orchestrator.tool_loop import _has_lexicon_hit

    assert _has_lexicon_hit("the air is clear", ("air",))
    assert not _has_lexicon_hit("we took the airway home", ("air",))
    # Multi-word collapsing.
    assert _has_lexicon_hit("Class  B   airspace", ("class b",))
    # Case insensitivity both ways.
    assert _has_lexicon_hit("VECTOR me out", ("vector",))
    # Empty lexicon → False (gate disabled signal).
    assert not _has_lexicon_hit("anything at all", ())


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


def test_router_skips_call_when_args_ungrounded() -> None:
    """harness-7od: router picks a tool with a domain in args that the
    user never named — grounding hook rejects it at the pre-tool gate
    and the main model handles the turn instead. Without this, the
    main model wraps confident prose around unrelated tool output."""
    from harness.router.intent import RouterIntent

    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web", output="1. Daily Drop — https://dailydrop.fm/"))
    adapter = _ScriptedAdapter(
        replies=[ModelReply(content="I can't help with that without more info.")]
    )
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="search_web", arguments={"query": "dailydrop.fm"})]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="go to stackoverflow and fetch the first question")],
        registry,
        router=router,  # type: ignore[arg-type]  # structural match
    )

    # Router's pick was skipped by grounding → no router_intent event,
    # no tool ran, main model took the turn directly.
    kinds = [e.kind for e in result.events]
    assert "router_intent" not in kinds
    assert "tool_call_start" not in kinds
    assert result.content == "I can't help with that without more info."


def test_main_model_tool_call_skipped_when_args_ungrounded() -> None:
    """Grounding also applies to main-model tool calls (not just the
    router prelude). An ungrounded domain arg gets Skipped with the
    re-plan nudge as the tool-role message; next round produces the
    final answer."""
    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web", output="search-result"))
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="search_web",
                        arguments={"query": "dailydrop.fm"},
                    ),
                ),
            ),
            ModelReply(content="I need the stackoverflow URL — can you paste it?"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="go to stackoverflow and summarize the first question")],
        registry,
    )

    # tool_call_deduped (the event kind used for any pre-tool Skip,
    # including grounding) replaces tool_call_start/end.
    kinds = [e.kind for e in result.events]
    assert "tool_call_deduped" in kinds
    # The grounding nudge landed in the tool-role message, not the
    # search_web output.
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "grounded-args check failed" in tool_msgs[0].content
    assert "dailydrop" in tool_msgs[0].content
    # Final answer is the recovery prose, not fabricated search prose.
    assert "stackoverflow" in result.content


def test_grounded_main_model_call_runs_normally() -> None:
    """Counter-case: when the main-model tool call's domain DOES trace
    back to the user message, grounding stays out of the way and the
    tool executes normally."""
    registry = ToolRegistry()
    registry.register(_ReadTool(name="search_web", output="SO result"))
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(
                        name="search_web",
                        arguments={"query": "stackoverflow.com python"},
                    ),
                ),
            ),
            ModelReply(content="here's what I found"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="search stackoverflow for python")],
        registry,
    )

    kinds = [e.kind for e in result.events]
    assert "tool_call_end" in kinds
    assert "tool_call_deduped" not in kinds
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert tool_msgs[0].content == "SO result"


# ---------- forced search_memory (harness-3uh) ----------


@dataclass
class _StubSearchMemoryTool:
    """Minimal search_memory stand-in. Avoids wiring a real EpisodicStore
    in these tests — we only care about the orchestrator calling the
    tool with the user message as `query` and threading the result into
    working history."""

    output: str = "(no memories matched)"
    calls: list[dict[str, object]] = field(default_factory=list)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_memory",
            description="stub search_memory",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            tier="read",
        )

    def call(self, *, query: str, k: int = 5) -> str:
        self.calls.append({"query": query, "k": k})
        return self.output


def test_force_search_memory_injects_call_before_model_round() -> None:
    """When force_search_memory=True, the orchestrator fires a
    search_memory tool call with the latest user message as `query`
    BEFORE the first complete_with_tools invocation. The tool result
    message must be in the thread the adapter sees on its first
    call, and the assistant-message carrying the forced tool call
    must precede it."""
    stub = _StubSearchMemoryTool(output="recalled: Class B airspace rules")
    registry = ToolRegistry()
    registry.register(stub)

    # The model replies with no tool calls — the forced call must
    # happen regardless of what the model would have done on its own.
    adapter = _ScriptedAdapter(replies=[ModelReply(content="here is an answer")])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="minimums in class B")],
        registry,
        force_search_memory=True,
    )

    # The forced call ran with the user message as `query`.
    assert len(stub.calls) == 1
    assert stub.calls[0]["query"] == "minimums in class B"

    # The adapter saw the forced tool-result in its first call.
    assert len(adapter.calls_seen) == 1
    seen = adapter.calls_seen[0]
    tool_msgs = [m for m in seen if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "search_memory"
    assert tool_msgs[0].content == "recalled: Class B airspace rules"

    # Working history ordering: assistant(tool_call) → tool(result) before the
    # model's final reply gets recorded.
    roles = [(m.role, m.name) for m in result.messages]
    assert ("assistant", None) in roles
    assert ("tool", "search_memory") in roles
    # Forced-call events fired at round_index=0, before round_start.
    kinds = [(e.kind, e.round_index) for e in result.events]
    forced_start = next(i for i, k in enumerate(kinds) if k == ("tool_call_start", 0))
    forced_end = next(i for i, k in enumerate(kinds) if k == ("tool_call_end", 0))
    round_start = next(i for i, k in enumerate(kinds) if k == ("round_start", 0))
    assert forced_start < forced_end < round_start


def test_force_search_memory_off_by_default_skips_injection() -> None:
    """Default (force_search_memory=False) preserves pre-harness-3uh
    behavior: no forced call, the adapter sees only system + user
    messages on its first complete_with_tools call."""
    stub = _StubSearchMemoryTool()
    registry = ToolRegistry()
    registry.register(stub)

    adapter = _ScriptedAdapter(replies=[ModelReply(content="plain answer")])

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="anything")],
        registry,
    )

    # No forced call ran.
    assert stub.calls == []
    # Adapter saw no tool messages before its first reply.
    assert len(adapter.calls_seen) == 1
    tool_msgs = [m for m in adapter.calls_seen[0] if m.role == "tool"]
    assert tool_msgs == []


def test_force_search_memory_degrades_when_tool_absent() -> None:
    """If `search_memory` isn't registered (e.g. tool-set minimal),
    the forced call is a no-op and the turn still completes normally.
    Graceful degradation is load-bearing: the character flag is
    advisory, not a hard requirement."""
    registry = ToolRegistry()  # empty — no search_memory
    adapter = _ScriptedAdapter(replies=[ModelReply(content="answered anyway")])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="question")],
        registry,
        force_search_memory=True,
    )

    assert result.content == "answered anyway"
    # No forced events.
    forced_events = [e for e in result.events if e.kind == "tool_call_start"]
    assert forced_events == []
    # Adapter saw no tool-role messages.
    assert len(adapter.calls_seen) == 1
    assert [m for m in adapter.calls_seen[0] if m.role == "tool"] == []


def test_force_search_memory_lights_up_tools_ran_for_finalize_hook() -> None:
    """Integration: with the forced call in place, a reply that
    contains a JO-style §X-Y-Z citation and no memory block should NOT
    be caught by UngroundedCitationHook — because tools_ran now
    includes `search_memory` from the forced injection. This is the
    load-bearing interaction with harness-oc8's finalize hook."""
    stub = _StubSearchMemoryTool(output="recalled: wake turbulence separation")
    registry = ToolRegistry()
    registry.register(stub)

    fake_reply = (
        "Standard separation is 3 miles. See JO 7110.65 §5-5-4 for the full wake turbulence rule."
    )
    adapter = _ScriptedAdapter(replies=[ModelReply(content=fake_reply)])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="how close can I put a heavy behind a small")],
        registry,
        force_search_memory=True,
        memory_block_attached=False,  # deliberately: the failure case
    )

    # UngroundedCitationHook would have replaced the reply with the
    # character-agnostic refusal. Assert it didn't — the forced call
    # legitimized the grounding signal.
    from harness.orchestrator.hooks import UNGROUNDED_CITATION_FALLBACK

    assert result.content == fake_reply
    assert UNGROUNDED_CITATION_FALLBACK not in result.content


def test_force_search_memory_skips_when_user_message_empty() -> None:
    """Empty / whitespace-only user message => no forced call. Rare
    (system-only bootstrap), but the prelude must not crash or emit an
    empty-query search."""
    stub = _StubSearchMemoryTool()
    registry = ToolRegistry()
    registry.register(stub)

    adapter = _ScriptedAdapter(replies=[ModelReply(content="ok")])

    run_tool_loop(
        adapter,
        [ChatMessage(role="system", content="system only")],
        registry,
        force_search_memory=True,
    )

    assert stub.calls == []


# ---------- forced assemble_context (harness-jkmk) ----------


@dataclass
class _StubAssembleContextTool:
    """Minimal assemble_context stand-in. Mirrors _StubSearchMemoryTool
    in shape but takes `role` and `variables` per the real
    AssembleContextTool API. We only care that the orchestrator hands
    us the expected arguments and threads the result back."""

    output: str = "Context Package — stub"
    calls: list[dict[str, object]] = field(default_factory=list)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="assemble_context",
            description="stub assemble_context",
            parameters={
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "variables": {"type": "object"},
                },
                "required": ["role"],
            },
            tier="read",
        )

    def call(self, *, role: str, variables: dict[str, object] | None = None) -> str:
        self.calls.append({"role": role, "variables": variables or {}})
        return self.output


def test_force_assemble_context_injects_call_before_model_round() -> None:
    """With force_assemble_context=<role>, the orchestrator fires an
    assemble_context call before the first model round. Variables map
    {request_summary: <user message>} so the contract's slot templates
    see the user's question without the model needing to extract it."""
    stub = _StubAssembleContextTool(output="Context Package — refund decision")
    registry = ToolRegistry()
    registry.register(stub)
    adapter = _ScriptedAdapter(replies=[ModelReply(content="here is an answer")])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="customer asked for a refund")],
        registry,
        force_assemble_context="returns_handler",
    )

    assert len(stub.calls) == 1
    assert stub.calls[0]["role"] == "returns_handler"
    assert stub.calls[0]["variables"] == {"request_summary": "customer asked for a refund"}

    # Adapter saw the forced tool result in working history before
    # generating its reply.
    assert len(adapter.calls_seen) == 1
    tool_msgs = [m for m in adapter.calls_seen[0] if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "assemble_context"
    assert tool_msgs[0].content == "Context Package — refund decision"

    # Forced-call events fire at round_index=0, before round_start.
    kinds = [(e.kind, e.round_index) for e in result.events]
    forced_start = next(i for i, k in enumerate(kinds) if k == ("tool_call_start", 0))
    round_start = next(i for i, k in enumerate(kinds) if k == ("round_start", 0))
    assert forced_start < round_start


def test_force_assemble_context_default_none_skips_injection() -> None:
    """Default (force_assemble_context=None) preserves the
    pre-harness-jkmk path. No forced call, no tool message in working."""
    stub = _StubAssembleContextTool()
    registry = ToolRegistry()
    registry.register(stub)
    adapter = _ScriptedAdapter(replies=[ModelReply(content="plain answer")])

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="anything")],
        registry,
    )

    assert stub.calls == []
    tool_msgs = [m for m in adapter.calls_seen[0] if m.role == "tool"]
    assert tool_msgs == []


def test_force_assemble_context_degrades_when_tool_absent() -> None:
    """If assemble_context isn't registered (e.g., the tool-set doesn't
    include it), the forced call is a no-op and the turn still
    completes. Character flag is advisory, not load-bearing."""
    # Empty registry — assemble_context not present.
    registry = ToolRegistry()
    adapter = _ScriptedAdapter(replies=[ModelReply(content="plain answer")])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="some question")],
        registry,
        force_assemble_context="some_role",
    )
    # No tool-call events for the forced call (tool not registered).
    forced_events = [e for e in result.events if e.kind == "tool_call_start"]
    assert forced_events == []
    # The turn still produced a reply.
    assert result.content == "plain answer"


def test_force_assemble_context_stacks_with_force_search_memory() -> None:
    """A character can opt into both forced calls. search_memory runs
    first (harness-3uh), assemble_context runs second (harness-jkmk),
    both visible in working history before the model round."""
    sm = _StubSearchMemoryTool(output="recalled: prior context")
    ac = _StubAssembleContextTool(output="Context Package — full bundle")
    registry = ToolRegistry()
    registry.register(sm)
    registry.register(ac)
    adapter = _ScriptedAdapter(replies=[ModelReply(content="answer")])

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="combined question")],
        registry,
        force_search_memory=True,
        force_assemble_context="some_role",
    )

    assert len(sm.calls) == 1
    assert len(ac.calls) == 1
    tool_msgs = [m for m in adapter.calls_seen[0] if m.role == "tool"]
    # search_memory FIRST, assemble_context SECOND — the contract order
    # is documented and tested so a character that wants the search
    # hit to be visible inside the contract's episodic slot gets the
    # right ordering.
    names = [m.name for m in tool_msgs]
    assert names == ["search_memory", "assemble_context"]


# ---------- synthesis-continue nudge (harness-b7yd) ----------


def test_synthesis_continue_nudge_injected_when_registry_has_tools(
    tmp_path: Path,
) -> None:
    """When the registry exposes at least one tool, run_tool_loop
    prepends the synthesis-continue rule as a system-role message.
    The model sees it on round 0 — exactly when the rule needs to
    bite (before any data-gathering call)."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="done")])

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="just text")],
        registry,
    )

    seen = adapter.calls_seen[0]
    nudge_msgs = [m for m in seen if m.role == "system" and "Tool-use rule" in (m.content or "")]
    assert len(nudge_msgs) == 1, f"expected one nudge system msg, got {seen!r}"
    assert "synthesis verb" in nudge_msgs[0].content
    assert "rank, prioritize, compare, summarize" in nudge_msgs[0].content


def test_tool_use_rules_nudge_includes_multipart_clause(tmp_path: Path) -> None:
    """The same system-prompt nudge carries BOTH the synthesis-verb
    rule (harness-b7yd) and the multi-part rule (harness-111v). Pin
    the multipart clause's keywords so it can't silently drop out
    if someone later refactors the constant."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    adapter = _ScriptedAdapter(replies=[ModelReply(content="done")])

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="just text")],
        registry,
    )

    seen = adapter.calls_seen[0]
    nudge_msgs = [m for m in seen if m.role == "system" and "Tool-use rule" in (m.content or "")]
    assert len(nudge_msgs) == 1
    body = nudge_msgs[0].content
    assert "Multi-part" in body or "multi-part" in body
    assert "another tool call" in body.lower()
    assert "partial answer" in body.lower()


def test_synthesis_continue_nudge_skipped_when_registry_empty() -> None:
    """Empty registry → no tools the model can call → the synthesis
    rule has nothing to gate, so it shouldn't appear. Keeps the
    minimal-orchestrator path quiet."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="just text")])

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        ToolRegistry(),
    )

    seen = adapter.calls_seen[0]
    nudge_msgs = [m for m in seen if m.role == "system" and "Tool-use rule" in (m.content or "")]
    assert not nudge_msgs


# ---------- meta-tool round-budget exemption (harness-rlza) ----------


def _meta_tool_call(name: str) -> ModelReply:
    """Synthetic reply that emits a meta-tool call. The tool itself
    isn't registered — execution fails with a 'tool not found' error
    but the call NAME drives the work-round accounting, which is what
    these tests pin."""
    return ModelReply(
        content="",
        tool_calls=(ToolCall(name=name, arguments={}),),
    )


def test_meta_tool_rounds_do_not_burn_work_budget(tmp_path: Path) -> None:
    """Three meta-tool rounds + content rounds = budget not exhausted.
    The 2026-05-19 Nairobi repro burned 3 of 8 rounds on discovery
    before the real work started; the exemption gives those rounds
    back."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    adapter = _ScriptedAdapter(
        replies=[
            _meta_tool_call("tool_search"),
            _meta_tool_call("tool_search"),
            _meta_tool_call("load_tool"),
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
            ),
            ModelReply(content="contents read; done"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="do the thing")],
        registry,
        max_rounds=3,
    )

    # Total iterations = 3 meta + 1 content + 1 text = 5. With the OLD
    # behavior (no exemption) and max_rounds=3 the loop would exit
    # after 3 iterations (all spent on meta tools); with the exemption
    # the meta rounds are free and the content round lands before
    # the work budget is touched. hard_ceiling = 2 * 3 = 6 covers the
    # 5 iterations comfortably.
    assert result.rounds == 5
    assert "contents read" in result.content
    meta_events = [e for e in result.events if e.kind == "meta_round"]
    assert len(meta_events) == 3
    assert [e.round_index for e in meta_events] == [0, 1, 2]


def test_meta_tool_round_event_round_index_matches_total_iteration() -> None:
    """meta_round events carry the absolute iteration index (not the
    work-round index), so observers see a monotonic counter aligned
    with round_start / round_complete events."""
    adapter = _ScriptedAdapter(
        replies=[
            _meta_tool_call("introspect"),
            ModelReply(content="ok"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="who are you")],
        ToolRegistry(),
        max_rounds=4,
    )

    meta = next(e for e in result.events if e.kind == "meta_round")
    assert meta.round_index == 0
    # Subsequent round_start event uses index 1 (the text-only round).
    round_starts = [e for e in result.events if e.kind == "round_start"]
    assert [e.round_index for e in round_starts] == [0, 1]


def test_pathological_meta_only_loop_terminates_at_hard_ceiling() -> None:
    """An agent that emits nothing but meta-tool calls forever must
    still terminate. The hard ceiling is 2 * max_rounds — without it,
    the work-budget exemption would let a malformed loop run
    indefinitely."""
    # 16 meta-tool calls — more than the 8-iteration hard ceiling.
    adapter = _ScriptedAdapter(replies=[_meta_tool_call("tool_search") for _ in range(16)])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="spin forever")],
        ToolRegistry(),
        max_rounds=4,
    )

    # hard_ceiling = max(2 * 4, 4 + 1) = 8. Loop must exit at 8
    # iterations regardless of work-round progress.
    assert result.rounds == 8
    # All 8 iterations were meta-only → 8 meta_round events.
    meta_events = [e for e in result.events if e.kind == "meta_round"]
    assert len(meta_events) == 8


def test_mixed_meta_and_content_in_same_round_counts_as_work(tmp_path: Path) -> None:
    """A round that emits BOTH a meta tool call AND a content tool
    call counts as work — the content piece is real progress. The
    exemption only fires when the round is meta-ONLY."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    adapter = _ScriptedAdapter(
        replies=[
            # Single round with both calls — mixed.
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(name="tool_search", arguments={}),
                    ToolCall(name="read_file", arguments={"path": "hi.txt"}),
                ),
            ),
            ModelReply(content="done"),
        ]
    )

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="do mixed work")],
        registry,
        max_rounds=4,
    )

    # No meta_round event — the mixed round counted as work.
    meta_events = [e for e in result.events if e.kind == "meta_round"]
    assert not meta_events
    assert result.content == "done"


# ---------- wrap-up round (harness-0gss) ----------


def test_wrap_up_runs_when_last_round_emits_tool_call(tmp_path: Path) -> None:
    """Loop terminates on a round that emitted a tool call; without
    the wrap-up the previous round's interim text would be the user-
    visible 'final' reply. With the wrap-up, one more synthesis-only
    model call runs and that reply is returned."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    last_round_tool_call = ModelReply(
        content="",
        tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
    )
    wrap_up_reply = ModelReply(
        content="Done — found contents in hi.txt.",
    )
    adapter = _ScriptedAdapter(replies=[last_round_tool_call, wrap_up_reply])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read hi.txt")],
        registry,
        max_rounds=1,
    )

    # Wrap-up fired exactly once.
    wrap_up_events = [e for e in result.events if e.kind == "wrap_up_forced"]
    assert len(wrap_up_events) == 1
    # And the wrap-up reply is what got returned, NOT empty / interim.
    assert "Done" in result.content
    assert "found contents" in result.content
    # Total iterations = 1 work round + 1 wrap-up = 2.
    assert result.rounds == 2


def test_wrap_up_skipped_when_last_round_is_text_only() -> None:
    """If the last round emitted a text-only reply, the normal exit
    path returns it. No wrap-up needed — the model already produced
    a final answer."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="all done")])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        ToolRegistry(),
        max_rounds=2,
    )

    wrap_up_events = [e for e in result.events if e.kind == "wrap_up_forced"]
    assert not wrap_up_events
    assert result.content == "all done"


def test_wrap_up_strips_tool_calls_from_wrap_up_reply(tmp_path: Path) -> None:
    """If the wrap-up reply itself emits a tool call, strip it —
    the wrap-up is synthesis only. Text content is kept; the tool
    call never executes."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    last_round_tool_call = ModelReply(
        content="",
        tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
    )
    wrap_up_with_extra_tool = ModelReply(
        content="From hi.txt: contents.",
        tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
    )
    adapter = _ScriptedAdapter(replies=[last_round_tool_call, wrap_up_with_extra_tool])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read hi.txt")],
        registry,
        max_rounds=1,
    )

    # Wrap-up fired, text content kept, no extra tool execution after.
    wrap_up_events = [e for e in result.events if e.kind == "wrap_up_forced"]
    assert len(wrap_up_events) == 1
    assert "contents" in result.content
    # The work round read hi.txt once; the wrap-up's tool call was stripped.
    # Count tool-result messages: should be exactly one (from the work round).
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1


def test_wrap_up_empty_reply_substitutes_fabrication_fallback(tmp_path: Path) -> None:
    """If the wrap-up reply has neither text nor a usable tool call,
    EmptyReplyAfterToolsHook (harness-uk34) fires Nudge in the
    wrap-up bail; fabrication_fallback then substitutes the canned
    refusal so the user sees a meaningful message instead of the
    "[tool loop exhausted without final reply]" debug sentinel."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    last_round_tool_call = ModelReply(
        content="",
        tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
    )
    # Wrap-up emits a tool call only — content empty, tool call stripped.
    empty_wrap_up = ModelReply(
        content="",
        tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
    )
    adapter = _ScriptedAdapter(replies=[last_round_tool_call, empty_wrap_up])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read hi.txt")],
        registry,
        max_rounds=1,
    )

    # New behavior: the canned refusal substitutes for the empty wrap-up.
    # The debug sentinel is reserved for cases where no catcher fires
    # (e.g. no tools ran AND wrap-up isn't even eligible).
    assert "[tool loop exhausted without final reply]" not in result.content
    assert "didn't land cleanly" in result.content or "couldn't" in result.content


def test_wrap_up_skipped_when_no_content_tool_succeeded() -> None:
    """If only meta tools ran (or no tools at all) and the loop
    exhausts at the hard ceiling, the wrap-up SHOULDN'T fire —
    there's no content for the model to synthesize. The exhausted
    message returns instead."""
    # 20 meta-only calls — meta exemption keeps the loop spinning
    # until the hard ceiling, never landing a work round.
    adapter = _ScriptedAdapter(replies=[_meta_tool_call("tool_search") for _ in range(20)])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="spin")],
        ToolRegistry(),
        max_rounds=4,
    )

    wrap_up_events = [e for e in result.events if e.kind == "wrap_up_forced"]
    assert not wrap_up_events
    # Last reply was a (failed) meta tool call, no content tool succeeded.
    assert "[tool loop exhausted without final reply]" in result.content


def test_wrap_up_event_round_index_aligns_with_total_iteration(tmp_path: Path) -> None:
    """The wrap-up event carries the absolute iteration index so
    observers see a monotonic counter aligned with round_start /
    round_complete events."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    last_round = ModelReply(
        content="",
        tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
    )
    wrap_up = ModelReply(content="synthesized")
    adapter = _ScriptedAdapter(replies=[last_round, wrap_up])

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read")],
        registry,
        max_rounds=1,
    )

    # Round 0 was the work round; wrap-up runs at iteration index 1.
    wrap_up_event = next(e for e in result.events if e.kind == "wrap_up_forced")
    assert wrap_up_event.round_index == 1
    # round_complete on the wrap-up matches.
    round_completes = [e for e in result.events if e.kind == "round_complete"]
    assert round_completes[-1].round_index == 1


# ---------- inbox (mid-turn injection, harness-6fr0) ----------


def test_inbox_drained_message_lands_in_next_round_context(tmp_path: Path) -> None:
    """Injection drained at the top of the second iteration must
    appear in the messages the adapter sees on that round."""
    (tmp_path / "hi.txt").write_text("contents")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "hi.txt"}),),
            ),
            ModelReply(content="ack with injected context"),
        ]
    )

    # Inbox yields once on the second consultation; consult counter
    # so it doesn't fire forever (the loop calls drain twice per
    # round — start of while + after _execute_tool_calls).
    calls: list[int] = [0]

    def inbox() -> list[ChatMessage]:
        calls[0] += 1
        if calls[0] == 2:
            return [ChatMessage(role="user", content="wait actually check 'hi.txt' twice")]
        return []

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read hi.txt")],
        registry,
        inbox=inbox,
    )

    # Adapter saw two rounds. Round 2's messages must include the
    # injected user-role line — it landed between the tool-role
    # result and the next model call.
    assert len(adapter.calls_seen) == 2
    round2_user_msgs = [m for m in adapter.calls_seen[1] if m.role == "user"]
    contents = [m.content for m in round2_user_msgs]
    assert "wait actually check 'hi.txt' twice" in contents
    # And one message_injected event was emitted with the text.
    injected = [e for e in result.events if e.kind == "message_injected"]
    assert len(injected) == 1
    assert injected[0].delta == "wait actually check 'hi.txt' twice"


def test_inbox_none_is_noop() -> None:
    """The default `inbox=None` path must not emit any
    message_injected events and must not alter the message
    thread."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="bye")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        ToolRegistry(),
        inbox=None,
    )
    assert result.content == "bye"
    assert not any(e.kind == "message_injected" for e in result.events)


def test_inbox_appends_n_separate_messages(tmp_path: Path) -> None:
    """Multiple injections drained at the same boundary must be
    appended as N separate user-role ChatMessages — no
    concatenation."""
    (tmp_path / "hi.txt").write_text("c")
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

    fired = [False]

    def inbox() -> list[ChatMessage]:
        if fired[0]:
            return []
        fired[0] = True
        return [
            ChatMessage(role="user", content="first injection"),
            ChatMessage(role="user", content="second injection"),
        ]

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read")],
        registry,
        inbox=inbox,
    )
    injected = [e for e in result.events if e.kind == "message_injected"]
    assert len(injected) == 2
    assert [e.delta for e in injected] == ["first injection", "second injection"]
    # Both also land in the working thread as separate user
    # messages — not merged.
    user_msgs = [m.content for m in result.messages if m.role == "user"]
    assert user_msgs.count("first injection") == 1
    assert user_msgs.count("second injection") == 1


def test_inbox_template_safe_position(tmp_path: Path) -> None:
    """Injection must NEVER land between an assistant(tool_calls)
    message and its tool-role result. The drain at the top of the
    next round is the earliest legal position."""
    (tmp_path / "hi.txt").write_text("c")
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

    fired = [False]

    def inbox() -> list[ChatMessage]:
        if fired[0]:
            return []
        fired[0] = True
        return [ChatMessage(role="user", content="mid-turn note")]

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="read")],
        registry,
        inbox=inbox,
    )

    # Walk the working thread. Every assistant(tool_calls=…) must be
    # immediately followed by a tool-role message — no user-role
    # message may interleave.
    for i, msg in enumerate(result.messages):
        if msg.role == "assistant" and msg.tool_calls:
            assert i + 1 < len(result.messages)
            assert result.messages[i + 1].role == "tool", (
                f"non-tool message {result.messages[i + 1].role!r} "
                "followed an assistant tool-call — chat template invariant broken"
            )


def test_inbox_empty_content_skipped() -> None:
    """A drained ChatMessage with empty content must NOT land in
    the working thread — just like the deque drain on the TUI side
    treats it as nothing typed."""
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(),
            ),
        ]
    )

    def inbox() -> list[ChatMessage]:
        return [ChatMessage(role="user", content="")]

    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        ToolRegistry(),
        inbox=inbox,
    )
    assert not any(e.kind == "message_injected" for e in result.events)
    assert all(m.content != "" or m.role != "user" for m in result.messages)


# Explicit import to confirm we can pass pytest from the tests folder
def test_tools_module_importable() -> None:
    import harness.tools  # noqa: F401 — import-for-side-effect check

    assert True
