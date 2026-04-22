"""Tests for the tool-result summarizer hook (sota punch #3, harness-zoz).

Exercises the hook in isolation (no tool-loop wiring) and through
the tool-loop path. Uses a scripted summarizer adapter so tests
stay fast and deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from harness.model.adapter import ChatMessage
from harness.orchestrator.hooks import (
    Continue,
    HookPipeline,
    PostToolContext,
    ReplaceResult,
    ToolResultSummarizerHook,
    default_hook_pipeline,
)
from harness.tools.base import ModelReply, ToolCall, ToolRegistry, ToolResult, ToolSpec

# ---------- fixtures ----------


@dataclass
class _ScriptedSummarizer:
    """Returns pre-queued strings from .complete(). `messages_seen`
    captures the prompts the hook sent so tests can assert they
    include the raw output + the hardening-preamble."""

    replies: list[str] = field(default_factory=list)
    messages_seen: list[list[ChatMessage]] = field(default_factory=list)
    raise_on_call: Exception | None = None

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.messages_seen.append(list(messages))
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.replies.pop(0) if self.replies else "(empty)"


def _spec(name: str = "grep", *, high_noise: bool = True, tier: str = "read") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="",
        parameters={"type": "object", "properties": {}},
        tier=tier,
        high_noise=high_noise,
    )


def _ctx(output: str, *, spec: ToolSpec | None = None, success: bool = True) -> PostToolContext:
    spec = spec or _spec()
    return PostToolContext(
        call=ToolCall(name=spec.name, arguments={"q": "x"}),
        result=ToolResult(
            tool_name=spec.name,
            output=output,
            success=success,
        ),
        spec=spec,
    )


# ---------- unit: hook ----------


def test_skips_when_output_under_threshold() -> None:
    summarizer = _ScriptedSummarizer(replies=["should not be used"])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=1024)
    outcome = hook.check(_ctx(output="tiny"))
    assert isinstance(outcome, Continue)
    assert summarizer.messages_seen == []


def test_skips_when_tool_not_high_noise() -> None:
    summarizer = _ScriptedSummarizer(replies=["should not be used"])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=100)
    spec = _spec(name="read_file", high_noise=False)
    outcome = hook.check(_ctx(output="x" * 500, spec=spec))
    assert isinstance(outcome, Continue)
    assert summarizer.messages_seen == []


def test_skips_errored_tool_calls() -> None:
    """Error messages are load-bearing — summarizing them would
    destroy the information the model needs to recover."""
    summarizer = _ScriptedSummarizer(replies=["would summarize"])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=10)
    outcome = hook.check(_ctx(output="traceback: …" * 100, success=False))
    assert isinstance(outcome, Continue)
    assert summarizer.messages_seen == []


def test_replaces_large_high_noise_output_with_summary() -> None:
    summarizer = _ScriptedSummarizer(replies=["grep.py:42 BeadsAdapter"])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=100)
    outcome = hook.check(_ctx(output="line\n" * 300))  # well over threshold
    assert isinstance(outcome, ReplaceResult)
    assert "grep.py:42 BeadsAdapter" in outcome.result.output
    # Annotation gives the model a signal that this isn't raw tool output.
    assert "summarized" in outcome.result.output
    # Preserves the identity + success of the original.
    assert outcome.result.tool_name == "grep"
    assert outcome.result.success is True


def test_preserves_error_field_on_replace() -> None:
    """Non-fatal errors on successful calls (e.g. `ToolResult` with
    success=True and error='warning') must survive the replace."""
    summarizer = _ScriptedSummarizer(replies=["summary"])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=10)
    ctx = PostToolContext(
        call=ToolCall(name="grep", arguments={}),
        result=ToolResult(
            tool_name="grep",
            output="x" * 200,
            success=True,
            error="truncated_at_200",
        ),
        spec=_spec(),
    )
    outcome = hook.check(ctx)
    assert isinstance(outcome, ReplaceResult)
    assert outcome.result.error == "truncated_at_200"


def test_falls_through_on_summarizer_exception() -> None:
    """Summarization is a throughput optimization; any failure must
    never break the turn — the original result reaches the model."""
    summarizer = _ScriptedSummarizer(raise_on_call=RuntimeError("boom"))
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=10)
    outcome = hook.check(_ctx(output="x" * 200))
    assert isinstance(outcome, Continue)


def test_falls_through_on_empty_summary() -> None:
    summarizer = _ScriptedSummarizer(replies=["   "])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=10)
    outcome = hook.check(_ctx(output="x" * 200))
    assert isinstance(outcome, Continue)


def test_prompt_includes_tool_name_and_raw_output() -> None:
    """The summarizer needs enough context to preserve identifiers.
    Regression test: the prompt template must carry both the tool
    name and the raw output."""
    summarizer = _ScriptedSummarizer(replies=["ok"])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=10)
    hook.check(_ctx(output="SOMETHING_UNIQUE_42" + "y" * 200))
    user_msg = summarizer.messages_seen[0][1]
    assert user_msg.role == "user"
    assert "grep" in user_msg.content
    assert "SOMETHING_UNIQUE_42" in user_msg.content


# ---------- pipeline integration ----------


def test_pipeline_post_tool_phase_is_empty_by_default() -> None:
    """The default pipeline must not include the summarizer — it
    requires an adapter and is opt-in via CLI flag. Shipping it
    on by default would load extra MLX weights on every session."""
    pipe = default_hook_pipeline()
    assert pipe.post_tool == []
    assert "tool_result_summarizer" not in pipe.names()


def test_pipeline_runs_summarizer_when_registered() -> None:
    summarizer = _ScriptedSummarizer(replies=["compressed"])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=50)
    pipe = HookPipeline(post_tool=[hook])
    outcome = pipe.run_post_tool(
        _ctx(output="verbose" * 100),
        disabled=frozenset(),
    )
    assert isinstance(outcome, ReplaceResult)
    assert outcome.result.output.endswith("compressed")


def test_pipeline_respects_disabled_set_for_summarizer() -> None:
    summarizer = _ScriptedSummarizer(replies=["should not fire"])
    hook = ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=10)
    pipe = HookPipeline(post_tool=[hook])
    outcome = pipe.run_post_tool(
        _ctx(output="verbose" * 100),
        disabled=frozenset({"tool_result_summarizer"}),
    )
    assert isinstance(outcome, Continue)


# ---------- tool-loop integration ----------


def _build_registry_with_grep(output: str) -> ToolRegistry:
    @dataclass
    class _FakeGrep:
        spec: ToolSpec = field(
            default_factory=lambda: ToolSpec(
                name="grep",
                description="",
                parameters={"type": "object", "properties": {}},
                tier="read",
                high_noise=True,
            )
        )

        def call(self, **_kw: object) -> str:
            return output

    reg = ToolRegistry()
    reg.register(_FakeGrep())
    return reg


def test_tool_loop_applies_summarizer_between_exec_and_append() -> None:
    """The thread the model sees on its next round should contain
    the SUMMARY, not the original blast-radius output."""
    from harness.orchestrator.tool_loop import run_tool_loop

    raw_output = "match line\n" * 500  # ~5500 chars
    registry = _build_registry_with_grep(raw_output)

    @dataclass
    class _ScriptedToolAdapter:
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

    adapter = _ScriptedToolAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="grep", arguments={"pattern": "x"}),),
            ),
            ModelReply(content="done"),
        ]
    )

    summarizer = _ScriptedSummarizer(replies=["grep.py:42 match"])
    hooks = default_hook_pipeline()
    hooks.post_tool.append(ToolResultSummarizerHook(summarizer=summarizer, threshold_chars=1024))

    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="find x")],
        registry,
        hooks=hooks,
    )

    # The SECOND model call (the wrap-up) should see the summarized
    # tool output in its thread, not the 5500-char original.
    wrap_up_messages = adapter.calls_seen[1]
    tool_msgs = [m for m in wrap_up_messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "grep.py:42 match" in tool_msgs[0].content
    # 50 chars of "match line" * 500 = 5500; annotated summary is much
    # shorter. Sanity-check the compression happened.
    assert len(tool_msgs[0].content) < 500
