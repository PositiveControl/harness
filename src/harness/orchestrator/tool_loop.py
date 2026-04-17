from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from harness.model.adapter import ChatMessage
from harness.tools.base import (
    ModelReply,
    StreamComplete,
    StreamText,
    ToolCall,
    ToolRegistry,
    ToolResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from harness.tools.base import StreamChunk, ToolSpec


# Matches "Let me check…", "I'll now read…", "Next, I'll…" etc. — the
# model announcing more work without actually emitting tool calls. Anchored
# to end of content so a teaser mid-paragraph (followed by real prose) doesn't
# trigger.
_TEASER_RE = re.compile(
    r"\b(let me|i'?ll|now i'?ll|now let me|next,?\s+i'?ll?)\b[^.\n]*[:.]\s*$",
    re.IGNORECASE,
)

_MAX_TOKENS_CEILING = 8192
_BAIL_RETRIES_PER_TURN = 2


def _diagnose_bail(reply: ModelReply) -> str | None:
    """Classify a 0-tool-calls reply. Returns:
    - "truncated" — caller should retry with a larger token budget
    - a nudge string — caller should append it as a user message and retry
    - None — genuine final reply, terminate normally"""
    if reply.was_truncated:
        return "truncated"
    if reply.had_unparseable_call:
        return (
            "Your last <tool_call> block was malformed and could not be parsed. "
            "Re-emit it as a single line of valid JSON inside <tool_call>…</tool_call>: "
            '<tool_call>{"name": "...", "arguments": {...}}</tool_call>'
        )
    if _TEASER_RE.search(reply.content.strip()):
        return (
            "Your reply announced more work but didn't include any tool calls. "
            "Either call the tool now, or give the user your final answer."
        )
    return None


class _ToolCapableAdapter(Protocol):
    """Structural type for adapters that support tool calls — narrower
    than the plain ModelAdapter Protocol so type-checkers know the
    orchestrator needs `complete_with_tools`.

    Adapters that additionally implement `stream_with_tools` get
    token-level streaming through the loop; those that don't fall back
    to the blocking `complete_with_tools` path."""

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply: ...


@dataclass(frozen=True)
class ToolLoopEvent:
    """Emitted synchronously to the optional observer so the CLI can
    print inline status. Kind is one of: round_start, model_call_start,
    token_delta, model_call_end, tool_call_start, tool_call_end,
    tool_call_failed, tool_call_declined, round_complete.

    `model_call_start` fires just before the adapter is invoked;
    `model_call_end` fires once it returns. When the adapter supports
    streaming, `token_delta` events fire between them, each carrying a
    `delta` of visible text (tool-call tag spans are already masked by
    the adapter). The CLI renders deltas into a live region and drops
    the spinner while tokens are flowing."""

    kind: str
    call: ToolCall | None = None
    result: ToolResult | None = None
    round_index: int = 0
    delta: str | None = None


@dataclass
class ToolLoopResult:
    content: str
    messages: list[ChatMessage]  # full thread including tool turns
    rounds: int
    events: list[ToolLoopEvent] = field(default_factory=list)


ConfirmFn = Callable[[ToolCall], bool]
ObserverFn = Callable[[ToolLoopEvent], None]


def run_tool_loop(
    adapter: _ToolCapableAdapter,
    messages: Iterable[ChatMessage],
    registry: ToolRegistry,
    *,
    max_rounds: int = 8,
    confirm: ConfirmFn | None = None,
    observe: ObserverFn | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.5,
) -> ToolLoopResult:
    """Drive a model + tool registry until the model emits a text-only
    reply or `max_rounds` rounds are spent.

    `confirm(call)` is called for every write-tier tool call. Return
    True to execute, False to refuse — the refusal is fed back to the
    model as a tool-role message so it can adjust.

    `observe(event)` is called synchronously on every state transition
    so a CLI can print inline status."""
    working: list[ChatMessage] = list(messages)
    events: list[ToolLoopEvent] = []
    last_reply: ModelReply = ModelReply(content="", tool_calls=())
    bail_retries = _BAIL_RETRIES_PER_TURN
    current_max_tokens = max_tokens

    def emit(event: ToolLoopEvent) -> None:
        events.append(event)
        if observe is not None:
            observe(event)

    stream_fn = getattr(adapter, "stream_with_tools", None)

    for round_idx in range(max_rounds):
        emit(ToolLoopEvent(kind="round_start", round_index=round_idx))
        emit(ToolLoopEvent(kind="model_call_start", round_index=round_idx))
        try:
            if callable(stream_fn):
                stream_iter: Iterator[StreamChunk] = stream_fn(
                    working,
                    tools=registry.specs(),
                    max_tokens=current_max_tokens,
                    temperature=temperature,
                )
                reply: ModelReply | None = None
                for chunk in stream_iter:
                    if isinstance(chunk, StreamText):
                        emit(
                            ToolLoopEvent(
                                kind="token_delta",
                                delta=chunk.text,
                                round_index=round_idx,
                            )
                        )
                    elif isinstance(chunk, StreamComplete):
                        reply = chunk.reply
                if reply is None:
                    raise RuntimeError(
                        "stream_with_tools exhausted without StreamComplete"
                    )
                last_reply = reply
            else:
                last_reply = adapter.complete_with_tools(
                    working,
                    tools=registry.specs(),
                    max_tokens=current_max_tokens,
                    temperature=temperature,
                )
        finally:
            emit(ToolLoopEvent(kind="model_call_end", round_index=round_idx))

        if not last_reply.tool_calls:
            recovery = _diagnose_bail(last_reply) if bail_retries > 0 else None
            if recovery is not None and round_idx + 1 < max_rounds:
                bail_retries -= 1
                if recovery == "truncated":
                    current_max_tokens = min(current_max_tokens * 2, _MAX_TOKENS_CEILING)
                else:
                    working.append(ChatMessage(role="user", content=recovery))
                continue
            emit(ToolLoopEvent(kind="round_complete", round_index=round_idx))
            return ToolLoopResult(
                content=last_reply.content,
                messages=working,
                rounds=round_idx + 1,
                events=events,
            )

        # Record the assistant's tool-call turn so the model can re-read it
        # on the next round via the chat template.
        working.append(
            ChatMessage(
                role="assistant",
                content=last_reply.content,
                tool_calls=last_reply.tool_calls,
            )
        )

        for call in last_reply.tool_calls:
            emit(ToolLoopEvent(kind="tool_call_start", call=call, round_index=round_idx))

            spec = registry.get(call.name).spec if call.name in registry else None
            needs_confirm = confirm is not None and spec is not None and spec.tier == "write"
            if needs_confirm and not confirm(call):  # type: ignore[misc]  # confirm is not None when needs_confirm is True
                result = ToolResult(
                    tool_name=call.name,
                    output="user declined to approve this tool call",
                    success=False,
                    error="user_declined",
                )
                kind = "tool_call_declined"
            else:
                result = registry.call(call.name, call.arguments)
                kind = "tool_call_end" if result.success else "tool_call_failed"
            emit(ToolLoopEvent(kind=kind, call=call, result=result, round_index=round_idx))

            working.append(ChatMessage(role="tool", content=result.output, name=call.name))

    # Loop exhausted — return what we have.
    return ToolLoopResult(
        content=last_reply.content or "[tool loop exhausted without final reply]",
        messages=working,
        rounds=max_rounds,
        events=events,
    )
