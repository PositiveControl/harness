from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from harness.model.adapter import ChatMessage
from harness.tools.base import ModelReply, ToolCall, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from harness.tools.base import ToolSpec


class _ToolCapableAdapter(Protocol):
    """Structural type for adapters that support tool calls — narrower
    than the plain ModelAdapter Protocol so type-checkers know the
    orchestrator needs `complete_with_tools`."""

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
    print inline status. Kind is one of: tool_call_start, tool_call_end,
    tool_call_declined, round_complete."""

    kind: str
    call: ToolCall | None = None
    result: ToolResult | None = None
    round_index: int = 0


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

    def emit(event: ToolLoopEvent) -> None:
        events.append(event)
        if observe is not None:
            observe(event)

    for round_idx in range(max_rounds):
        last_reply = adapter.complete_with_tools(
            working,
            tools=registry.specs(),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if not last_reply.tool_calls:
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
            if confirm is not None and spec is not None and spec.tier == "write":
                if not confirm(call):
                    result = ToolResult(
                        tool_name=call.name,
                        output="user declined to approve this tool call",
                        success=False,
                        error="user_declined",
                    )
                    emit(
                        ToolLoopEvent(
                            kind="tool_call_declined",
                            call=call,
                            result=result,
                            round_index=round_idx,
                        )
                    )
                else:
                    result = registry.call(call.name, call.arguments)
                    emit(
                        ToolLoopEvent(
                            kind="tool_call_end",
                            call=call,
                            result=result,
                            round_index=round_idx,
                        )
                    )
            else:
                result = registry.call(call.name, call.arguments)
                emit(
                    ToolLoopEvent(
                        kind="tool_call_end",
                        call=call,
                        result=result,
                        round_index=round_idx,
                    )
                )

            working.append(ChatMessage(role="tool", content=result.output, name=call.name))

    # Loop exhausted — return what we have.
    return ToolLoopResult(
        content=last_reply.content or "[tool loop exhausted without final reply]",
        messages=working,
        rounds=max_rounds,
        events=events,
    )
