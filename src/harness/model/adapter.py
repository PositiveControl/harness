from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from harness.tools.base import ToolCall

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None
    # Populated on assistant messages that issued tool calls; preserved
    # across rounds so the model's chat template can reconstruct the turn.
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    # Populated on tool-role messages (the result of a specific call).
    tool_call_id: str | None = None


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal adapter contract. Implementations wrap a specific runtime
    (MLX, llama.cpp, Ollama, OpenAI-compatible endpoints, etc.).

    Phase 0 is sync + non-streaming. Streaming and async arrive in Phase 1
    when the orchestrator becomes asyncio-native."""

    id: str
    context_window: int

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...
