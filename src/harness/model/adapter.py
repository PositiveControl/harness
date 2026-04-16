from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None


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
