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


@runtime_checkable
class GrammarCapableAdapter(Protocol):
    """Optional extension for adapters that support grammar-constrained
    decoding — generation where every sampled token must continue a
    valid parse of the supplied JSON schema.

    Used by the GrammarRouter to guarantee valid JSON + valid
    `tool_name` choice by construction. Adapters that don't support
    this don't need to implement it; GrammarRouter checks for the
    method structurally and falls through if it isn't present."""

    id: str
    context_window: int

    def complete_grammar(
        self,
        messages: Iterable[ChatMessage],
        schema: dict[str, object],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str: ...


def approx_token_count(messages: Iterable[ChatMessage]) -> int:
    """Char-heuristic token count: ~4 chars per token plus ~4 tokens of
    role/delimiter overhead per message. Shared fallback for adapters
    that don't have a tokenizer on hand."""
    total = 0
    for m in messages:
        total += len(m.content) // 4 + 4
    return total


def count_tokens(adapter: object, messages: Iterable[ChatMessage]) -> int:
    """Ask the adapter for an exact token count when it can produce one;
    otherwise fall back to `approx_token_count`. Kept off the core
    `ModelAdapter` protocol so test stubs and alternative adapters don't
    all have to implement it."""
    materialized = list(messages)
    fn = getattr(adapter, "count_tokens", None)
    if callable(fn):
        try:
            return int(fn(materialized))
        except Exception:
            return approx_token_count(materialized)
    return approx_token_count(materialized)
