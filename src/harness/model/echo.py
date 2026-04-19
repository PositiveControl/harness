from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from harness.model.adapter import ChatMessage, approx_token_count
from harness.tools.base import ModelReply, ToolSpec


@dataclass
class EchoAdapter:
    """Deterministic adapter that echoes the last user message with a
    preamble. Exists so the full loop (character → transcript → model →
    transcript) can be tested before any real model is configured."""

    id: str = "echo"
    context_window: int = 8192

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        last_user = next(
            (m.content for m in reversed(list(messages)) if m.role == "user"),
            "",
        )
        return f"[echo] {last_user}"

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Trivial streaming shim: yield the echo reply as a single chunk.
        Exists so any CLI path that prefers the streaming API can still
        use the echo adapter for wiring tests."""
        yield self.complete(messages, max_tokens=max_tokens, temperature=temperature)

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        """Tool-loop stub: returns the echo reply with no tool calls so
        `--tools` paths can be exercised for wiring tests without
        requiring a real model. Actual tool invocation needs MLX or
        Ollama — echo can't decide when to route a prompt at a tool."""
        content = self.complete(messages, max_tokens=max_tokens, temperature=temperature)
        return ModelReply(content=content, tool_calls=())

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        return approx_token_count(messages)
