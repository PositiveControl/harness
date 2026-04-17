from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from harness.model.adapter import ChatMessage, approx_token_count


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

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        return approx_token_count(messages)
