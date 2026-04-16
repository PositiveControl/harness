from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from harness.model.adapter import ChatMessage


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
