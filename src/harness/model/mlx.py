from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from harness.model.adapter import ChatMessage


def _messages_to_dicts(messages: Iterable[ChatMessage]) -> list[dict[str, str]]:
    """Project ChatMessage records down to the {role, content} dicts that
    tokenizer.apply_chat_template expects. Tool-role messages and the
    `name` field are ignored here — Phase 1a does not do tool-calling."""
    return [{"role": m.role, "content": m.content} for m in messages]


class MLXAdapter:
    """MLX-backed adapter. Load is deferred until the first `complete()`
    call, or until `.load()` is called explicitly. This keeps construction
    cheap (useful for the CLI factory and for unit tests that only need
    to assert protocol compliance).

    Only this module imports `mlx_lm`. The adapter boundary must stay clean
    so other runtimes (llama.cpp, Ollama, OpenAI-compatible) can swap in
    without any other file having to change."""

    def __init__(
        self,
        repo: str = "mlx-community/Qwen2.5-32B-Instruct-4bit",
        *,
        context_window: int = 131_072,
    ) -> None:
        self.repo = repo
        self.id = f"mlx:{repo.split('/')[-1]}"
        self.context_window = context_window
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        """Load model + tokenizer from the HF cache (downloading if
        missing). Safe to call more than once; no-op after the first."""
        if self._model is not None:
            return
        from mlx_lm import load as _load

        self._model, self._tokenizer = _load(self.repo)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self._ensure_loaded()
        from mlx_lm import generate as _generate
        from mlx_lm.sample_utils import make_sampler

        dicts = _messages_to_dicts(messages)
        assert self._tokenizer is not None  # for type narrowing; _ensure_loaded guarantees
        prompt = self._tokenizer.apply_chat_template(
            dicts, tokenize=False, add_generation_prompt=True
        )
        sampler = make_sampler(temp=temperature, top_p=0.9)
        output: str = _generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            sampler=sampler,
            max_tokens=max_tokens,
        )
        return output
