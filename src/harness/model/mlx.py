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
        adapter_path: str | None = None,
        context_window: int = 131_072,
    ) -> None:
        self.repo = repo
        self.adapter_path = adapter_path
        # Suffix the id with the LoRA adapter name so evals and logs can tell
        # a LoRA-adapted run from the base model.
        if adapter_path:
            from pathlib import Path

            lora_name = Path(adapter_path).stem or "lora"
            self.id = f"mlx:{repo.split('/')[-1]}+lora:{lora_name}"
        else:
            self.id = f"mlx:{repo.split('/')[-1]}"
        self.context_window = context_window
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> None:
        """Load model + tokenizer from the HF cache (downloading if
        missing). Safe to call more than once; no-op after the first.

        When `adapter_path` is set, mlx_lm applies the LoRA weights on
        top of the base model at load time. The result behaves like any
        other adapter from our perspective — no changes downstream."""
        if self._model is not None:
            return
        from mlx_lm import load as _load

        # mlx_lm.load() returns a 2- or 3-tuple depending on `return_config`;
        # we only need model + tokenizer. Index instead of unpacking so the
        # Union type-checks cleanly.
        if self.adapter_path:
            loaded = _load(self.repo, adapter_path=self.adapter_path)
        else:
            loaded = _load(self.repo)
        self._model = loaded[0]
        self._tokenizer = loaded[1]

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
