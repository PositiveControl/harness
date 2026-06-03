from __future__ import annotations

from typing import Literal

from harness.model.adapter import ModelAdapter
from harness.model.echo import EchoAdapter

AdapterName = Literal["echo", "mlx", "ollama", "vllm"]


def make_adapter(name: AdapterName) -> ModelAdapter:
    """Single resolution point so `chat` and `eval` pick adapters the same
    way. Runtime-specific modules are imported lazily so environments
    without a given runtime (MLX, Ollama, vLLM) can still use the echo
    adapter (tests, headless verification, CI)."""
    if name == "echo":
        return EchoAdapter()
    if name == "mlx":
        from harness.model.mlx import MLXAdapter

        return MLXAdapter()
    if name == "ollama":
        from harness.model.ollama import OllamaAdapter

        return OllamaAdapter()
    if name == "vllm":
        from harness.model.vllm import VllmAdapter

        return VllmAdapter()
    raise ValueError(f"Unknown adapter: {name!r}")


# Window of the gx10-served Qwen3-VL VLM (max_model_len). Smaller than the
# VllmAdapter 32k default, so the budget clamp would otherwise over-promise.
_VISION_CONTEXT_WINDOW = 16_384


def make_vision_adapter(base_url: str | None) -> ModelAdapter | None:
    """Resolve the advisory browser-QA vision adapter (harness-ke4hx).

    A SEPARATE endpoint from the drive's reasoning model — a vLLM server
    hosting a VLM (gx10 Qwen3-VL). Returns None when `base_url` is unset
    (vision-QA disabled), so callers degrade gracefully without a vision
    endpoint. Construction is lazy/network-free (VllmAdapter contract), so
    an unreachable endpoint surfaces only on first use, where the caller
    treats it as a skip rather than a drive failure.

    `context_window` is pinned to the VLM's `max_model_len`, not the
    adapter's 32k default, so the output-budget clamp stays honest."""
    if not base_url:
        return None
    from harness.model.vllm import VllmAdapter

    return VllmAdapter(base_url=base_url, context_window=_VISION_CONTEXT_WINDOW)
