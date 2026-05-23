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
