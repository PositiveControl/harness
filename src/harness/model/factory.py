from __future__ import annotations

from typing import Literal

from harness.model.adapter import ModelAdapter
from harness.model.echo import EchoAdapter

AdapterName = Literal["echo", "mlx"]


def make_adapter(name: AdapterName) -> ModelAdapter:
    """Single resolution point so `chat` and `eval` pick adapters the same
    way. MLX is imported lazily so environments without MLX can still use
    the echo adapter (tests, headless verification, CI)."""
    if name == "echo":
        return EchoAdapter()
    if name == "mlx":
        from harness.model.mlx import MLXAdapter

        return MLXAdapter()
    raise ValueError(f"Unknown adapter: {name!r}")
