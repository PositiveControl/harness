from __future__ import annotations

from typing import cast

import pytest

from harness.model import AdapterName, make_adapter
from harness.model.adapter import ModelAdapter


def test_make_adapter_echo() -> None:
    adapter = make_adapter("echo")
    assert isinstance(adapter, ModelAdapter)
    assert adapter.id == "echo"


def test_make_adapter_mlx_is_lazy() -> None:
    """Construction must stay cheap — no mlx_lm import on instantiation."""
    adapter = make_adapter("mlx")
    assert isinstance(adapter, ModelAdapter)
    assert adapter.id.startswith("mlx:")


def test_make_adapter_ollama_is_lazy() -> None:
    """Instantiation must not contact the Ollama daemon — network I/O
    is deferred until the first .complete() call."""
    adapter = make_adapter("ollama")
    assert isinstance(adapter, ModelAdapter)
    assert adapter.id.startswith("ollama:")


def test_make_adapter_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown adapter"):
        make_adapter(cast(AdapterName, "not-a-real-adapter"))
