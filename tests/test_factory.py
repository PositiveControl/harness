from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest

from harness.model import AdapterName, make_adapter, make_vision_adapter
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


def test_make_vision_adapter_none_when_unset() -> None:
    """No vision_base_url → vision-QA disabled, resolver returns None so
    callers degrade gracefully without a VLM endpoint."""
    assert make_vision_adapter(None) is None
    assert make_vision_adapter("") is None


def test_make_vision_adapter_builds_vllm_when_set() -> None:
    adapter = make_vision_adapter("http://gx10-5fb9:8001/v1")
    assert adapter is not None
    assert isinstance(adapter, ModelAdapter)
    assert adapter.id == "vllm:http://gx10-5fb9:8001/v1"
    # Pinned to the VLM's max_model_len, not the 32k adapter default.
    assert adapter.context_window == 16_384


def test_make_vision_adapter_is_network_free() -> None:
    """Resolution must not touch the network — discovery is deferred to
    first use (the VllmAdapter lazy-client contract)."""
    with patch("httpx.Client") as mock_client:
        adapter = make_vision_adapter("http://nowhere:8001/v1")
        assert adapter is not None
    mock_client.assert_not_called()
