from __future__ import annotations

import pytest

from harness.model.adapter import ChatMessage, ModelAdapter
from harness.model.mlx import _messages_to_dicts


def test_messages_to_dicts_shape() -> None:
    msgs = [
        ChatMessage(role="system", content="be airton"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="morning"),
    ]
    assert _messages_to_dicts(msgs) == [
        {"role": "system", "content": "be airton"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "morning"},
    ]


def test_messages_to_dicts_drops_name_field() -> None:
    msgs = [ChatMessage(role="user", content="hi", name="mark")]
    out = _messages_to_dicts(msgs)
    assert out == [{"role": "user", "content": "hi"}]
    assert "name" not in out[0]


def test_mlx_adapter_satisfies_protocol() -> None:
    """Construction must not require mlx_lm to be importable — the
    adapter is lazy. Protocol compliance is a structural isinstance check."""
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter()
    assert isinstance(adapter, ModelAdapter)
    assert adapter.id.startswith("mlx:")
    assert adapter.context_window > 0


def test_mlx_adapter_id_derived_from_repo() -> None:
    from harness.model.mlx import MLXAdapter

    a = MLXAdapter(repo="mlx-community/Qwen2.5-32B-Instruct-4bit")
    assert a.id == "mlx:Qwen2.5-32B-Instruct-4bit"

    b = MLXAdapter(repo="foo/bar")
    assert b.id == "mlx:bar"


def test_mlx_adapter_accepts_custom_repo() -> None:
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter(repo="mlx-community/Qwen2.5-7B-Instruct-4bit")
    assert adapter.repo == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert adapter.id == "mlx:Qwen2.5-7B-Instruct-4bit"
    assert adapter.adapter_path is None


def test_mlx_adapter_with_lora_path_ids_differently() -> None:
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter(
        repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        adapter_path="/path/to/airton-v1/adapters.npz",
    )
    assert adapter.adapter_path == "/path/to/airton-v1/adapters.npz"
    assert adapter.id == "mlx:Qwen2.5-7B-Instruct-4bit+lora:adapters"


def test_mlx_adapter_lora_stem_comes_from_filename() -> None:
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter(
        repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        adapter_path="/loras/airton-v2/fine-tuned.npz",
    )
    assert adapter.id == "mlx:Qwen2.5-7B-Instruct-4bit+lora:fine-tuned"


def test_mlx_adapter_does_not_load_on_construction() -> None:
    """Instantiation must not pay the ~30s model-load cost."""
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter()
    assert adapter._model is None
    assert adapter._tokenizer is None


@pytest.mark.skipif(
    "os.environ.get('HARNESS_TEST_MLX') != '1'",
    reason="set HARNESS_TEST_MLX=1 to run (loads the real 32B model)",
)
def test_mlx_adapter_smoke() -> None:
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter()
    reply = adapter.complete(
        [ChatMessage(role="user", content="Say 'ok' and nothing else.")],
        max_tokens=16,
        temperature=0.0,
    )
    assert reply
