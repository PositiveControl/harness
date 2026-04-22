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


def test_mlx_adapter_cache_limit_defaults_to_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing set, cache_limit_mb is None so MLX keeps its default
    cache behavior. With HARNESS_MLX_CACHE_LIMIT_MB=1024 the adapter
    picks it up without needing an explicit ctor argument — matches the
    pattern every other model knob uses (embedder_repo, router_repo)."""
    from harness.config import Settings
    from harness.model.mlx import MLXAdapter

    assert Settings().mlx_cache_limit_mb is None
    assert MLXAdapter().cache_limit_mb is None

    monkeypatch.setenv("HARNESS_MLX_CACHE_LIMIT_MB", "1024")
    assert Settings().mlx_cache_limit_mb == 1024
    # Re-import to pick up fresh Settings() in the adapter's inline import.
    import importlib

    import harness.config

    importlib.reload(harness.config)
    import harness.model.mlx as mlx_module

    importlib.reload(mlx_module)
    assert mlx_module.MLXAdapter().cache_limit_mb == 1024


def test_mlx_adapter_explicit_cache_limit_overrides_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ctor arg beats the env — useful when a single process
    wants one adapter capped and another unlimited."""
    monkeypatch.setenv("HARNESS_MLX_CACHE_LIMIT_MB", "4096")
    from harness.model.mlx import MLXAdapter

    assert MLXAdapter(cache_limit_mb=512).cache_limit_mb == 512


def test_mlx_adapter_draft_repo_defaults_to_none() -> None:
    """Without Settings and without an explicit kwarg, draft is disabled
    so every existing install keeps today's ~4 GB 7B-only footprint."""
    from harness.model.mlx import MLXAdapter

    assert MLXAdapter().draft_repo is None
    assert MLXAdapter()._draft_model is None


def test_mlx_adapter_draft_repo_reads_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """HARNESS_MLX_DRAFT_MODEL_REPO flows through Settings to the adapter
    without needing an explicit ctor arg — same pattern as
    HARNESS_MLX_CACHE_LIMIT_MB."""
    monkeypatch.setenv("HARNESS_MLX_DRAFT_MODEL_REPO", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    import importlib

    import harness.config

    importlib.reload(harness.config)
    import harness.model.mlx as mlx_module

    importlib.reload(mlx_module)
    adapter = mlx_module.MLXAdapter()
    assert adapter.draft_repo == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def test_mlx_adapter_explicit_draft_repo_overrides_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_MLX_DRAFT_MODEL_REPO", "mlx-community/default-from-env")
    from harness.model.mlx import MLXAdapter

    adapter = MLXAdapter(draft_repo="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    assert adapter.draft_repo == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def test_mlx_adapter_forwards_draft_model_to_stream_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stream paths must pass `draft_model=<loaded>` through to
    mlx_lm.stream_generate so speculative decoding actually engages.
    Monkeypatches mlx_lm.load + stream_generate so no weights download."""
    from harness.model.mlx import MLXAdapter

    # Fake tokenizer with a vocab_size + apply_chat_template used by the
    # prompt path. Minimal surface: both real tokenizers and ours expose
    # these attrs.
    class _FakeTokenizer:
        vocab_size = 151_936

        def apply_chat_template(
            self, dicts: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
        ) -> str:
            return "PROMPT"

        def encode(self, text: str) -> list[int]:
            return [0] * len(text.split())

    main_model = object()
    draft_model = object()
    main_tokenizer = _FakeTokenizer()
    draft_tokenizer = _FakeTokenizer()

    def fake_load(repo: str, **_kw: object) -> tuple[object, _FakeTokenizer]:
        if "0.5B" in repo:
            return (draft_model, draft_tokenizer)
        return (main_model, main_tokenizer)

    import mlx_lm

    monkeypatch.setattr(mlx_lm, "load", fake_load)

    received_kwargs: dict[str, object] = {}

    class _FakeResp:
        def __init__(self, text: str) -> None:
            self.text = text

    def fake_stream_generate(
        model: object,
        tokenizer: object,
        **kwargs: object,
    ) -> list[_FakeResp]:
        received_kwargs.update(kwargs)
        received_kwargs["__model__"] = model
        return [_FakeResp("hello"), _FakeResp("")]

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)

    adapter = MLXAdapter(
        repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        draft_repo="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    )
    out = "".join(
        adapter.stream(
            [ChatMessage(role="user", content="hi")],
            max_tokens=8,
            temperature=0.0,
        )
    )
    assert out == "hello"
    assert received_kwargs.get("__model__") is main_model
    assert received_kwargs.get("draft_model") is draft_model
    assert adapter._draft_model is draft_model


def test_mlx_adapter_disables_draft_on_vocab_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draft tokenizer with a different vocab size means the draft
    can't share verification with the main. Warn + disable rather than
    raise; speculative decoding is never a correctness requirement."""
    from harness.model.mlx import MLXAdapter

    class _FakeTokenizer:
        def __init__(self, vocab_size: int) -> None:
            self.vocab_size = vocab_size

        def apply_chat_template(self, *a: object, **kw: object) -> str:
            return "PROMPT"

        def encode(self, text: str) -> list[int]:
            return [0]

    def fake_load(repo: str, **_kw: object) -> tuple[object, _FakeTokenizer]:
        if "draft" in repo:
            return (object(), _FakeTokenizer(vocab_size=32_000))  # different vocab
        return (object(), _FakeTokenizer(vocab_size=151_936))

    import mlx_lm

    monkeypatch.setattr(mlx_lm, "load", fake_load)

    adapter = MLXAdapter(
        repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        draft_repo="mlx-community/bogus-draft",
    )
    with pytest.warns(UserWarning, match="vocab"):
        adapter.load()
    assert adapter._draft_model is None


def test_mlx_adapter_disables_draft_on_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draft-model download / load failure must not kill the session
    — warn and continue without speculative decoding."""
    from harness.model.mlx import MLXAdapter

    class _FakeTokenizer:
        vocab_size = 151_936

        def apply_chat_template(self, *a: object, **kw: object) -> str:
            return "PROMPT"

        def encode(self, text: str) -> list[int]:
            return [0]

    def fake_load(repo: str, **_kw: object) -> tuple[object, _FakeTokenizer]:
        if "missing" in repo:
            raise OSError(f"cannot reach {repo}")
        return (object(), _FakeTokenizer())

    import mlx_lm

    monkeypatch.setattr(mlx_lm, "load", fake_load)

    adapter = MLXAdapter(
        repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        draft_repo="mlx-community/missing-draft",
    )
    with pytest.warns(UserWarning, match="failed to load"):
        adapter.load()
    assert adapter._draft_model is None


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
