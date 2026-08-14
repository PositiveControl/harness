"""Behavioral contract for `_resolve_adapter`.

Step 6 of docs/cli-extraction-plan.md moves this into `cli_adapter.py`.
The plan calls it "already well-isolated", and it is — but the coverage
sweep for harness-z4k1.1 still put it at ~47% against the adapter-facing
tests: what WAS covered is the echo + persona-wrap path, and what wasn't
is every flag-validation error and the whole custom-config dispatch that
maps CLI flags onto adapter constructor kwargs.

That mapping is the part a move can silently scramble (swap
`adapter_path` for `repo` and MLX loads the wrong thing), so the real
adapter classes are stubbed out here — the tests assert on the kwargs
the CLI hands them, without loading a multi-GB model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer

from harness.character import load_character
from harness.cli import _resolve_adapter

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AIRTON = _REPO_ROOT / "character" / "airton"


class _StubAdapter:
    """Records the kwargs the CLI constructed it with."""

    id = "stub"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def complete(self, *args: Any, **kwargs: Any) -> str:
        return ""


def _stub_model_module(monkeypatch: pytest.MonkeyPatch, module: str, cls_name: str) -> None:
    """Replace a model module with a namespace exposing the stub.

    `_resolve_adapter` imports these lazily inside the branch, so
    patching sys.modules intercepts the import without the real MLX /
    Ollama / vLLM package ever being touched.
    """
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{cls_name: _StubAdapter}))


# ---------- flag validation ----------


@pytest.mark.parametrize("flag", ["lora_path", "draft_repo"])
def test_mlx_only_flags_are_rejected_for_other_backends(flag: str) -> None:
    """--lora-path / --draft-repo are MLX-only. Rejecting up front beats
    constructing an Ollama adapter that quietly ignores them."""
    with pytest.raises(typer.BadParameter, match="requires --model mlx"):
        _resolve_adapter("ollama", **{flag: "/some/path"})  # type: ignore[arg-type]


def test_model_repo_is_rejected_for_backends_that_have_no_repo_concept() -> None:
    with pytest.raises(typer.BadParameter, match="mlx, ollama, or vllm"):
        _resolve_adapter("echo", model_repo="some/repo")


def test_unknown_backend_name_becomes_a_parameter_error() -> None:
    """make_adapter's ValueError must surface as CLI usage feedback."""
    with pytest.raises(typer.BadParameter):
        _resolve_adapter("not-a-backend")


def test_persona_without_a_character_is_a_parameter_error() -> None:
    with pytest.raises(typer.BadParameter, match="requires a character"):
        _resolve_adapter("echo", persona=True, character=None)


# ---------- custom-config dispatch ----------


def test_mlx_flags_map_onto_the_adapter_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    """repo / adapter_path / draft_repo are three different things.
    Crossing them wires the wrong weights with no error."""
    _stub_model_module(monkeypatch, "harness.model.mlx", "MLXAdapter")

    adapter = _resolve_adapter(
        "mlx",
        model_repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        lora_path="/adapters/voice",
        draft_repo="mlx-community/Qwen2.5-0.5B",
    )

    assert adapter.kwargs == {  # type: ignore[attr-defined]
        "repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "adapter_path": "/adapters/voice",
        "draft_repo": "mlx-community/Qwen2.5-0.5B",
    }


def test_mlx_omits_kwargs_that_were_not_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset flags must not reach the constructor as None — the adapter's
    own defaults are what pick the model."""
    _stub_model_module(monkeypatch, "harness.model.mlx", "MLXAdapter")

    adapter = _resolve_adapter("mlx", lora_path="/adapters/voice")

    assert adapter.kwargs == {"adapter_path": "/adapters/voice"}  # type: ignore[attr-defined]


def test_ollama_repo_becomes_the_model_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_model_module(monkeypatch, "harness.model.ollama", "OllamaAdapter")

    adapter = _resolve_adapter("ollama", model_repo="qwen2.5-coder:32b-instruct")

    assert adapter.kwargs == {"model": "qwen2.5-coder:32b-instruct"}  # type: ignore[attr-defined]


def test_vllm_repo_becomes_the_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM's --model-repo is a URL, not a repo id — same flag, third
    meaning."""
    _stub_model_module(monkeypatch, "harness.model.vllm", "VllmAdapter")

    adapter = _resolve_adapter("vllm", model_repo="http://gx10-1.tailnet:8000/v1")

    assert adapter.kwargs == {"base_url": "http://gx10-1.tailnet:8000/v1"}  # type: ignore[attr-defined]


# ---------- persona wrapping + eager load ----------


def test_rewriter_temperature_reaches_the_persona_adapter() -> None:
    character = load_character(_AIRTON)

    adapter = _resolve_adapter("echo", persona=True, character=character, rewriter_temperature=0.05)

    assert adapter.rewriter_temperature == 0.05  # type: ignore[attr-defined]


def test_rewriter_temperature_left_unset_keeps_the_persona_default() -> None:
    character = load_character(_AIRTON)

    adapter = _resolve_adapter("echo", persona=True, character=character)

    assert adapter.rewriter_temperature == 0.2  # type: ignore[attr-defined]


def test_eager_load_is_called_when_the_adapter_exposes_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`.load()` isn't part of the ModelAdapter protocol — it's honored
    duck-typed so heavyweight backends warm up before the first turn."""
    loaded: list[bool] = []

    class _EagerAdapter(_StubAdapter):
        def load(self) -> None:
            loaded.append(True)

    monkeypatch.setitem(
        sys.modules, "harness.model.ollama", SimpleNamespace(OllamaAdapter=_EagerAdapter)
    )

    _resolve_adapter("ollama", model_repo="gemma4:latest")

    assert loaded == [True]
