"""Smoke tests for the central Settings object.

The defaults encode the 32GB-box trade-off (harness-e4m, harness-5b3):
bge-small for the embedder, Llama-3.2-1B for the router. Both are
overridable via HARNESS_EMBEDDER_REPO / HARNESS_ROUTER_REPO env vars
so the heavier variants stay one export away — these tests lock in
both the defaults and the override surface so a refactor can't
silently revert them.
"""

from __future__ import annotations

import pytest

from harness.config import Settings


def test_default_embedder_repo_is_small() -> None:
    s = Settings()
    assert s.embedder_repo == "BAAI/bge-small-en-v1.5"


def test_default_router_repo_is_hermes3() -> None:
    """Router stays on Hermes-3 3B. Generic-chat small models under-
    route in router eval (Qwen-1.5B 0.85, Llama-1B 0.65, SmolLM2-1.7B
    0.40 vs Hermes-3 1.00). Revisit if a function-call-tuned 1B lands
    in mlx-community."""
    s = Settings()
    assert s.router_repo == "mlx-community/Hermes-3-Llama-3.2-3B-4bit"


def test_embedder_repo_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_EMBEDDER_REPO", "mixedbread-ai/mxbai-embed-large-v1")
    assert Settings().embedder_repo == "mixedbread-ai/mxbai-embed-large-v1"


def test_router_repo_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_ROUTER_REPO", "mlx-community/Hermes-3-Llama-3.2-3B-4bit")
    assert Settings().router_repo == "mlx-community/Hermes-3-Llama-3.2-3B-4bit"


def test_embedder_constructor_picks_up_default() -> None:
    """SentenceTransformersEmbedder() with no args must resolve to the
    Settings default, not a hardcoded string — that's the whole point of
    centralizing."""
    from harness.retrieval.st_embedder import SentenceTransformersEmbedder

    e = SentenceTransformersEmbedder()
    assert e.model_name == Settings().embedder_repo
