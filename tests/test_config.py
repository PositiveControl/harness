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
from pydantic import ValidationError

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


# ab (airton_b) thought-graph budget knobs - Phase 3.6 C5 (harness-6y5).
# Defaults + env overrides + range validation are pinned here so
# downstream consumers (C1-C4) can rely on the Settings surface without
# drift.


def test_ab_budget_defaults() -> None:
    s = Settings()
    assert s.ab_turn_cap == 3
    assert s.ab_inflight_cap == 10
    assert s.ab_stall_defers == 3
    assert s.ab_drift_days == 7


def test_ab_turn_cap_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_AB_TURN_CAP", "5")
    assert Settings().ab_turn_cap == 5


def test_ab_inflight_cap_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_AB_INFLIGHT_CAP", "25")
    assert Settings().ab_inflight_cap == 25


def test_ab_stall_defers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_AB_STALL_DEFERS", "6")
    assert Settings().ab_stall_defers == 6


def test_ab_drift_days_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_AB_DRIFT_DAYS", "30")
    assert Settings().ab_drift_days == 30


@pytest.mark.parametrize(
    ("env_var", "bad_value"),
    [
        ("HARNESS_AB_TURN_CAP", "0"),
        ("HARNESS_AB_TURN_CAP", "11"),
        ("HARNESS_AB_INFLIGHT_CAP", "4"),
        ("HARNESS_AB_INFLIGHT_CAP", "51"),
        ("HARNESS_AB_STALL_DEFERS", "0"),
        ("HARNESS_AB_STALL_DEFERS", "11"),
        ("HARNESS_AB_DRIFT_DAYS", "0"),
        ("HARNESS_AB_DRIFT_DAYS", "91"),
    ],
)
def test_ab_budget_rejects_out_of_range(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValidationError):
        Settings()
