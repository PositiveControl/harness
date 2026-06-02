"""Output-budget clamping (harness-2epb).

A fixed-window model sizes a request as `prompt_tokens + max_tokens`; if
that sum exceeds the window the runtime rejects the whole turn (vLLM
returns HTTP 422). `budget_max_tokens` clamps the generation budget so
the invariant `prompt + result <= context_window` always holds, and
raises when the prompt leaves no usable room.
"""

from __future__ import annotations

import pytest

from harness.model.adapter import (
    DEFAULT_OUTPUT_SAFETY_MARGIN,
    MIN_OUTPUT_TOKENS,
    PromptBudgetError,
    budget_max_tokens,
)


def test_passes_through_when_request_fits() -> None:
    assert budget_max_tokens(context_window=32768, prompt_tokens=1000, requested_max=2048) == 2048


def test_clamps_when_request_would_overflow() -> None:
    # 30721 + 2048 = 32769 > 32768. Clamp to window - prompt - margin.
    result = budget_max_tokens(
        context_window=32768,
        prompt_tokens=30721,
        requested_max=2048,
        safety_margin=32,
    )
    assert result == 32768 - 30721 - 32
    assert 30721 + result <= 32768  # the invariant holds, with margin to spare


def test_invariant_holds_at_exact_boundary() -> None:
    # Prompt one token under "window - requested - margin": no clamp.
    cw, margin, req = 32768, 32, 2048
    prompt = cw - req - margin
    assert (
        budget_max_tokens(
            context_window=cw, prompt_tokens=prompt, requested_max=req, safety_margin=margin
        )
        == req
    )
    # One token more: clamp kicks in by exactly one.
    assert (
        budget_max_tokens(
            context_window=cw, prompt_tokens=prompt + 1, requested_max=req, safety_margin=margin
        )
        == req - 1
    )


def test_raises_when_no_room_for_generation() -> None:
    with pytest.raises(PromptBudgetError, match="window remain"):
        budget_max_tokens(
            context_window=32768,
            prompt_tokens=32768 - MIN_OUTPUT_TOKENS,  # leaves < MIN after margin
            requested_max=2048,
        )


def test_raises_when_prompt_exceeds_window() -> None:
    with pytest.raises(PromptBudgetError):
        budget_max_tokens(context_window=32768, prompt_tokens=40000, requested_max=512)


def test_unknown_window_passes_through() -> None:
    # context_window <= 0 means the adapter doesn't advertise a window.
    assert budget_max_tokens(context_window=0, prompt_tokens=999999, requested_max=512) == 512


def test_default_margin_is_reserved() -> None:
    result = budget_max_tokens(context_window=1000, prompt_tokens=900, requested_max=512)
    assert result == 1000 - 900 - DEFAULT_OUTPUT_SAFETY_MARGIN
