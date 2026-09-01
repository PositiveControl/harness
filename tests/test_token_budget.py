"""Output-budget clamping (harness-2epb).

A fixed-window model sizes a request as `prompt_tokens + max_tokens`; if
that sum exceeds the window the runtime rejects the whole turn (vLLM
returns HTTP 422). `budget_max_tokens` clamps the generation budget so
the invariant `prompt + result <= context_window` always holds, and
raises when the prompt leaves no usable room.
"""

from __future__ import annotations

import math

import pytest

from harness.model.adapter import (
    DEFAULT_OUTPUT_SAFETY_MARGIN,
    DRIFT_MARGIN_FRACTION,
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


# ---------- proportional drift margin (harness-ccksu) --------------------


def test_margin_scales_with_prompt_size() -> None:
    """A flat 32 tokens can't absorb the char heuristic's error, because
    that error scales with the prompt. Measured against gx10: a prose
    prompt drifted +2%, but once a tool loop had ASCII art in the message
    list it drifted 6% — 33 tokens past a 32-token margin, and vLLM
    rejected the whole request."""
    window, prompt = 65_536, 10_000
    result = budget_max_tokens(context_window=window, prompt_tokens=prompt, requested_max=window)
    applied_margin = window - prompt - result
    assert applied_margin == math.ceil(prompt * DRIFT_MARGIN_FRACTION)
    assert applied_margin > DEFAULT_OUTPUT_SAFETY_MARGIN


def test_the_measured_gx10_overflow_is_now_covered() -> None:
    """The exact shape that failed live: a 65536-token window, a prompt
    the heuristic put at 503 that the server tokenized at 536. The old
    flat margin sent 65001 and got
    'you requested 65001 output tokens and your prompt contains at least
    536 input tokens, for a total of at least 65537'."""
    result = budget_max_tokens(context_window=65_536, prompt_tokens=503, requested_max=65_536)
    assert result == 65_536 - 503 - math.ceil(503 * DRIFT_MARGIN_FRACTION)
    # What the server would actually have had to fit: 536 real input tokens.
    assert 536 + result <= 65_536


def test_a_fitting_request_is_never_touched_by_the_margin() -> None:
    """The margin only ever costs anything at the window boundary."""
    assert (
        budget_max_tokens(context_window=65_536, prompt_tokens=40_000, requested_max=2048) == 2048
    )


def test_cushion_shrinks_rather_than_starving_the_budget() -> None:
    """The proportional part is a cushion, not a reservation. Where it
    would leave less than MIN_OUTPUT_TOKENS, it falls back to the floor
    margin instead of raising — a big prompt near the window must not
    start erroring where it used to generate."""
    window, prompt = 1000, 900
    proportional = math.ceil(prompt * DRIFT_MARGIN_FRACTION)
    assert window - prompt - proportional < MIN_OUTPUT_TOKENS  # cushion can't fit

    result = budget_max_tokens(context_window=window, prompt_tokens=prompt, requested_max=512)
    assert result == window - prompt - DEFAULT_OUTPUT_SAFETY_MARGIN
    assert result >= MIN_OUTPUT_TOKENS


def test_still_raises_when_even_the_floor_margin_cannot_fit() -> None:
    """Shrinking the cushion doesn't make an impossible prompt possible."""
    with pytest.raises(PromptBudgetError, match="window remain"):
        budget_max_tokens(context_window=1000, prompt_tokens=990, requested_max=512)
