"""Tool-loop banter intercept tests (harness-vadq).

Verifies that `run_tool_loop` short-circuits before any model round
when a banter-shaped user prompt arrives and a tracker is attached.
The tracker owns the joke / redirect cycle; the loop's only job here
is to consume one tier and bail.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from harness.model.adapter import ChatMessage
from harness.orchestrator import run_tool_loop
from harness.persona.banter import (
    REDIRECT_LADDER,
    BanterCorpus,
    BanterStreakTracker,
    JokeEntry,
)
from harness.tools import ModelReply, ToolRegistry, ToolSpec


@dataclass
class _ScriptedAdapter:
    """Returns pre-queued ModelReply objects in order."""

    replies: list[ModelReply]
    calls_seen: list[list[ChatMessage]] = field(default_factory=list)

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        self.calls_seen.append(list(messages))
        return self.replies.pop(0) if self.replies else ModelReply(content="model ran")


def _two_joke_tracker() -> BanterStreakTracker:
    return BanterStreakTracker(
        corpus=BanterCorpus(
            jokes=(
                JokeEntry(id="alpha", text="alpha joke"),
                JokeEntry(id="beta", text="beta joke"),
            )
        ),
        rng=random.Random(42),  # noqa: S311 — deterministic non-crypto pick
    )


# ---------- intercept fires ----------


def test_banter_prompt_short_circuits_before_model_round() -> None:
    adapter = _ScriptedAdapter(replies=[ModelReply(content="model should NOT run")])
    tracker = _two_joke_tracker()
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="this page intentionally left blank")],
        ToolRegistry(),
        banter_tracker=tracker,
    )
    # Joke composed; model never invoked.
    assert result.content in {"alpha joke", "beta joke"}
    assert result.rounds == 0
    assert adapter.calls_seen == []
    assert tracker.streak == 1


def test_banter_intercept_cycles_through_joke_then_redirects() -> None:
    adapter = _ScriptedAdapter(replies=[])
    tracker = _two_joke_tracker()
    outputs = []
    for _ in range(4):
        result = run_tool_loop(
            adapter,
            [ChatMessage(role="user", content="test")],
            ToolRegistry(),
            banter_tracker=tracker,
        )
        outputs.append(result.content)
    assert outputs[0] in {"alpha joke", "beta joke"}
    assert outputs[1] == REDIRECT_LADDER[0]
    assert outputs[2] == REDIRECT_LADDER[1]
    assert outputs[3] == REDIRECT_LADDER[2]
    # Adapter never ran.
    assert adapter.calls_seen == []


def test_banter_intercept_returns_zero_round_result() -> None:
    adapter = _ScriptedAdapter(replies=[])
    tracker = _two_joke_tracker()
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="ping")],
        ToolRegistry(),
        banter_tracker=tracker,
    )
    assert result.rounds == 0
    assert result.events == []
    assert result.tool_results == []


# ---------- intercept does NOT fire ----------


def test_real_prompt_falls_through_to_model() -> None:
    adapter = _ScriptedAdapter(replies=[ModelReply(content="real reply")])
    tracker = _two_joke_tracker()
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="explain wake turbulence separation minima")],
        ToolRegistry(),
        banter_tracker=tracker,
    )
    assert result.content == "real reply"
    assert result.rounds == 1
    assert len(adapter.calls_seen) == 1


def test_real_prompt_resets_tracker_streak() -> None:
    adapter = _ScriptedAdapter(replies=[ModelReply(content="real reply")])
    tracker = _two_joke_tracker()
    # Two banter prompts → streak=2.
    for _ in range(2):
        run_tool_loop(
            adapter,
            [ChatMessage(role="user", content="test")],
            ToolRegistry(),
            banter_tracker=tracker,
        )
    assert tracker.streak == 2
    # Real prompt → tracker resets.
    run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="how does wake turbulence separation work")],
        ToolRegistry(),
        banter_tracker=tracker,
    )
    assert tracker.streak == 0
    # Next banter prompt → joke (not redirect tier 3).
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="ping")],
        ToolRegistry(),
        banter_tracker=tracker,
    )
    assert result.content in {"alpha joke", "beta joke"}


def test_no_tracker_means_no_intercept() -> None:
    """Backward compat: tracker omitted → loop behaves as before."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="model ran")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="this page intentionally left blank")],
        ToolRegistry(),
    )
    assert result.content == "model ran"
    assert result.rounds == 1
    assert len(adapter.calls_seen) == 1


def test_no_user_message_means_no_intercept() -> None:
    """Edge case: system-only thread (no user turn yet) skips intercept."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="model ran")])
    tracker = _two_joke_tracker()
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="system", content="you are airton")],
        ToolRegistry(),
        banter_tracker=tracker,
    )
    assert result.content == "model ran"
    assert tracker.streak == 0  # never consumed


def test_greeting_falls_through_to_model_not_intercept() -> None:
    """Greetings get a separate path (return False from is_banter_prompt),
    so the model handles them — no joke."""
    adapter = _ScriptedAdapter(replies=[ModelReply(content="morning")])
    tracker = _two_joke_tracker()
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="morning")],
        ToolRegistry(),
        banter_tracker=tracker,
    )
    assert result.content == "morning"
    assert tracker.streak == 0
