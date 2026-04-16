from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.character import load_character
from harness.evals.voice_judge import judge_voice
from harness.model.adapter import ChatMessage

REPO = Path(__file__).resolve().parents[1]
AIRTON = REPO / "character" / "airton"


@dataclass
class _CannedAdapter:
    """Returns a pre-set reply on every call. Used to drive the judge
    without loading a real model."""

    reply: str = "8"
    id: str = "canned"
    context_window: int = 8192
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(list(messages))
        return self.reply


def test_judge_voice_parses_integer_reply() -> None:
    character = load_character(AIRTON)
    adapter = _CannedAdapter(reply="7")

    score = judge_voice(
        adapter,
        character,
        actual="some reply",
        gold="some gold",
        prompt="some prompt",
    )
    assert score == 7


def test_judge_voice_parses_integer_with_surrounding_text() -> None:
    character = load_character(AIRTON)
    adapter = _CannedAdapter(reply=" 9 ")
    assert judge_voice(adapter, character, actual="a", gold="b", prompt="c") == 9


def test_judge_voice_returns_none_on_unparseable_reply() -> None:
    character = load_character(AIRTON)
    adapter = _CannedAdapter(reply="not a number at all")
    assert judge_voice(adapter, character, actual="a", gold="b", prompt="c") is None


def test_judge_voice_rejects_out_of_range_bare_number() -> None:
    character = load_character(AIRTON)
    # "42" has no 1-10 substring at a word boundary, so the regex finds
    # no match and judge_voice returns None.
    adapter = _CannedAdapter(reply="42")
    assert judge_voice(adapter, character, actual="a", gold="b", prompt="c") is None


def test_judge_voice_extracts_first_in_range_integer() -> None:
    character = load_character(AIRTON)
    # When the model volunteers context, the regex still picks up the
    # first in-range integer.
    adapter = _CannedAdapter(reply="Solid match — 8 out of 10.")
    assert judge_voice(adapter, character, actual="a", gold="b", prompt="c") == 8


def test_judge_voice_includes_all_context_in_user_message() -> None:
    character = load_character(AIRTON)
    adapter = _CannedAdapter(reply="6")

    judge_voice(
        adapter,
        character,
        actual="ACTUAL_SENTINEL",
        gold="GOLD_SENTINEL",
        prompt="PROMPT_SENTINEL",
    )

    assert len(adapter.calls) == 1
    user_msg = next(m for m in adapter.calls[0] if m.role == "user")
    assert "ACTUAL_SENTINEL" in user_msg.content
    assert "GOLD_SENTINEL" in user_msg.content
    assert "PROMPT_SENTINEL" in user_msg.content
