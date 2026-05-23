"""Tests for the drive-executor system prompt (harness-d6ak).

These pin the contract that the drive loop's executor uses the
character-agnostic tool-engineer prompt — not the chat persona's
``character.system_prompt()`` — so the persona's style rules ("prose
by default", "1-4 sentences", "say 'Don't know' plainly") cannot
prime the model to narrate rather than call tools.
"""

from __future__ import annotations

from pathlib import Path

from harness.character import load_character
from harness.driver.system_prompt import EXECUTOR_SYSTEM_PROMPT

_REPO = Path(__file__).resolve().parents[1]


def test_executor_prompt_demands_a_tool_call_per_turn() -> None:
    """The headline contract — every turn must produce at least one
    substantive tool call. The model should see this as RULE 1."""
    assert "CALLING TOOLS" in EXECUTOR_SYSTEM_PROMPT
    assert "at least one substantive tool call" in EXECUTOR_SYSTEM_PROMPT


def test_executor_prompt_explicitly_forbids_preamble() -> None:
    """The pathology that motivated this prompt (harness-d6ak)
    was the model opening every turn with 'Let me first check…' /
    'I need to create…' / 'I'll help you drive…'. The prompt
    names those exact opening patterns as forbidden."""
    assert "Let me first check" in EXECUTOR_SYSTEM_PROMPT
    assert "I need to create" in EXECUTOR_SYSTEM_PROMPT
    assert "I'll help you drive" in EXECUTOR_SYSTEM_PROMPT


def test_executor_prompt_overrides_competing_style_guidance() -> None:
    """Other system messages in the turn (a leftover persona prompt,
    a future shared style block, etc.) MUST yield to these rules.
    Without the explicit override clause the small model could
    resolve the conflict in favor of the larger persona block, which
    is the exact failure mode this prompt is here to prevent."""
    assert "override" in EXECUTOR_SYSTEM_PROMPT.lower()


def test_executor_prompt_directs_close_via_bd() -> None:
    """When the work is done the model should close the bd issue via
    shell, not narrate completion. The driver enforces this on its
    end too (bd-show post-turn check); the prompt makes it explicit
    on the model's end."""
    assert "bd close" in EXECUTOR_SYSTEM_PROMPT


def test_executor_prompt_demands_concrete_blocker_over_silence() -> None:
    """When stuck the model must NAME the blocker in one sentence —
    not stay silent and not narrate a plan it can't execute. The
    empty_reply_after_tools catcher (harness-5zjj) is the defensive
    layer; this is the preventive layer."""
    assert "blocker" in EXECUTOR_SYSTEM_PROMPT.lower()
    assert "do not stay silent" in EXECUTOR_SYSTEM_PROMPT.lower()


def test_executor_prompt_does_not_carry_persona_style_rules() -> None:
    """harness-d6ak smoking gun: the persona's style rules ("prose
    by default", "1-4 sentences", "say 'Don't know' plainly")
    actively conflict with the agent's role. The drive prompt must
    NOT carry them. The chat character's full prompt still has them
    — that's correct for chat — but they must not leak into drive."""
    persona_anti_patterns = [
        "Prose by default",
        "1-4 sentences",
        'say "Don\'t know" plainly',
        "Voice examples",
    ]
    for anti_pattern in persona_anti_patterns:
        assert anti_pattern not in EXECUTOR_SYSTEM_PROMPT, (
            f"persona style rule leaked into drive prompt: {anti_pattern!r}"
        )


def test_chat_persona_still_carries_style_rules() -> None:
    """Symmetric check: the chat persona path is unchanged. Voice
    eval + chat REPL still get the full style guidance — the drive
    prompt is a NEW path, not a replacement."""
    airton = load_character(_REPO / "character" / "airton")
    persona = airton.system_prompt(include_samples=())
    # The exact phrases the drive prompt explicitly omits.
    assert "Prose by default" in persona
    assert "1-4 sentences" in persona
