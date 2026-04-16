from __future__ import annotations

import re
from typing import TYPE_CHECKING

from harness.model.adapter import ChatMessage

if TYPE_CHECKING:
    from harness.character import Character
    from harness.model.adapter import ModelAdapter


_JUDGE_SYSTEM_TEMPLATE = """\
You are a voice-matching judge for a character named {name}. You will
be shown a CHARACTER SHEET describing how {name} talks, a GOLD response
that IS {name}'s voice, and an ACTUAL model output attempting to match
it. Score how close ACTUAL is to GOLD on a scale of 1 to 10:

  1-2: completely off — reads like a generic assistant; no {name}
       register at all.
  3-4: recognizable attempt, but fundamental tells remain (assistant
       openers, tutorial shape, excessive length).
  5-6: right register in places but key moves missing; substance
       preserved but voice diluted.
  7-8: strong match on length, openers, structure. Minor filler or
       one or two of {name}'s specific turns of phrase missing.
  9-10: indistinguishable from gold on voice; includes {name}'s
        specific punchlines and concrete actions where relevant.

Respond with ONLY a single integer between 1 and 10. No explanation,
no preamble, no trailing text."""


_NUMBER_PATTERN = re.compile(r"\b(10|[1-9])\b")


def judge_voice(
    adapter: ModelAdapter,
    character: Character,
    *,
    actual: str,
    gold: str,
    prompt: str,
    max_tokens: int = 8,
) -> int | None:
    """Ask a model to score voice-match 1-10. Returns the parsed integer
    or None if parsing failed. Pure — no side effects beyond the
    adapter call.

    For now we reuse the same adapter that generated the response being
    judged. This is circular in principle (same priors produce and
    evaluate), but still catches egregious drift because the judge is
    grounded in the character sheet, not just its own defaults. A
    future refinement is a dedicated smaller judge model or a
    different-family cloud adapter."""
    system = ChatMessage(
        role="system",
        content=_JUDGE_SYSTEM_TEMPLATE.format(name=character.name),
    )
    user_content = (
        f"CHARACTER SHEET:\n{character.system_prompt()}\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"GOLD:\n{gold.strip()}\n\n"
        f"ACTUAL:\n{actual.strip()}\n\n"
        "Score 1-10 (integer only):"
    )
    user = ChatMessage(role="user", content=user_content)

    response = adapter.complete([system, user], temperature=0.0, max_tokens=max_tokens)
    match = _NUMBER_PATTERN.search(response.strip())
    if match is None:
        return None
    score = int(match.group(1))
    if not 1 <= score <= 10:
        return None
    return score
