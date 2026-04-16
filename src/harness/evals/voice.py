from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from harness.character import Character
from harness.model.adapter import ChatMessage, ModelAdapter


@dataclass(frozen=True)
class VoiceEvalResult:
    sample_id: str
    prompt: str
    gold: str
    actual: str


def run_voice_eval(
    character: Character,
    adapter: ModelAdapter,
    *,
    temperature: float = 0.5,
    sample_ids: Iterable[str] | None = None,
) -> list[VoiceEvalResult]:
    """Run the canonical voice prompts through an adapter and pair each
    model reply with the gold response. Pure; the only side effect is the
    adapter call itself.

    Temperature defaults to 0.5 — voice evaluation wants consistency, not
    creativity."""
    selected = (
        character.voice_samples
        if sample_ids is None
        else tuple(s for s in character.voice_samples if s.id in set(sample_ids))
    )
    system = ChatMessage(role="system", content=character.system_prompt())
    results: list[VoiceEvalResult] = []
    for sample in selected:
        user = ChatMessage(role="user", content=sample.prompt)
        actual = adapter.complete([system, user], temperature=temperature)
        results.append(
            VoiceEvalResult(
                sample_id=sample.id,
                prompt=sample.prompt,
                gold=sample.gold,
                actual=actual,
            )
        )
    return results
