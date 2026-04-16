from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.character import Character
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.persona.rewriter import build_rewriter_messages

if TYPE_CHECKING:
    from harness.retrieval.voice_retriever import VoiceRetriever


@dataclass(frozen=True)
class VoiceEvalResult:
    sample_id: str
    prompt: str
    gold: str
    actual: str
    draft: str | None = None  # pass-1 output when persona is on; None otherwise


def run_voice_eval(
    character: Character,
    adapter: ModelAdapter,
    *,
    temperature: float = 0.5,
    sample_ids: Iterable[str] | None = None,
    leave_one_out: bool = True,
    persona: bool = False,
    rewriter_temperature: float = 0.2,
    retriever: VoiceRetriever | None = None,
    top_k: int = 6,
) -> list[VoiceEvalResult]:
    """Run the canonical voice prompts through an adapter and pair each
    model reply with the gold response. Pure; the only side effect is the
    adapter call itself.

    `leave_one_out` (default True) excludes the current sample from its
    own few-shot examples — the fair generalization test. Set to False
    to measure the ceiling: what the model produces with the full set
    in view. Useful for diagnosing whether drift is a few-shot dosing
    issue or a model-register issue.

    `persona` (default False) runs the voice-rewrite post-pass after the
    substance pass. Both passes share the same example selection so the
    eval stays honest end-to-end.

    `retriever` (optional): when provided, each prompt gets its own top-K
    few-shot set via similarity. Without a retriever, all non-excluded
    samples are shown (the pre-retrieval behavior). `top_k` is ignored
    when `retriever` is None.

    Temperature defaults to 0.5 — voice evaluation wants consistency,
    not creativity."""
    selected = (
        character.voice_samples
        if sample_ids is None
        else tuple(s for s in character.voice_samples if s.id in set(sample_ids))
    )
    results: list[VoiceEvalResult] = []
    for sample in selected:
        excluded = frozenset({sample.id}) if leave_one_out else frozenset()

        if retriever is not None:
            retrieved = retriever.top_k(sample.prompt, k=top_k, exclude_ids=excluded)
            prompt = character.system_prompt(include_samples=retrieved)
        else:
            retrieved = None
            prompt = character.system_prompt(exclude_example_ids=excluded)

        system = ChatMessage(role="system", content=prompt)
        user = ChatMessage(role="user", content=sample.prompt)
        draft = adapter.complete([system, user], temperature=temperature)

        if persona:
            if retrieved is not None:
                rewrite_msgs = build_rewriter_messages(character, draft, include_samples=retrieved)
            else:
                rewrite_msgs = build_rewriter_messages(
                    character, draft, exclude_example_ids=excluded
                )
            actual = adapter.complete(rewrite_msgs, temperature=rewriter_temperature)
            results.append(
                VoiceEvalResult(
                    sample_id=sample.id,
                    prompt=sample.prompt,
                    gold=sample.gold,
                    actual=actual,
                    draft=draft,
                )
            )
        else:
            results.append(
                VoiceEvalResult(
                    sample_id=sample.id,
                    prompt=sample.prompt,
                    gold=sample.gold,
                    actual=draft,
                )
            )
    return results
