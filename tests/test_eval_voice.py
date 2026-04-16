from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.character import load_character
from harness.evals.voice import run_voice_eval
from harness.model.adapter import ChatMessage
from harness.model.echo import EchoAdapter

REPO = Path(__file__).resolve().parents[1]
AIRTON = REPO / "character" / "airton"


@dataclass
class _RecordingAdapter:
    """Captures the system prompt passed to each call so tests can assert
    on what the model actually saw."""

    id: str = "recording"
    context_window: int = 8192
    seen_system_prompts: list[str] = field(default_factory=list)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        for m in messages:
            if m.role == "system":
                self.seen_system_prompts.append(m.content)
                break
        return "ok"


def test_voice_eval_produces_one_result_per_sample() -> None:
    character = load_character(AIRTON)
    results = run_voice_eval(character, EchoAdapter())

    assert len(results) == len(character.voice_samples)
    sample_ids = {s.id for s in character.voice_samples}
    assert {r.sample_id for r in results} == sample_ids


def test_voice_eval_results_carry_prompt_gold_and_actual() -> None:
    character = load_character(AIRTON)
    results = run_voice_eval(character, EchoAdapter())

    for r in results:
        assert r.prompt
        assert r.gold
        assert r.actual.startswith("[echo] ")
        assert r.prompt in r.actual


def test_voice_eval_respects_sample_id_filter() -> None:
    character = load_character(AIRTON)
    results = run_voice_eval(character, EchoAdapter(), sample_ids=["self_reference"])

    assert len(results) == 1
    assert results[0].sample_id == "self_reference"


def test_voice_eval_empty_filter_returns_nothing() -> None:
    character = load_character(AIRTON)
    results = run_voice_eval(character, EchoAdapter(), sample_ids=["does_not_exist"])

    assert results == []


def test_leave_one_out_excludes_current_sample_gold() -> None:
    character = load_character(AIRTON)
    adapter = _RecordingAdapter()
    target = next(s for s in character.voice_samples if s.id == "self_reference")

    run_voice_eval(character, adapter, sample_ids=["self_reference"], leave_one_out=True)

    assert len(adapter.seen_system_prompts) == 1
    prompt = adapter.seen_system_prompts[0]
    assert target.gold.strip() not in prompt
    for other in character.voice_samples:
        if other.id == "self_reference":
            continue
        assert other.gold.strip() in prompt


def test_no_leave_one_out_includes_every_sample() -> None:
    character = load_character(AIRTON)
    adapter = _RecordingAdapter()

    run_voice_eval(character, adapter, sample_ids=["self_reference"], leave_one_out=False)

    assert len(adapter.seen_system_prompts) == 1
    prompt = adapter.seen_system_prompts[0]
    for sample in character.voice_samples:
        assert sample.gold.strip() in prompt
