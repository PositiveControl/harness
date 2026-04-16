from __future__ import annotations

from pathlib import Path

from harness.character import load_character
from harness.evals.voice import run_voice_eval
from harness.model.echo import EchoAdapter

REPO = Path(__file__).resolve().parents[1]
AIRTON = REPO / "character" / "airton"


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
