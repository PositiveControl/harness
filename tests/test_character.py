from __future__ import annotations

from pathlib import Path

from harness.character import load_character

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON = REPO_ROOT / "character" / "airton"


def test_load_airton_shape() -> None:
    character = load_character(AIRTON)

    assert character.name == "airton"
    assert character.pronouns == "it"
    assert len(character.values) == 5
    assert len(character.taboos) >= 6
    assert len(character.seed_memories) == 5
    assert len(character.voice_samples) >= 20
    assert all(s.principle for s in character.seed_memories)


def test_voice_samples_have_unique_ids() -> None:
    character = load_character(AIRTON)
    ids = [s.id for s in character.voice_samples]
    assert len(ids) == len(set(ids))


def test_self_awareness_is_explicit() -> None:
    character = load_character(AIRTON)
    assert "program on your Mac" in character.self_awareness


def test_self_reference_voice_sample_present() -> None:
    character = load_character(AIRTON)
    ids = {s.id for s in character.voice_samples}
    assert "self_reference" in ids


def test_system_prompt_includes_values_and_taboos() -> None:
    character = load_character(AIRTON)
    prompt = character.system_prompt()

    for v in character.values:
        assert v.rule in prompt
    for t in character.taboos:
        assert t in prompt
    assert character.premise in prompt


def test_system_prompt_embeds_voice_examples() -> None:
    character = load_character(AIRTON)
    prompt = character.system_prompt()

    for sample in character.voice_samples:
        assert sample.prompt in prompt
        assert sample.gold.strip() in prompt
    assert "Voice examples" in prompt
    assert "How you speak" in prompt


def test_system_prompt_excludes_named_voice_examples() -> None:
    character = load_character(AIRTON)
    excluded = frozenset({"self_reference"})
    prompt = character.system_prompt(exclude_example_ids=excluded)

    excluded_sample = next(s for s in character.voice_samples if s.id == "self_reference")
    assert excluded_sample.gold.strip() not in prompt

    for sample in character.voice_samples:
        if sample.id in excluded:
            continue
        assert sample.gold.strip() in prompt
