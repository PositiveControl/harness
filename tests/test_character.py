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
    assert len(character.voice_samples) == 6
    assert all(s.principle for s in character.seed_memories)


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
