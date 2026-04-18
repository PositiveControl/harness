"""Character-load shape tests for ab (airton_b) — harness-inj.1.

Verifies the persona-data files load cleanly with the expected surface
and that ab's identity is distinct from Airton's (no accidental copy).
Voice corpus is stubbed by design in this task; harness-inj.2 fills it.
"""

from __future__ import annotations

from pathlib import Path

from harness.character import load_character

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON = REPO_ROOT / "character" / "airton"
AIRTON_B = REPO_ROOT / "character" / "airton_b"


def test_load_airton_b_shape() -> None:
    ab = load_character(AIRTON_B)

    assert ab.name == "airton_b"
    assert ab.pronouns == "it"
    assert ab.era == "personal operations lead"
    assert ab.relationship["mark"] == "operations steward"
    assert len(ab.values) == 7
    assert len(ab.taboos) == 8
    assert len(ab.directives) >= 4
    assert len(ab.seed_memories) == 5
    assert ab.constitution
    assert ab.premise
    assert ab.self_awareness
    assert ab.on_being_wrong


def test_ab_voice_corpus_is_stubbed() -> None:
    """harness-inj.2 fills the corpus. Until then canonical + captured
    are empty; character load must still succeed."""
    ab = load_character(AIRTON_B)
    assert ab.canonical_voice_count == 0
    assert ab.captured_voice_count == 0
    assert len(ab.voice_samples) == 0


def test_ab_seed_principles_match_values() -> None:
    """Each seed memory's principle must match one of ab's value rules
    verbatim. Ties the formative experiences back to the rule they teach."""
    ab = load_character(AIRTON_B)
    value_rules = {v.rule for v in ab.values}
    for memory in ab.seed_memories:
        assert memory.principle in value_rules, (
            f"seed {memory.id!r} principle {memory.principle!r} is not in ab's value rules"
        )


def test_ab_seed_principle_coverage() -> None:
    """The five seed memories must cover five distinct value rules —
    no two memories teaching the same lesson."""
    ab = load_character(AIRTON_B)
    principles = [m.principle for m in ab.seed_memories]
    assert len(principles) == len(set(principles))


def test_ab_distinct_from_airton() -> None:
    """ab and Airton must not share taboos or premises verbatim."""
    airton = load_character(AIRTON)
    ab = load_character(AIRTON_B)
    assert airton.premise != ab.premise
    assert not (set(airton.taboos) & set(ab.taboos))
    assert not ({v.rule for v in airton.values} & {v.rule for v in ab.values})
