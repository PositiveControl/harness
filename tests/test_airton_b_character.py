"""Character-load shape tests for ab (airton_b) — harness-inj.1 +
harness-inj.2.

Verifies the persona-data files load cleanly with the expected surface,
that ab's identity is distinct from Airton's (no accidental copy), and
that the voice corpus meets the shipped minimum (26 gold samples
covering every taboo). A later pass (live voice capture) will accrue
additional samples into voice/captured.yaml.
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
    assert len(ab.taboos) == 9
    assert len(ab.directives) >= 4
    assert len(ab.seed_memories) == 5
    assert ab.constitution
    assert ab.premise
    assert ab.self_awareness
    assert ab.on_being_wrong


def test_ab_voice_corpus_ships_minimum() -> None:
    """ab's canonical corpus must carry at least 26 gold samples — the
    v1 spec allocation across plan / capture / re-plan / drift / status
    / clarifying / auto-clarity / retraction / register-stress. Live
    captures accrue separately into voice/captured.yaml over time."""
    ab = load_character(AIRTON_B)
    assert ab.canonical_voice_count >= 26
    assert ab.captured_voice_count == 0
    assert len(ab.voice_samples) >= 26


def test_ab_voice_samples_have_unique_ids() -> None:
    ab = load_character(AIRTON_B)
    ids = [s.id for s in ab.voice_samples]
    assert len(ids) == len(set(ids))


def test_ab_voice_corpus_covers_surfaces() -> None:
    """The corpus must include at least one sample per first-class
    surface so the voice eval exercises every register intensity the
    rewriter supports."""
    ab = load_character(AIRTON_B)
    ids = {s.id for s in ab.voice_samples}
    required_prefixes = {
        "plan_",
        "capture_",
        "replan_",
        "drift_",
        "status_",
        "ambiguous_",
        "confirm_",
        "retraction_",
        "stress_",
    }
    for prefix in required_prefixes:
        assert any(sample_id.startswith(prefix) for sample_id in ids), (
            f"no sample with id prefix {prefix!r} — register coverage gap"
        )


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
