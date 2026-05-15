"""Character-load shape tests for airton_g (the reckoner) —
harness-1u2h/cvg8.

Verifies the persona files load cleanly with the expected shape, that
airton_g's identity is distinct from siblings, and that the voice
corpus ships the documented stub set.
"""

from __future__ import annotations

from pathlib import Path

from harness.character import load_character

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON_G = REPO_ROOT / "character" / "airton_g"


def test_load_airton_g_shape() -> None:
    g = load_character(AIRTON_G)

    assert g.name == "airton_g"
    assert g.pronouns == "it"
    assert g.era == "the desk calculator"
    assert g.relationship["mark"] == "operator"
    assert len(g.values) == 5
    assert len(g.taboos) == 7
    assert len(g.directives) >= 4
    assert len(g.seed_memories) == 4
    assert g.constitution
    assert g.premise
    assert g.self_awareness
    assert g.on_being_wrong


def test_airton_g_values_include_load_bearing_ones() -> None:
    """The four pillars (units, timezones, no-fabrication, show-work)
    must be present — the constitution + tools assume them."""
    g = load_character(AIRTON_G)
    value_ids = {v.id for v in g.values}
    assert {
        "units_attached",
        "dates_with_timezones",
        "no_fabricated_numbers",
        "show_your_work",
    } <= value_ids


def test_airton_g_voice_corpus_ships_stubs() -> None:
    """airton_g ships minimum-viable voice scaffolding. Stub count is
    pinned so a future PR can't accidentally drop the documented
    samples without updating this test."""
    g = load_character(AIRTON_G)
    assert len(g.voice_samples) >= 6


def test_airton_g_is_not_a_copy_of_airton_d() -> None:
    """Guard against accidental sibling-paste — premise should be
    reckoning, not notes."""
    g = load_character(AIRTON_G)
    assert "calculator" in g.premise.lower() or "reckon" in g.premise.lower()
    assert "notes" not in g.premise.lower()
