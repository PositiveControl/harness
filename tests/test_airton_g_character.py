"""Character-load shape tests for airton_g (the reckoner) —
harness-1u2h/cvg8, slimmed under harness-2fr1.

The slimmed shape (post-harness-2fr1) is intentional: Qwen 2.5-7B
was overruled by a long prose-form constitution and started parroting
example replies instead of emitting tool calls. The fix dropped most
of the persona surface area; tool schemas + chat-template carry the
load. These tests pin the new minimum.
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
    # Slimmed shape: 2 values, 3 taboos, 1 directive (harness-2fr1).
    assert len(g.values) == 2
    assert len(g.taboos) == 3
    assert len(g.directives) >= 1
    assert len(g.seed_memories) == 4
    assert g.constitution
    assert g.premise
    assert g.self_awareness
    assert g.on_being_wrong


def test_airton_g_values_include_load_bearing_ones() -> None:
    """The two slim-shape values must be present — the constitution
    + tool schemas assume them."""
    g = load_character(AIRTON_G)
    value_ids = {v.id for v in g.values}
    assert {"tool_or_silence", "units_attached"} <= value_ids


def test_airton_g_voice_corpus_intentionally_empty_or_small() -> None:
    """Voice samples are blanked while we work out why Qwen 7B parrots
    them as completed-turn templates instead of emitting tool calls.
    Test pins the upper bound, not the lower, so re-adding curated
    samples in a later PR doesn't trip the test."""
    g = load_character(AIRTON_G)
    assert len(g.voice_samples) <= 6


def test_airton_g_is_not_a_copy_of_airton_d() -> None:
    """Guard against accidental sibling-paste — premise should be
    reckoning, not notes."""
    g = load_character(AIRTON_G)
    assert "calculator" in g.premise.lower() or "reckon" in g.premise.lower()
    assert "notes" not in g.premise.lower()
