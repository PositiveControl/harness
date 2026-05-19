"""Tests for the GeographyTool — harness-e83u.

Companion to ScopeViolationHook (harness-lyyr): the tool exposes the
same gazetteer as the catcher so the model can verify scope before
answering instead of being caught after.
"""

from __future__ import annotations

import pytest

from harness.tools import GeographyTool, ToolSpec

# --- spec --------------------------------------------------------------


def test_spec_shape() -> None:
    spec = GeographyTool().spec
    assert isinstance(spec, ToolSpec)
    assert spec.name == "geography"
    assert spec.tier == "read"
    props = spec.parameters["properties"]
    assert set(props) == {"country", "region"}
    # Neither is required by JSONSchema — the tool itself enforces
    # mutual exclusion at call-time.
    assert spec.parameters["required"] == []


def test_spec_description_calls_out_proactive_verification() -> None:
    """The description must explicitly tell the model to verify scope
    BEFORE answering — that's the load-bearing use case."""
    desc = GeographyTool().spec.description.lower()
    assert "before answering" in desc
    assert "region" in desc
    assert "country" in desc


# --- country lookup ----------------------------------------------------


def test_country_lookup_returns_regions_for_mexico() -> None:
    out = GeographyTool().call(country="Mexico")
    assert "Mexico" in out
    assert "central america" in out
    assert "latin america" in out
    # NOT a generic 'not in gazetteer' fallback.
    assert "not in gazetteer" not in out


def test_country_lookup_case_insensitive() -> None:
    out_upper = GeographyTool().call(country="MEXICO")
    out_lower = GeographyTool().call(country="mexico")
    out_mixed = GeographyTool().call(country="MeXiCo")
    # All return canonical 'Mexico'.
    for out in (out_upper, out_lower, out_mixed):
        assert "Mexico" in out
        assert "latin america" in out


def test_country_lookup_handles_short_forms() -> None:
    """USA / US / United States all resolve."""
    for variant in ("United States", "USA", "US"):
        out = GeographyTool().call(country=variant)
        assert "not in gazetteer" not in out, f"variant {variant!r} should resolve"
        assert "north america" in out


def test_country_lookup_unknown_country_reports_clearly() -> None:
    out = GeographyTool().call(country="Atlantis")
    assert "not in gazetteer" in out
    assert "Atlantis" in out


# --- region lookup -----------------------------------------------------


def test_region_lookup_returns_countries_for_south_america() -> None:
    out = GeographyTool().call(region="south america")
    # Expected SA members surface.
    for country in ("Peru", "Brazil", "Chile", "Argentina"):
        assert country in out, f"{country} should be in south america list"
    # Count line present.
    assert "count=" in out
    # Mexico is NOT in strict UN M49 South America.
    assert "Mexico" not in out


def test_region_lookup_handles_adjective_form() -> None:
    """'south american' as the region key should work just like
    'south america'."""
    out_noun = GeographyTool().call(region="south america")
    out_adj = GeographyTool().call(region="south american")
    # Both resolve to the same set — just compare count.
    import re

    count_noun = re.search(r"count=(\d+)", out_noun)
    count_adj = re.search(r"count=(\d+)", out_adj)
    assert count_noun is not None
    assert count_adj is not None
    assert count_noun.group(1) == count_adj.group(1)


def test_region_lookup_handles_latin_america() -> None:
    """Latin America is the broader grouping (Mexico + Central America
    + South America + Caribbean). Mexico must appear here even though
    it doesn't appear in strict South America."""
    out = GeographyTool().call(region="latin america")
    assert "Mexico" in out
    assert "Peru" in out  # South America is a subset
    assert "Cuba" in out  # Caribbean is a subset


def test_region_lookup_unknown_region_reports_clearly() -> None:
    out = GeographyTool().call(region="atlantis")
    assert "unknown" in out.lower()
    # Hint includes a list of known regions.
    assert "south america" in out


# --- mutual exclusion --------------------------------------------------


def test_neither_arg_raises() -> None:
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        GeographyTool().call()


def test_both_args_raises() -> None:
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        GeographyTool().call(country="Peru", region="south america")


def test_empty_strings_raise() -> None:
    """Empty strings on both args — treat as 'neither provided', raise
    the same mutual-exclusion error so the model can self-correct."""
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        GeographyTool().call(country="", region="")


# --- end-to-end ScopeViolationHook integration -------------------------


def test_geography_tool_data_matches_hook_gazetteer() -> None:
    """The tool and the ScopeViolationHook must consume the SAME data
    so the tool's answers and the catcher's verdicts agree. Smoke test
    by spot-checking a case the catcher targets (Mexico ∉ SA)."""
    from harness.geography import REGION_COUNTRIES, in_region

    # Hook-side facts.
    assert not in_region("Mexico", "south america")
    assert in_region("Mexico", "latin america")

    # Tool-side answers reflect the same.
    out_mexico = GeographyTool().call(country="Mexico")
    assert "south america" not in out_mexico
    assert "latin america" in out_mexico

    # Region lookup parity.
    out_sa = GeographyTool().call(region="south america")
    for country in REGION_COUNTRIES["south america"]:
        assert country in out_sa, (
            f"tool's south america list missing {country!r} from REGION_COUNTRIES"
        )
