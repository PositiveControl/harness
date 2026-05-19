"""geography — country / region lookup over the harness gazetteer.

Companion to the ScopeViolationHook (harness-lyyr): exposes the same
REGION_COUNTRIES data through a model-callable tool so the model can
verify scope PROACTIVELY before committing to an answer instead of
being caught by the bail catcher after.

Two call shapes (exactly one of `country` / `region` required):

  geography(country='Mexico')
    -> {
         country: 'Mexico',
         in_regions: ['central america', 'latin america'],
       }

  geography(region='south america')
    -> {
         region: 'south america',
         countries: ['Argentina', 'Bolivia', 'Brazil', ...],
         count: 14,
       }

Read-tier, deterministic, no network, no compiled deps. Schema cost
~150 tokens. Tagged as discovery so tool_search surfaces it for
'is X in Y' / 'what countries are in Y' shaped questions.

For the broader country-attribute lookup (alpha2/alpha3 codes,
capital, neighbors, area, population), see the follow-up bead — this
ships the minimal region-lookup surface that pairs with the catcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.geography import REGION_COUNTRIES, countries_in_region
from harness.tools.base import ToolSpec


def _country_lookup(country: str) -> dict[str, Any]:
    """Return the regions a country belongs to, plus a canonical
    spelling. Unknown country: in_regions is empty + a hint message."""
    country_lower = country.strip().lower()
    matches: list[str] = []
    canonical: str | None = None
    for region, members in REGION_COUNTRIES.items():
        for member in members:
            if member.lower() == country_lower:
                matches.append(region)
                if canonical is None:
                    canonical = member
                break
    if canonical is None:
        return {
            "country": country,
            "in_regions": [],
            "note": (
                f"'{country}' is not in the gazetteer. The harness "
                f"covers ~250 commonly-named countries; obscure "
                f"microstates or dependencies may be missing. Try a "
                f"different spelling or use search_web for verification."
            ),
        }
    # Deduplicate and sort for stable output (a country can appear in
    # multiple regions, e.g. Mexico in central america + latin america).
    unique = sorted(set(matches))
    return {"country": canonical, "in_regions": unique}


def _region_lookup(region: str) -> dict[str, Any]:
    """Return the country list for a region, plus the count. Unknown
    region: countries is empty + a hint listing known regions."""
    region_lower = region.strip().lower()
    members = countries_in_region(region_lower)
    if not members:
        known = sorted(REGION_COUNTRIES.keys())
        return {
            "region": region,
            "countries": [],
            "count": 0,
            "note": (f"'{region}' is not a known region phrase. Known: {', '.join(known)}."),
        }
    return {
        "region": region_lower,
        "countries": sorted(members),
        "count": len(members),
    }


def _format_country_result(result: dict[str, Any]) -> str:
    if result.get("note"):
        return f"country={result['country']!r}: not in gazetteer.\n{result['note']}"
    regions = result["in_regions"]
    if not regions:
        return f"country={result['country']!r}: no regions matched."
    return f"country={result['country']!r} -> in_regions: {', '.join(regions)}"


def _format_region_result(result: dict[str, Any]) -> str:
    if result.get("note"):
        return f"region={result['region']!r}: unknown.\n{result['note']}"
    countries = result["countries"]
    count = result["count"]
    # Cap the visible list at 60 to keep tool output token-bounded while
    # showing every entry for the largest region (Africa has ~58 entries).
    # The count line tells the model the total regardless.
    visible = countries[:60]
    listing = "\n".join(f"  - {c}" for c in visible)
    overflow = "" if len(countries) <= 60 else f"\n  ... (+{len(countries) - 60} more)"
    return f"region={result['region']!r} (count={count}):\n{listing}{overflow}"


@dataclass
class GeographyTool:
    """Look up which region(s) a country belongs to, or which
    countries are in a region. Static gazetteer; no network."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="geography",
            description=(
                "Country/region lookup over a static gazetteer (UN M49 "
                "aligned). Call with EXACTLY ONE of `country` or "
                "`region`. country='Mexico' returns the list of regions "
                "Mexico is in (e.g. 'central america', 'latin america'); "
                "region='south america' returns the countries in South "
                "America. Use this BEFORE answering 'top X in region Y' "
                "questions to verify that your pick is actually in Y. "
                "No network — purely deterministic data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": (
                            "A country name (e.g. 'Mexico', 'Peru', "
                            "'United States', 'USA'). Common short forms "
                            "are accepted."
                        ),
                    },
                    "region": {
                        "type": "string",
                        "description": (
                            "A region phrase (e.g. 'south america', "
                            "'north america', 'africa', 'middle east', "
                            "'scandinavia'). Adjective forms ('south "
                            "american') also work."
                        ),
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Geography",
        )

    def call(
        self,
        *,
        country: str | None = None,
        region: str | None = None,
    ) -> str:
        country = (country or "").strip()
        region = (region or "").strip()
        if bool(country) == bool(region):
            raise ValueError(
                "geography: provide EXACTLY ONE of `country` or `region` (not both, not neither)"
            )
        if country:
            return _format_country_result(_country_lookup(country))
        return _format_region_result(_region_lookup(region))
