"""Tests for the tool_search meta-tool — harness-ozx1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from harness.tools import (
    ToolCatalog,
    ToolCatalogEntry,
    ToolRegistry,
    ToolSearchTool,
    ToolSpec,
)


@dataclass
class _StubTool:
    """Minimal Tool — spec + call. Same pattern as test_tool_working_set."""

    _spec: ToolSpec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def call(self, **_: Any) -> str:
        return "ok"


def _seeded_catalog() -> ToolCatalog:
    """Catalog with a mix of reckon, filesystem, and meta tools — gives
    every test a non-trivial corpus to search across."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="now",
            family="reckon",
            tags=("time", "clock", "timezone"),
            description="Return the current wall-clock time.",
        )
    )
    cat.register(
        ToolCatalogEntry(
            name="calc",
            family="reckon",
            tags=("arithmetic", "math", "unit-convert"),
            description="Evaluate arithmetic expressions or unit conversions.",
        )
    )
    cat.register(
        ToolCatalogEntry(
            name="grep",
            family="filesystem",
            tags=("read", "search", "regex"),
            description="Find text in files via regex.",
        )
    )
    cat.register(
        ToolCatalogEntry(
            name="search_web",
            family="research",
            tags=("read", "search", "web"),
            description="DuckDuckGo HTML search across the public web.",
        )
    )
    return cat


# --- spec --------------------------------------------------------------


def test_spec_shape() -> None:
    tool = ToolSearchTool(catalog=ToolCatalog())
    spec = tool.spec
    assert spec.name == "tool_search"
    assert spec.tier == "read"
    props = spec.parameters["properties"]
    assert set(props) == {"query", "tag", "family", "limit"}
    assert spec.parameters["required"] == ["query"]


# --- query path --------------------------------------------------------


def test_query_matches_tool_name() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="calc")
    assert "calc" in out
    assert "arithmetic" in out  # from tags


def test_query_matches_description_substring() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="regex")
    assert "grep" in out


def test_query_matches_tag_substring() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="web")
    assert "search_web" in out


def test_query_is_case_insensitive() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="CLOCK")
    assert "now" in out


def test_query_returns_no_tools_found_on_miss() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="nonexistent-term")
    assert "no tools found" in out
    # harness-mkzk: the recovery hint must travel with the
    # empty-result line so the model can pivot on the very next
    # emission. Echo the failed query, teach capability framing,
    # and seed at least one example capability term.
    assert "nonexistent-term" in out
    assert "capability" in out.lower()
    assert "search" in out  # the most-common recovery verb


def test_no_results_recovery_hint_lists_example_capability_terms() -> None:
    """harness-mkzk repro shape: model queries by topic ('game engine'),
    gets no hits, must see at least several capability tokens to anchor
    its next attempt on the right axis."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="game engine")
    # The motivating session-trace query is echoed verbatim.
    assert "query='game engine'" in out
    # Several capability tokens must appear as examples; checking a
    # representative spread (data-gathering + compute + IO + time).
    for verb in ("search", "lookup", "fetch", "compute", "time"):
        assert verb in out, f"recovery hint missing capability example {verb!r}"
    # And the load_tool follow-up reminder so a recovered query can
    # actually progress to activation on the NEXT emission.
    assert "load_tool" in out


def test_no_results_recovery_hint_calls_out_topic_vs_capability() -> None:
    """The hint must make the topic-vs-capability distinction explicit
    so the model understands WHY its topic-shaped query missed. Without
    that framing, a model just rephrases the same topic and gets the
    same miss (the 2026-05-20 'framework' / 'engine' / 'Python game
    framework' iterative repro)."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="2D game framework")
    # The hint contrasts "subject matter" / "topic" with "capability".
    assert "topic" in out.lower()
    assert "capability" in out.lower()
    # And calls out at least one of the session-trace failure shapes
    # by name so the model recognizes its own pattern.
    assert "game engine" in out or "weather in" in out


def test_no_results_hint_only_when_no_candidates() -> None:
    """The recovery hint MUST NOT bleed into the normal results path —
    a non-empty result page should not carry the topic-vs-capability
    framing (the model would read it as a critique of a successful
    search and get confused)."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="search")
    # Non-empty result; hint phrases must be absent.
    assert "no tools found" not in out
    assert "Recovery:" not in out
    assert "subject matter" not in out


# --- tag + family filters ----------------------------------------------


def test_tag_filter_without_query() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="", tag="time")
    assert "now" in out
    assert "calc" not in out


def test_family_filter_without_query() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="", family="reckon")
    # Both reckon tools surface.
    assert "now" in out
    assert "calc" in out
    assert "grep" not in out


def test_query_plus_tag_intersects() -> None:
    """Query 'search' matches grep + search_web (both have 'search' in
    tags). Tag='web' narrows to search_web only."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="search", tag="web")
    assert "search_web" in out
    assert "grep" not in out


def test_query_plus_family_intersects() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="search", family="filesystem")
    assert "grep" in out
    assert "search_web" not in out


def test_no_constraints_raises() -> None:
    """Empty query AND no tag AND no family — caller error.
    'Substring of everything' isn't a useful default."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    with pytest.raises(ValueError, match="at least one of"):
        tool.call(query="", tag=None, family=None)


def test_non_positive_limit_rejected() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    with pytest.raises(ValueError, match="limit must be positive"):
        tool.call(query="now", limit=0)


# --- output shape ------------------------------------------------------


def test_output_includes_family_in_parens() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="now")
    assert "now (reckon)" in out


def test_output_lists_tags() -> None:
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="now")
    assert "tags: time, clock, timezone" in out


def test_output_truncates_long_descriptions() -> None:
    """Each entry's description gets capped at ~140 chars; the
    `(no description)` placeholder appears when stored description
    is empty AND no registry override exists."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="verbose",
            family="meta",
            description="x" * 500,
        )
    )
    out = ToolSearchTool(catalog=cat).call(query="verbose")
    # Verify the description is truncated (full 500 chars not present)
    # and the ellipsis marker shows.
    assert "x" * 500 not in out
    assert "..." in out


def test_output_shows_no_description_marker_when_blank() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="bare", family="meta"))
    out = ToolSearchTool(catalog=cat).call(query="bare")
    assert "(no description)" in out


def test_output_caps_at_limit_with_overflow_marker() -> None:
    """Results sort by name (the catalog's all() guarantee), so a
    zero-padded suffix keeps the ordering predictable for the
    overflow-marker assertion."""
    cat = ToolCatalog()
    for i in range(15):
        cat.register(
            ToolCatalogEntry(
                name=f"search-{i:02d}",
                family="research",
                tags=("search",),
            )
        )
    tool = ToolSearchTool(catalog=cat)
    out = tool.call(query="search", limit=3)
    # N < M shape (harness-1zun): "showing 3 of 15 matching tool(s)" so
    # the reader can tell M is the candidates pool, not catalog growth.
    assert "showing 3 of 15 matching tool(s)" in out
    assert "search-00" in out
    assert "search-02" in out
    assert "(+12 more" in out
    assert "search-10" not in out  # past the cap


def test_default_limit_is_ten() -> None:
    cat = ToolCatalog()
    for i in range(20):
        cat.register(ToolCatalogEntry(name=f"x-{i}", tags=("foo",)))
    out = ToolSearchTool(catalog=cat).call(query="x-")
    assert "showing 10 of 20 matching tool(s)" in out


def test_header_phrasing_when_limit_meets_candidates() -> None:
    """harness-1zun: when N == M (all candidates fit under limit), the
    header drops the 'showing N of M' framing for the cleaner '{M}
    matching tool(s)' shape. The asymmetric phrasing is what makes
    the limit < candidates case read unambiguously."""
    cat = ToolCatalog()
    for i in range(3):
        cat.register(ToolCatalogEntry(name=f"x-{i}", tags=("foo",)))
    out = ToolSearchTool(catalog=cat).call(query="x-", limit=10)
    # No 'showing N of M' line — everything fit.
    assert "showing" not in out.splitlines()[0]
    assert "3 matching tool(s)" in out
    # And no '+N more' overflow marker.
    assert "+0 more" not in out
    assert "more —" not in out


def test_header_phrasing_when_limit_below_candidates() -> None:
    """harness-1zun positive: when limit < candidates, the header
    spells out the count split so a second call with a higher limit
    doesn't read as 'the catalog grew between calls' (the original
    2026-05-19 session repro)."""
    cat = ToolCatalog()
    for i in range(7):
        cat.register(ToolCatalogEntry(name=f"k-{i:02d}", tags=("search",)))
    tool = ToolSearchTool(catalog=cat)

    # Same query, limit=1 → '1 of 7 matching' (rendered count vs pool).
    one = tool.call(query="search", limit=1)
    assert "showing 1 of 7 matching tool(s)" in one
    # Same query, limit=10 → '7 matching tool(s)' (all fit).
    ten = tool.call(query="search", limit=10)
    assert "7 matching tool(s)" in ten
    assert "showing" not in ten.splitlines()[0]


# --- registry integration ----------------------------------------------


def test_live_description_overrides_catalog_when_registry_supplied() -> None:
    """The catalog's seeded description is '' for builtins — the
    registry's live spec is the source of truth. tool_search prefers
    the live one when a registry is supplied."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="now",
            family="reckon",
            tags=("time",),
            description="",
        )
    )
    reg = ToolRegistry()
    reg.register(
        _StubTool(
            _spec=ToolSpec(
                name="now",
                description="Live registry description.",
                parameters={"type": "object", "properties": {}},
                tier="read",
            )
        )
    )
    out = ToolSearchTool(catalog=cat, registry=reg).call(query="now")
    assert "Live registry description." in out


def test_falls_back_to_catalog_description_when_not_in_registry() -> None:
    """A synthesized tool catalog entry not yet hot-loaded into the
    registry shows the catalog's stored description."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="synth_tool",
            family="meta",
            tags=("custom",),
            description="Stored in catalog only.",
            origin="synthesized",
        )
    )
    reg = ToolRegistry()  # synth_tool NOT registered
    out = ToolSearchTool(catalog=cat, registry=reg).call(query="synth")
    assert "Stored in catalog only." in out


def test_query_falls_back_to_tag_when_query_yields_zero() -> None:
    """harness-wwki: a verbatim user-phrase query that doesn't match
    anything must fall back to the explicit tag/family filter rather
    than dead-end. The fallback note tells the agent the query was
    dropped so it can refine next time."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="search_web",
            family="research",
            description="Search the web.",
            tags=("web", "search"),
        )
    )
    tool = ToolSearchTool(catalog=cat)
    out = tool.call(query="completely unrelated phrase xyz", tag="web")
    assert "search_web" in out
    assert "fell back to tag='web'" in out
    # Phase-3a fallback now carries a diagnostic warning so the model
    # treats the rescue as suggestive, not authoritative (harness-l0wg).
    assert "VERIFY" in out


def test_query_alone_fallback_wins_over_tag_alone_when_both_have_hits() -> None:
    """harness-8rqw: Mark's 'sunset + time' repro. Query 'sunset'
    matches sun (via the 'sunset' tag); tag 'time' matches now and
    tz_convert. The intersection is empty (sun isn't 'time'-tagged).
    Old fallback ordering surfaced the tag-alone hits, dropping the
    correct query match. New ordering: query-alone wins.

    Reasoning: when the agent passes a real query, that's the better
    signal of intent. Tag/family is often over-restrictive."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="sun",
            family="reckon",
            description="Compute sunrise / sunset / civil twilight.",
            tags=("astronomy", "sunrise", "sunset", "twilight"),
        )
    )
    cat.register(
        ToolCatalogEntry(
            name="now",
            family="reckon",
            description="Current wall-clock.",
            tags=("time", "clock"),
        )
    )
    cat.register(
        ToolCatalogEntry(
            name="tz_convert",
            family="reckon",
            description="Timezone conversion.",
            tags=("time", "timezone"),
        )
    )
    tool = ToolSearchTool(catalog=cat)
    out = tool.call(query="sunset", tag="time")
    assert "sun" in out
    assert "now" not in out
    assert "tz_convert" not in out
    assert "fell back to query='sunset' only" in out


def test_query_falls_back_to_query_alone_when_filter_is_fabricated() -> None:
    """Mark's recurring transcript: agent passes a real query plus a
    fabricated tag like 'fs-read' that exists in no catalog entry.
    Both the intersection path AND the tag-alone path yield zero.
    Phase 3b drops the bad filter and retries with the query alone."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="search_web",
            family="research",
            description="Search the web.",
            tags=("web", "weather", "search"),
        )
    )
    tool = ToolSearchTool(catalog=cat)
    out = tool.call(query="weather", tag="fs-read")
    assert "search_web" in out
    assert "fell back to query='weather' only" in out


def test_query_fallback_does_not_fire_when_query_matches() -> None:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="search_web", family="research", tags=("web",)))
    tool = ToolSearchTool(catalog=cat)
    out = tool.call(query="search_web", tag="web")
    assert "search_web" in out
    assert "fell back" not in out


def test_no_registry_uses_catalog_description() -> None:
    """When no registry is supplied, the catalog description is the
    only source — empty descriptions show '(no description)'."""
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now", family="reckon", description=""))
    out = ToolSearchTool(catalog=cat).call(query="now")
    assert "(no description)" in out


# --- schema content (harness-l0wg) -------------------------------------


def test_tag_description_does_not_advertise_arithmetic_as_example() -> None:
    """The earlier schema listed 'arithmetic' as a tag example, which
    led small models to pass tag='arithmetic' for factual lookups (the
    user asked them to compute a percentage AFTER finding the data).
    The example pulled the model toward picking SOME tag instead of
    sending query alone. Repro: 'population of Nairobi, percent
    female vs male' → tag='arithmetic' → calc → fabrication.
    Pin the fix so the bait example can't sneak back in."""
    spec = ToolSearchTool(catalog=ToolCatalog()).spec
    tag_desc = spec.parameters["properties"]["tag"]["description"]
    # Must NOT advertise arithmetic as an opt-in example.
    assert "'arithmetic'" not in tag_desc, (
        "tag examples must not include 'arithmetic' — models latch onto "
        "it for percent-math questions and skip data-gathering tools"
    )
    # Must include the IMMEDIATE-step framing.
    assert "NOW" in tag_desc or "immediate" in tag_desc.lower()


def test_top_level_description_steers_toward_data_tools_for_lookups() -> None:
    """The tool_search top-level description must explicitly call out
    that factual lookups should reach for search/lookup/fetch tools,
    not arithmetic tools. Without this, the model treats 'I need to
    compute a percentage' as the framing and picks calc."""
    spec = ToolSearchTool(catalog=ToolCatalog()).spec
    desc = spec.description
    # Must mention the data-side options for factual lookups.
    lower = desc.lower()
    assert "search" in lower, (
        "tool_search description must mention 'search' as a data-gathering option"
    )
    assert "lookup" in lower, (
        "tool_search description must mention 'lookup' as a data-gathering option"
    )
    # Must warn against picking arithmetic just because computation
    # happens later.
    assert "arithmetic" in lower, (
        "description must explicitly call out the arithmetic trap so the model can recognize it"
    )


# --- tag-equals-family auto-rescue (harness-hbvm) ----------------------


def test_tag_equals_family_name_auto_swaps_to_family() -> None:
    """harness-hbvm: tag and family are different axes. The common
    small-model mistake is passing tag=<family-name> ('filesystem',
    'research') — that never intersects because no tool has its own
    family in its tag tuple. Auto-correct by swapping tag → family."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="", tag="filesystem")
    # The single filesystem-family tool surfaces.
    assert "grep" in out
    # Non-filesystem tools are excluded by the swapped family filter.
    assert "now (reckon)" not in out
    assert "search_web" not in out
    # The swap is announced inline so the model learns the distinction.
    assert "auto-corrected" in out.lower()
    assert "tag='filesystem'" in out
    assert "family='filesystem'" in out
    assert "FAMILY name" in out


def test_tag_equals_family_swap_combines_with_query() -> None:
    """The swap fires before phase 1, then query still narrows. So
    query='read' + tag='filesystem' yields filesystem-family tools
    that match 'read' (grep does; search_web also has the 'read' tag
    but is research-family — excluded post-swap)."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="read", tag="filesystem")
    assert "grep" in out
    # search_web has 'read' as a tag but isn't filesystem-family.
    assert "search_web" not in out


def test_tag_equals_family_swap_does_not_fire_when_family_already_set() -> None:
    """When family= is independently set, don't swap — the caller had
    a reason. The intersection is likely empty (no tool has family X
    AND tag = X-the-family-name), and the empty-result hint will
    explain (covered by the 8gk8 test below)."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="zzz-no-match", tag="filesystem", family="research")
    # Swap message must NOT appear — swap didn't fire.
    assert "auto-corrected" not in out.lower()


def test_tag_value_not_matching_any_family_does_not_swap() -> None:
    """tag='time' is a real tag (now has it), not a family. Behavior
    unchanged from pre-hbvm: by_tag('time') returns 'now'."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    out = tool.call(query="", tag="time")
    assert "now" in out
    assert "calc" not in out
    assert "auto-corrected" not in out.lower()


# --- schema ordering (harness-u2ap) ------------------------------------


def test_schema_lists_family_before_tag() -> None:
    """harness-u2ap: family= is the higher-leverage filter for small
    models; the schema should advertise it before tag=. JSON-schema
    property order is visible to the model as the field listing,
    and reordering nudges them to reach for family first."""
    spec = ToolSearchTool(catalog=ToolCatalog()).spec
    keys = list(spec.parameters["properties"].keys())
    assert keys.index("family") < keys.index("tag"), (
        f"family must come before tag in schema, got order: {keys}"
    )


def test_family_description_lists_known_families() -> None:
    """The family= description should name the actual families a
    caller can reach for — small models won't otherwise know which
    strings are valid family names."""
    spec = ToolSearchTool(catalog=ToolCatalog()).spec
    family_desc = spec.parameters["properties"]["family"]["description"]
    for name in ("filesystem", "research", "memory", "reckon"):
        assert name in family_desc, f"family description must mention {name!r}"
    # And the desc should call out that family= is the primary axis.
    assert "PRIMARY" in family_desc or "primary" in family_desc.lower()


def test_tag_description_warns_about_family_swap() -> None:
    """The tag= description should warn that passing a family name as
    tag is auto-corrected, so callers learn the distinction from the
    schema alone instead of needing an empty-result round-trip."""
    spec = ToolSearchTool(catalog=ToolCatalog()).spec
    tag_desc = spec.parameters["properties"]["tag"]["description"]
    assert "auto-correct" in tag_desc.lower()
    assert "family" in tag_desc.lower()


# --- empty-result family-aware hint (harness-8gk8) --------------------


def test_empty_result_hint_names_family_when_tag_was_family() -> None:
    """harness-8gk8: the narrow case where hbvm's swap didn't fire
    (because family= was independently set) AND nothing matched.
    The empty-result hint must still tell the caller their tag=
    value was a family name."""
    tool = ToolSearchTool(catalog=_seeded_catalog())
    # tag='filesystem' AND family='research' — swap blocked (family
    # already set). Intersection: empty. Phase 3b: query yields zero.
    # Phase 3a: by_tag('filesystem') is empty (no tool has 'filesystem'
    # as a tag). Empty-result fires.
    out = tool.call(query="zzz-no-match", tag="filesystem", family="research")
    assert "no tools found" in out
    assert "'filesystem' is a FAMILY name" in out
    assert "family='filesystem'" in out


def test_phase_3a_fallback_note_carries_verification_hint() -> None:
    """When a query yields zero and tool_search falls back to
    tag-alone, the note must signal that this is a rescue, not a
    confirmation. Without the warning, the model treats whatever came
    back as authoritative. Targets the harness-l0wg repro shape."""
    cat = ToolCatalog()
    cat.register(
        ToolCatalogEntry(
            name="calc",
            family="reckon",
            tags=("arithmetic",),
            description="Arithmetic.",
        )
    )
    cat.register(
        ToolCatalogEntry(
            name="search_web",
            family="research",
            tags=("search", "lookup", "web"),
            description="Search the web.",
        )
    )
    tool = ToolSearchTool(catalog=cat)
    out = tool.call(query="population count Nairobi Kenya", tag="arithmetic")
    # 3a fired and surfaced calc.
    assert "calc" in out
    assert "fell back to tag='arithmetic'" in out
    # And the diagnostic hint is present.
    assert "VERIFY" in out
    assert "search" in out
    assert "lookup" in out


def test_builtin_catalog_surfaces_outline_and_symbol_read() -> None:
    """harness-roia: outline + read_file's symbol mode must be
    discoverable via tool_search so the model finds them when it wants to
    navigate or read a whole function, rather than over-reading a file."""
    from harness.tools.catalog import seed_builtins_into

    cat = ToolCatalog()
    seed_builtins_into(cat, now_iso="2026-01-01T00:00:00Z")
    tool = ToolSearchTool(catalog=cat)

    # Natural-language capability queries (not distinctively-worded). Post
    # harness-53jm these rank the precise tool into the displayed window
    # even though the query floods every read-tier tool via the shared
    # "read" tag — match-strength ranking puts name + specific-tag hits on
    # top, so the alphabetical-cap burial is gone.
    nav = tool.call(query="navigate functions and classes in a file")
    assert "outline" in nav

    whole = tool.call(query="read a whole function by symbol name")
    assert "read_file" in whole
