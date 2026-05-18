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
    assert "3 of 15 tool(s)" in out
    assert "search-00" in out
    assert "search-02" in out
    assert "(+12 more" in out
    assert "search-10" not in out  # past the cap


def test_default_limit_is_ten() -> None:
    cat = ToolCatalog()
    for i in range(20):
        cat.register(ToolCatalogEntry(name=f"x-{i}", tags=("foo",)))
    out = ToolSearchTool(catalog=cat).call(query="x-")
    assert "10 of 20" in out


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


def test_no_registry_uses_catalog_description() -> None:
    """When no registry is supplied, the catalog description is the
    only source — empty descriptions show '(no description)'."""
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now", family="reckon", description=""))
    out = ToolSearchTool(catalog=cat).call(query="now")
    assert "(no description)" in out
