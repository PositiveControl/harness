"""Tests for the ToolRegistry working-set + tag-based selection — harness-fzvg."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from harness.tools import (
    ToolCatalog,
    ToolCatalogEntry,
    ToolRegistry,
    ToolSpec,
    resolve_active,
)


@dataclass
class _StubTool:
    """Minimal Tool — spec + call. Used to exercise the registry's
    working-set semantics without depending on the real tool surface."""

    _spec: ToolSpec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def call(self, **_: Any) -> str:
        return f"ok:{self._spec.name}"


def _tool(name: str, description: str = "stub", tier: str = "read") -> _StubTool:
    return _StubTool(
        _spec=ToolSpec(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {}},
            tier=tier,
        )
    )


# --- working-set semantics ------------------------------------------------


def test_default_active_is_all_registered() -> None:
    """Back-compat: a fresh registry with no set_active call returns
    every registered tool from specs() — existing behavior."""
    reg = ToolRegistry()
    reg.register(_tool("now"))
    reg.register(_tool("calc"))
    specs = reg.specs()
    assert {s.name for s in specs} == {"now", "calc"}


def test_set_active_restricts_specs_output() -> None:
    """After set_active({a}), specs() returns only that one tool's
    spec — the rest stay registered but hidden."""
    reg = ToolRegistry()
    reg.register(_tool("now"))
    reg.register(_tool("calc"))
    reg.register(_tool("grep"))
    reg.set_active({"now", "calc"})
    specs = reg.specs()
    assert {s.name for s in specs} == {"now", "calc"}


def test_set_active_drops_unregistered_names_silently() -> None:
    """Names not in the registry get filtered out — the caller may
    pass catalog-level names that this session didn't register, and
    that's fine."""
    reg = ToolRegistry()
    reg.register(_tool("now"))
    reg.set_active({"now", "future_tool"})
    assert {s.name for s in reg.specs()} == {"now"}
    assert reg.active_names() == ("now",)


def test_clear_active_restores_all() -> None:
    reg = ToolRegistry()
    reg.register(_tool("now"))
    reg.register(_tool("calc"))
    reg.set_active({"now"})
    assert len(reg.specs()) == 1
    reg.clear_active()
    assert {s.name for s in reg.specs()} == {"now", "calc"}


def test_clear_active_is_idempotent() -> None:
    """Calling clear_active on a registry that was never restricted
    is a no-op."""
    reg = ToolRegistry()
    reg.register(_tool("now"))
    reg.clear_active()  # no raise
    assert {s.name for s in reg.specs()} == {"now"}


def test_tool_registered_after_set_active_stays_inactive() -> None:
    """Working set is frozen at set_active time — a later register()
    doesn't auto-add to the active set. Caller must re-call set_active
    (or clear_active) to include the new tool."""
    reg = ToolRegistry()
    reg.register(_tool("now"))
    reg.set_active({"now"})
    reg.register(_tool("calc"))
    # calc registered but not active.
    assert "calc" in reg.names()
    assert {s.name for s in reg.specs()} == {"now"}


def test_active_names_sorted_when_set() -> None:
    reg = ToolRegistry()
    reg.register(_tool("calc"))
    reg.register(_tool("now"))
    reg.set_active({"now", "calc"})
    assert reg.active_names() == ("calc", "now")


def test_active_names_returns_all_when_unset() -> None:
    """When no working set is set, active_names mirrors specs()'s
    full-registry view."""
    reg = ToolRegistry()
    reg.register(_tool("now"))
    reg.register(_tool("calc"))
    assert reg.active_names() == ("calc", "now")


def test_call_still_works_on_inactive_tool() -> None:
    """Working-set filters specs() but NOT call(). A tool that isn't
    in the schema this turn can still be invoked — the working set is
    a model-visibility constraint, not a sandbox. (The orchestrator
    won't generate a call to an inactive tool because the model
    doesn't see its schema, but tests + meta-tools may invoke
    directly.)"""
    reg = ToolRegistry()
    reg.register(_tool("now"))
    reg.set_active(set())  # nothing active
    assert reg.specs() == []
    result = reg.call("now", {})
    assert result.success is True
    assert "ok:now" in result.output


def test_override_description_still_applied_to_active_subset() -> None:
    """The existing description-override path layers on top of the
    working set."""
    reg = ToolRegistry()
    reg.register(_tool("now", description="default desc"))
    reg.register(_tool("calc"))
    reg.set_active({"now"})
    reg.override_description("now", "rewritten desc")
    [spec] = reg.specs()
    assert spec.name == "now"
    assert spec.description == "rewritten desc"


# --- resolve_active -------------------------------------------------------


def _catalog_with_three_families() -> ToolCatalog:
    cat = ToolCatalog()
    cat.register(ToolCatalogEntry(name="now", family="reckon", tags=("time", "clock")))
    cat.register(ToolCatalogEntry(name="calc", family="reckon", tags=("math",)))
    cat.register(ToolCatalogEntry(name="grep", family="filesystem", tags=("search", "regex")))
    cat.register(ToolCatalogEntry(name="search_web", family="research", tags=("search", "web")))
    return cat


def test_resolve_active_profile_only() -> None:
    """No catalog, no tags — just the profile. Equivalent to
    resolve_tool_names but returns a set, not a sorted tuple."""
    names = resolve_active(profile="core")
    assert "read_file" in names
    assert "list_dir" in names


def test_resolve_active_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        resolve_active(profile="does-not-exist")


def test_resolve_active_explicit_names_pass_through() -> None:
    names = resolve_active(names=("custom-tool", "another"))
    assert names == {"custom-tool", "another"}


def test_resolve_active_tag_selection() -> None:
    """Tag filter pulls every catalog entry tagged with that
    keyword."""
    cat = _catalog_with_three_families()
    names = resolve_active(catalog=cat, tags=("search",))
    assert names == {"grep", "search_web"}


def test_resolve_active_family_selection() -> None:
    cat = _catalog_with_three_families()
    names = resolve_active(catalog=cat, families=("reckon",))
    assert names == {"now", "calc"}


def test_resolve_active_combines_profile_and_tags() -> None:
    """All three layers union — profile + names + tag."""
    cat = _catalog_with_three_families()
    names = resolve_active(
        catalog=cat,
        profile=None,
        names=("custom-tool",),
        tags=("time",),
    )
    assert names == {"custom-tool", "now"}


def test_resolve_active_no_catalog_skips_tag_layer() -> None:
    """When catalog is None, tags are ignored — no way to resolve
    them without the metadata source."""
    names = resolve_active(catalog=None, names=("a",), tags=("anywhere",))
    assert names == {"a"}


def test_resolve_active_empty_inputs_returns_empty() -> None:
    assert resolve_active() == set()


def test_resolve_active_wired_into_registry() -> None:
    """End-to-end: resolve_active produces a name set; the registry
    accepts it as the working set."""
    cat = _catalog_with_three_families()
    reg = ToolRegistry()
    for name in ("now", "calc", "grep", "search_web"):
        reg.register(_tool(name))
    reg.set_active(resolve_active(catalog=cat, tags=("search",)))
    assert {s.name for s in reg.specs()} == {"grep", "search_web"}
