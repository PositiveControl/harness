"""Tests for the load_tool meta-tool (harness-atsz).

Six unit tests cover the four-way state matrix (catalog hit/miss x
registry hit/miss, plus already-active and inactive variants). A
seventh integration test exercises the working-set + specs()
round-trip — what the orchestrator depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.tools.base import ToolRegistry, ToolSpec
from harness.tools.catalog import ToolCatalog, ToolCatalogEntry
from harness.tools.load_tool import LoadToolTool


@dataclass
class _StubTool:
    """Minimal Tool: a fixed spec + a no-op call. Used to seed the
    registry without pulling real tool modules into these tests."""

    _spec: ToolSpec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def call(self) -> str:
        return f"stub:{self._spec.name}"


def _stub_spec(name: str, *, tier: str = "read") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"stub description for {name}",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        tier=tier,
    )


def _build(
    *,
    catalog_entries: list[ToolCatalogEntry] | None = None,
    registered: list[ToolSpec] | None = None,
    active: list[str] | None = None,
) -> LoadToolTool:
    catalog = ToolCatalog()
    for entry in catalog_entries or []:
        catalog.register(entry)
    registry = ToolRegistry()
    for spec in registered or []:
        registry.register(_StubTool(_spec=spec))
    if active is not None:
        registry.set_active(active)
    return LoadToolTool(catalog=catalog, registry=registry)


# --- 1. catalog miss → point at tool_search ----------------------------


def test_load_tool_unknown_name_points_at_tool_search() -> None:
    tool = _build()
    out = tool.call(name="ghost_tool")
    assert "no tool named 'ghost_tool'" in out
    assert "tool_search" in out


# --- 2. catalog hit, registry miss → restart hints ---------------------


def test_load_tool_in_catalog_but_not_loaded_builtin_hint() -> None:
    tool = _build(
        catalog_entries=[ToolCatalogEntry(name="calc", family="reckon", origin="builtin")]
    )
    out = tool.call(name="calc")
    assert "not loaded this session" in out
    assert "--tools-add calc" in out


def test_load_tool_in_catalog_but_not_loaded_synthesized_hint() -> None:
    """Synthesized tools land via the t5kx hot-reload pass, which only
    runs at session start. The error tells the agent to wait for a
    restart, not retry."""
    tool = _build(
        catalog_entries=[
            ToolCatalogEntry(
                name="my_synth",
                family="meta",
                origin="synthesized",
                source_path="/var/folders/whatever.py",
            )
        ]
    )
    out = tool.call(name="my_synth")
    assert "not loaded this session" in out
    assert "harness-t5kx" in out
    assert "restart" in out


# --- 3. registry hit, already in active set → no-op --------------------


def test_load_tool_already_active_is_noop_confirmation() -> None:
    spec = _stub_spec("calc")
    tool = _build(registered=[spec], active=["calc"])
    out = tool.call(name="calc")
    assert "already in your active working set" in out
    # Active set unchanged.
    assert tool.registry.active_names() == ("calc",)


# --- 4. registry hit, inactive → add to working set --------------------


def test_load_tool_expands_working_set_when_inactive() -> None:
    """The orchestrator depends on this: set_active(active + {name})
    must make the new spec appear in registry.specs() on the next
    call. That's what makes the schema flow into the next round's
    chat template."""
    spec = _stub_spec("calc")
    other = _stub_spec("now")
    tool = _build(registered=[spec, other], active=["now"])

    out = tool.call(name="calc")

    assert "now active" in out
    assert "tier       : read" in out
    assert "parameters : q" in out
    # specs() now exposes both names.
    spec_names = {s.name for s in tool.registry.specs()}
    assert spec_names == {"now", "calc"}


# --- 5. shape guards ----------------------------------------------------


def test_load_tool_rejects_blank_or_non_string_name() -> None:
    tool = _build()
    assert "non-empty string" in tool.call(name="   ")
    assert "non-empty string" in tool.call(name="")


# --- 6. spec contract ---------------------------------------------------


def test_load_tool_spec_is_read_tier_and_takes_name() -> None:
    tool = _build()
    spec = tool.spec
    assert spec.name == "load_tool"
    assert spec.tier == "read"
    assert spec.parameters["required"] == ["name"]


# --- 7. catalog-miss-but-registry-hit edge case ------------------------


def test_load_tool_works_when_name_in_registry_but_not_in_catalog() -> None:
    """A tool registered but never catalogued (e.g. a profile-specific
    helper added after catalog seed) should still be expandable —
    presence in the registry is what we actually care about."""
    spec = _stub_spec("ephemeral")
    tool = _build(registered=[spec], active=[])  # empty working set
    out = tool.call(name="ephemeral")
    assert "now active" in out
    assert "ephemeral" in {s.name for s in tool.registry.specs()}
