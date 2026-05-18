"""Tests for the load_tool meta-tool (harness-atsz).

Six unit tests cover the four-way state matrix (catalog hit/miss x
registry hit/miss, plus already-active and inactive variants). A
seventh integration test exercises the working-set + specs()
round-trip — what the orchestrator depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

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


# --- 8-11. cm4v: lazy build via builders ---------------------------------


def _build_with_builders(
    *,
    catalog_entries: list[ToolCatalogEntry],
    builders: dict[str, object],
    registered: list[ToolSpec] | None = None,
) -> LoadToolTool:
    catalog = ToolCatalog()
    for e in catalog_entries:
        catalog.register(e)
    registry = ToolRegistry()
    for spec in registered or []:
        registry.register(_StubTool(_spec=spec))
    return LoadToolTool(
        catalog=catalog,
        registry=registry,
        builders=builders,  # type: ignore[arg-type]  # accepting Callable typing in tests
    )


def test_load_tool_builds_builtin_lazily_when_builder_supplied() -> None:
    """The cm4v happy path: builtin in catalog, not in registry, but
    a builder is wired. Tool should be constructed, registered, and
    show up in subsequent specs() output."""
    target_spec = _stub_spec("on_demand")

    def _builder() -> _StubTool:
        return _StubTool(_spec=target_spec)

    tool = _build_with_builders(
        catalog_entries=[ToolCatalogEntry(name="on_demand", family="reckon", origin="builtin")],
        builders={"on_demand": _builder},
    )
    out = tool.call(name="on_demand")
    assert "built and activated 'on_demand' on demand" in out
    assert "on_demand" in tool.registry.names()
    # Specs reflect the new tool on the next round.
    assert "on_demand" in {s.name for s in tool.registry.specs()}


def test_load_tool_builder_returning_none_explains_state_gap() -> None:
    """Builders that need session state (memory store, etc.) return
    None when state is missing. load_tool must surface that as a
    clear error pointing at the CLI flag — not a silent failure."""

    def _stateful_builder() -> _StubTool | None:
        return None  # state not enabled

    tool = _build_with_builders(
        catalog_entries=[ToolCatalogEntry(name="search_memory", family="memory", origin="builtin")],
        builders={"search_memory": _stateful_builder},
    )
    out = tool.call(name="search_memory")
    assert "requires session state" in out
    assert "--memories" in out or "--facts" in out
    # Tool NOT registered — must wait for a restart with state enabled.
    assert "search_memory" not in tool.registry.names()


def test_load_tool_builder_raising_surfaces_exception() -> None:
    """A builder that crashes (missing dep, bad config) must not crash
    load_tool — surface the exception type + message so the operator
    can diagnose."""

    def _broken_builder() -> _StubTool:
        raise ImportError("cannot import 'optional_dep'")

    tool = _build_with_builders(
        catalog_entries=[ToolCatalogEntry(name="broken_tool", family="research", origin="builtin")],
        builders={"broken_tool": _broken_builder},
    )
    out = tool.call(name="broken_tool")
    assert "builder for 'broken_tool' raised" in out
    assert "ImportError" in out
    assert "optional_dep" in out
    assert "broken_tool" not in tool.registry.names()


def test_load_tool_without_builders_keeps_old_restart_hint() -> None:
    """Backward compat: when load_tool is constructed without a
    builders map (the v0 invocation), the catalog-hit-registry-miss
    branch still produces the 'restart with --tools-add' hint."""
    tool = _build_with_builders(
        catalog_entries=[ToolCatalogEntry(name="not_built", family="filesystem", origin="builtin")],
        builders={},  # explicit empty
    )
    out = tool.call(name="not_built")
    assert "not loaded this session" in out
    assert "--tools-add not_built" in out


# --- harness-h6ve: companion auto-load ----------------------------------


def test_load_tool_auto_loads_known_companion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harness-h6ve: loading search_web pulls fetch_url along. The
    pair travels together because snippets are previews and the
    actual page content lives behind the URL — small models bail
    rather than do another discovery cycle."""
    from harness.tools import load_tool as load_tool_mod

    primary_spec = _stub_spec("search_web")
    companion_spec = _stub_spec("fetch_url")

    monkeypatch.setattr(
        load_tool_mod,
        "_TOOL_COMPANIONS",
        {"search_web": ("fetch_url",)},
    )

    tool = _build_with_builders(
        catalog_entries=[
            ToolCatalogEntry(name="search_web", family="research", origin="builtin"),
            ToolCatalogEntry(name="fetch_url", family="research", origin="builtin"),
        ],
        builders={
            "search_web": lambda: _StubTool(_spec=primary_spec),
            "fetch_url": lambda: _StubTool(_spec=companion_spec),
        },
    )
    out = tool.call(name="search_web")
    assert "built and activated 'search_web'" in out
    assert "also loaded: fetch_url (peer tools)" in out
    assert "search_web" in tool.registry.names()
    assert "fetch_url" in tool.registry.names()


def test_load_tool_companion_skipped_when_builder_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Companion declared but its builder isn't wired — silent skip.
    Primary load still succeeds; the success summary just omits the
    'also loaded' line."""
    from harness.tools import load_tool as load_tool_mod

    primary_spec = _stub_spec("search_web")
    monkeypatch.setattr(load_tool_mod, "_TOOL_COMPANIONS", {"search_web": ("fetch_url",)})

    tool = _build_with_builders(
        catalog_entries=[
            ToolCatalogEntry(name="search_web", family="research", origin="builtin"),
        ],
        builders={"search_web": lambda: _StubTool(_spec=primary_spec)},  # no fetch_url
    )
    out = tool.call(name="search_web")
    assert "built and activated 'search_web'" in out
    assert "also loaded:" not in out
    assert "fetch_url" not in tool.registry.names()


def test_load_tool_companion_skipped_when_already_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the companion was already explicitly loaded earlier in the
    session, don't double-register. The 'also loaded' line should
    omit it — there was no companion load to report."""
    from harness.tools import load_tool as load_tool_mod

    primary_spec = _stub_spec("search_web")
    companion_spec = _stub_spec("fetch_url")
    monkeypatch.setattr(load_tool_mod, "_TOOL_COMPANIONS", {"search_web": ("fetch_url",)})

    tool = _build_with_builders(
        catalog_entries=[
            ToolCatalogEntry(name="search_web", family="research", origin="builtin"),
            ToolCatalogEntry(name="fetch_url", family="research", origin="builtin"),
        ],
        builders={
            "search_web": lambda: _StubTool(_spec=primary_spec),
            "fetch_url": lambda: _StubTool(_spec=companion_spec),
        },
        registered=[companion_spec],  # fetch_url already there
    )
    out = tool.call(name="search_web")
    assert "built and activated 'search_web'" in out
    assert "also loaded:" not in out


def test_load_tool_companion_raise_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Companion builder crashing must not break the primary load.
    Silently swallow the exception; the primary load reports success
    without claiming the companion landed."""
    from harness.tools import load_tool as load_tool_mod

    primary_spec = _stub_spec("search_web")

    def _broken_companion() -> _StubTool:
        raise ImportError("ddgs not installed")

    monkeypatch.setattr(load_tool_mod, "_TOOL_COMPANIONS", {"search_web": ("fetch_url",)})

    tool = _build_with_builders(
        catalog_entries=[
            ToolCatalogEntry(name="search_web", family="research", origin="builtin"),
            ToolCatalogEntry(name="fetch_url", family="research", origin="builtin"),
        ],
        builders={
            "search_web": lambda: _StubTool(_spec=primary_spec),
            "fetch_url": _broken_companion,
        },
    )
    out = tool.call(name="search_web")
    assert "built and activated 'search_web'" in out
    assert "also loaded:" not in out
    assert "search_web" in tool.registry.names()
    assert "fetch_url" not in tool.registry.names()
