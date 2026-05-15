"""Tests for the request-scoped tool registry helper (harness-3jz1.9)."""

from __future__ import annotations

from dataclasses import dataclass

from harness.tools.base import ToolRegistry, ToolResult, ToolSpec
from harness.web.tool_registry import build_request_registry


@dataclass(frozen=True)
class _StaticTool:
    """Minimal Tool implementation that returns a fixed string. Captures
    enough Tool surface (.spec, .call(**kwargs)) for ToolRegistry's
    dispatch without dragging in a real tool implementation."""

    spec: ToolSpec
    fixed_output: str

    def call(self, **_kwargs: object) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output=self.fixed_output, success=True)


def _spec(name: str, description: str = "test tool") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {}, "additionalProperties": True},
        tier="read",
    )


def test_base_tools_passed_through_to_fresh_registry() -> None:
    base = ToolRegistry()
    base.register(_StaticTool(spec=_spec("alpha"), fixed_output="alpha-base"))
    base.register(_StaticTool(spec=_spec("beta"), fixed_output="beta-base"))

    fresh = build_request_registry(base, [])
    assert sorted(fresh.names()) == ["alpha", "beta"]
    assert fresh.call("alpha", {}).output == "alpha-base"
    assert fresh.call("beta", {}).output == "beta-base"


def test_request_tools_shadow_base_tools_by_name() -> None:
    base = ToolRegistry()
    base.register(_StaticTool(spec=_spec("alpha"), fixed_output="alpha-base"))
    request_alpha = _StaticTool(spec=_spec("alpha"), fixed_output="alpha-REQUEST")

    fresh = build_request_registry(base, [request_alpha])
    # Same name; request version wins.
    assert fresh.call("alpha", {}).output == "alpha-REQUEST"


def test_request_tools_can_add_new_names() -> None:
    base = ToolRegistry()
    base.register(_StaticTool(spec=_spec("alpha"), fixed_output="alpha-base"))
    request_gamma = _StaticTool(
        spec=_spec("gamma", description="bound this request"),
        fixed_output="gamma-REQUEST",
    )

    fresh = build_request_registry(base, [request_gamma])
    assert sorted(fresh.names()) == ["alpha", "gamma"]
    assert fresh.call("gamma", {}).output == "gamma-REQUEST"


def test_base_is_not_mutated() -> None:
    base = ToolRegistry()
    base.register(_StaticTool(spec=_spec("alpha"), fixed_output="alpha-base"))
    starting = set(base.names())

    extra = _StaticTool(spec=_spec("zeta"), fixed_output="z")
    _fresh = build_request_registry(base, [extra])

    # Base unchanged.
    assert set(base.names()) == starting
    assert "zeta" not in base


def test_description_overrides_carry_over_for_non_shadowed_tools() -> None:
    base = ToolRegistry()
    base.register(_StaticTool(spec=_spec("alpha", description="default desc"), fixed_output=""))
    base.override_description("alpha", "OVERRIDE — TFR-framed retrieval")

    fresh = build_request_registry(base, [])
    [alpha_spec] = [s for s in fresh.specs() if s.name == "alpha"]
    assert alpha_spec.description == "OVERRIDE — TFR-framed retrieval"


def test_description_overrides_dropped_when_request_tool_shadows() -> None:
    """A request-scoped tool brings its own description; the base override
    no longer applies because the request tool replaced the slot."""
    base = ToolRegistry()
    base.register(_StaticTool(spec=_spec("alpha", description="default"), fixed_output=""))
    base.override_description("alpha", "OVERRIDE on base")

    request_alpha = _StaticTool(
        spec=_spec("alpha", description="request-bound description"),
        fixed_output="",
    )
    fresh = build_request_registry(base, [request_alpha])
    [alpha_spec] = [s for s in fresh.specs() if s.name == "alpha"]
    assert alpha_spec.description == "request-bound description"
