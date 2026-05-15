"""Request-scoped tool registry helper.

Character extensions sometimes need to bind a tool's behavior to *this*
request's input — the TFR explainer's `read_parsed_notam` returns this
request's parsed NOTAM struct, not a globally registered one. This
helper builds a fresh `ToolRegistry` per request that overlays a base
registry with one or more request-scoped `Tool` instances.

Design choice — a fresh registry rather than a mutating wrapper:
  - The orchestrator's tool loop reads from a `ToolRegistry`. Building a
    fresh instance per request keeps the orchestrator path identical to
    the CLI path; no need for the orchestrator to know about scoping.
  - The base registry stays immutable from the request's point of view
    — concurrent requests can't race each other on shared mutable
    state.
  - Cost is small: copying ~20 tool references is cheap.

Description overrides from the base registry are preserved on non-
shadowed tools so per-character profile reframings (e.g. airton_c's
search_memory → "search the FAA JO 7110.65 rulebook" override) still
surface to the model.
"""

from __future__ import annotations

from collections.abc import Sequence

from harness.tools.base import Tool, ToolRegistry


def build_request_registry(
    base: ToolRegistry,
    request_tools: Sequence[Tool] = (),
) -> ToolRegistry:
    """Construct a fresh `ToolRegistry` populated from `base` and
    overlaid with `request_tools`. Request tools shadow base tools by
    name (a request-scoped `read_parsed_notam` wins over a generic one
    registered at the base).

    `base` is not mutated. The returned registry is intended to be
    used for the duration of one HTTP request and discarded.
    """
    fresh = ToolRegistry()
    shadowed = {tool.spec.name for tool in request_tools}

    # Copy base tools, skipping any that the request layer shadows.
    for name in base.names():
        if name in shadowed:
            continue
        fresh.register(base.get(name))

    # Preserve base description overrides on non-shadowed tools so
    # per-character profile reframings still reach the model. The base
    # registry's `specs()` already merges overrides into its returned
    # specs; comparing against the underlying tool's own spec tells us
    # which names carry an override.
    for spec in base.specs():
        if spec.name in shadowed:
            continue
        if spec.name not in fresh:
            continue
        tool_default = fresh.get(spec.name).spec
        if spec.description != tool_default.description:
            fresh.override_description(spec.name, spec.description)

    # Register request-scoped tools last; these are the ones bound to
    # this request's inputs.
    for tool in request_tools:
        fresh.register(tool)

    return fresh


__all__ = ["build_request_registry"]
