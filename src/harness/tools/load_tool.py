"""load_tool — agent-driven working-set expansion (harness-atsz).

The companion to `tool_search` (harness-ozx1):

  tool_search → "what tools exist that fit my need right now?"
  load_tool   → "bring tool X into this session's active working set
                 so I can call it on the next round."

Why this is needed: the working set (harness-fzvg) is fixed at
session start by the CLI's `--tool-set` + `--tools-add` flags. The
agent can call `tool_search` and learn that `calc` exists in the
catalog, but on the *next* round its chat-template tool list still
doesn't include `calc`, so it can't emit the call. `load_tool`
closes that gap by expanding the registry's `_active_names` set.
The orchestrator already calls `registry.specs()` on every round
(see `_run_round` in `tool_loop.py`), so a successful `load_tool`
call surfaces the new tool's schema starting next round.

Three states, three outcomes:

  catalog miss        — "no tool named X. Try tool_search to find one."
  in catalog,
    not in registry   — "tool exists but isn't loaded this session.
                         Restart with --tools-add X (or, if origin=
                         synthesized, wait for the t5kx hot-reload
                         pass on next session start)."
  in registry,
    already active    — "X is already in your working set."
  in registry,
    inactive          — set_active(active + {X}); return spec preview
                         + 'now active'.

Tier=read because the only mutation is to the in-memory working set;
no filesystem or network involved. The orchestrator's confirm gate
correctly does not prompt for this.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.tools.base import ToolRegistry, ToolSpec
from harness.tools.catalog import ToolCatalog


@dataclass
class LoadToolTool:
    """Expand the live registry's working set by name. Reads the
    catalog for the description/origin hint when the name isn't
    currently loaded — that's where the 'restart with --tools-add'
    error string gets its tailoring."""

    catalog: ToolCatalog
    registry: ToolRegistry

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="load_tool",
            description=(
                "Add a tool from the catalog to this session's active "
                "working set so you can call it on subsequent rounds. "
                "Use after `tool_search` reveals a tool that fits your "
                "need. Returns the tool's spec preview + a confirmation; "
                "next round's chat template will include the tool's "
                "schema. Read-only with respect to disk and network."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Catalog entry name to activate.",
                    },
                },
                "required": ["name"],
            },
            tier="read",
            display_name="Load tool",
        )

    def call(self, *, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            return "load_tool: name must be a non-empty string."

        name = name.strip()

        # 1. Catalog miss: point at tool_search.
        entry = self.catalog.get(name)
        if entry is None and name not in self.registry:
            return (
                f"load_tool: no tool named {name!r} in the catalog. "
                "Use `tool_search` to find the right name."
            )

        # 2. In catalog but not in registry: needs --tools-add (or hot-
        #    reload, depending on origin).
        if name not in self.registry:
            # entry is non-None here because we returned above otherwise.
            assert entry is not None  # for type narrowing
            if entry.origin == "synthesized":
                hint = (
                    "It's a synthesized tool from a previous session. The "
                    "hot-reload pass runs at session start (harness-t5kx); "
                    "restart the harness to pick it up."
                )
            else:
                hint = (
                    "Restart the session with `--tools-add "
                    f"{name}` (or a profile that includes it)."
                )
            return f"load_tool: {name!r} is in the catalog but not loaded this session. {hint}"

        # 3. In registry, already in the active set: no-op confirmation.
        active = set(self.registry.active_names())
        if name in active:
            return f"load_tool: {name!r} is already in your active working set; no change."

        # 4. In registry, not active: add to the working set.
        new_active = active | {name}
        self.registry.set_active(new_active)
        spec = self.registry.get(name).spec
        return _format_loaded(spec)


def _format_loaded(spec: ToolSpec) -> str:
    lines = [
        f"load_tool: {spec.name!r} is now active.",
        f"  description: {_clip(spec.description, 200)}",
        f"  tier       : {spec.tier}",
    ]
    props = spec.parameters.get("properties") if isinstance(spec.parameters, dict) else None
    if isinstance(props, dict) and props:
        joined = ", ".join(sorted(props.keys()))
        lines.append(f"  parameters : {joined}")
    lines.append("Next round will see its schema in the tool list.")
    return "\n".join(lines)


def _clip(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


__all__ = ["LoadToolTool"]
