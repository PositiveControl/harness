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

Five states, five outcomes:

  catalog miss        — "no tool named X. Try tool_search to find one."
  in catalog,
    not in registry,
    builder available — build via the builder, register, activate.
                         This is what makes core_minimal actually
                         pay-on-demand: tools live in the catalog
                         until the agent asks for them (harness-cm4v).
  in catalog,
    not in registry,
    builder missing   — "tool exists but isn't loaded this session.
                         Restart with --tools-add X (or, if origin=
                         synthesized, wait for the t5kx hot-reload
                         pass on next session start)."
  in registry,
    already active    — "X is already in your working set."
  in registry,
    inactive          — set_active(active + {X}); return spec preview.

Tier=read because the only mutation is to the in-memory working set;
no filesystem or network involved. The orchestrator's confirm gate
correctly does not prompt for this.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from harness.tools.base import Tool, ToolRegistry, ToolSpec
from harness.tools.catalog import ToolCatalog

# Tools that travel together — loading one auto-loads the others
# (harness-h6ve). Pure pragmatism: small models bail on multi-step
# discovery, and these pairings are 'almost always wanted together.'
# Reasoning, not a hard convention:
#   search_web → fetch_url    Snippets are previews; the agent almost
#                             always wants to read one of the URLs.
# Add cautiously: pairings that aren't tight enough cause the wrong
# tools to leak into a session.
_TOOL_COMPANIONS: dict[str, tuple[str, ...]] = {
    "search_web": ("fetch_url",),
    # write_file ↔ edit_file (harness-hnt7). A model that loads one
    # almost always needs the other within the same task — the GTA2
    # session 2026-05-20 burned multiple round-trips because the model
    # had write_file but not edit_file when it tried to modify the
    # file it just created. Bidirectional pairing keeps the
    # WriteFileRedirectHook's `ensure_edit_file_active` cheap (the
    # companion already landed at load time) and lets the model treat
    # the pair as one capability.
    "write_file": ("edit_file",),
    "edit_file": ("write_file",),
}


@dataclass
class LoadToolTool:
    """Expand the live registry's working set by name.

    When `builders` is supplied (an injected map keyed by tool name,
    callable returns a Tool or None), the catalog-hit-registry-miss
    branch will try to construct and register the tool on the fly
    rather than dead-ending at 'restart with --tools-add'. That's
    what makes the core_minimal profile (harness-sbia) deliver real
    pay-on-demand discovery instead of a hard restart for every
    capability.

    Builders that return None signal 'tool requires session state
    that isn't enabled' (e.g. search_memory without --memories) —
    surfaced as a clear error pointing at the right CLI flag.
    """

    catalog: ToolCatalog
    registry: ToolRegistry
    # Optional builder map. Provided by the CLI session-bootstrap path
    # (cli.py:_build_tool_registry_for_tui) so the same dict that
    # constructs the profile's initial tools is reused for on-demand
    # construction. Without it, load_tool keeps its v0 'restart' hint.
    builders: dict[str, Callable[[], Tool | None]] = field(default_factory=dict)

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

        # 2. In catalog but not in registry: try to build it on demand
        #    (the cm4v path), else fall back to a 'restart' hint.
        if name not in self.registry:
            # entry is non-None here because we returned above otherwise.
            assert entry is not None  # for type narrowing
            if entry.origin == "synthesized":
                hint = (
                    "It's a synthesized tool from a previous session. The "
                    "hot-reload pass runs at session start (harness-t5kx); "
                    "restart the harness to pick it up."
                )
                return f"load_tool: {name!r} is in the catalog but not loaded this session. {hint}"

            # Builtin in catalog, not registered, builder available:
            # construct it lazily, register, ensure it's in the working
            # set, and return the spec preview. This is the cm4v path
            # that closes the discovery loop for core_minimal.
            builder = self.builders.get(name)
            if builder is not None:
                try:
                    instance = builder()
                except Exception as exc:
                    return (
                        f"load_tool: builder for {name!r} raised "
                        f"{type(exc).__name__}: {exc}. Likely a missing dep; "
                        f"restart with `--tools-add {name}` to see the full error."
                    )
                if instance is None:
                    return (
                        f"load_tool: tool {name!r} requires session state that "
                        "isn't enabled (memory store, semantic store, or similar). "
                        "Restart with the matching CLI flag — see `--memories` / "
                        "`--facts` / `--workspace` for the common ones."
                    )
                try:
                    self.registry.register(instance)
                except ValueError as exc:
                    return f"load_tool: registry rejected {name!r}: {exc}"
                # Ensure the freshly-registered tool participates in
                # the working set. If a working set was already
                # explicitly configured (active_names != None), add to
                # it; if no working set was configured (all-active),
                # registration alone is sufficient.
                if self.registry._active_names is not None:
                    self.registry.set_active(set(self.registry.active_names()) | {name})

                # Auto-load known companions (harness-h6ve). search_web
                # → fetch_url etc. Silent on failure: the primary load
                # already succeeded, so we don't want a companion
                # builder hiccup to mask that. The caller sees which
                # companions actually landed in the success summary.
                companions_loaded = self._load_companions(name)
                return _format_loaded(instance.spec, lazy_built=True, companions=companions_loaded)
            hint = f"Restart the session with `--tools-add {name}` (or a profile that includes it)."
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

    def _load_companions(self, primary_name: str) -> tuple[str, ...]:
        """Build + register peer tools that travel with `primary_name`.

        Quietly skips: companions already in the registry, missing
        builders, builders that return None (state not enabled),
        builders that raise, or `register` failures. Returns the names
        that actually landed."""
        wanted = _TOOL_COMPANIONS.get(primary_name, ())
        if not wanted:
            return ()
        loaded: list[str] = []
        for comp in wanted:
            if comp in self.registry:
                continue
            builder = self.builders.get(comp)
            if builder is None:
                continue
            try:
                instance = builder()
            except Exception:  # noqa: S112 — companion failure must not break primary load
                continue
            if instance is None:
                continue
            try:
                self.registry.register(instance)
            except ValueError:
                continue
            if self.registry._active_names is not None:
                self.registry.set_active(set(self.registry.active_names()) | {comp})
            loaded.append(comp)
        return tuple(loaded)


def _format_loaded(
    spec: ToolSpec,
    *,
    lazy_built: bool = False,
    companions: tuple[str, ...] = (),
) -> str:
    headline = (
        f"load_tool: built and activated {spec.name!r} on demand."
        if lazy_built
        else f"load_tool: {spec.name!r} is now active."
    )
    lines = [
        headline,
        f"  description: {_clip(spec.description, 200)}",
        f"  tier       : {spec.tier}",
    ]
    props = spec.parameters.get("properties") if isinstance(spec.parameters, dict) else None
    if isinstance(props, dict) and props:
        joined = ", ".join(sorted(props.keys()))
        lines.append(f"  parameters : {joined}")
    if companions:
        lines.append(f"  also loaded: {', '.join(companions)} (peer tools)")
    lines.append("Next round will see its schema in the tool list.")
    return "\n".join(lines)


def _clip(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


__all__ = ["LoadToolTool"]
