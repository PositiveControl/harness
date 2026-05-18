"""tool_search — agent-side discovery primitive — harness-ozx1.

The discovery primitive that pairs with the catalog (hfa7) + working
set (fzvg). With a larger persistent tool library and a bounded
per-turn schema budget, the model only sees a SUBSET of tools at any
given turn. tool_search lets it find what's available beyond the
current working set, without burning schema budget on tools it
doesn't need yet.

Read-tier, cheap (~150 tokens of schema overhead), and intended for
*every* profile so it's always-available without filling the budget.
Returns a compact text list — name + family + description excerpt +
tags — similar in feel to `search_memory`.

When the agent realizes the current working set doesn't have what
the user is asking for:

  user: convert 5 ft to m
  agent: (current working set has no calc tool) tool_search('unit conversion')
  -> calc (reckon) — Evaluate an arithmetic expression OR convert units...
       tags: arithmetic, math, unit-convert, expression
  agent: (adds calc to working set via a later mechanism, then calls it)
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.tools.base import ToolRegistry, ToolSpec
from harness.tools.catalog import ToolCatalog, ToolCatalogEntry

# How much of an entry's description to surface per result. Tuned so a
# 10-result return stays under ~700 tokens — leaves room in the
# orchestrator's per-tool-result budget for the surrounding text the
# model wraps the search output in.
_DESCRIPTION_EXCERPT_CHARS = 140


def _truncate_desc(desc: str, *, limit: int = _DESCRIPTION_EXCERPT_CHARS) -> str:
    if len(desc) <= limit:
        return desc
    return desc[: limit - 3] + "..."


def _format_entry(entry: ToolCatalogEntry, *, live_description: str | None = None) -> str:
    """One result block — name + family + description excerpt + tags.

    `live_description` overrides the catalog entry's stored
    description. The catalog seeds builtins with description="", so
    the registry's live spec is the source for actively-loaded tools;
    catalog-only entries (synthesized tools not yet hot-loaded) fall
    back to their stored description."""
    family = f" ({entry.family})" if entry.family else ""
    raw_desc = live_description if live_description else entry.description
    desc = _truncate_desc(raw_desc) if raw_desc else "(no description)"
    tags = ", ".join(entry.tags) if entry.tags else "(none)"
    return f"- {entry.name}{family} — {desc}\n    tags: {tags}"


@dataclass
class ToolSearchTool:
    """Look up tools in the catalog by query / tag / family.

    The catalog is the session-level superset of every tool the
    harness knows about. The model's current working set is a subset
    rendered as schemas; tool_search lets the model discover tools
    outside that subset without paying schema cost for them.

    Returns a compact text list — name, family, description excerpt,
    tags. Empty results return a clear 'no tools found' line so the
    model knows to fall back instead of fabricating an answer.

    `registry` is optional. When supplied, the tool prefers live
    descriptions from `registry.get(name).spec.description` over the
    catalog's stored description — handy because the builtin catalog
    seed records empty descriptions (the live spec is the source of
    truth for actively-loaded tools). For tools in the catalog but
    not yet registered (e.g. synthesized tools awaiting hot-reload),
    the catalog's description is the fallback.
    """

    catalog: ToolCatalog
    registry: ToolRegistry | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="tool_search",
            description=(
                "Find tools in the catalog by keyword query, tag, or family. "
                "Use this BEFORE fabricating an answer when you suspect "
                "a tool might fit the user's need but isn't in the current "
                "turn's schema. Returns name + family + description "
                "excerpt + tags for up to `limit` matching tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Substring match against tool name + description "
                            "+ tags. Case-insensitive. Empty query returns "
                            "the empty list unless `tag` or `family` is set."
                        ),
                    },
                    "tag": {
                        "type": "string",
                        "description": (
                            "Restrict results to tools carrying this tag "
                            "(e.g. 'arithmetic', 'time', 'fs-read'). "
                            "Optional."
                        ),
                    },
                    "family": {
                        "type": "string",
                        "description": (
                            "Restrict results to tools in this family "
                            "(e.g. 'reckon', 'filesystem', 'memory'). "
                            "Optional."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": ("Maximum number of results to return. Default 10."),
                    },
                },
                "required": ["query"],
            },
            tier="read",
            display_name="Tool search",
        )

    def call(
        self,
        *,
        query: str = "",
        tag: str | None = None,
        family: str | None = None,
        limit: int = 10,
    ) -> str:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit!r}")

        # Phase 1: pull a candidate set. Each source contributes; we
        # intersect with subsequent filters below. Empty query + no
        # tag + no family is a usage error.
        if query.strip():
            candidates = self.catalog.search(query)
        elif tag is not None:
            candidates = self.catalog.by_tag(tag)
            tag = None  # already applied
        elif family is not None:
            candidates = self.catalog.by_family(family)
            family = None  # already applied
        else:
            raise ValueError("tool_search: at least one of query, tag, or family must be set")

        # Phase 2: apply additional filters not consumed in phase 1.
        if tag is not None:
            candidates = [e for e in candidates if tag in e.tags]
        if family is not None:
            candidates = [e for e in candidates if e.family == family]

        # Phase 3: if query was over-specific and yielded zero but the
        # caller also passed tag or family, retry with just those.
        # Small models routinely pass a verbatim user-prompt fragment
        # as the query AND a relevant tag — we'd rather surface the
        # tag/family match than dead-end on the literal phrase
        # (harness-wwki).
        fallback_note = ""
        if not candidates and query.strip() and (tag is not None or family is not None):
            if tag is not None:
                candidates = self.catalog.by_tag(tag)
            elif family is not None:
                candidates = self.catalog.by_family(family)
            if family is not None and tag is not None:
                candidates = [e for e in candidates if e.family == family]
            if candidates:
                dropped = f"query={query!r}"
                fallback_note = f"\n(no match for {dropped}; fell back to "
                if tag is not None:
                    fallback_note += f"tag={tag!r}"
                elif family is not None:
                    fallback_note += f"family={family!r}"
                fallback_note += ")"

        if not candidates:
            constraints = []
            if query.strip():
                constraints.append(f"query={query!r}")
            if tag is not None:
                constraints.append(f"tag={tag!r}")
            if family is not None:
                constraints.append(f"family={family!r}")
            return f"no tools found ({', '.join(constraints) or 'no constraints'})"

        visible = candidates[:limit]
        lines = [_format_entry(e, live_description=self._live_description(e.name)) for e in visible]
        remaining = len(candidates) - len(visible)
        if remaining > 0:
            lines.append(f"  (+{remaining} more — increase `limit` or narrow the search)")
        header = f"{len(visible)} of {len(candidates)} tool(s)"
        # Nudge the agent toward the discovery flow's next step. Without
        # this, small models that found a candidate often replied
        # "you'd want X but I can't run it" instead of activating it.
        # See harness-huwj for the core_minimal reproduction.
        suffix = (
            "\nNext step: call `load_tool(name=<one of the above>)` to "
            "activate it for the next round."
        )
        return f"{header}:{fallback_note}\n" + "\n".join(lines) + suffix

    def _live_description(self, name: str) -> str | None:
        """Pull the description off the registry's live spec when the
        tool is registered. Returns None when the tool isn't in the
        registry (catalog-only); the formatter falls back to the
        catalog entry's stored description."""
        if self.registry is None:
            return None
        if name not in self.registry:
            return None
        return self.registry.get(name).spec.description
