"""synthesize_tool — runtime tool authoring meta-tool (harness-yari).

The write-tier meta-tool the agent calls to author + register a new
tool at runtime. Flow:

    model → synthesize_tool(name, source, family, tags, smoke_args)
          → validate_tool_module(source, smoke_args)       (rqg0.3)
          → write source to <synth_dir>/<name>.py
          → catalog.register(ToolCatalogEntry(origin="synthesized", ...))
          → save_catalog(...)
          → success summary back to model

Three invariants make this safe:

1. Write-tier — the orchestrator's confirm gate prompts the operator
   on the first call per session. The tool itself doesn't reimplement
   confirm — it trusts the gate above.
2. Validator is the security boundary — the candidate runs in
   `validate_tool_module`'s sandboxed subprocess before any
   filesystem mutation. If validation fails, nothing is written or
   registered.
3. The new tool is NOT hot-loaded into the current session — the
   hot-reload path (harness-t5kx) picks it up at session start. The
   model can call `tool_search` to find its own creation next turn,
   but can't invoke it inside the same session.

Collision policy: refuses to overwrite an existing catalog entry —
the model must `drop_tool` first if it really wants to replace one.
Refusing here keeps the catalog audit-trail honest (every entry has
exactly one author + registered_at).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.tools.base import ToolSpec
from harness.tools.catalog import (
    ToolCatalog,
    ToolCatalogEntry,
    save_catalog,
)
from harness.tools.sandbox_validator import validate_tool_module

# Conservative name shape: python identifier, snake_case, ≤40 chars.
# Catalog entries with weirder names work fine but would surprise the
# operator-facing CLI grouping; rejecting at synthesis keeps the
# catalog tidy.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


@dataclass
class SynthesizeToolTool:
    """Author + register a new tool. Write-tier; per-session confirm.

    Constructed with a catalog handle + paths so the orchestrator can
    inject the character-scoped catalog (each character has its own
    synthesized-tools directory under `character/<name>/tools/
    synthesized/`).
    """

    catalog: ToolCatalog
    catalog_path: Path
    synth_dir: Path
    # Caller-curated allowlist extras passed into the validator. The
    # default already covers stdlib subset + harness.tools.base; this
    # field lets a character extend it (e.g. an atc character that
    # wants its synthesised tools to import `pyproj`).
    allowed_imports: tuple[str, ...] = ()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="synthesize_tool",
            description=(
                "Author a new tool at runtime. Provide a Python module "
                "that defines a single Tool class (a class with a "
                "`spec` property returning a ToolSpec and a `call` "
                "method) plus a smoke-test input. The candidate is "
                "validated in a sandbox before being persisted to the "
                "catalog. Use this when an existing tool would not "
                "fit the need and the new shape would be reusable. "
                "The new tool will NOT be available until the next "
                "session — confirm it with `tool_search` next turn."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Snake_case identifier, ≤40 chars. Becomes "
                            "the catalog key and the source filename."
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Full Python module source. Must define "
                            "exactly one top-level class exposing "
                            "`spec` + `call`. ≤8 KB."
                        ),
                    },
                    "family": {
                        "type": "string",
                        "description": (
                            "Broad category for catalog grouping "
                            "(e.g. 'reckon', 'research', 'meta'). "
                            "Free-form; used by `tool list --family`."
                        ),
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Keywords for agent-side discovery via "
                            "`tool_search`. Lowercase, short, prefer "
                            "verbs + nouns the agent would query."
                        ),
                    },
                    "smoke_args": {
                        "type": "object",
                        "description": (
                            "kwargs dict passed to the candidate's "
                            "`call(**smoke_args)`. Required — the "
                            "validator runs one real invocation to "
                            "catch contract bugs early."
                        ),
                    },
                },
                "required": ["name", "source", "family", "smoke_args"],
            },
            tier="write",
            display_name="Synthesize tool",
        )

    def call(
        self,
        *,
        name: str,
        source: str,
        family: str,
        smoke_args: dict[str, Any],
        tags: list[str] | None = None,
    ) -> str:
        # Validate the requested name shape before anything expensive.
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(
                f"synthesize_tool: name {name!r} must be snake_case, "
                "start with a-z, and be 2-40 chars (a-z, 0-9, _ only)."
            )
        if name in self.catalog.entries:
            existing = self.catalog.entries[name]
            raise ValueError(
                f"tool {name!r} already in catalog (origin={existing.origin}). "
                f"Use `drop_tool` to remove it first, or pick a different name."
            )
        if not isinstance(family, str) or not family.strip():
            raise ValueError("synthesize_tool: family must be a non-empty string.")
        if not isinstance(smoke_args, dict):
            raise ValueError("synthesize_tool: smoke_args must be an object (dict).")

        # Validate in the sandbox. This is the security boundary —
        # nothing is written until the candidate compiles cleanly,
        # exposes a single Tool, and survives one smoke call.
        validation = validate_tool_module(
            source=source,
            smoke_args=smoke_args,
            allowed_imports=tuple(self.allowed_imports),
        )
        if not validation.ok or validation.spec is None:
            return _format_failure(name, validation.error or "unknown error")

        # The spec the candidate emitted must name itself the same as
        # the catalog key — otherwise `tool_search` and operator CLI
        # would disagree on what the tool is called.
        emitted_name = validation.spec.name
        if emitted_name != name:
            return _format_failure(
                name,
                f"the candidate's ToolSpec.name is {emitted_name!r} but "
                f"the synthesize_tool call set name={name!r}. They must match.",
            )

        # Persist the source first, then register. Inverted order would
        # leave a catalog entry pointing at a non-existent file if the
        # write failed.
        self.synth_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.synth_dir / f"{name}.py"
        source_path.write_text(source, encoding="utf-8")

        normalised_tags = tuple(_normalise_tags(tags))
        entry = ToolCatalogEntry(
            name=name,
            family=family.strip(),
            tags=normalised_tags,
            description=validation.spec.description,
            tier=validation.spec.tier,
            origin="synthesized",
            source_path=str(source_path),
            registered_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        )
        self.catalog.register(entry)
        save_catalog(self.catalog, self.catalog_path)

        return _format_success(entry, validation.smoke_output)


def _normalise_tags(tags: Any) -> list[str]:
    """Lowercase + dedup. Accepts Any because the JSON envelope can
    carry oddly-typed tags from the model (None, str-of-strs, mixed
    list). Garbage in → empty list out, not a crash."""
    if not isinstance(tags, list):
        return []
    out: list[str] = []
    for raw in tags:
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _format_success(entry: ToolCatalogEntry, smoke_output: str | None) -> str:
    lines = [
        f"synthesize_tool: registered {entry.name!r}",
        f"  family       : {entry.family}",
        f"  tags         : {', '.join(entry.tags) or '(none)'}",
        f"  tier         : {entry.tier}",
        f"  source_path  : {entry.source_path}",
        f"  registered_at: {entry.registered_at}",
    ]
    if smoke_output:
        snippet = smoke_output if len(smoke_output) < 200 else smoke_output[:197] + "..."
        lines.append(f"  smoke_output : {snippet!r}")
    lines.append("Not loaded into this session. Call `tool_search` next turn to confirm.")
    return "\n".join(lines)


def _format_failure(name: str, reason: str) -> str:
    return f"synthesize_tool: refusing to register {name!r}. Validation failed: {reason}"


__all__ = ["SynthesizeToolTool"]
