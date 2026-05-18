"""Hot-reload synthesized tools from the catalog at session start.

The complement to `synthesize_tool` (harness-yari). Synthesized tools
land on disk + in the catalog at synthesis time but are deliberately
*not* hot-loaded into the live session — confirming that the new tool
worked happens next session. This module runs at session bootstrap:
scan the catalog for `origin=synthesized` entries, import each source
file, register the Tool with the live `ToolRegistry`.

Failure mode: if a synthesised module won't import (a dep got removed,
the file was hand-edited and now has a syntax error, the file is
missing), don't crash the session. Mark the catalog entry
`quarantined=True` with a reason, save, and skip. The next
`harness tool list` shows the quarantine; the operator can drop the
entry or repair the source.

Already-quarantined entries are skipped without retrying — preserves
the "tried once, gave up" semantics so a permanently broken module
doesn't lengthen every session start.

Trusted-input note: this loader does *not* re-run `validate_tool_module`.
The catalog is the source of truth for "this passed validation when it
was synthesised"; re-validating on every boot would impose seconds of
subprocess overhead for no security benefit (the threat model is
incompetent synthesis, not tampered files between sessions).
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from harness.tools.base import ToolRegistry
from harness.tools.catalog import (
    ToolCatalog,
    ToolCatalogEntry,
    save_catalog,
)


@dataclass(frozen=True)
class LoadReport:
    """Summary of one hot-reload pass.

    `loaded`: names registered into the live registry, in catalog order.
    `quarantined`: (name, reason) for entries marked broken THIS pass.
    `already_quarantined`: names skipped because they were already
        quarantined from a prior session — surfaced so the operator
        can see the carryover without re-reading the catalog file.
    """

    loaded: tuple[str, ...] = ()
    quarantined: tuple[tuple[str, str], ...] = ()
    already_quarantined: tuple[str, ...] = ()


def load_synthesized_tools(
    *,
    catalog: ToolCatalog,
    catalog_path: Path,
    registry: ToolRegistry,
) -> LoadReport:
    """Import every synthesized catalog entry; register surviving Tools.

    Mutates `catalog` in place when entries quarantine (and writes the
    new state via `save_catalog`). Caller's catalog handle remains
    valid — quarantined entries are still in `catalog.entries`, just
    with `quarantined=True`.
    """
    loaded: list[str] = []
    quarantined: list[tuple[str, str]] = []
    already_quarantined: list[str] = []

    entries = catalog.by_origin("synthesized")
    catalog_dirty = False

    for entry in entries:
        if entry.quarantined:
            already_quarantined.append(entry.name)
            continue
        if not entry.source_path:
            _quarantine(catalog, entry, "no source_path on catalog entry")
            quarantined.append((entry.name, "no source_path on catalog entry"))
            catalog_dirty = True
            continue

        source_path = Path(entry.source_path)
        if not source_path.exists():
            reason = f"source file missing: {source_path}"
            _quarantine(catalog, entry, reason)
            quarantined.append((entry.name, reason))
            catalog_dirty = True
            continue

        try:
            tool_instance = _import_and_instantiate(entry, source_path)
        except _LoadError as exc:
            _quarantine(catalog, entry, str(exc))
            quarantined.append((entry.name, str(exc)))
            catalog_dirty = True
            continue

        if entry.name in registry:
            reason = f"a tool named {entry.name!r} is already registered (builtin collision?)"
            _quarantine(catalog, entry, reason)
            quarantined.append((entry.name, reason))
            catalog_dirty = True
            continue

        try:
            registry.register(tool_instance)
        except ValueError as exc:
            # Defensive — the `in registry` check above should have
            # caught duplicates, but a race or refactor could surface a
            # different ValueError here.
            _quarantine(catalog, entry, f"registry rejected tool: {exc}")
            quarantined.append((entry.name, f"registry rejected tool: {exc}"))
            catalog_dirty = True
            continue

        loaded.append(entry.name)

    if catalog_dirty:
        save_catalog(catalog, catalog_path)

    return LoadReport(
        loaded=tuple(loaded),
        quarantined=tuple(quarantined),
        already_quarantined=tuple(already_quarantined),
    )


class _LoadError(Exception):
    """Internal — bubbles importlib / inspection failures up to the
    quarantine path with a clean reason string."""


def _import_and_instantiate(entry: ToolCatalogEntry, source_path: Path) -> Any:
    """Import the source file, locate the single Tool class, instantiate
    with no args, and return the instance. Raises `_LoadError` with a
    human-readable reason on any failure."""
    # Unique module name per (name, path) so re-loading the same file
    # under a different catalog entry doesn't collide in sys.modules.
    # `synth.` prefix makes hot-reloaded tools easy to spot in
    # introspection.
    mod_name = f"_harness_synth_{entry.name}"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, source_path)
    except Exception as exc:
        raise _LoadError(f"spec_from_file_location failed: {exc}") from exc
    if spec is None or spec.loader is None:
        raise _LoadError("importlib could not build a spec for the source file")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        # Drop the partial module from sys.modules so a later repair +
        # reload doesn't see a stale cached version.
        sys.modules.pop(mod_name, None)
        raise _LoadError(f"module import failed: {type(exc).__name__}: {exc}") from exc

    candidates: list[tuple[str, type]] = []
    for cls_name, obj in vars(module).items():
        if cls_name.startswith("_"):
            continue
        if not inspect.isclass(obj):
            continue
        if obj.__module__ != mod_name:
            continue  # ignore re-exported imports
        if not hasattr(obj, "spec") or not hasattr(obj, "call"):
            continue
        candidates.append((cls_name, obj))

    if not candidates:
        raise _LoadError("module exposes no Tool class (need a class with `spec` + `call`)")
    if len(candidates) > 1:
        names = ", ".join(n for n, _ in candidates)
        raise _LoadError(f"module exposes multiple Tool candidates: {names}")

    cls_name, cls = candidates[0]
    try:
        instance = cls()
    except Exception as exc:
        raise _LoadError(
            f"could not instantiate {cls_name}(): {type(exc).__name__}: {exc}"
        ) from exc

    # The catalog's name must match the live ToolSpec.name; if not,
    # discoverability tools would lie about the tool's identity.
    try:
        spec_name = instance.spec.name
    except Exception as exc:
        raise _LoadError(f"reading .spec raised {type(exc).__name__}: {exc}") from exc

    if spec_name != entry.name:
        raise _LoadError(f"ToolSpec.name {spec_name!r} disagrees with catalog key {entry.name!r}")

    return instance


def _quarantine(catalog: ToolCatalog, entry: ToolCatalogEntry, reason: str) -> None:
    """Mark `entry` quarantined in `catalog` with `reason`. ToolCatalogEntry
    is frozen, so we replace the dict slot with a fresh entry."""
    catalog.entries[entry.name] = replace(entry, quarantined=True, quarantine_reason=reason)


__all__ = ["LoadReport", "load_synthesized_tools"]
