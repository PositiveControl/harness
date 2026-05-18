"""Tool catalog — harness-hfa7.

The durable record of every tool the system *knows about*, distinct
from the `ToolRegistry`, which is the subset currently *active* in a
session. The two work in concert (rqg0.2 wires them together):

  catalog.all() — every tool we could load (built-in + synthesized
    + external). Persisted as JSON; grows as `synthesize_tool` (rqg0.4)
    or operator file-drops add entries.

  registry.specs() — the subset rendered as schemas this turn.
    Bounded by the ~1.5k-token budget; rotates by profile / tag /
    explicit names (rqg0.2).

Lightweight grouping + discoverability flows from the catalog:
  - `family` — broad category for the operator-facing CLI grouping
    (`harness tool list --family reckon`).
  - `tags` — keywords for the agent's `tool_search` discovery
    primitive (rqg0.6): "I need to compute a percentile — what tools
    are available?" matches against tags.
  - `description` — the model-facing text the agent reads to decide
    whether the tool fits the need. Catalog stores it once, the
    registry inherits it.

Persistence: JSON sidecar, atomic write (`.tmp` + `Path.replace()`),
defensive load (missing / malformed / unknown fields all survive).
Same pattern as runtime/state.py + plan/store.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# Catalog-entry origin — declares where the tool came from. `builtin`
# tools are seeded at first launch from a hardcoded metadata table;
# `synthesized` tools are written by the synthesize_tool meta-tool
# (rqg0.4); `external` is the operator-drop path (someone places a
# tool module in the right directory and registers it manually).
Origin = Literal["builtin", "synthesized", "external"]
ORIGINS: tuple[Origin, ...] = ("builtin", "synthesized", "external")


class ToolCatalogError(Exception):
    """Raised on catalog corruption that the caller can't safely
    recover from. Missing files + unknown fields are tolerated; this
    fires only when load can't produce a coherent catalog at all."""


@dataclass(frozen=True)
class ToolCatalogEntry:
    """One tool's metadata. Carries enough for grouping +
    discoverability + reload without re-running the tool's source.

    `source_path` is None for built-ins (they live in src/harness/
    tools/ and import-resolve through Python's normal mechanism).
    Synthesized + external tools point at their source files so the
    hot-reload slice (rqg0.5) can pick them up at session start.
    """

    name: str
    family: str = ""
    tags: tuple[str, ...] = ()
    description: str = ""
    tier: str = "read"
    origin: Origin = "builtin"
    source_path: str | None = None
    registered_at: str = ""
    quarantined: bool = False
    quarantine_reason: str | None = None


@dataclass
class ToolCatalog:
    """Mutable collection of catalog entries with grouping +
    discovery helpers. Construct empty; populate via `register` /
    `seed_builtins`; persist via `save_catalog`.

    Lookups are deliberately exhaustive scans — the catalog is small
    enough (low hundreds of entries at the upper bound) that an
    in-memory dict + list filter beats indexing complexity.
    """

    entries: dict[str, ToolCatalogEntry] = field(default_factory=dict)

    # --- mutation ----------------------------------------------------

    def register(self, entry: ToolCatalogEntry) -> None:
        """Insert or replace `entry` by name. The caller is
        responsible for collision checks where they matter (e.g.
        synthesize_tool refuses to overwrite an existing entry)."""
        if not entry.name:
            raise ValueError("ToolCatalogEntry.name must be non-empty")
        self.entries[entry.name] = entry

    def drop(self, name: str) -> None:
        """Idempotent removal — dropping an absent name is a no-op
        (mirrors the rest of the harness's drop semantics)."""
        self.entries.pop(name, None)

    # --- lookups -----------------------------------------------------

    def get(self, name: str) -> ToolCatalogEntry | None:
        """Return the named entry or None. Use `.get()` rather than
        indexing when 'not in catalog' is a normal case."""
        return self.entries.get(name)

    def all(self) -> list[ToolCatalogEntry]:
        """Every entry, sorted by name for deterministic output."""
        return sorted(self.entries.values(), key=lambda e: e.name)

    def by_family(self, family: str) -> list[ToolCatalogEntry]:
        return [e for e in self.all() if e.family == family]

    def by_tag(self, tag: str) -> list[ToolCatalogEntry]:
        return [e for e in self.all() if tag in e.tags]

    def by_origin(self, origin: Origin) -> list[ToolCatalogEntry]:
        return [e for e in self.all() if e.origin == origin]

    def search(self, query: str) -> list[ToolCatalogEntry]:
        """Substring match (case-folded) against name + description +
        tags. Used by the tool_search meta-tool (rqg0.6) for
        agent-side discovery.

        Empty query returns the empty list — substring-of-everything
        isn't a useful default. Callers that want 'show everything'
        should use `all()`.
        """
        q = query.strip().lower()
        if not q:
            return []
        out: list[ToolCatalogEntry] = []
        for entry in self.all():
            if q in entry.name.lower():
                out.append(entry)
                continue
            if q in entry.description.lower():
                out.append(entry)
                continue
            if any(q in t.lower() for t in entry.tags):
                out.append(entry)
        return out


# --- persistence -----------------------------------------------------


def load_catalog(path: Path) -> ToolCatalog:
    """Read the catalog JSON. Missing file -> empty catalog (first
    launch). Malformed file -> empty catalog (defensive — same policy
    as runtime/state.py)."""
    if not path.exists():
        return ToolCatalog()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ToolCatalog()
    raw_entries = raw.get("entries", {}) if isinstance(raw, dict) else {}
    if not isinstance(raw_entries, dict):
        return ToolCatalog()

    allowed_keys = {
        "name",
        "family",
        "tags",
        "description",
        "tier",
        "origin",
        "source_path",
        "registered_at",
        "quarantined",
        "quarantine_reason",
    }
    entries: dict[str, ToolCatalogEntry] = {}
    for raw_name, raw_entry in raw_entries.items():
        if not isinstance(raw_name, str) or not isinstance(raw_entry, dict):
            continue
        filtered = {k: v for k, v in raw_entry.items() if k in allowed_keys}
        # Tags is a tuple in-memory but a list on disk.
        if "tags" in filtered:
            tags_raw = filtered["tags"]
            if isinstance(tags_raw, list):
                filtered["tags"] = tuple(str(t) for t in tags_raw if isinstance(t, str))
            else:
                filtered["tags"] = ()
        # Origin guard — fall back to 'external' for unknown values
        # so the catalog stays consistent under schema drift.
        if "origin" in filtered and filtered["origin"] not in ORIGINS:
            filtered["origin"] = "external"
        try:
            entry = ToolCatalogEntry(
                name=raw_name, **{k: v for k, v in filtered.items() if k != "name"}
            )
        except TypeError:
            continue
        entries[raw_name] = entry
    return ToolCatalog(entries=entries)


def save_catalog(catalog: ToolCatalog, path: Path) -> None:
    """Atomic write — serialize to `.tmp`, then `Path.replace()`. A
    crash mid-write leaves either the previous file or no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": {name: _entry_to_json(entry) for name, entry in catalog.entries.items()}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _entry_to_json(entry: ToolCatalogEntry) -> dict[str, object]:
    raw = asdict(entry)
    # Serialize the tuple as a list for JSON-natural output.
    raw["tags"] = list(entry.tags)
    return raw


# --- builtin metadata seeding ---------------------------------------


# Lightweight family + tags for every built-in tool. Lives here rather
# than on each ToolSpec so the existing tool surface stays untouched
# (frozen ToolSpec dataclasses + ~25 import sites would balloon). When
# a built-in's metadata changes, edit this table — synthesized tools
# get their metadata from the synthesize_tool call directly.
#
# Tags should be the keywords a model would naturally use when
# searching for the capability: 'time' over 'clock', 'arithmetic'
# over 'math'. Aim for 3-5 tags per tool — discoverable without
# being noisy.
BUILTIN_TOOL_METADATA: dict[str, tuple[str, tuple[str, ...]]] = {
    # reckon family — deterministic time + compute
    "now": ("reckon", ("time", "clock", "timezone", "date")),
    "date_math": ("reckon", ("date", "arithmetic", "duration", "business-days", "weekday")),
    "calc": ("reckon", ("arithmetic", "math", "unit-convert", "expression")),
    "python_eval": ("reckon", ("python", "sandbox", "compute", "scripting")),
    "tz_convert": ("reckon", ("timezone", "time", "convert", "iana")),
    "stats": ("reckon", ("statistics", "math", "percentile", "summary")),
    "sun": ("reckon", ("astronomy", "sunrise", "sunset", "twilight", "latlon")),
    # filesystem read
    "read_file": ("filesystem", ("read", "file", "open")),
    "list_dir": ("filesystem", ("read", "directory", "ls")),
    "grep": ("filesystem", ("read", "search", "regex", "text-match")),
    "glob": ("filesystem", ("read", "filename-pattern", "find")),
    # filesystem write
    "edit_file": ("filesystem", ("write", "file", "edit", "patch")),
    "write_file": ("filesystem", ("write", "file", "create")),
    "shell": ("filesystem", ("write", "shell", "execute", "subprocess")),
    # git
    "git_status": ("git", ("read", "status", "diff-summary")),
    "git_diff": ("git", ("read", "diff", "changes")),
    "git_log": ("git", ("read", "log", "history")),
    # memory + retrieval
    "search_memory": ("memory", ("read", "search", "episodic", "retrieval")),
    "search_facts": ("memory", ("read", "search", "semantic", "facts", "retrieval")),
    "remember_fact": ("memory", ("write", "facts", "semantic", "store")),
    "remember_event": ("memory", ("write", "episodic", "store", "summary")),
    "scribe_session": ("memory", ("write", "scribe", "extract", "transcript")),
    "consolidate_memory": ("memory", ("write", "consolidate", "promote")),
    "transcript_ingest": ("memory", ("write", "transcript", "ingest")),
    "assemble_context": ("memory", ("read", "contract", "retrieval-bundle")),
    # research
    "search_web": ("research", ("read", "search", "web", "ddg")),
    "fetch_url": ("research", ("read", "http", "fetch", "url")),
    "search_scholar": ("research", ("read", "search", "academic", "papers")),
    # self / meta
    "introspect": ("meta", ("read", "self-inspection", "capabilities")),
    "spawn_subagent": ("meta", ("read", "subagent", "delegate")),
    "tool_search": ("meta", ("read", "discovery", "search", "catalog")),
    # ab ops (personal-operations data plane)
    "plan": ("ops", ("write", "bd", "task", "plan-issue")),
    "capture": ("ops", ("write", "bd", "task", "capture")),
    "status": ("ops", ("read", "bd", "ready", "open")),
    "drift": ("ops", ("read", "bd", "drift", "stale")),
    "reprioritize": ("ops", ("write", "bd", "priority")),
    "close": ("ops", ("write", "bd", "close")),
    "defer": ("ops", ("write", "bd", "defer")),
    "retro": ("ops", ("write", "bd", "retro", "lesson")),
    "reopen": ("ops", ("write", "bd", "reopen")),
    "delete": ("ops", ("write", "bd", "delete")),
    "update": ("ops", ("write", "bd", "update", "edit")),
    "search": ("ops", ("read", "bd", "search")),
    "list": ("ops", ("read", "bd", "list")),
    "memories": ("ops", ("read", "bd", "memories")),
    "remember": ("ops", ("write", "bd", "memory", "store")),
    "forget": ("ops", ("write", "bd", "memory", "delete")),
    "dep": ("ops", ("write", "bd", "dependency", "blocks")),
    "label": ("ops", ("write", "bd", "label", "tag")),
    "comments": ("ops", ("read", "bd", "comments")),
    "find_duplicates": ("ops", ("read", "bd", "duplicate")),
    "persist_focus_note": ("ops", ("write", "bd", "focus", "note")),
    # phraseology + atc
    "phraseology_lint": ("atc", ("read", "atc", "phraseology", "verify")),
    # query
    "query_table": ("research", ("read", "sqlite", "query", "table")),
    # citations
    "citation_lookup": ("research", ("read", "citation", "lookup")),
}


def seed_builtins_into(catalog: ToolCatalog, *, now_iso: str) -> int:
    """Populate `catalog` with built-in metadata entries for every
    name in `BUILTIN_TOOL_METADATA`. Existing entries (e.g. from a
    prior session) are LEFT UNCHANGED — operator edits + synthesized
    entries with the same name survive.

    Returns the number of new entries inserted. Used at first launch
    + by tests to materialize a fresh catalog with the canonical
    built-in surface."""
    inserted = 0
    for name, (family, tags) in BUILTIN_TOOL_METADATA.items():
        if name in catalog.entries:
            continue
        entry = ToolCatalogEntry(
            name=name,
            family=family,
            tags=tags,
            description="",  # filled at runtime from spec.description when convenient
            tier="read",  # also a runtime fill; default conservative
            origin="builtin",
            source_path=None,
            registered_at=now_iso,
        )
        catalog.register(entry)
        inserted += 1
    return inserted
