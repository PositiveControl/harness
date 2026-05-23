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
        """Tokenized substring match against name + description + tags.

        The query is split on whitespace; an entry matches if ANY
        token substring-matches one of its fields (case-insensitive).
        OR-of-words rather than AND because agents reach for the
        catalog with full natural-language fragments — "weather in
        phoenix" should land on search_web via the `weather` tag even
        though "phoenix" matches nothing. AND-of-words would have
        failed this case (harness-wwki).

        Empty / whitespace-only query returns the empty list —
        substring-of-everything isn't a useful default. Callers that
        want 'show everything' should use `all()`.
        """
        # Drop short tokens (<3 chars) — "in", "to", "of" etc. produce
        # noise by substring-matching unrelated tags ("in" hits "find",
        # "diff-summary", etc.). 3 chars is the threshold below which
        # English connectives + prepositions live.
        tokens = [t for t in query.lower().split() if len(t) >= 3]
        if not tokens:
            # If every token was filtered out (e.g. query="a b c"),
            # fall back to a single whole-phrase substring match — at
            # least let the literal query try to land on something.
            stripped = query.strip().lower()
            return [
                e
                for e in self.all()
                if stripped
                and (
                    stripped in e.name.lower()
                    or stripped in e.description.lower()
                    or any(stripped in t.lower() for t in e.tags)
                )
            ]
        out: list[ToolCatalogEntry] = []
        for entry in self.all():
            haystacks = (
                entry.name.lower(),
                entry.description.lower(),
                *(t.lower() for t in entry.tags),
            )
            if any(token in hay for token in tokens for hay in haystacks):
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
#
# Schema: name -> (family, tags, description, tier). Description is
# the one-line capability summary the model reads in `tool_search`
# output to decide whether to activate the tool. Keep it under ~120
# chars and lead with the verb ("Search the web for...", "Read a
# file...") so partial-match queries land on the right tool.
BUILTIN_TOOL_METADATA: dict[str, tuple[str, tuple[str, ...], str, str]] = {
    # reckon family — deterministic time + compute
    "now": (
        "reckon",
        ("time", "clock", "timezone", "date"),
        "Get the current time and date in a named timezone.",
        "read",
    ),
    "date_math": (
        "reckon",
        ("date", "arithmetic", "duration", "business-days", "weekday"),
        "Add/subtract durations to dates; compute weekday, business days, and date deltas.",
        "read",
    ),
    "calc": (
        "reckon",
        ("arithmetic", "math", "unit-convert", "expression"),
        "Evaluate an arithmetic expression with unit conversions (length, mass, temperature, etc).",
        "read",
    ),
    "python_eval": (
        "reckon",
        ("python", "sandbox", "compute", "scripting"),
        "Run a short Python snippet in a sandbox for compute beyond calc's grammar.",
        "read",
    ),
    "tz_convert": (
        "reckon",
        ("timezone", "time", "convert", "iana"),
        "Convert a time between IANA timezones.",
        "read",
    ),
    "stats": (
        "reckon",
        ("statistics", "math", "percentile", "summary"),
        "Compute summary statistics (mean, median, stdev, percentiles) on a list of numbers.",
        "read",
    ),
    "sun": (
        "reckon",
        ("astronomy", "sunrise", "sunset", "twilight", "latlon"),
        "Compute sunrise / sunset / civil twilight / solar noon at a lat-lon and date.",
        "read",
    ),
    # filesystem read
    "read_file": (
        "filesystem",
        ("read", "file", "open"),
        "Read a file from the workspace.",
        "read",
    ),
    "list_dir": (
        "filesystem",
        ("read", "directory", "ls"),
        "List the contents of a directory in the workspace.",
        "read",
    ),
    "grep": (
        "filesystem",
        ("read", "search", "regex", "text-match"),
        "Search workspace files for a regex pattern.",
        "read",
    ),
    "glob": (
        "filesystem",
        ("read", "filename-pattern", "find"),
        "Find files in the workspace matching a glob pattern.",
        "read",
    ),
    # filesystem write
    "edit_file": (
        "filesystem",
        ("write", "file", "edit", "patch"),
        "Edit a file in the workspace by replacing old_string with new_string.",
        "write",
    ),
    "write_file": (
        "filesystem",
        ("write", "file", "create"),
        "Create a new file in the workspace.",
        "write",
    ),
    "shell": (
        "filesystem",
        ("write", "shell", "execute", "subprocess"),
        "Run a shell command in the workspace.",
        "write",
    ),
    "stream_edit": (
        "filesystem",
        ("file-ops", "awk", "sed", "cut", "tr", "transform", "in-place"),
        (
            "Stream-edit workspace files through awk/sed/cut/tr "
            "(optionally in-place); ideal for find-and-replace across many files."
        ),
        "read",
    ),
    "python_stream": (
        "filesystem",
        ("file-ops", "python", "transform", "json", "in-place"),
        (
            "Run a Python expression over workspace files (optionally in-place); "
            "use when awk/sed don't fit — JSON parsing, multi-line block rewrites."
        ),
        "read",
    ),
    # git
    "git_status": (
        "git",
        ("read", "status", "diff-summary"),
        "Show working-tree status for the workspace git repo.",
        "read",
    ),
    "git_diff": (
        "git",
        ("read", "diff", "changes"),
        "Show staged or unstaged changes in the workspace git repo.",
        "read",
    ),
    "git_log": (
        "git",
        ("read", "log", "history"),
        "Show recent commits in the workspace git repo.",
        "read",
    ),
    # memory + retrieval
    "search_memory": (
        "memory",
        ("read", "search", "episodic", "retrieval"),
        "Search episodic memory (past events, seed memories) for relevant entries.",
        "read",
    ),
    "search_facts": (
        "memory",
        ("read", "search", "semantic", "facts", "retrieval"),
        "Search semantic facts (subject-predicate-object) for relevant entries.",
        "read",
    ),
    "remember_fact": (
        "memory",
        ("write", "facts", "semantic", "store"),
        "Store a semantic fact (subject, predicate, object).",
        "write",
    ),
    "remember_event": (
        "memory",
        ("write", "episodic", "store", "summary"),
        "Store an episodic memory (something that happened, a decision, a context note).",
        "write",
    ),
    "scribe_session": (
        "memory",
        ("write", "scribe", "extract", "transcript"),
        "Extract memory candidates from the current session's unprocessed transcript turns.",
        "write",
    ),
    "consolidate_memory": (
        "memory",
        ("write", "consolidate", "promote"),
        "Cluster near-duplicate working-tier memories and promote them to consolidated tier.",
        "write",
    ),
    "transcript_ingest": (
        "memory",
        ("write", "transcript", "ingest"),
        "Ingest a transcript file into the episodic store.",
        "write",
    ),
    "assemble_context": (
        "memory",
        ("read", "contract", "retrieval-bundle"),
        "Assemble a retrieval bundle (memory + facts) for a query.",
        "read",
    ),
    # research
    "search_web": (
        "research",
        ("read", "search", "web", "ddg", "weather", "news", "lookup"),
        (
            "Search the web (DuckDuckGo) for current information — "
            "weather, news, facts, recent events."
        ),
        "read",
    ),
    "fetch_url": (
        "research",
        ("read", "http", "fetch", "url"),
        "Fetch a single HTTPS URL and return its text content.",
        "read",
    ),
    "search_scholar": (
        "research",
        ("read", "search", "academic", "papers"),
        "Search academic papers across Semantic Scholar and OpenAlex.",
        "read",
    ),
    "geography": (
        "research",
        ("read", "geography", "lookup", "verify", "country", "region", "continent"),
        (
            "Country/region lookup over a static gazetteer. Verify a "
            "country's region BEFORE answering 'top X in region Y' "
            "questions, or list the countries in a named region."
        ),
        "read",
    ),
    # self / meta
    "introspect": (
        "meta",
        ("read", "self-inspection", "capabilities"),
        (
            "Describe this session's tools, model, memory stores, "
            "character config, or available commands."
        ),
        "read",
    ),
    "spawn_subagent": (
        "meta",
        ("read", "subagent", "delegate"),
        "Spawn a depth-1 read-only subagent to handle a focused sub-task.",
        "read",
    ),
    "tool_search": (
        "meta",
        ("read", "discovery", "search", "catalog"),
        (
            "Find tools available in the catalog by keyword, tag, or family. "
            "Returns name + description matches; pair with load_tool to activate one."
        ),
        "read",
    ),
    "load_tool": (
        "meta",
        ("read", "discovery", "activate", "working-set"),
        (
            "Activate a tool from the catalog into this session's working set. "
            "Use after tool_search reveals a candidate; next round will see its schema."
        ),
        "read",
    ),
    # ab ops (personal-operations data plane)
    "plan": (
        "ops",
        ("write", "bd", "task", "plan-issue"),
        "Plan a bd issue (the task tracker).",
        "write",
    ),
    "capture": (
        "ops",
        ("write", "bd", "task", "capture"),
        "Capture a new bd task or note.",
        "write",
    ),
    "status": (
        "ops",
        ("read", "bd", "ready", "open"),
        "Show ready/open bd issues for this character.",
        "read",
    ),
    "drift": (
        "ops",
        ("read", "bd", "drift", "stale"),
        "Show stale or drifting bd issues.",
        "read",
    ),
    "reprioritize": (
        "ops",
        ("write", "bd", "priority"),
        "Change priority of a bd issue.",
        "write",
    ),
    "close": ("ops", ("write", "bd", "close"), "Close a bd issue.", "write"),
    "defer": ("ops", ("write", "bd", "defer"), "Defer a bd issue to a later date.", "write"),
    "retro": (
        "ops",
        ("write", "bd", "retro", "lesson"),
        "Record a retro lesson against a bd issue.",
        "write",
    ),
    "reopen": ("ops", ("write", "bd", "reopen"), "Reopen a closed bd issue.", "write"),
    "delete": ("ops", ("write", "bd", "delete"), "Delete a bd issue.", "write"),
    "update": (
        "ops",
        ("write", "bd", "update", "edit"),
        "Update a bd issue's title, body, or assignee.",
        "write",
    ),
    "search": (
        "ops",
        ("read", "bd", "search"),
        "Search bd issues by keyword.",
        "read",
    ),
    "list": (
        "ops",
        ("read", "bd", "list"),
        "List bd issues by status / assignee.",
        "read",
    ),
    "memories": (
        "ops",
        ("read", "bd", "memories"),
        "List bd-remembered notes for this character.",
        "read",
    ),
    "remember": (
        "ops",
        ("write", "bd", "memory", "store"),
        "Store a bd memory (persistent note across sessions).",
        "write",
    ),
    "forget": (
        "ops",
        ("write", "bd", "memory", "delete"),
        "Delete a bd-remembered note by key.",
        "write",
    ),
    "dep": (
        "ops",
        ("write", "bd", "dependency", "blocks"),
        "Add or remove a dependency between two bd issues.",
        "write",
    ),
    "label": (
        "ops",
        ("write", "bd", "label", "tag"),
        "Add or remove a label on a bd issue.",
        "write",
    ),
    "comments": ("ops", ("read", "bd", "comments"), "Read comments on a bd issue.", "read"),
    "find_duplicates": (
        "ops",
        ("read", "bd", "duplicate"),
        "Find candidate duplicate bd issues.",
        "read",
    ),
    "persist_focus_note": (
        "ops",
        ("write", "bd", "focus", "note"),
        "Persist a focus note for the current bd-tracked thread.",
        "write",
    ),
    # phraseology + atc
    "phraseology_lint": (
        "atc",
        ("read", "atc", "phraseology", "verify"),
        "Lint an utterance against FAA ATC phraseology conventions.",
        "read",
    ),
    # query
    "query_table": (
        "research",
        ("read", "sqlite", "query", "table"),
        "Run a read-only SQL query against a registered SQLite table.",
        "read",
    ),
    # citations
    "citation_lookup": (
        "research",
        ("read", "citation", "lookup"),
        "Resolve a citation identifier (DOI, arXiv id, etc.) to its full record.",
        "read",
    ),
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
    for name, (family, tags, description, tier) in BUILTIN_TOOL_METADATA.items():
        if name in catalog.entries:
            continue
        entry = ToolCatalogEntry(
            name=name,
            family=family,
            tags=tags,
            description=description,
            tier=tier,
            origin="builtin",
            source_path=None,
            registered_at=now_iso,
        )
        catalog.register(entry)
        inserted += 1
    return inserted
