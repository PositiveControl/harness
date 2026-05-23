"""Named collections of tools the CLI can enable as a group.

Each profile is a tuple of tool *names* (not instances). The CLI holds
the name → constructor map and only instantiates the tools the chosen
profile asks for. Profiles may list tools that are not yet implemented;
unknown names are warned about and skipped rather than raising, so
profiles stay forward-compatible as new tools land.

Token-cost budget target on a Qwen 2.5 chat template (measured via
`scripts/bench_tool_use.py --measure-tokens`):

  * minimal / diagnostic / phraseology: ≤ 1,000 tokens.
  * core (read-only chat):              ≤ 2,000 tokens.
  * coding / atc / scholar / ops:       ≤ 3,500 tokens.
  * full:                               unbounded (kitchen sink).

The targets relaxed when harness-s8sw added the reckon trio
(now / date_math / calc) to core + coding — the schema cost of three
reckon primitives is ~775 tokens on its own. Beyond the trio, the
heavier reckon tools (python_eval / tz_convert / stats / sun) stay
opt-in via `--tools-add reckon` (or per-tool) so casual chat doesn't
pay for them. Overloading the schema dilutes the model's tool-picking
and wastes every turn's context — the targets are guidance, not hard
limits, but a profile that drifts past should justify itself."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.character import Character

TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    "minimal": (),
    # Discovery-only bootstrap (harness-sbia). The agent boots with
    # almost no schema overhead (~550 tokens for these three specs vs
    # ~2,000 for core, measured on Qwen 2.5 7B-Instruct-4bit), then
    # drives its own tool acquisition:
    #
    #   1. tool_search "I need to compute X" → returns matching catalog entries
    #   2. load_tool name=<best fit>          → expands the working set
    #   3. <invoke the tool on the next round>
    #
    # The price is 1-2 extra rounds per turn the first time the agent
    # reaches for a new tool — once a tool is in the working set, calls
    # are free. Best for sessions where most turns are pure conversation
    # but the model needs the option to escalate to real tools without
    # paying for them on every prompt. See harness-atsz for the rationale.
    "core_minimal": (
        "tool_search",
        "load_tool",
        "introspect",
    ),
    # Explicit web-research set — search_web is the new member. Not in
    # core/coding because search queries leave the trust boundary.
    "research": (
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "search_memory",
        "search_facts",
        "search_web",
        "fetch_url",
        "geography",
        "spawn_subagent",
        "tool_search",
        "load_tool",
    ),
    # Read-only everyday chat: open a file, find files, grep, recall.
    # The three highest-leverage reckon primitives (now / date_math /
    # calc) ride along so the main persona stops fabricating dates
    # and arithmetic in everyday chat — the original goal of
    # harness-1u2h. python_eval + tz_convert + stats + sun are
    # deliberately kept out of core to stay inside the ~1500-token
    # schema budget; opt in via `--tools-add reckon` (or per-tool)
    # when needed (harness-s8sw — bench shows reckon-5 alone is ~1260
    # tokens, which pushed core over budget).
    "core": (
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "search_memory",
        "search_facts",
        "introspect",
        "now",
        "date_math",
        "calc",
        # tool_search rides in every non-minimal profile so the agent
        # can always answer "is there a tool for X?" before fabricating
        # (harness-ozx1 + the always-available decision documented in
        # harness-atsz's discussion of load_tool — until load_tool ships,
        # tool_search is read-only discovery + operator runs
        # --tools-add to actually expand the working set).
        "tool_search",
        "load_tool",
    ),
    # Active code collaboration — full read/write/shell/memory/git +
    # the same three reckon primitives (now/date_math/calc) so quick
    # date and unit math doesn't force a python_eval round-trip
    # (harness-s8sw). Coding's pre-existing schema was already large;
    # python_eval / tz_convert / stats / sun stay off-by-default and
    # available via --tools-add reckon.
    "coding": (
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "edit_file",
        "write_file",
        "stream_edit",
        "python_stream",
        "shell",
        "git_status",
        "git_diff",
        "git_log",
        "search_memory",
        "search_facts",
        "fetch_url",
        "introspect",
        "spawn_subagent",
        "now",
        "date_math",
        "calc",
        "tool_search",
        "load_tool",
    ),
    # ab's personal-operations tool set — harness-inj.5. Every tool
    # dispatches through the BeadsAdapter to ab's isolated beads DB
    # (HARNESS_AB_BD_DIR). No filesystem / shell / git / web — the
    # experience plane only touches the data plane through bd.
    "ops": (
        "plan",
        "capture",
        "status",
        "drift",
        "reprioritize",
        "close",
        "defer",
        "retro",
        "reopen",
        "delete",
        "update",
        "search",
        "list",
        "memories",
        "remember",
        "forget",
        "dep",
        "label",
        "comments",
        "find_duplicates",
        "persist_focus_note",
        "introspect",
        "tool_search",
        "load_tool",
    ),
    # atc (airton_c) — educational FAA-documentation expert. Read-tier
    # filesystem + scoped write (sandboxed to character/airton_c/workspace/
    # via --workspace), memory tools, search_web + fetch_url with an
    # aviation-source allowlist wired in the CLI builder, and
    # spawn_subagent for depth-1 read-only delegation (e.g. "go find
    # every IFR approach-procedure requirement in §91.175"). Excludes
    # shell and git_* — atc is a student-facing reference, not a
    # general-purpose agent. harness-xbk.3.
    "atc": (
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "edit_file",
        "write_file",
        "search_memory",
        "search_facts",
        "remember_fact",
        "remember_event",
        "transcript_ingest",
        "search_web",
        "fetch_url",
        "introspect",
        "spawn_subagent",
        "tool_search",
        "load_tool",
    ),
    # Phraseology-lint focused profile (harness-q35t). Single-purpose
    # mode for ATC controllers (or training scenarios) verifying
    # transmissions against JO 7110.65. Pairs the lint tool with read-
    # tier rulebook search so the model can cross-check its own
    # verdict; introspect for self-inspection. No write tier — this is
    # a verifier, not an authoring tool. Inherits the atc profile's
    # search_memory description override at registry-build time so the
    # router still treats search_memory as rulebook-first.
    "phraseology": (
        "phraseology_lint",
        "search_memory",
        "introspect",
        "tool_search",
        "load_tool",
    ),
    # airton_d (notes character) — curator-over-filesystem tool set.
    # Read-tier fs to query the notes tree (grep/glob/list_dir/read_file),
    # write-tier fs scoped to the workspace for capture + triage
    # (edit_file/write_file). search_memory + search_facts for "what did
    # we capture about X last week" recall across sessions.
    # remember_event lets the character commit a capture summary to
    # episodic so future sessions can find it. introspect for self-
    # inspection. No shell / git / web / subagent — notes are a closed
    # workspace, not a general-purpose agent.
    "notes": (
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "edit_file",
        "write_file",
        "search_memory",
        "search_facts",
        "remember_event",
        "introspect",
        "tool_search",
        "load_tool",
    ),
    # Memory-curation sessions. supersede_fact will join once
    # implemented (see bd issue harness-5tz).
    "memory": (
        "search_memory",
        "search_facts",
        "remember_fact",
        "remember_event",
        "scribe_session",
        "consolidate_memory",
        "tool_search",
        "load_tool",
    ),
    # Self-inspection. `stats` and `transcript_recent` will join once
    # implemented (see bd issues harness-m2e, harness-2mi).
    "diagnostic": (
        "search_memory",
        "search_facts",
        "introspect",
        "spawn_subagent",
        "tool_search",
        "load_tool",
    ),
    # Contract-driven retrieval (harness-xysp). For role-specialized
    # agents whose work is contract-shaped — the contract YAML declares
    # what each slot needs, `assemble_context` materializes a packaged
    # context bundle in one call. Pairs with search_memory / search_facts
    # for the cases the contract doesn't predict.
    "contract": (
        "assemble_context",
        "search_memory",
        "search_facts",
        "introspect",
        "tool_search",
        "load_tool",
    ),
    # airton_f (scholar) — contract over a markdown doc tree, plus
    # bounded external lookup. Two web surfaces:
    #   - search_scholar: structured paper search across Semantic
    #     Scholar + OpenAlex (free APIs, no auth required for
    #     personal use). Primary tool for academic / research
    #     queries.
    #   - search_web: general DDG search for non-academic
    #     orientation (Wikipedia, blogs, news).
    # fetch_url is host-allowlisted in the character's core.yaml
    # (scholar.google.com, arxiv.org, doi.org, en.wikipedia.org).
    # Pair with citation_grammar + lead_with_citation: the corpus
    # path emits §<path> (<doc>) citations, the external path emits
    # [arxiv:…] / [scholar:…] / [doi:…] / [wiki:…] tags (constitution
    # defines the form). No shell / git / write-tier fs — scholar
    # reads, doesn't author. `remember_event` is the one write-tier
    # tool in the roster: scholar uses it to persist multi-source
    # research summaries so the contract's `prior_discussion` slot
    # can recall them on future turns instead of re-searching.
    "scholar": (
        "assemble_context",
        "search_memory",
        "search_facts",
        "search_scholar",
        "search_web",
        "fetch_url",
        "remember_event",
        "introspect",
        "tool_search",
        "load_tool",
    ),
    # airton_g (the reckoner) — deterministic time + compute. The
    # four primitives kill date and math hallucinations across every
    # character that loads this profile. Companion memory + introspect
    # tools so the reckoner can record context across sessions and
    # explain its own roster on request. No fs / shell / git / web —
    # reckoning never needs the filesystem or network (harness-1u2h).
    "reckon": (
        "now",
        "date_math",
        "calc",
        "python_eval",
        "tz_convert",
        "stats",
        "sun",
        "search_memory",
        "search_facts",
        "introspect",
        "tool_search",
        "load_tool",
    ),
    # Kitchen-sink — every built-in tool the registry knows about.
    # Intended as the starting point for scripts/chat.sh + power users
    # who prefer to prune with --tools-drop rather than opt in to each
    # tool with --tools-add. Overshoots the ~1,500-token schema budget;
    # trim write-tier / web / shell for everyday chat. Ordering here
    # mirrors the other profiles (fs read → fs write → shell → git →
    # memory → web → self) so a side-by-side diff is readable
    # (harness-1nu).
    "full": (
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "edit_file",
        "write_file",
        "shell",
        "git_status",
        "git_diff",
        "git_log",
        "search_memory",
        "search_facts",
        "remember_fact",
        "remember_event",
        "scribe_session",
        "consolidate_memory",
        "search_web",
        "fetch_url",
        "introspect",
        "spawn_subagent",
        "assemble_context",
        "now",
        "date_math",
        "calc",
        "python_eval",
        "tz_convert",
        "stats",
        "sun",
        "tool_search",
        "load_tool",
    ),
}

DEFAULT_PROFILE = "core"


# Profile-scoped description overrides shipped with the harness.
# Currently empty — historical FAA-rulebook reframings for the atc
# and phraseology profiles moved to character data
# (`character/airton_c{,1}/tool_descriptions.yaml`, harness-xo7v) so
# any corpus-grounded persona can ship its own override without
# editing this module. Kept defined (rather than deleted) as a
# back-stop: a description universal across every character that
# uses a given profile would still belong here. Layered with
# `Character.tool_descriptions[profile]` at registry build time;
# character entries win on key collision.
BUILTIN_PROFILE_DESCRIPTIONS: dict[str, dict[str, str]] = {}


def apply_profile_descriptions(
    registry: object,
    profile: str,
    *,
    character: Character | None = None,
) -> None:
    """Apply description overrides for `profile` to `registry`. Sources
    layered in order (later wins on key collision):

      1. `BUILTIN_PROFILE_DESCRIPTIONS[profile]` — historical
         module-level entries; currently empty.
      2. `character.tool_descriptions[profile]` — per-character file
         loaded from `character/<name>/tool_descriptions.yaml`.

    No-op when neither layer has an entry for `profile`. Silently
    skips tools that aren't registered (a layer can name overrides
    for optional tools that --tools-drop may have removed). Typed as
    `object` for the registry to avoid an import cycle — the only
    call is .override_description(name, desc), enforced by duck-typing
    (and by tests).
    """
    overrides: dict[str, str] = {}
    overrides.update(BUILTIN_PROFILE_DESCRIPTIONS.get(profile, {}))
    if character is not None:
        overrides.update(character.tool_descriptions.get(profile, {}))
    if not overrides:
        return
    for name, desc in overrides.items():
        try:
            registry.override_description(name, desc)  # type: ignore[attr-defined]
        except KeyError:
            # Tool isn't in the final registry (e.g. dropped via
            # --tools-drop, or store not enabled) — silent skip is
            # correct; there's nothing to reframe.
            continue


def _expand(tokens: tuple[str, ...]) -> set[str]:
    """Expand a comma-split list of add/drop tokens. A token that names a
    profile contributes every tool in that profile; everything else
    passes through as a literal tool name (possibly a future tool not
    yet implemented — warned at registration time, not here). Empty
    strings dropped so CLI split of "" → ("",) is a no-op."""
    out: set[str] = set()
    for raw in tokens:
        t = raw.strip()
        if not t:
            continue
        if t in TOOL_PROFILES:
            out.update(TOOL_PROFILES[t])
        else:
            out.add(t)
    return out


def resolve_tool_names(
    profile: str,
    *,
    add: tuple[str, ...] = (),
    drop: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the sorted tuple of tool names for `profile`, with any
    user overrides from `add` / `drop` applied. Unknown profile raises
    ValueError; unknown tool names in add/drop pass through (they'll be
    caught + warned at registration time, which is forward-compatible
    with tools that haven't shipped yet). Profile names in add/drop
    expand to every member tool."""
    if profile not in TOOL_PROFILES:
        raise ValueError(f"unknown tool-set {profile!r}; available: {sorted(TOOL_PROFILES)}")
    names = set(TOOL_PROFILES[profile])
    names.update(_expand(add))
    names.difference_update(_expand(drop))
    return tuple(sorted(names))


def resolve_active(
    *,
    catalog: object | None = None,
    profile: str | None = None,
    names: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    families: tuple[str, ...] = (),
) -> set[str]:
    """Resolve a working-set spec into a concrete set of tool names —
    harness-fzvg.

    Three selection layers, unioned:
      1. Profile membership: every tool in TOOL_PROFILES[profile].
         Skipped when profile is None.
      2. Explicit names: passed through verbatim.
      3. Catalog tags + families: every catalog entry matching any
         supplied tag or family contributes its name. Skipped when
         catalog is None or tags+families are both empty.

    The result is the union; callers decide whether to further
    restrict (e.g. by intersecting with the registered set). Designed
    for ToolRegistry.set_active(resolve_active(...)) at session
    start.

    `catalog` is typed as `object` to avoid an import cycle with
    `harness.tools.catalog`. The duck-typed surface this function
    uses: `.by_tag(tag) -> Iterable` and `.by_family(family) -> Iterable`,
    each returning records with a `.name` attribute. Either method
    failing (catalog wrong type) raises at call time.
    """
    out: set[str] = set()
    if profile is not None:
        if profile not in TOOL_PROFILES:
            raise ValueError(f"unknown profile {profile!r}; available: {sorted(TOOL_PROFILES)}")
        out.update(TOOL_PROFILES[profile])
    out.update(names)
    if catalog is not None and (tags or families):
        for tag in tags:
            for entry in catalog.by_tag(tag):  # type: ignore[attr-defined]
                out.add(entry.name)
        for family in families:
            for entry in catalog.by_family(family):  # type: ignore[attr-defined]
                out.add(entry.name)
    return out
