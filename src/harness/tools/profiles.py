"""Named collections of tools the CLI can enable as a group.

Each profile is a tuple of tool *names* (not instances). The CLI holds
the name → constructor map and only instantiates the tools the chosen
profile asks for. Profiles may list tools that are not yet implemented;
unknown names are warned about and skipped rather than raising, so
profiles stay forward-compatible as new tools land.

Token-cost budget target: each profile should cost ~1,500 tokens of
tool-schema overhead or less on a Qwen 2.5 chat template (measured via
`scripts/bench_tool_use.py --measure-tokens`). Overloading the schema
dilutes the model's tool-picking and wastes every turn's context."""

from __future__ import annotations

TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    "minimal": (),
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
        "spawn_subagent",
    ),
    # Read-only everyday chat: open a file, find files, grep, recall.
    "core": (
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "search_memory",
        "search_facts",
        "introspect",
    ),
    # Active code collaboration — full read/write/shell/memory/git.
    "coding": (
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
        "fetch_url",
        "introspect",
        "spawn_subagent",
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
    ),
    # Self-inspection. `stats` and `transcript_recent` will join once
    # implemented (see bd issues harness-m2e, harness-2mi).
    "diagnostic": (
        "search_memory",
        "search_facts",
        "introspect",
        "spawn_subagent",
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
    ),
}

DEFAULT_PROFILE = "core"


# Profile-scoped description overrides. Reframes a generic tool for a
# character whose episodic / semantic substrate is domain-specific and
# whose router-tier intent distribution differs from the default chat
# use case. Applied via ToolRegistry.override_description() at build
# time; missing tools are ignored so a profile can list overrides for
# tools that may or may not make the final cut after --tools-drop.
#
# atc rationale: episodic seeds ARE the FAA JO 7110.65 rulebook chunks
# (tier=seed, loaded from character/<name>/seed_memories/). Generic
# description "past events and lessons" reads as autobiographical,
# which pushes the small-model router toward search_facts on rule
# questions (observed hallucination where "can a ground controller
# clear takeoff" misrouted to search_facts → empty store → fabricated
# affirmative). The rewording names the rulebook explicitly so the
# router picks search_memory when it must go hunt a section, and —
# more importantly — stops picking search_facts for rule-shaped
# questions that should have been answered from pre-retrieved context.
TOOL_PROFILE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "atc": {
        "search_memory": (
            "Search the FAA JO 7110.65 air-traffic control rulebook for "
            "sections, procedures, phraseology, separation minimums, and "
            "controller responsibilities. Use for ANY rule-shaped "
            "question — anchored ('what does §5-5-4 say') or bareword "
            "('can ground clear takeoff', 'who issues go-arounds'). "
            "Search-first beats guess-and-answer; pre-retrieved context "
            "is not guaranteed to carry the needed chunk. NOT for "
            "user-relationship facts — use search_facts for those. "
            "Empty-signal prompts ('test', 'ping', 'this page "
            "intentionally left blank', 'are you alive', repeat-char "
            "mash, lorem ipsum) are NOT rule-shaped — return null and "
            "let the banter intercept handle them."
        ),
        "search_facts": (
            "Search atomic facts about the user (study plans, exam "
            "timelines, stated preferences). Returns (subject, "
            "predicate, object) triples. NOT for ATC rules or "
            "procedures — use search_memory for those."
        ),
    },
    # Phraseology profile inherits atc's rulebook-first search_memory
    # framing — the lint tool's pre-model retrieval and the model's
    # cross-check both target JO 7110.65, not autobiographical memory.
    "phraseology": {
        "search_memory": (
            "Search the FAA JO 7110.65 air-traffic control rulebook for "
            "the canonical phraseology, slot templates, or section text "
            "behind a controller utterance. Use to fetch the rulebook "
            "context BEFORE forming a verdict; the phraseology_lint tool "
            "calls retrieval internally but exposes only the structured "
            "verdict. Use this when the user wants to read the section "
            "itself."
        ),
    },
}


def apply_profile_descriptions(registry: object, profile: str) -> None:
    """Apply TOOL_PROFILE_DESCRIPTIONS[profile] to `registry` via its
    override_description() method. No-op for profiles with no override
    map; silently skips tools that aren't registered (a profile can
    name overrides for optional tools). Typed as `object` to avoid an
    import cycle — the only call is .override_description(name, desc),
    enforced by duck-typing (and by tests)."""
    overrides = TOOL_PROFILE_DESCRIPTIONS.get(profile)
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
