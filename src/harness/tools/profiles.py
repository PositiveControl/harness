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
        "introspect",
    ),
}

DEFAULT_PROFILE = "core"


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
