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
    # Read-only everyday chat: open a file, find files, grep, recall.
    "core": (
        "read_file",
        "list_dir",
        "grep",
        "glob",
        "search_memory",
        "search_facts",
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
    ),
    # Memory-curation sessions. Live-write tools will join once
    # implemented (see bd issues harness-vn5, harness-1tu, harness-5tz).
    "memory": (
        "search_memory",
        "search_facts",
    ),
    # Self-inspection. `stats` and `transcript_recent` will join once
    # implemented (see bd issues harness-m2e, harness-2mi).
    "diagnostic": (
        "search_memory",
        "search_facts",
    ),
}

DEFAULT_PROFILE = "core"


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
    with tools that haven't shipped yet)."""
    if profile not in TOOL_PROFILES:
        raise ValueError(f"unknown tool-set {profile!r}; available: {sorted(TOOL_PROFILES)}")
    names = set(TOOL_PROFILES[profile])
    names.update(n.strip() for n in add if n.strip())
    names.difference_update(n.strip() for n in drop if n.strip())
    return tuple(sorted(names))
