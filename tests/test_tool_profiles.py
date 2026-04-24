from __future__ import annotations

import pytest

from harness.tools.base import ToolRegistry, ToolSpec
from harness.tools.profiles import (
    DEFAULT_PROFILE,
    TOOL_PROFILE_DESCRIPTIONS,
    TOOL_PROFILES,
    apply_profile_descriptions,
    resolve_tool_names,
)


class _StubTool:
    """Minimal Tool-protocol implementation for registry tests."""

    def __init__(self, name: str, description: str, tier: str = "read") -> None:
        self._spec = ToolSpec(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {}, "required": []},
            tier=tier,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def call(self) -> str:
        return ""


def test_profiles_registered() -> None:
    assert "minimal" in TOOL_PROFILES
    assert "core" in TOOL_PROFILES
    assert "coding" in TOOL_PROFILES
    assert "memory" in TOOL_PROFILES
    assert "diagnostic" in TOOL_PROFILES


def test_default_profile_resolves() -> None:
    assert DEFAULT_PROFILE in TOOL_PROFILES


def test_minimal_is_empty() -> None:
    assert resolve_tool_names("minimal") == ()


def test_core_is_read_only() -> None:
    names = resolve_tool_names("core")
    assert "read_file" in names
    assert "write_file" not in names
    assert "shell" not in names


def test_coding_is_superset_of_core() -> None:
    core = set(resolve_tool_names("core"))
    coding = set(resolve_tool_names("coding"))
    assert core <= coding
    assert "write_file" in coding
    assert "shell" in coding


def test_add_override() -> None:
    names = resolve_tool_names("core", add=("write_file",))
    assert "write_file" in names
    assert "read_file" in names


def test_drop_override() -> None:
    names = resolve_tool_names("coding", drop=("shell",))
    assert "shell" not in names
    assert "write_file" in names


def test_add_and_drop_compose() -> None:
    names = resolve_tool_names("core", add=("write_file",), drop=("search_facts",))
    assert "write_file" in names
    assert "search_facts" not in names


def test_add_accepts_unknown_names() -> None:
    """Unknown tool names in add/drop pass through — they're filtered to
    available tools at registration time, keeping profiles forward-compatible
    with tools that haven't shipped yet."""
    names = resolve_tool_names("core", add=("future_tool",))
    assert "future_tool" in names


def test_drop_of_missing_name_is_noop() -> None:
    names = resolve_tool_names("core", drop=("not_there",))
    assert names == resolve_tool_names("core")


def test_empty_add_or_drop_strings_ignored() -> None:
    # Simulates a CLI splitting "" → ("",) which would otherwise add a
    # blank entry.
    names = resolve_tool_names("core", add=("",), drop=("",))
    assert names == resolve_tool_names("core")


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown tool-set"):
        resolve_tool_names("does-not-exist")


def test_sorted_output() -> None:
    names = resolve_tool_names("coding")
    assert list(names) == sorted(names)


def test_add_expands_profile_name() -> None:
    """A profile name passed to --tools-add expands to every tool in
    that profile, not a literal tool named 'research'. Regression test
    for harness-akq where profile-name adds were silently dropped."""
    base = set(resolve_tool_names("core"))
    expanded = set(resolve_tool_names("core", add=("research",)))
    assert "search_web" in expanded
    assert expanded >= base


def test_drop_expands_profile_name() -> None:
    """Symmetric: dropping 'memory' as a profile name removes each of
    its tools, not a literal tool called 'memory'."""
    base = set(resolve_tool_names("coding", add=("research",)))
    dropped = set(resolve_tool_names("coding", add=("research",), drop=("research",)))
    assert "search_web" in base
    assert "search_web" not in dropped


def test_add_mixed_profile_and_literal() -> None:
    names = set(resolve_tool_names("minimal", add=("research", "edit_file")))
    assert "search_web" in names  # from research profile
    assert "edit_file" in names  # literal


def test_introspect_included_in_default_profiles() -> None:
    """harness-e9i: introspect is a read-tier tool the agent calls to
    describe its own capabilities. It ships in the default chat
    profiles (core, coding, diagnostic) so 'what can you do?' doesn't
    hallucinate."""
    for profile in ("core", "coding", "diagnostic"):
        assert "introspect" in resolve_tool_names(profile), profile


def test_introspect_not_in_minimal_or_research() -> None:
    """Minimal is deliberately empty; research is web-focused. Neither
    carries introspect — opt in via --tools-add."""
    assert "introspect" not in resolve_tool_names("minimal")
    assert "introspect" not in resolve_tool_names("research")


# ---------- atc (airton_c) profile — harness-xbk.3 ----------


def test_atc_profile_registered() -> None:
    assert "atc" in TOOL_PROFILES


def test_atc_includes_read_search_and_memory_tools() -> None:
    """atc's Phase-1 tool set: read-tier fs, scoped write, memory +
    retrieval, web research, introspect. The profile encodes the
    audience: PPL/IFR students asking rule questions against a RAG
    corpus."""
    names = set(resolve_tool_names("atc"))
    for read_tool in ("read_file", "list_dir", "grep", "glob"):
        assert read_tool in names, read_tool
    for mem_tool in ("search_memory", "search_facts", "remember_fact", "remember_event"):
        assert mem_tool in names, mem_tool
    for web_tool in ("search_web", "fetch_url"):
        assert web_tool in names, web_tool
    assert "introspect" in names


# ---------- description overrides (harness-80h7) ----------


def test_override_description_replaces_spec_description() -> None:
    """override_description() updates what specs() emits without mutating
    the underlying Tool instance. Downstream consumers (router, model
    schema, introspect) all flow through specs(), so this is the only
    choke point an override needs to touch."""
    registry = ToolRegistry()
    tool = _StubTool("search_memory", "original description")
    registry.register(tool)
    registry.override_description("search_memory", "rulebook-specific description")

    [spec] = registry.specs()
    assert spec.description == "rulebook-specific description"
    # Underlying tool spec is unchanged — override is applied at render time.
    assert tool.spec.description == "original description"


def test_override_description_raises_for_unregistered_tool() -> None:
    """Overrides target real tools only; typos surface immediately
    rather than silently no-op'ing."""
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.override_description("not_a_tool", "whatever")


def test_specs_without_override_unchanged() -> None:
    """Tools without an override round-trip through specs() untouched."""
    registry = ToolRegistry()
    registry.register(_StubTool("read_file", "read a file"))
    [spec] = registry.specs()
    assert spec.description == "read a file"


def test_apply_profile_descriptions_atc_reframes_search_tools() -> None:
    """atc profile maps search_memory → rulebook framing and search_facts
    → user-facts-only framing. Regression guard for the hallucination
    where 'can a ground controller clear takeoff' misrouted to
    search_facts because the generic descriptions made it look
    fact-shaped."""
    registry = ToolRegistry()
    registry.register(_StubTool("search_memory", "generic episodic search"))
    registry.register(_StubTool("search_facts", "generic fact search"))
    apply_profile_descriptions(registry, "atc")

    specs = {s.name: s.description for s in registry.specs()}
    assert "JO 7110.65" in specs["search_memory"]
    assert "rule-shaped" in specs["search_memory"].lower() or "rule" in specs["search_memory"]
    assert "user" in specs["search_facts"].lower()


def test_apply_profile_descriptions_skips_unregistered_tools() -> None:
    """A profile can list overrides for optional tools (--tools-drop may
    have removed them). Missing tools are silently skipped — the
    remaining overrides still apply."""
    registry = ToolRegistry()
    # Only search_memory is registered; search_facts is absent.
    registry.register(_StubTool("search_memory", "generic"))
    apply_profile_descriptions(registry, "atc")  # must not raise

    [spec] = registry.specs()
    assert "JO 7110.65" in spec.description


def test_apply_profile_descriptions_noop_for_profile_without_overrides() -> None:
    """Profiles not in TOOL_PROFILE_DESCRIPTIONS leave specs untouched."""
    registry = ToolRegistry()
    registry.register(_StubTool("read_file", "read a file"))
    apply_profile_descriptions(registry, "core")  # no override map

    [spec] = registry.specs()
    assert spec.description == "read a file"


def test_atc_description_overrides_declared() -> None:
    """The atc map exists and covers both search tools the router confuses."""
    assert "atc" in TOOL_PROFILE_DESCRIPTIONS
    assert "search_memory" in TOOL_PROFILE_DESCRIPTIONS["atc"]
    assert "search_facts" in TOOL_PROFILE_DESCRIPTIONS["atc"]


def test_atc_allows_scoped_write_subagent_but_excludes_shell_and_git() -> None:
    """atc can write into its workspace (edit_file / write_file,
    sandboxed to character/airton_c/workspace/ by --workspace) and
    spawn read-only subagents for depth-1 research, but never invokes
    shell or git_*. atc is a student-facing reference, not a general-
    purpose agent."""
    names = set(resolve_tool_names("atc"))
    assert "edit_file" in names
    assert "write_file" in names
    assert "spawn_subagent" in names
    assert "shell" not in names
    for git_tool in ("git_status", "git_diff", "git_log"):
        assert git_tool not in names, git_tool
