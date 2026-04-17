from __future__ import annotations

import pytest

from harness.tools.profiles import (
    DEFAULT_PROFILE,
    TOOL_PROFILES,
    resolve_tool_names,
)


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
