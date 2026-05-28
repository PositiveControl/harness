"""Tests for the bd auto-export warning filter (harness-rtwm)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.driver.bd_warning_filter import (
    BdAutoExportWarningFilter,
    _strip_warning,
    make_bd_auto_export_warning_filter,
    workspace_is_gitignored,
)
from harness.orchestrator.hooks import (
    Continue,
    PostToolContext,
    ReplaceResult,
)
from harness.tools.base import ToolCall, ToolResult, ToolSpec


def _spec(name: str = "shell") -> ToolSpec:
    return ToolSpec(name=name, description="", parameters={}, tier="write")


def _post_ctx(name: str, args: dict[str, object], result: ToolResult) -> PostToolContext:
    return PostToolContext(
        call=ToolCall(name=name, arguments=dict(args)),
        result=result,
        spec=_spec(name),
    )


# ---- _strip_warning ----------------------------------------------------


def test_strip_removes_warning_line() -> None:
    raw = (
        "exit=0\n"
        "✓ Closed harness-yd3m — §15b Controls — pause toggle: Closed "
        "Warning: auto-export: git add failed: exit status 1"
    )
    out = _strip_warning(raw)
    assert "auto-export" not in out
    assert "✓ Closed harness-yd3m" in out


def test_strip_handles_warning_on_own_line() -> None:
    raw = "exit=0\n✓ Closed thing\nWarning: auto-export: git add failed: exit status 1"
    out = _strip_warning(raw)
    assert "auto-export" not in out
    assert "✓ Closed thing" in out


def test_strip_preserves_unrelated_warnings() -> None:
    raw = "exit=0\nWarning: something else important"
    assert _strip_warning(raw) == raw


# ---- workspace_is_gitignored ------------------------------------------


def test_workspace_is_gitignored_true_for_ignored_path(tmp_path: Path) -> None:
    """When the workspace is under a gitignored prefix, returns True."""
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)  # noqa: S607
    (tmp_path / ".gitignore").write_text("scratch/\n")
    scratch = tmp_path / "scratch" / "workspace"
    scratch.mkdir(parents=True)
    assert workspace_is_gitignored(scratch) is True


def test_workspace_is_gitignored_false_for_tracked_path(tmp_path: Path) -> None:
    """A path inside a git repo that is NOT ignored returns False."""
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)  # noqa: S607
    sub = tmp_path / "src"
    sub.mkdir()
    assert workspace_is_gitignored(sub) is False


def test_workspace_is_gitignored_true_outside_any_repo(tmp_path: Path) -> None:
    """A path outside a git repo (exit 128) is treated as 'no git
    cares' — same effective outcome as 'gitignored'."""
    # tmp_path itself is not in a git repo (pytest's tmp dir).
    assert workspace_is_gitignored(tmp_path) is True


# ---- BdAutoExportWarningFilter -----------------------------------------


def _shell_result(output: str) -> ToolResult:
    return ToolResult(tool_name="shell", output=output, success=True)


def test_hook_no_op_when_disabled() -> None:
    hook = BdAutoExportWarningFilter(enabled=False)
    result = _shell_result("exit=0\n✓ Closed x Warning: auto-export: git add failed: exit status 1")
    outcome = hook.check(_post_ctx("shell", {"cmd": "bd close x"}, result))
    assert isinstance(outcome, Continue)
    assert hook.filtered_count == 0


def test_hook_no_op_for_non_shell_calls() -> None:
    hook = BdAutoExportWarningFilter(enabled=True)
    result = _shell_result("Warning: auto-export: git add failed: exit status 1")
    outcome = hook.check(_post_ctx("read_file", {"path": "x"}, result))
    assert isinstance(outcome, Continue)


def test_hook_no_op_when_warning_absent() -> None:
    hook = BdAutoExportWarningFilter(enabled=True)
    result = _shell_result("exit=0\nordinary shell output")
    outcome = hook.check(_post_ctx("shell", {"cmd": "ls"}, result))
    assert isinstance(outcome, Continue)


def test_hook_replaces_result_when_warning_present() -> None:
    hook = BdAutoExportWarningFilter(enabled=True)
    raw = "exit=0\n✓ Closed harness-yd3m Warning: auto-export: git add failed: exit status 1"
    outcome = hook.check(_post_ctx("shell", {"cmd": "bd close harness-yd3m"}, _shell_result(raw)))
    assert isinstance(outcome, ReplaceResult)
    assert "auto-export" not in outcome.result.output
    assert "✓ Closed harness-yd3m" in outcome.result.output
    assert outcome.result.success is True
    assert hook.filtered_count == 1


def test_hook_counts_multiple_filters() -> None:
    hook = BdAutoExportWarningFilter(enabled=True)
    raw = "exit=0\n✓ Closed x Warning: auto-export: git add failed: exit status 1"
    for _ in range(3):
        hook.check(_post_ctx("shell", {"cmd": "bd close x"}, _shell_result(raw)))
    assert hook.filtered_count == 3


# ---- make_bd_auto_export_warning_filter --------------------------------


def test_factory_enables_when_gitignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harness.driver.bd_warning_filter.workspace_is_gitignored",
        lambda _: True,
    )
    hook = make_bd_auto_export_warning_filter(tmp_path)
    assert hook.enabled is True


def test_factory_disables_when_tracked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harness.driver.bd_warning_filter.workspace_is_gitignored",
        lambda _: False,
    )
    hook = make_bd_auto_export_warning_filter(tmp_path)
    assert hook.enabled is False
