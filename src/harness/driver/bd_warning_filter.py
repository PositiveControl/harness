"""Bd auto-export warning filter (harness-rtwm).

When the driver runs in a workspace that's outside or under a
.gitignored path in the operator's repo, every `bd <write>` command
emits ``Warning: auto-export: git add failed: exit status 1`` to
stderr. The warning is bd's own non-fatal status — `git add` fails
because the file is in an ignored tree — but it appears verbatim in
the shell tool's output and lands in the model's conversation.
loop_run=3e295564 carried it on twelve different `bd close` results.

This module installs a post_tool hook that strips the warning from
shell-tool output when the workspace is gitignored (or outside a git
repo entirely). User-facing bd usage is untouched; only the driver's
executor pipeline filters.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from harness.orchestrator.hooks import (
    Continue,
    PostToolContext,
    PostToolOutcome,
    ReplaceResult,
)
from harness.tools.base import ToolResult

# The exact warning bd emits when auto-export's `git add` step fails.
# Match the prefix only — bd has stamped variations of exit-status text
# across versions, but the "Warning: auto-export: git add failed"
# header is stable.
_WARNING_RE = re.compile(r"(?:\n)?Warning:\s*auto-export:\s*git add failed[^\n]*", re.MULTILINE)


def workspace_is_gitignored(workspace: Path) -> bool:
    """True iff `workspace` is ignored by git OR isn't inside a git
    work tree at all. Either case means bd's auto-export `git add` is
    futile and its warning is uninteresting. Returns False on any git
    error so we err toward NOT filtering."""
    try:
        # `git check-ignore` exits 0 when the path is ignored, 1 when
        # not ignored. From inside the workspace itself it may not be
        # in any git repo, so run from the parent so we hit the
        # outer repo's gitignore rules.
        parent = workspace.parent if workspace.parent != workspace else workspace
        result = subprocess.run(  # noqa: S603 — fixed git argv, no shell
            ["git", "check-ignore", "-q", str(workspace)],  # noqa: S607 — git from PATH
            cwd=str(parent),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return False
    if result.returncode == 0:
        return True
    # Exit 128: not a git repo. From the driver's POV the warning is
    # still meaningless — bd's `git add` will fail the same way.
    return result.returncode == 128


def _strip_warning(output: str) -> str:
    return _WARNING_RE.sub("", output).rstrip()


@dataclass
class BdAutoExportWarningFilter:
    """post_tool hook: strip the auto-export warning from shell tool
    output. Pre-tested against the workspace at construction time —
    `enabled=False` skips every check after that, so the hook is a
    cheap no-op on tracked workspaces."""

    name: str = "bd_auto_export_filter"
    enabled: bool = True
    filtered_count: int = field(default=0)

    def check(self, ctx: PostToolContext) -> PostToolOutcome:
        if not self.enabled or ctx.call.name != "shell":
            return Continue()
        if "auto-export: git add failed" not in ctx.result.output:
            return Continue()
        filtered = _strip_warning(ctx.result.output)
        if filtered == ctx.result.output:
            return Continue()
        self.filtered_count += 1
        return ReplaceResult(
            ToolResult(
                tool_name=ctx.result.tool_name,
                output=filtered,
                success=ctx.result.success,
                error=ctx.result.error,
                hits=ctx.result.hits,
                citations_grounded=ctx.result.citations_grounded,
            )
        )


def make_bd_auto_export_warning_filter(workspace: Path) -> BdAutoExportWarningFilter:
    """Construct a filter hook keyed to whether `workspace` is
    gitignored. The check runs once per pipeline construction (cheap;
    one `git check-ignore` subprocess). Tracked workspaces get a hook
    with `enabled=False`."""
    return BdAutoExportWarningFilter(enabled=workspace_is_gitignored(workspace))


__all__ = [
    "BdAutoExportWarningFilter",
    "make_bd_auto_export_warning_filter",
    "workspace_is_gitignored",
]
