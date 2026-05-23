"""Read-tier git tools. Scoped to the workspace root; run `git` as a
subprocess. Kept as three separate tool classes (not one dispatcher)
because status/diff/log have genuinely different arguments and the
model picks better when each intent has its own schema slot."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec

_GIT_TIMEOUT_S = 10


def _run_git(cwd: Path, args: list[str]) -> str:
    # Strip inherited ``GIT_*`` env vars so the explicit ``cwd=`` actually
    # points at the workspace's repo. When the harness is invoked from a
    # parent process that already set ``GIT_DIR`` / ``GIT_WORK_TREE``
    # (pre-commit hooks, a driver loop's pre-flight wrapper, a wrapping
    # editor), git ignores ``cwd=`` and silently talks to the parent
    # repo instead. The tools target ``cwd=`` by contract.
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        result = subprocess.run(  # noqa: S603 — args assembled from trusted tool params
            ["git", *args],  # noqa: S607 — `git` on PATH is expected
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            cwd=str(cwd),
            env=clean_env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired:
        return f"(git command timed out after {_GIT_TIMEOUT_S}s)"
    body = (result.stdout or "") + (result.stderr or "")
    return body.rstrip() or "(no output)"


def _truncate(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = "\n".join(lines[:max_lines])
    return f"{head}\n… [truncated at {max_lines} lines of {len(lines)}]"


@dataclass
class GitStatusTool:
    """`git status --short` scoped to the workspace."""

    root: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="git_status",
            description=(
                "Show the working-tree status: which files are "
                "modified, staged, untracked. Short format (one line "
                "per file). Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional paths to restrict status to. Default: whole working tree."
                        ),
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Git status",
        )

    def call(self, *, paths: list[str] | None = None) -> str:
        args = ["status", "--short"]
        if paths:
            args += ["--", *paths]
        out = _run_git(self.root, args)
        if out == "(no output)":
            return "(working tree clean)"
        return out


@dataclass
class GitDiffTool:
    """`git diff` scoped to the workspace. Truncated to keep tool
    output under control."""

    root: Path
    default_max_lines: int = 500

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="git_diff",
            description=(
                "Show unstaged changes (or staged, with "
                "`staged=true`). Truncated at 500 lines by default; "
                "pass `max_lines` to change that. Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "staged": {
                        "type": "boolean",
                        "description": "Show the staged diff instead. Default false.",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional paths to restrict the diff to.",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Line cap. Default 500.",
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Git diff",
        )

    def call(
        self,
        *,
        staged: bool = False,
        paths: list[str] | None = None,
        max_lines: int | None = None,
    ) -> str:
        args = ["diff"]
        if staged:
            args.append("--staged")
        if paths:
            args += ["--", *paths]
        out = _run_git(self.root, args)
        return _truncate(out, max_lines or self.default_max_lines)


@dataclass
class GitLogTool:
    """`git log --oneline` scoped to the workspace."""

    root: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="git_log",
            description=(
                "Show recent commits in oneline format (sha + "
                "subject). Default 20 entries. Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of commits. Default 20.",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional paths to restrict history to.",
                    },
                    "author": {
                        "type": "string",
                        "description": "Filter by author substring.",
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Git log",
        )

    def call(
        self,
        *,
        n: int = 20,
        paths: list[str] | None = None,
        author: str | None = None,
    ) -> str:
        args = ["log", "--oneline", f"-n{n}"]
        if author:
            args += [f"--author={author}"]
        if paths:
            args += ["--", *paths]
        return _run_git(self.root, args)
