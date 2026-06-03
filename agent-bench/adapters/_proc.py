"""Subprocess + git helpers shared by every CLI-driven adapter.

Adapters shell out to a framework's CLI under a fixed argv (no shell), capture
stdout/stderr, and snapshot the resulting git diff as the run artifact.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

TIMEOUT_RC = -999


def run(
    cmd: list[str], *, cwd: Path, timeout_s: int, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run argv to completion; on timeout return a synthetic result with TIMEOUT_RC."""
    env = {**os.environ, **(env_extra or {})}
    try:
        return subprocess.run(  # noqa: S603 — fixed argv, no shell; benchmark runner by design
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = as_text(exc.stdout or "")
        err = as_text(exc.stderr or "") + f"\n[bench] timed out after {timeout_s}s"
        return subprocess.CompletedProcess(cmd, TIMEOUT_RC, out, err)


def git(cwd: Path, *args: str) -> None:
    run(["git", *args], cwd=cwd, timeout_s=60)


def init_repo(workspace: Path) -> None:
    """Fresh git repo with an empty baseline commit, so `git diff` captures all work."""
    workspace.mkdir(parents=True, exist_ok=True)
    git(workspace, "init", "-q")
    git(workspace, "commit", "-q", "--allow-empty", "-m", "bench: empty baseline")


def snapshot_diff(workspace: Path) -> str:
    git(workspace, "add", "-A")
    return run(["git", "diff", "--cached"], cwd=workspace, timeout_s=60).stdout


def transcript_of(cmd: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    return f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}"


def as_text(value: str | bytes) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value
