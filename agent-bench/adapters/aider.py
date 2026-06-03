"""Aider adapter — drives the `aider` CLI non-interactively against gx10.

https://github.com/Aider-AI/aider
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .base import Endpoint, RunArtifacts

# Pin and record. Bump deliberately; the value lands in every result row's manifest.
AIDER_VERSION = "0.86.1"


class AiderAdapter:
    name = "aider"
    version = AIDER_VERSION

    def prepare(self, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        _git(workspace, "init", "-q")
        _git(workspace, "commit", "-q", "--allow-empty", "-m", "bench: empty baseline")

    def invoke(self, spec: str, gx10: Endpoint, workspace: Path, timeout_s: int) -> RunArtifacts:
        env_extra = {
            "OPENAI_API_BASE": gx10.base_url,
            "OPENAI_API_KEY": gx10.api_key,
        }
        cmd = [
            "aider",
            "--model",
            f"openai/{gx10.model}",
            "--yes-always",  # non-interactive: accept edits/commits
            "--no-auto-commits",  # we snapshot the diff ourselves
            "--no-gitignore",
            "--no-check-update",
            "--no-show-model-warnings",
            "--map-tokens",
            "1024",
            "--message",
            spec,  # the whole task as a single instruction
        ]
        started = time.perf_counter()
        proc = _run(cmd, cwd=workspace, timeout_s=timeout_s, env_extra=env_extra)
        duration = time.perf_counter() - started

        _git(workspace, "add", "-A")
        diff = _capture(["git", "diff", "--cached"], cwd=workspace)

        transcript = f"$ {' '.join(cmd)}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}"
        return RunArtifacts(
            exit_ok=proc.returncode == 0,
            transcript=transcript,
            diff=diff,
            duration_s=duration,
            extra={"returncode": proc.returncode, "timed_out": proc.returncode == _TIMEOUT_RC},
        )


_TIMEOUT_RC = -999


def _git(cwd: Path, *args: str) -> None:
    _run(["git", *args], cwd=cwd, timeout_s=60, env_extra={})


def _run(
    cmd: list[str], *, cwd: Path, timeout_s: int, env_extra: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    import os

    env = {**os.environ, **env_extra}
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
        out = _as_text(exc.stdout or "")
        err = _as_text(exc.stderr or "") + f"\n[bench] timed out after {timeout_s}s"
        return subprocess.CompletedProcess(cmd, _TIMEOUT_RC, out, err)


def _capture(cmd: list[str], *, cwd: Path) -> str:
    return _run(cmd, cwd=cwd, timeout_s=60, env_extra={}).stdout


def _as_text(value: str | bytes) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else value
