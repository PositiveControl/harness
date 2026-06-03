"""Aider adapter — drives the `aider` CLI non-interactively against gx10.

https://github.com/Aider-AI/aider
"""

from __future__ import annotations

import time
from pathlib import Path

from ._proc import TIMEOUT_RC, init_repo, run, snapshot_diff, transcript_of
from .base import Endpoint, RunArtifacts

# Pin and record. Bump deliberately; the value lands in every result row's manifest.
AIDER_VERSION = "0.86.2"


class AiderAdapter:
    name = "aider"
    version = AIDER_VERSION

    def prepare(self, workspace: Path) -> None:
        init_repo(workspace)

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
        proc = run(cmd, cwd=workspace, timeout_s=timeout_s, env_extra=env_extra)
        duration = time.perf_counter() - started

        return RunArtifacts(
            exit_ok=proc.returncode == 0,
            transcript=transcript_of(cmd, proc),
            diff=snapshot_diff(workspace),
            duration_s=duration,
            extra={"returncode": proc.returncode, "timed_out": proc.returncode == TIMEOUT_RC},
        )
