"""Goose adapter — drives the `goose run` CLI (Block) against gx10.

https://github.com/block/goose · https://goose-docs.ai/docs/getting-started/providers/

gx10 is wired via goose's OpenAI provider pointed at a custom host. goose splits
the endpoint into OPENAI_HOST (scheme://host:port) + OPENAI_BASE_PATH (the request
path), so the bench's single `base_url` (a `/v1` root) is decomposed here.

`goose run --no-session -t <text>` runs one shot and exits — no session file.
NOTE: goose's provider/keyring env names shift across releases; verify
GOOSE_PROVIDER / GOOSE_DISABLE_KEYRING against the installed goose before a real run.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlsplit

from ._proc import TIMEOUT_RC, init_repo, run, snapshot_diff, transcript_of
from .base import Endpoint, RunArtifacts

GOOSE_VERSION = "1.x"  # pin to the installed release before a real run


class GooseAdapter:
    name = "goose"
    version = GOOSE_VERSION

    def prepare(self, workspace: Path) -> None:
        init_repo(workspace)

    def invoke(self, spec: str, gx10: Endpoint, workspace: Path, timeout_s: int) -> RunArtifacts:
        host, base_path = _split_endpoint(gx10.base_url)
        env_extra = {
            "GOOSE_PROVIDER": "openai",
            "GOOSE_MODEL": gx10.model,
            "GOOSE_DISABLE_KEYRING": "1",  # read the key from env, not the OS keyring
            "OPENAI_API_KEY": gx10.api_key,
            "OPENAI_HOST": host,
            "OPENAI_BASE_PATH": base_path,
        }
        cmd = [
            "goose",
            "run",
            "--no-session",  # headless: no session file
            "--quiet",  # print only the model response
            "--model",
            gx10.model,
            "-t",
            spec,
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


def _split_endpoint(base_url: str) -> tuple[str, str]:
    """`http://gx10:8000/v1` -> ("http://gx10:8000", "v1/chat/completions")."""
    parts = urlsplit(base_url)
    host = f"{parts.scheme}://{parts.netloc}"
    prefix = parts.path.strip("/")
    base_path = f"{prefix}/chat/completions" if prefix else "v1/chat/completions"
    return host, base_path
