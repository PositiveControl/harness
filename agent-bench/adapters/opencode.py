"""OpenCode adapter — drives the `opencode run` CLI against gx10.

https://github.com/sst/opencode · https://opencode.ai/docs/cli/

gx10 is wired as a custom OpenAI-compatible provider written into the workspace's
`opencode.json` (the `@ai-sdk/openai-compatible` npm provider). `opencode run`
completes after the single message and exits — ideal for headless benchmarking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ._proc import TIMEOUT_RC, init_repo, run, snapshot_diff, transcript_of
from .base import Endpoint, RunArtifacts

OPENCODE_VERSION = "0.6.x"  # pin to the installed release before a real run
_PROVIDER_ID = "gx10"


class OpencodeAdapter:
    name = "opencode"
    version = OPENCODE_VERSION

    def prepare(self, workspace: Path) -> None:
        init_repo(workspace)

    def invoke(self, spec: str, gx10: Endpoint, workspace: Path, timeout_s: int) -> RunArtifacts:
        _write_config(workspace, gx10)
        # provider/model: opencode splits on the first '/', so the model id may itself
        # contain slashes (e.g. "Qwen/Qwen3-Coder-...").
        model_ref = f"{_PROVIDER_ID}/{gx10.model}"
        cmd = [
            "opencode",
            "run",
            "--model",
            model_ref,
            "--format",
            "json",  # raw events -> token usage parseable by the cost scorer
            spec,
        ]
        # opencode reads opencode.json from the cwd; keep the dummy key in env too.
        env_extra = {"OPENAI_API_KEY": gx10.api_key}
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


def _write_config(workspace: Path, gx10: Endpoint) -> None:
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            _PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "gx10",
                "options": {"baseURL": gx10.base_url, "apiKey": gx10.api_key},
                "models": {gx10.model: {"name": gx10.model}},
            }
        },
    }
    (workspace / "opencode.json").write_text(json.dumps(config, indent=2))
