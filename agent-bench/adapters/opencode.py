"""OpenCode adapter — drives the `opencode run` CLI against gx10.

https://github.com/sst/opencode · https://opencode.ai/docs/cli/

gx10 is wired as a custom OpenAI-compatible provider written into the workspace's
`opencode.json` (the `@ai-sdk/openai-compatible` npm provider). `opencode run`
completes after the single message and exits — ideal for headless benchmarking.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

        usage = _parse_usage(proc.stdout)
        return RunArtifacts(
            exit_ok=proc.returncode == 0,
            transcript=transcript_of(cmd, proc),
            diff=snapshot_diff(workspace),
            duration_s=duration,
            tokens_prompt=usage.tokens_prompt,
            tokens_completion=usage.tokens_completion,
            turns=usage.turns,
            extra={
                "returncode": proc.returncode,
                "timed_out": proc.returncode == TIMEOUT_RC,
                "opencode_cost_usd": usage.cost_usd,
                "cache_read_tokens": usage.cache_read,
                "cache_write_tokens": usage.cache_write,
                "per_step_tokens": usage.per_step,  # audit trail; see _parse_usage note
            },
        )


@dataclass
class _Usage:
    tokens_prompt: int = 0
    tokens_completion: int = 0
    turns: int = 0
    cost_usd: float = 0.0
    cache_read: int = 0
    cache_write: int = 0
    per_step: list[dict[str, int]] = field(default_factory=list)


def _parse_usage(stdout: str) -> _Usage:
    """Extract exact token usage from `opencode run --format json` (JSONL) output.

    opencode emits one JSON object per line; `step_finish` events carry
    `part.tokens.{input,output,reasoning,cache.{read,write}}` and `part.cost`.
    Token/cost are summed across step_finish events (per-step semantics, AI-SDK
    convention — undocumented by opencode, so the per-step breakdown is kept in
    `extra.per_step_tokens` for audit; if a release reports cumulative totals
    instead, switch the sum to a max here). Tolerant: skips non-JSON lines and
    also accepts a single top-level JSON array.
    """
    usage = _Usage()
    for event in _iter_events(stdout):
        if not _is_step_finish(event):
            continue
        part = event.get("part", {})
        tokens = part.get("tokens", {}) if isinstance(part, dict) else {}
        if not isinstance(tokens, dict):
            continue
        prompt = _as_int(tokens.get("input"))
        output = _as_int(tokens.get("output"))
        reasoning = _as_int(tokens.get("reasoning"))
        cache = tokens.get("cache", {}) if isinstance(tokens.get("cache"), dict) else {}
        usage.tokens_prompt += prompt
        usage.tokens_completion += output + reasoning  # reasoning is generated output
        usage.cache_read += _as_int(cache.get("read"))
        usage.cache_write += _as_int(cache.get("write"))
        usage.cost_usd += _as_float(part.get("cost"))
        usage.turns += 1
        usage.per_step.append({"input": prompt, "output": output, "reasoning": reasoning})
    return usage


def _iter_events(stdout: str) -> Iterator[dict[str, Any]]:
    text = stdout.strip()
    if not text:
        return
    # Some releases may emit a single JSON array rather than JSONL.
    if text[0] == "[":
        try:
            arr = json.loads(text)
        except json.JSONDecodeError:
            arr = []
        for obj in arr:
            if isinstance(obj, dict):
                yield obj
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] != "{":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _is_step_finish(event: dict[str, Any]) -> bool:
    if event.get("type") in {"step_finish", "step-finish"}:
        return True
    part = event.get("part")
    return isinstance(part, dict) and part.get("type") == "step-finish"


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


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
