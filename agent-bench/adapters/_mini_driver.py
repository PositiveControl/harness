"""Headless driver for mini-SWE-agent, run inside the tool's own interpreter.

The `mini` CLI is interactive-only (it wires asyncio stdin readers and crashes
on a non-tty), so the sanctioned headless path is the Python API documented in
mini-SWE-agent's own `run/hello_world.py`. This script is that path: build a
DefaultAgent against gx10 via LiteLLM, run the task to a terminal state, and
emit a metrics JSON the adapter reads back.

It imports `minisweagent` and so must run under the tool venv's python, NOT the
bench's. The adapter locates that interpreter and shells out to this file. Keep
it pure stdlib + minisweagent — it never imports bench code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import LimitsExceeded, TimeExceeded
from minisweagent.models.litellm_model import LitellmModel

# Per-command shell timeout inside the agent's environment. The build task only
# writes files; generous enough that a stray `node`/test invocation won't trip it.
ENV_COMMAND_TIMEOUT_S = 120


def _sum_tokens(messages: list[dict[str, object]]) -> tuple[int, int]:
    """Sum prompt/completion tokens across every model response in the trajectory.

    Matches the cost scorer's convention (sum every usage block), so the
    `artifacts` and `transcript` token sources agree.
    """
    prompt = completion = 0
    for msg in messages:
        extra = msg.get("extra")
        response = extra.get("response") if isinstance(extra, dict) else None
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
    return prompt, completion


def _render_transcript(messages: list[dict[str, object]]) -> str:
    """Flatten the message log into a readable transcript for replay."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):  # multimodal — keep the text parts
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
        lines.append(f"### {role}\n{content}")
    return "\n\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--metrics-out", required=True)
    ap.add_argument("--step-limit", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    task = Path(args.task_file).read_text()
    agent_cfg = yaml.safe_load((package_dir / "config" / "mini.yaml").read_text())["agent"]
    # Non-interactive: never prompt; bound by our step limit; cost limit off
    # (litellm can't price the local model, so dollar-cost gating is meaningless).
    agent_cfg.update(step_limit=args.step_limit, cost_limit=0.0, mode="yolo")

    model = LitellmModel(
        model_name=args.model,
        model_kwargs={"temperature": args.temperature, "drop_params": True},
    )
    env = LocalEnvironment(cwd=args.cwd, timeout=ENV_COMMAND_TIMEOUT_S)
    agent = DefaultAgent(model, env, **agent_cfg)

    exit_status = "ok"
    try:
        agent.run(task)
        exit_status = (agent.messages[-1].get("extra") or {}).get("exit_status", "ok")
    except (LimitsExceeded, TimeExceeded) as exc:
        exit_status = type(exc).__name__
    except Exception as exc:  # a framework crash is a run outcome, not a driver bug
        exit_status = f"error:{type(exc).__name__}: {exc}"

    prompt_tokens, completion_tokens = _sum_tokens(agent.messages)
    Path(args.metrics_out).write_text(
        json.dumps(
            {
                "exit_status": exit_status,
                "turns": agent.n_calls,
                "cost": agent.cost,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
            }
        )
    )
    # The transcript goes to stdout; the adapter captures it via the subprocess.
    print(_render_transcript(agent.messages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
