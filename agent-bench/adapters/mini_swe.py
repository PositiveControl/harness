"""mini-SWE-agent adapter — drives the agent headlessly against gx10.

https://github.com/SWE-agent/mini-swe-agent

mini-SWE-agent ships as a standalone CLI (`uv tool install mini-swe-agent`), but
its `mini` command is interactive-only and crashes on a non-tty. So we drive its
Python API through `_mini_driver.py`, executed by the tool venv's own
interpreter (minisweagent is never a dependency of the bench itself). The driver
runs the agent in a LocalEnvironment rooted at the workspace; we snapshot the
resulting git diff and read a metrics JSON the driver writes.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from ._proc import TIMEOUT_RC, init_repo, run, snapshot_diff, transcript_of
from .base import Endpoint, RunArtifacts

# Pin and record. Bump deliberately; the value lands in every result row's manifest.
MINI_VERSION = "2.4.1"
# Bound the agent's turns. Mirrors bench.yaml's advisory turn_cap; the runner's
# wall-clock timeout is the outer backstop.
MINI_STEP_LIMIT = 60

_DRIVER = Path(__file__).resolve().parent / "_mini_driver.py"


class MiniSweAdapter:
    name = "mini-swe-agent"
    version = MINI_VERSION

    def prepare(self, workspace: Path) -> None:
        init_repo(workspace)

    def invoke(self, spec: str, gx10: Endpoint, workspace: Path, timeout_s: int) -> RunArtifacts:
        interpreter = _tool_interpreter()
        env_extra = {
            "OPENAI_API_BASE": gx10.base_url,
            "OPENAI_API_KEY": gx10.api_key,
            "MSWEA_COST_TRACKING": "ignore_errors",  # local model has no litellm price
            "MSWEA_SILENT_STARTUP": "1",
        }
        # Pass the (large) spec and collect metrics via temp files — no argv escaping,
        # and kept out of the workspace so they never pollute the diff snapshot.
        with tempfile.TemporaryDirectory(prefix="mini-bench-") as tmp:
            task_file = Path(tmp) / "task.txt"
            task_file.write_text(spec)
            metrics_file = Path(tmp) / "metrics.json"

            cmd = [
                *interpreter,
                str(_DRIVER),
                "--task-file",
                str(task_file),
                "--model",
                f"openai/{gx10.model}",
                "--cwd",
                str(workspace),
                "--metrics-out",
                str(metrics_file),
                "--step-limit",
                str(MINI_STEP_LIMIT),
                "--temperature",
                str(gx10.temperature),
            ]
            started = time.perf_counter()
            proc = run(cmd, cwd=workspace, timeout_s=timeout_s, env_extra=env_extra)
            duration = time.perf_counter() - started
            metrics = _read_metrics(metrics_file)

        timed_out = proc.returncode == TIMEOUT_RC
        return RunArtifacts(
            exit_ok=proc.returncode == 0,
            transcript=transcript_of(cmd, proc),
            diff=snapshot_diff(workspace),
            duration_s=duration,
            tokens_prompt=_opt_int(metrics, "tokens_prompt"),
            tokens_completion=_opt_int(metrics, "tokens_completion"),
            turns=_opt_int(metrics, "turns"),
            extra={
                "returncode": proc.returncode,
                "timed_out": timed_out,
                "exit_status": metrics.get("exit_status"),
                "model_cost": metrics.get("cost"),
            },
        )


def _opt_int(metrics: dict[str, object], key: str) -> int | None:
    """A metric value as int, or None when absent/non-int (bools excluded)."""
    value = metrics.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read_metrics(path: Path) -> dict[str, object]:
    """Driver metrics, or empty on timeout/crash before it could write them."""
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _tool_interpreter() -> list[str]:
    """Locate the tool venv python that has minisweagent installed.

    The `mini` console-script shim's shebang points straight at it (stable, no
    uv cache hash). Fall back to `uv tool run` if the shim isn't found.
    """
    shim = shutil.which("mini")
    if shim:
        first_line = Path(shim).read_text(errors="replace").splitlines()[:1]
        if first_line and first_line[0].startswith("#!"):
            interpreter = first_line[0][2:].strip()
            if interpreter and Path(interpreter).exists():
                return [interpreter]
    return ["uv", "tool", "run", "--from", "mini-swe-agent", "python"]
