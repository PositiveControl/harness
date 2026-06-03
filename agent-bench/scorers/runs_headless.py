"""`runs_headless` — boot main.py under the SDL dummy driver.

A pygame game loops forever, so "ran" means: started and survived a short window
without crashing. We launch it, wait `BOOT_WINDOW_S`, and treat *still-running*
(killed by us) or a clean exit as success; an early non-zero exit with a traceback
is failure.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from adapters.base import RunArtifacts

from scorers.base import Scores

BOOT_WINDOW_S = 8.0
ENTRYPOINTS = ("main.py", "game.py", "run.py")


class RunsHeadlessScorer:
    name = "runs_headless"

    def score(self, workspace: Path, artifacts: RunArtifacts) -> Scores:
        entry = next((workspace / e for e in ENTRYPOINTS if (workspace / e).exists()), None)
        if entry is None:
            return {"runs_headless": False, "reason": "no entrypoint"}

        env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell; sandboxed bench run
            ["python", entry.name],  # noqa: S607 — `python` from the bench venv on PATH
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.perf_counter() + BOOT_WINDOW_S
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        survived = proc.poll() is None
        if survived:
            _kill_group(proc)
            return {"runs_headless": True, "reason": "survived boot window"}

        out = proc.stdout.read() if proc.stdout else ""
        crashed = proc.returncode != 0 or "Traceback" in out
        return {
            "runs_headless": not crashed,
            "reason": "clean early exit" if not crashed else "crash",
            "exit_code": proc.returncode,
        }


def _kill_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait(timeout=10)
