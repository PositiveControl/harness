"""Tests for `harness daemon` — harness-ppi0.

Subprocess-based integration coverage. The daemon command wires
runtime.Heartbeat (covered exhaustively in test_heartbeat.py) into
a Typer subcommand with SIGTERM hygiene + an alive-tick placeholder
task. These tests verify the CLI plumbing — flags, exit codes,
logged output — not the heartbeat internals.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# `harness` script lives next to the venv's python (pyproject.toml
# [project.scripts] harness = "harness.cli:app"). Resolve via
# sys.executable so the subprocess inherits the same env the test runner uses.
HARNESS_BIN = Path(sys.executable).parent / "harness"
HARNESS_CMD = [str(HARNESS_BIN)]


def _env() -> dict[str, str]:
    """A minimal env that lets the daemon find the default character
    without dragging in the user's HARNESS_* environment overrides."""
    base = os.environ.copy()
    # Drop overrides that could redirect the character path; leave
    # PYTHONPATH/PATH/HOME alone so the venv resolves.
    for key in ("HARNESS_CHARACTER_NAME", "HARNESS_DATA_DIR"):
        base.pop(key, None)
    return base


def test_daemon_tick_once_exits_clean_with_heartbeat_log() -> None:
    """--tick-once: fires each task once, prints the alive-tick line,
    and exits 0. Smoke test for the CLI plumbing."""
    proc = subprocess.run(
        [*HARNESS_CMD, "daemon", "--tick-once"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert "heartbeat daemon starting" in proc.stdout
    assert "heartbeat tick" in proc.stdout
    assert "tick-once complete" in proc.stdout


def _norm(s: str) -> str:
    """Flatten whitespace — Rich wraps long lines at the terminal width
    even with capture_output, so substring assertions against the raw
    stdout are flaky."""
    return " ".join(s.split())


def test_daemon_tick_once_registers_built_in_tasks_by_default() -> None:
    """Default daemon registers 3 tasks: heartbeat_alive + compaction +
    consolidation. tick-once fires all three; the alive line + each
    task's outcome line all appear in stdout."""
    proc = subprocess.run(
        [*HARNESS_CMD, "daemon", "--tick-once"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    assert "3 task(s) registered" in out
    assert "heartbeat compaction" in out
    assert "heartbeat consolidation" in out


def test_daemon_compaction_disabled_when_interval_zero() -> None:
    """--compaction-interval 0 drops compaction; consolidation still
    registers."""
    proc = subprocess.run(
        [*HARNESS_CMD, "daemon", "--tick-once", "--compaction-interval", "0"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    assert "2 task(s) registered" in out
    assert "heartbeat compaction" not in out
    assert "heartbeat consolidation" in out


def test_daemon_consolidation_disabled_when_interval_zero() -> None:
    """--consolidation-interval 0 drops consolidation; compaction still
    registers."""
    proc = subprocess.run(
        [*HARNESS_CMD, "daemon", "--tick-once", "--consolidation-interval", "0"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    assert "2 task(s) registered" in out
    assert "heartbeat compaction" in out
    assert "heartbeat consolidation" not in out


def test_daemon_both_optional_tasks_disabled_leaves_alive_only() -> None:
    """Disable both maintenance tasks: only heartbeat_alive remains."""
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "daemon",
            "--tick-once",
            "--compaction-interval",
            "0",
            "--consolidation-interval",
            "0",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    out = _norm(proc.stdout)
    assert "1 task(s) registered" in out
    assert "heartbeat compaction" not in out
    assert "heartbeat consolidation" not in out


def test_daemon_tick_once_with_explicit_character() -> None:
    """--character airton is the documented form; verify it doesn't
    blow up when supplied explicitly."""
    proc = subprocess.run(
        [*HARNESS_CMD, "daemon", "--character", "airton", "--tick-once"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert "for airton" in proc.stdout


def test_daemon_help_lists_flags() -> None:
    """`--help` is the documentation surface; pin the flag names so
    they don't silently rename."""
    proc = subprocess.run(
        [*HARNESS_CMD, "daemon", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    # Typer renders option names verbatim; assert each documented flag
    # appears at least once.
    for flag in (
        "--character",
        "--interval-default",
        "--compaction-interval",
        "--compaction-model",
        "--consolidation-interval",
        "--consolidation-min-working",
        "--tick-once",
    ):
        assert flag in proc.stdout, f"missing flag {flag!r} in --help output"


def test_daemon_sigterm_exits_zero() -> None:
    """Start the daemon as a long-running subprocess (no --tick-once),
    let it tick once, then send SIGTERM. Verify clean shutdown with
    exit code 0 + the 'stopped' line printed."""
    proc = subprocess.Popen(
        [*HARNESS_CMD, "daemon", "--interval-default", "0.5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
    )
    try:
        # Wait long enough for at least one tick line. Poll stdout
        # cheaply; bail at 6s so a hung daemon doesn't wedge the test.
        deadline = time.monotonic() + 6.0
        tick_seen = False
        out_chunks: list[str] = []
        while time.monotonic() < deadline:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                if proc.poll() is not None:
                    break
                continue
            out_chunks.append(line)
            if "heartbeat tick" in line:
                tick_seen = True
                break
        assert tick_seen, f"no tick line within 6s. captured:\n{''.join(out_chunks)}"
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
        assert proc.returncode == 0, f"SIGTERM did not produce clean exit; rc={proc.returncode}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_daemon_sigint_exits_zero() -> None:
    """SIGINT (Ctrl+C) path mirrors SIGTERM — both should clean-exit
    via Heartbeat.stop()."""
    proc = subprocess.Popen(
        [*HARNESS_CMD, "daemon", "--interval-default", "0.5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
    )
    try:
        # Give it a moment to start the loop.
        time.sleep(1.5)
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        assert proc.returncode == 0, f"SIGINT did not produce clean exit; rc={proc.returncode}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
