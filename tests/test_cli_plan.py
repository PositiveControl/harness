"""Tests for `harness plan bootstrap` — harness-snn2.

Subprocess-based integration coverage. The bootstrap command wires
bd_source.build_plan_from_bd + JsonPlanStore.save into a CLI; tests
verify the CLI plumbing (flags, exit codes, dry-run vs save, file
shape, idempotency) rather than re-testing the underlying components
(which have their own unit tests).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_BIN = Path(sys.executable).parent / "harness"
HARNESS_CMD = [str(HARNESS_BIN)]


def _env() -> dict[str, str]:
    """Clean env, dropping character-redirecting HARNESS_* vars (same
    pattern as test_cli_daemon.py)."""
    base = os.environ.copy()
    for key in ("HARNESS_CHARACTER_NAME", "HARNESS_DATA_DIR"):
        base.pop(key, None)
    return base


def _norm(s: str) -> str:
    return " ".join(s.split())


def test_plan_bootstrap_dry_run_prints_plan_without_writing(tmp_path: Path) -> None:
    """--dry-run: command prints the plan JSON but doesn't create any
    file under --plans-dir."""
    plans_dir = tmp_path / "plans"
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "plan",
            "bootstrap",
            "--dry-run",
            "--plans-dir",
            str(plans_dir),
            "--assignee",
            "mark",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = _norm(proc.stdout)
    assert "plan bootstrap" in out
    assert "dry-run; nothing written" in out
    # Plan JSON appears (the root subgoal at minimum).
    assert "bd:mark" in out
    # No file written.
    assert not plans_dir.exists() or not list(plans_dir.iterdir())


def test_plan_bootstrap_writes_plan_to_plans_dir(tmp_path: Path) -> None:
    """Non-dry-run: a JSON file appears at <plans_dir>/<plan_id>.json
    with the expected shape."""
    plans_dir = tmp_path / "plans"
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "plan",
            "bootstrap",
            "--plans-dir",
            str(plans_dir),
            "--assignee",
            "mark",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    plan_file = plans_dir / "bd:mark.json"
    assert plan_file.exists()
    payload = json.loads(plan_file.read_text())
    assert payload["id"] == "bd:mark"
    assert payload["root_subgoal_id"] == "bd:mark:root"
    assert "subgoals" in payload
    assert "bd:mark:root" in payload["subgoals"]


def test_plan_bootstrap_idempotent_on_plan_id_and_structure(tmp_path: Path) -> None:
    """Re-running on the same bd state produces a Plan with the same
    id, same subgoal ids, and same statuses. Timestamps may update —
    structural idempotency is the contract."""
    plans_dir = tmp_path / "plans"
    cmd = [
        *HARNESS_CMD,
        "plan",
        "bootstrap",
        "--plans-dir",
        str(plans_dir),
        "--assignee",
        "mark",
    ]
    for _ in range(2):
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=_env(),
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
    payload = json.loads((plans_dir / "bd:mark.json").read_text())
    assert payload["id"] == "bd:mark"
    # Structural shape — same on first and second writes.
    subgoal_ids = sorted(payload["subgoals"].keys())
    statuses = {sid: payload["subgoals"][sid]["status"] for sid in subgoal_ids}
    # Re-run; compare against the same field set.
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    second = json.loads((plans_dir / "bd:mark.json").read_text())
    assert sorted(second["subgoals"].keys()) == subgoal_ids
    assert {sid: second["subgoals"][sid]["status"] for sid in subgoal_ids} == statuses


def test_plan_bootstrap_explicit_plan_id_used(tmp_path: Path) -> None:
    """--plan-id overrides the default 'bd:<assignee>' anchor."""
    plans_dir = tmp_path / "plans"
    proc = subprocess.run(
        [
            *HARNESS_CMD,
            "plan",
            "bootstrap",
            "--plans-dir",
            str(plans_dir),
            "--plan-id",
            "main",
            "--assignee",
            "mark",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (plans_dir / "main.json").exists()
    payload = json.loads((plans_dir / "main.json").read_text())
    assert payload["id"] == "main"
    assert payload["root_subgoal_id"] == "main:root"


def test_plan_bootstrap_help_lists_flags() -> None:
    proc = subprocess.run(
        [*HARNESS_CMD, "plan", "bootstrap", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    for flag in (
        "--character",
        "--plan-id",
        "--assignee",
        "--plans-dir",
        "--include-closed",
        "--dry-run",
    ):
        assert flag in proc.stdout, f"missing flag {flag!r} in --help output"


def test_plan_subcommand_listed_in_top_level_help() -> None:
    """`harness --help` shows the `plan` subcommand exists."""
    proc = subprocess.run(
        [*HARNESS_CMD, "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=_env(),
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert "plan" in proc.stdout
