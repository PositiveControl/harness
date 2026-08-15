"""Smoke contract for the `denylist`, `web` and `phraseology` commands.

Step 5d of docs/cli-extraction-plan.md moves the last five subcommand
groups out of cli.py. Two of them already had tests — `tool`
(tests/test_cli_tool.py) and `plan` (tests/test_cli_plan.py) drive the
real binary through subprocess, so a move can't break them silently;
they'd fail loudly. These three had nothing.

Same smoke bar as the eval group: --help for every command (Typer
resolves Options at import time, so this is what catches a signature
that didn't survive the move), plus the guard paths that run without
standing up a server or loading a model.

`web serve` is only ever invoked with a rejected --model. Past that
guard it calls uvicorn.run and would block the suite forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import app
from harness.config import settings

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tmp root with a copy of the airton character, so denylist writes
    land in a throwaway sqlite rather than the real data dir."""
    import shutil

    shutil.copytree(_REPO_ROOT / "character" / "airton", tmp_path / "character" / "airton")
    monkeypatch.setattr(settings, "root", tmp_path)
    return settings.character_db_path


# ---------- --help across all five groups ----------


@pytest.mark.parametrize(
    ("group", "command"),
    [
        ("denylist", "list"),
        ("denylist", "add"),
        ("denylist", "clear"),
        ("web", "serve"),
        ("phraseology", "lint"),
        ("tool", "list"),
        ("tool", "show"),
        ("tool", "drop"),
        ("plan", "bootstrap"),
    ],
)
def test_command_help_renders(group: str, command: str) -> None:
    result = _run(group, command, "--help")

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "group", ["denylist", "web", "phraseology", "tool", "plan", "session", "memory", "voice"]
)
def test_group_help_renders(group: str) -> None:
    """Catches a handler module that never gets imported: its sub-app
    would render with no commands under it."""
    result = _run(group, "--help")

    assert result.exit_code == 0, result.output
    assert "Commands" in result.output


# ---------- denylist ----------


def test_denylist_list_reports_an_empty_store(cli_env: Path) -> None:
    result = _run("denylist", "list")

    assert result.exit_code == 0, result.output
    assert "no active denylist entries" in result.output


def test_denylist_add_then_list_shows_the_host(cli_env: Path) -> None:
    added = _run("denylist", "add", "example.com")

    assert added.exit_code == 0, added.output

    listed = _run("denylist", "list")
    assert "example.com" in listed.output


def test_denylist_clear_removes_a_single_host(cli_env: Path) -> None:
    _run("denylist", "add", "example.com")
    _run("denylist", "add", "other.test")

    cleared = _run("denylist", "clear", "--host", "example.com")

    assert cleared.exit_code == 0, cleared.output
    listed = _run("denylist", "list")
    assert "example.com" not in listed.output
    assert "other.test" in listed.output


def test_denylist_list_include_expired_is_accepted(cli_env: Path) -> None:
    """Nothing has expired in a fresh store, so this pins the flag's
    plumbing rather than the TTL logic (which store tests cover)."""
    result = _run("denylist", "list", "--include-expired")

    assert result.exit_code == 0, result.output
    assert "no any denylist entries" in result.output


# ---------- web ----------


def test_web_serve_rejects_an_unknown_adapter_before_binding_a_port() -> None:
    """The adapter is resolved before uvicorn starts, so a bad --model
    is a usage error rather than a server that comes up broken."""
    result = _run("web", "serve", "--host", "127.0.0.1", "--model", "nope")

    assert result.exit_code != 0
    assert "nope" in result.output


# ---------- phraseology ----------


def test_phraseology_lint_refuses_a_character_without_the_profile() -> None:
    """The lint is airton_c1-only. Against the default character it must
    say so — and name the env var that fixes it — rather than running a
    retrieval stack that has no JO 7110.65 corpus behind it."""
    result = _run("phraseology", "lint", "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF.")

    assert result.exit_code != 0
    assert "airton_c1" in result.output
