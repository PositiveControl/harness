"""The Typer sub-apps are shared singletons, and the tree stays whole.

Step 5a of docs/cli-extraction-plan.md moved the app objects into
`cli_apps.py` so handler modules can register against them without
importing `cli.py`. That only works while every module reaches the SAME
object: `@sub_app.command()` binds at import time, so a second
`eval_app` somewhere would register its commands onto an app nothing
mounts — and the commands would vanish from `--help` with no error
anywhere.

These tests exist for steps 5b-5d, which move handlers out one group at
a time. A group that ends up importing the wrong object, or a module
that never gets imported for its side effects, fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from harness import cli, cli_apps

_SUB_APPS = (
    "eval",
    "memory",
    "voice",
    "session",
    "phraseology",
    "web",
    "plan",
    "tool",
    "denylist",
    "drive",
)


def test_every_sub_app_is_mounted_on_the_root() -> None:
    mounted = {group.name for group in cli_apps.app.registered_groups}

    assert mounted == set(_SUB_APPS)


@pytest.mark.parametrize("name", ["app", *[f"{n}_app" for n in _SUB_APPS]])
def test_cli_re_exports_the_same_object_not_a_copy(name: str) -> None:
    """`harness.cli.eval_app is harness.cli_apps.eval_app`.

    Identity, not equality — a copy would collect handler registrations
    that the mounted app never sees.
    """
    assert getattr(cli, name) is getattr(cli_apps, name)


def test_cli_apps_does_not_import_cli() -> None:
    """The dependency runs one way. `cli.py` imports `cli_apps`; the
    reverse would reintroduce the cycle this module exists to break
    (and which step 4 had to paper over with a late-import shim)."""
    tree = ast.parse(Path(cli_apps.__file__).read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "harness.cli" not in imported


def test_importing_cli_registers_commands_on_every_sub_app() -> None:
    """Importing `harness.cli` is what populates the sub-apps today.

    As handlers move out in 5b-5d, this is the test that catches a
    handler module nobody imports: its sub-app comes back empty.
    """
    empty = [name for name in _SUB_APPS if not _commands_of(getattr(cli_apps, f"{name}_app"))]

    assert not empty, f"sub-apps with no registered commands: {empty}"


def _commands_of(app: typer.Typer) -> list[str]:
    names: list[str] = []
    for command in app.registered_commands:
        callback = command.callback
        names.append(command.name or (callback.__name__ if callback is not None else "?"))
    for group in app.registered_groups:
        if group.typer_instance is not None:
            names.extend(_commands_of(group.typer_instance))
    return names


def test_the_whole_command_tree_is_reachable_from_the_entry_point() -> None:
    """`harness --help` lists every top-level command. This is the
    user-visible consequence of the invariant above, asserted through
    the actual entry point rather than the registry."""
    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0, result.output
    for name in ("chat", "describe", "daemon", *_SUB_APPS):
        assert name in result.output, f"{name} missing from --help"
