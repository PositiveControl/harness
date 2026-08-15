"""Smoke contract for the `harness eval` commands.

Step 5c of docs/cli-extraction-plan.md moves ten eval handlers — 1,959
LOC, 2% covered — into cli_eval.py. The bar here is deliberately a smoke
bar, not characterization, and that's a scoping decision worth stating:

Each handler is ~90 lines of typer.Option declarations plus wiring plus
Rich rendering over `harness.evals.*`, which already has seven dedicated
test files of its own. Characterizing the handlers would mean
materializing FAA corpora and loading MLX models to re-test logic that
is already covered. What a MOVE can actually break is narrower — the
Option declarations and the flag plumbing — so that is what these
tests cover:

  * every command's --help renders (a broken default, annotation or
    duplicated flag fails here, and nothing else would catch it);
  * --fixture reaches the code that validates it;
  * the two evals that run offline (tool-loop, and file-ops on the echo
    adapter) run end to end and emit parseable --json.

`eval router` is never invoked without --fixture: it loads an MLX
router model and would hang the suite.
"""

from __future__ import annotations

import json

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import app

_COMMANDS = (
    "voice",
    "router",
    "session-resume",
    "tfr",
    "atc",
    "atc-retrieval",
    "phraseology",
    "atc-audio",
    "tool-loop",
    "file-ops",
)

# Commands that take --fixture and validate it before building anything
# expensive. Verified by hand: each returns the "not found" error
# immediately rather than loading a model first.
_FIXTURE_COMMANDS = (
    "router",
    "session-resume",
    "tfr",
    "atc",
    "atc-retrieval",
    "tool-loop",
)


def _run(*args: str) -> Result:
    return CliRunner().invoke(app, list(args))


@pytest.mark.parametrize("command", _COMMANDS)
def test_eval_command_help_renders(command: str) -> None:
    """The cheapest test that catches the most likely move damage.

    Typer builds every Option at import time; --help is what forces the
    whole signature to resolve. A parameter that lost its default, or an
    annotation that no longer imports in the new module, fails right
    here.
    """
    result = _run("eval", command, "--help")

    assert result.exit_code == 0, result.output
    assert command in result.output


def test_eval_group_lists_every_command() -> None:
    result = _run("eval", "--help")

    assert result.exit_code == 0, result.output
    for command in _COMMANDS:
        assert command in result.output, f"{command} missing from `eval --help`"


@pytest.mark.parametrize("command", _FIXTURE_COMMANDS)
def test_fixture_flag_reaches_the_validator(command: str) -> None:
    """--fixture lands on the parameter that checks the path exists, and
    the rejection names the path back. Proves the flag is wired without
    running the eval — and that validation happens BEFORE any adapter is
    constructed, which is what keeps this test offline."""
    result = _run("eval", command, "--fixture", "/nonexistent/fixture.yaml")

    assert result.exit_code != 0
    assert "/nonexistent/fixture.yaml" in result.output


def test_phraseology_rejects_a_character_without_the_profile() -> None:
    """Its own guard fires ahead of fixture handling: the eval is
    airton_c1-only and says so, naming the character it actually got."""
    result = _run("eval", "phraseology", "--fixture", "/nonexistent/fixture.yaml")

    assert result.exit_code != 0
    assert "phraseology" in result.output
    assert "airton" in result.output


# ---------- the two that run offline ----------


def test_tool_loop_eval_runs_and_emits_json() -> None:
    """Scripted adapters + mock tools, no model, no corpus — so the whole
    handler path is exercised for real, not stubbed."""
    result = _run("eval", "tool-loop", "--json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload


def test_file_ops_eval_runs_on_the_echo_adapter_and_emits_json() -> None:
    result = _run("eval", "file-ops", "--model", "echo", "--json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload


def test_tool_loop_eval_renders_a_human_table_without_json() -> None:
    """The default path is Rich rendering, which --json bypasses. Both
    branches move together, so both get a smoke."""
    result = _run("eval", "tool-loop")

    assert result.exit_code == 0, result.output
    assert result.output.strip()
