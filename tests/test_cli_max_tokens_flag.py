"""`harness chat --max-tokens` reaches both chat front-ends (harness-gebo5).

The flag exists because the tool loop's per-round budget was hardcoded at
2048 with no way through from the CLI. A `write_file` whose content runs
past that cap is cut mid-argument, arrives with no closing tag, stops
parsing as a tool call, and the turn burns its retries on a file that
never lands (harness-4s6fv). Both the classic REPL and the TUI run their
own tool loop, so a flag that only reaches one of them is half a fix —
these pin the forwarding on each path.
"""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from harness import cli, cli_classic, cli_tui
from harness.orchestrator import DEFAULT_ROUND_MAX_TOKENS

runner = CliRunner()


@pytest.fixture
def captured_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub both chat entry points so `chat` returns before it loads a
    model, opens a store, or draws a UI. Records what it was handed."""
    seen: dict[str, Any] = {}

    def fake_classic(**kwargs: Any) -> None:
        seen.update(kwargs)
        seen["front_end"] = "classic"

    def fake_tui(**kwargs: Any) -> None:
        seen.update(kwargs)
        seen["front_end"] = "tui"

    monkeypatch.setattr(cli_classic, "run_classic_chat", fake_classic)
    monkeypatch.setattr(cli_tui, "run_tui", fake_tui)
    return seen


def test_max_tokens_reaches_the_classic_repl(captured_kwargs: dict[str, Any]) -> None:
    result = runner.invoke(cli.app, ["chat", "--session", "t", "--max-tokens", "8192"])

    assert result.exit_code == 0, result.output
    assert captured_kwargs["front_end"] == "classic"
    assert captured_kwargs["round_max_tokens"] == 8192


def test_max_tokens_reaches_the_tui(captured_kwargs: dict[str, Any]) -> None:
    result = runner.invoke(cli.app, ["chat", "--session", "t", "--tui", "--max-tokens", "4096"])

    assert result.exit_code == 0, result.output
    assert captured_kwargs["front_end"] == "tui"
    assert captured_kwargs["round_max_tokens"] == 4096


def test_default_is_the_orchestrator_default(captured_kwargs: dict[str, Any]) -> None:
    """Unset flag must not move the tuned local-MLX operating point."""
    result = runner.invoke(cli.app, ["chat", "--session", "t"])

    assert result.exit_code == 0, result.output
    assert captured_kwargs["round_max_tokens"] == DEFAULT_ROUND_MAX_TOKENS


def test_max_tokens_is_range_checked() -> None:
    """The ceiling matches _MAX_TOKENS_CEILING; below the floor a round
    can't hold a tool call at all. Typer rejects both without reaching
    the chat body."""
    for bad in ("0", "65536"):
        result = runner.invoke(cli.app, ["chat", "--session", "t", "--max-tokens", bad])
        assert result.exit_code != 0
