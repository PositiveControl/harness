"""Unit tests for driver/smoke_runner.py pure helpers — harness-7bxm.

The settle-window resolver and the blank-canvas off-switch parse env
vars that the operator (or a per-workspace driver config) sets to tune
the headless smoke gate. These are pure functions — no Playwright /
Chromium needed — so they run in every environment. The end-to-end
gate behavior (real headless load, blank-canvas detection) is pinned
in test_workspace_verify.py behind the _requires_playwright skip.
"""

from __future__ import annotations

import pytest

from harness.driver.smoke_runner import (
    _DEFAULT_SETTLE_MS,
    _blank_canvas_check_enabled,
    _settle_ms,
)


def test_settle_ms_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_SMOKE_SETTLE_MS", raising=False)
    assert _settle_ms() == _DEFAULT_SETTLE_MS


def test_settle_ms_honors_valid_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_SETTLE_MS", "2500")
    assert _settle_ms() == 2500


def test_settle_ms_clamps_low(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_SETTLE_MS", "1")
    assert _settle_ms() == 100


def test_settle_ms_clamps_high(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_SETTLE_MS", "999999")
    assert _settle_ms() == 30000


def test_settle_ms_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_SETTLE_MS", "soon")
    assert _settle_ms() == _DEFAULT_SETTLE_MS


def test_blank_canvas_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_SMOKE_BLANK_CANVAS", raising=False)
    assert _blank_canvas_check_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "False"])
def test_blank_canvas_disabled_by_falsey_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_BLANK_CANVAS", value)
    assert _blank_canvas_check_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_blank_canvas_enabled_by_truthy_or_empty(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_BLANK_CANVAS", value)
    assert _blank_canvas_check_enabled() is True


def test_blank_canvas_cli_flag_disables_even_when_env_enables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harness-6dsn: the driver's --no-blank-canvas flag wins even if
    the env var would enable the check. The flag is how the driver
    suppresses the check during early-phase incremental builds."""
    monkeypatch.setenv("HARNESS_SMOKE_BLANK_CANVAS", "1")
    assert _blank_canvas_check_enabled(cli_disabled=True) is False


def test_blank_canvas_cli_flag_absent_keeps_env_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_SMOKE_BLANK_CANVAS", raising=False)
    assert _blank_canvas_check_enabled(cli_disabled=False) is True
