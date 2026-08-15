"""Unit tests for driver/smoke_runner.py pure helpers — harness-7bxm.

The settle-window resolver and the blank-canvas off-switch parse env
vars that the operator (or a per-workspace driver config) sets to tune
the headless smoke gate. These are pure functions — no Playwright /
Chromium needed — so they run in every environment. The end-to-end
gate behavior (real headless load, blank-canvas detection) is pinned
in test_workspace_verify.py behind the _requires_playwright skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.driver.smoke_runner import (
    _ASSERT_POLL_MS,
    _DEFAULT_ASSERT_RETRY_MS,
    _DEFAULT_SETTLE_MS,
    _assert_retry_ms,
    _blank_canvas_check_enabled,
    _capture_screenshot,
    _is_undefined_ref_error,
    _partition_tolerated_errors,
    _read_assert,
    _read_path_arg,
    _read_setup,
    _run_behavioral_assert,
    _settle_ms,
)


class _FakePage:
    """Stub playwright Page exposing just .screenshot(path=...)."""

    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.shots: list[str] = []

    def screenshot(self, *, path: str) -> None:
        if self.raises:
            raise RuntimeError("capture boom")
        Path(path).write_bytes(b"\x89PNG fake")
        self.shots.append(path)


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


# --- harness-k1kx: behavioral-assert retry --------------------------


class _AssertPage:
    """Stub playwright Page for the assert path: `evaluate` replays a
    scripted list of verdicts (a verdict that is an Exception is
    raised), and `wait_for_timeout` just records the sleep the runner
    asked for instead of taking it."""

    def __init__(self, verdicts: list[object]) -> None:
        self._verdicts = list(verdicts)
        self.reads = 0
        self.waits: list[int] = []

    def evaluate(self, _source: str) -> object:
        self.reads += 1
        verdict = self._verdicts[min(self.reads - 1, len(self._verdicts) - 1)]
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)


def test_assert_retry_ms_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_SMOKE_ASSERT_RETRY_MS", raising=False)
    assert _assert_retry_ms() == _DEFAULT_ASSERT_RETRY_MS


def test_assert_retry_ms_honors_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_ASSERT_RETRY_MS", "500")
    assert _assert_retry_ms() == 500


def test_assert_retry_ms_allows_zero_as_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 is a legal value (read once) — unlike the settle window, which
    floors at 100ms. A negative clamps up to 0, not to a floor."""
    monkeypatch.setenv("HARNESS_SMOKE_ASSERT_RETRY_MS", "0")
    assert _assert_retry_ms() == 0
    monkeypatch.setenv("HARNESS_SMOKE_ASSERT_RETRY_MS", "-5")
    assert _assert_retry_ms() == 0


def test_assert_retry_ms_clamps_high_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_ASSERT_RETRY_MS", "999999")
    assert _assert_retry_ms() == 30000
    monkeypatch.setenv("HARNESS_SMOKE_ASSERT_RETRY_MS", "later")
    assert _assert_retry_ms() == _DEFAULT_ASSERT_RETRY_MS


def test_behavioral_assert_passes_on_first_read_without_waiting() -> None:
    """A probe that passes immediately costs nothing — the retry budget
    is only spent by runs that were failing."""
    page = _AssertPage([[]])
    assert _run_behavioral_assert(page, "return [];", budget_ms=3000) == []
    assert page.reads == 1
    assert page.waits == []


def test_behavioral_assert_retries_until_state_catches_up() -> None:
    """The load-under-pressure case (harness-k1kx): the game loop hadn't
    ticked yet on the first two reads, so the assert must re-check
    rather than report a failure that is merely early."""
    page = _AssertPage([["fire did not set window.fired"], ["fire did not set window.fired"], []])
    assert _run_behavioral_assert(page, "return [];", budget_ms=3000) == []
    assert page.reads == 3
    assert page.waits == [_ASSERT_POLL_MS, _ASSERT_POLL_MS]


def test_behavioral_assert_reports_failures_when_budget_runs_out() -> None:
    """A genuinely broken app still fails — the retry only moves WHEN
    it fails, never WHETHER."""
    page = _AssertPage([["fire did not set window.fired"]])
    failures = _run_behavioral_assert(page, "return [...];", budget_ms=2 * _ASSERT_POLL_MS)
    assert failures == ["assert failure: fire did not set window.fired"]
    assert page.reads == 3  # reads at 0ms, 100ms, 200ms — then the budget is spent


def test_behavioral_assert_zero_budget_reads_once() -> None:
    page = _AssertPage([["still broken"]])
    assert _run_behavioral_assert(page, "return [...];", budget_ms=0) == [
        "assert failure: still broken"
    ]
    assert page.reads == 1
    assert page.waits == []


def test_behavioral_assert_throw_is_a_failure_and_is_retried() -> None:
    """A throwing probe is a failure, not a skip — and a throw can also
    be an early read ("X is not defined" before init finished), so it
    rides the same retry."""
    page = _AssertPage([RuntimeError("nonexistentGlobal is not defined")])
    failures = _run_behavioral_assert(page, "return nope.value;", budget_ms=_ASSERT_POLL_MS)
    assert len(failures) == 1
    assert "assert-scenario error" in failures[0]
    assert "not defined" in failures[0]
    assert page.reads == 2


def test_behavioral_assert_truthy_non_list_is_a_failure() -> None:
    """A probe that returns something other than an array of failure
    strings is misconfigured — fail loud rather than coerce."""
    page = _AssertPage(["oops"])
    failures = _run_behavioral_assert(page, "return 'oops';", budget_ms=0)
    assert len(failures) == 1
    assert "non-list truthy value" in failures[0]


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


# --- harness-u1il5: --setup / --assert script-arg resolution ---------


def test_read_setup_absent_returns_none() -> None:
    """No --setup flag → (None, None): the scenario is optional."""
    assert _read_setup(["prog", "index.html"]) == (None, None)


def test_read_assert_absent_returns_none() -> None:
    """No --assert flag → (None, None): the behavioral check is optional."""
    assert _read_assert(["prog", "index.html"]) == (None, None)


def test_read_assert_reads_file_contents(tmp_path: Path) -> None:
    """--assert=<file> → the file body is returned verbatim for eval."""
    probe = tmp_path / "smoke_assert.js"
    probe.write_text("return window.fired ? [] : ['fire did not trigger'];\n")
    source, error = _read_assert(["prog", f"--assert={probe}", "index.html"])
    assert error is None
    assert source is not None
    assert "fire did not trigger" in source


def test_read_assert_missing_file_is_loud(tmp_path: Path) -> None:
    """--assert pointing at a missing file → an error string, NOT a
    silent skip — a misconfigured probe must fail the gate, not pass it."""
    missing = tmp_path / "nope.js"
    source, error = _read_assert(["prog", f"--assert={missing}", "index.html"])
    assert source is None
    assert error is not None
    assert "unreadable" in error


def test_read_setup_missing_file_is_loud(tmp_path: Path) -> None:
    """Same loud-skip contract for --setup (regression guard for the
    shared _read_file_arg helper)."""
    missing = tmp_path / "nope.js"
    source, error = _read_setup(["prog", f"--setup={missing}", "index.html"])
    assert source is None
    assert error is not None
    assert "unreadable" in error


# ---- harness-ke4hx.1: screenshot path arg + capture --------------------


def test_read_path_arg_absent_returns_none() -> None:
    assert _read_path_arg(["prog", "index.html"], "screenshot") is None


def test_read_path_arg_absolute_passes_through() -> None:
    p = _read_path_arg(["prog", "--screenshot=/srv/x/out.png"], "screenshot")
    assert p == Path("/srv/x/out.png")


def test_read_path_arg_relative_resolves_to_cwd() -> None:
    p = _read_path_arg(["prog", "--screenshot=shots/out.png"], "screenshot")
    assert p is not None
    assert p.is_absolute()
    assert p.name == "out.png"


def test_capture_screenshot_none_path_is_noop() -> None:
    page = _FakePage()
    _capture_screenshot(page, None, "x")  # must not raise / not call page
    assert page.shots == []


def test_capture_screenshot_writes_and_makes_parent(tmp_path: Path) -> None:
    page = _FakePage()
    out = tmp_path / "nested" / "dir" / "shot.png"
    _capture_screenshot(page, out, "after")
    assert out.is_file()
    assert page.shots == [str(out)]


def test_capture_screenshot_swallows_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A capture failure is best-effort — it logs a skip and must NOT
    propagate (never turns a clean load red)."""
    page = _FakePage(raises=True)
    _capture_screenshot(page, tmp_path / "shot.png", "before")
    err = capsys.readouterr().err
    assert "screenshot (before) skipped" in err


# --- harness-wngfm: render-milestone undefined-ref tolerance ---------


def test_is_undefined_ref_error_matches_reference_error() -> None:
    """A JS ReferenceError for a not-yet-defined symbol is the tolerable
    class while a render milestone is open."""
    assert _is_undefined_ref_error("pageerror: ReferenceError: render is not defined")
    assert _is_undefined_ref_error("pageerror: drawTile is not defined")


def test_is_undefined_ref_error_rejects_real_errors() -> None:
    """Other runtime errors must NOT be tolerated — they're real bugs
    regardless of milestone state."""
    assert not _is_undefined_ref_error("pageerror: TypeError: x.foo is not a function")
    assert not _is_undefined_ref_error("console.error /game.js:9 SyntaxError: bad token")
    assert not _is_undefined_ref_error("setup-scenario error: boom")


def test_partition_tolerates_undefined_only_when_flagged() -> None:
    """With the flag off (no milestone open) nothing is tolerated; with
    it on, only undefined-ref errors move to advisory."""
    errs = [
        "pageerror: ReferenceError: render is not defined",
        "pageerror: TypeError: ctx.fillRct is not a function",
    ]
    # Flag off: everything gates.
    gating, tolerated = _partition_tolerated_errors(list(errs), tolerate_undefined_refs=False)
    assert gating == errs
    assert tolerated == []
    # Flag on: the ReferenceError is advisory, the TypeError still gates.
    gating, tolerated = _partition_tolerated_errors(list(errs), tolerate_undefined_refs=True)
    assert gating == ["pageerror: TypeError: ctx.fillRct is not a function"]
    assert tolerated == ["pageerror: ReferenceError: render is not defined"]
