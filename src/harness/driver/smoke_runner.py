"""Headless browser smoke-load — harness-4b8v.

Subprocess entry point for the workspace-verify gate. Loads
``file://<index>`` via Playwright's headless Chromium, waits for the
``load`` event plus a short settle window, and exits non-zero if any
``console.error`` or unhandled page error fired during the window.

The canonical failure this catches::

    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('d');   // typo -> ctx is null
    ctx.clearRect(0, 0, w, h);             // TypeError on first frame

``node --check`` says the file is fine. Only running the artifact
catches it. See harness-3jo1 / harness-4b8v for the original incident.

Usage (matches the contract of ``_exec_test_cmd`` in fsm_turn.py)::

    python -m harness.driver.smoke_runner <index.html>

Exit codes:
    0  page loaded cleanly OR Playwright/Chromium not installed (a
       stale install should not turn every verify red).
    1  page error, console.error fired, or unrecoverable Playwright
       failure. The first error messages are written to stderr to
       feed the verify-fail handoff tail.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Window in ms the runner waits AFTER the load event for animation-
# frame and microtask-driven init bugs to surface. Default 1500 ms
# covers many requestAnimationFrame ticks plus queued Promise
# resolutions, so a bug that throws a few frames into the render loop
# (not just on the very first tick) still surfaces. Override via
# HARNESS_SMOKE_SETTLE_MS for slower machines or longer warm-up
# sequences; longer windows trade verify cost for coverage.
_DEFAULT_SETTLE_MS = 1500


def _blank_canvas_check_enabled(*, cli_disabled: bool = False) -> bool:
    """Whether to fail the smoke step when a canvas renders entirely
    one color. Default ON — it catches the "draw loop never wired into
    requestAnimationFrame" class (all-black map).

    Two off-switches, either disables:
    - ``cli_disabled`` (the ``--no-blank-canvas`` flag): the driver
      passes this for early phases of an incremental from-scratch build
      where a §1-style skeleton legitimately renders nothing until a
      later render milestone lands (harness-6dsn). The blank canvas is
      EXPECTED there, not a bug.
    - ``HARNESS_SMOKE_BLANK_CANVAS=0`` (or ``false`` / ``no`` / ``off``):
      operator-level off-switch for a canvas app that intentionally
      renders nothing until user interaction."""
    if cli_disabled:
        return False
    raw = os.environ.get("HARNESS_SMOKE_BLANK_CANVAS", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _settle_ms() -> int:
    """Resolve the post-load settle window from
    ``HARNESS_SMOKE_SETTLE_MS`` (clamped to [100, 30000]) or the
    default. A malformed / non-positive value falls back to the
    default rather than failing the verify on a misconfigured env."""
    raw = os.environ.get("HARNESS_SMOKE_SETTLE_MS", "").strip()
    if not raw:
        return _DEFAULT_SETTLE_MS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_SETTLE_MS
    if value < 100:
        return 100
    if value > 30000:
        return 30000
    return value


# JS evaluated in the page after the settle window to detect a canvas
# that loaded clean but rendered nothing — the all-uniform-color
# "blank canvas" failure mode (e.g. a game whose draw loop was never
# wired into requestAnimationFrame, so the map never paints). Returns
# the list of canvas descriptors that are entirely one color. A canvas
# is "blank" when every sampled pixel is identical. We downsample to a
# small grid (toDataURL is avoided — getImageData is cheaper and
# doesn't need a data URL round-trip) and compare RGBA tuples.
#
# Canvases smaller than 2x2 or with a zero dimension are skipped (not
# meaningfully renderable). Cross-origin / tainted canvases throw on
# getImageData; we swallow that and treat them as non-blank (we can't
# inspect them, so we don't fail on them).
_BLANK_CANVAS_JS = r"""
() => {
  const blanks = [];
  const canvases = Array.from(document.querySelectorAll('canvas'));
  for (let i = 0; i < canvases.length; i++) {
    const c = canvases[i];
    const w = c.width, h = c.height;
    if (!w || !h || w < 2 || h < 2) continue;
    const ctx = c.getContext('2d');
    if (!ctx) continue;
    let data;
    try {
      data = ctx.getImageData(0, 0, w, h).data;
    } catch (e) {
      continue;  // tainted/cross-origin — can't inspect, don't fail
    }
    const r0 = data[0], g0 = data[1], b0 = data[2], a0 = data[3];
    let uniform = true;
    const stride = Math.max(4, Math.floor(data.length / 4 / 256) * 4);
    for (let p = 0; p < data.length; p += stride) {
      if (data[p] !== r0 || data[p+1] !== g0 || data[p+2] !== b0 || data[p+3] !== a0) {
        uniform = false;
        break;
      }
    }
    if (uniform) {
      blanks.push({ index: i, width: w, height: h, rgba: [r0, g0, b0, a0] });
    }
  }
  return blanks;
}
"""


def _format_console_error(msg: Any) -> str:
    """Render a Playwright ``ConsoleMessage`` as
    ``console.error <file:line> <text>``. The ``msg`` parameter is
    typed ``Any`` because Playwright is imported lazily inside
    ``main()`` so the module type-checks even when Playwright isn't
    installed in the dev environment."""
    text = getattr(msg, "text", "") or "<no message>"
    location = getattr(msg, "location", None)
    url = ""
    line: int | str = ""
    if isinstance(location, dict):
        url = str(location.get("url", "") or "")
        raw_line = location.get("lineNumber", "")
        line = raw_line if raw_line not in (None, "") else ""
    if isinstance(url, str) and url.startswith("file://"):
        url = url[len("file://") :]
    locator = f"{url}:{line}" if url else "<unknown>"
    return f"console.error {locator} {text}"


def _format_page_error(err: Any) -> str:
    """Render a Playwright pageerror event as ``pageerror: <message>``.
    The exception's ``__str__`` typically includes the JS stack head."""
    return f"pageerror: {err}"


def _is_undefined_ref_error(err: str) -> bool:
    """True for a runtime error caused by a not-yet-defined identifier —
    a JS ``ReferenceError: X is not defined`` (harness-wngfm).

    While a render milestone is still open, the entry symbol it will
    define (``render`` / ``drawTile`` …) legitimately doesn't exist yet,
    so a pre-milestone bead's scaffold that references it throws this on
    load. That absence is expected, not a regression — the caller passes
    ``--tolerate-undefined-refs`` during that window so these downgrade to
    advisory while every other runtime error (TypeError, SyntaxError, a
    real console.error) still gates."""
    return "is not defined" in err.lower()


def _partition_tolerated_errors(
    errors: list[str], *, tolerate_undefined_refs: bool
) -> tuple[list[str], list[str]]:
    """Split collected runtime errors into ``(gating, tolerated)``.

    When ``tolerate_undefined_refs`` is False (the default — no render
    milestone open), nothing is tolerated and every error gates. When
    True (harness-wngfm), ``ReferenceError: X is not defined`` errors move
    to the tolerated list (advisory) while all others still gate."""
    if not tolerate_undefined_refs:
        return errors, []
    gating = [e for e in errors if not _is_undefined_ref_error(e)]
    tolerated = [e for e in errors if _is_undefined_ref_error(e)]
    return gating, tolerated


def _read_file_arg(argv: list[str], flag: str, label: str) -> tuple[str | None, str | None]:
    """Resolve an optional ``--<flag>=<file>`` script argument.

    Returns ``(source, error)``: ``source`` is the file contents (None
    when the flag is absent), ``error`` is a one-line failure string when
    the flag was given but the file is missing/unreadable (so a
    misconfigured scenario fails loud rather than silently skipping)."""
    prefix = f"--{flag}="
    arg = next((a for a in argv[1:] if a.startswith(prefix)), None)
    if arg is None:
        return None, None
    path = Path(arg[len(prefix) :])
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, f"smoke-execute: {label} script unreadable ({path}): {exc}"


def _read_path_arg(argv: list[str], flag: str) -> Path | None:
    """Resolve an optional ``--<flag>=<path>`` OUTPUT path. Unlike
    ``_read_file_arg`` the target need not exist — we write it. Relative
    paths resolve against CWD. harness-ke4hx.1 (vision-QA screenshots)."""
    prefix = f"--{flag}="
    arg = next((a for a in argv[1:] if a.startswith(prefix)), None)
    if arg is None:
        return None
    p = Path(arg[len(prefix) :])
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _capture_screenshot(page: Any, path: Path | None, label: str) -> None:
    """Best-effort PNG capture to ``path``. A screenshot failure must
    never turn a clean load red — mirrors the blank-canvas best-effort
    policy. No-op when ``path`` is None (flag absent). harness-ke4hx.1."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path))
    except Exception as exc:
        print(f"smoke-execute: screenshot ({label}) skipped: {exc}", file=sys.stderr)


def _read_setup(argv: list[str]) -> tuple[str | None, str | None]:
    """Resolve the optional ``--setup=<file>`` scenario (run after load,
    before the settle window — drives conditionally-spawned entities and
    synthetic input so their code paths execute during the smoke)."""
    return _read_file_arg(argv, "setup", "setup")


def _read_assert(argv: list[str]) -> tuple[str | None, str | None]:
    """Resolve the optional ``--assert=<file>`` behavioral check (run
    AFTER the settle window — harness-u1il5). The script body is wrapped
    in an arrow function and must RETURN an array of failure strings
    (empty = pass); a throw or a non-empty array fails the smoke. Pairs
    with ``--setup`` (setup drives the input; assert checks the effect
    once the game loop has had frames to process it). Closes the
    render-only blind spot where input handlers no draw check exercises
    can close blind."""
    return _read_file_arg(argv, "assert", "assert")


def main(argv: list[str]) -> int:
    flags = {a for a in argv[1:] if a.startswith("--") and "=" not in a}
    positionals = [a for a in argv[1:] if not a.startswith("--")]
    no_blank_canvas = "--no-blank-canvas" in flags
    # harness-wngfm: while a render milestone is open, tolerate
    # "X is not defined" ReferenceErrors (the entry symbol the milestone
    # will define doesn't exist yet) — downgrade them to advisory instead
    # of gating. Other runtime errors still fail the smoke.
    tolerate_undefined_refs = "--tolerate-undefined-refs" in flags
    setup_source, setup_error = _read_setup(argv)
    if setup_error is not None:
        print(setup_error, file=sys.stderr)
        return 1
    assert_source, assert_error = _read_assert(argv)
    if assert_error is not None:
        print(assert_error, file=sys.stderr)
        return 1
    # harness-ke4hx.1: opt-in screenshots for advisory vision-QA.
    #   --screenshot=        post-settle final-state shot
    #   --screenshot-before= after load, BEFORE setup drives input
    #   --screenshot-after=  post-settle (alias of --screenshot; lets a
    #                        before/after pair use symmetric names)
    shot_path = _read_path_arg(argv, "screenshot")
    shot_before = _read_path_arg(argv, "screenshot-before")
    shot_after = _read_path_arg(argv, "screenshot-after")
    if not positionals:
        print(
            "usage: python -m harness.driver.smoke_runner "
            "[--no-blank-canvas] [--tolerate-undefined-refs] "
            "[--setup=<scenario.js>] "
            "[--assert=<checks.js>] [--screenshot[-before|-after]=<out.png>] "
            "<index.html>",
            file=sys.stderr,
        )
        return 1
    raw = Path(positionals[0])
    index_path = raw if raw.is_absolute() else (Path.cwd() / raw).resolve()
    if not index_path.is_file():
        print(f"smoke-execute: index not found at {index_path}", file=sys.stderr)
        return 1

    try:
        from playwright.sync_api import (  # pyright: ignore[reportMissingImports]
            sync_playwright,
        )
    except ImportError as exc:
        # The verify gate should not have invoked us here (it gates
        # on import availability), but if Playwright has been removed
        # out from under a running session, treat as a clean skip
        # rather than failing every drive verify.
        print(f"smoke-execute: Playwright not importable: {exc}", file=sys.stderr)
        return 0

    errors: list[str] = []
    blanks: list[Any] = []

    def _on_console(msg: Any) -> None:
        if getattr(msg, "type", "") == "error":
            errors.append(_format_console_error(msg))

    def _on_pageerror(err: Any) -> None:
        errors.append(_format_page_error(err))

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_context().new_page()
                page.on("console", _on_console)
                page.on("pageerror", _on_pageerror)
                page.goto(f"file://{index_path}", wait_until="load")
                # harness-ke4hx.1: pre-interaction shot — captured after
                # load but BEFORE setup drives input, so the advisory
                # vision-QA can diff it against the post-settle shot to
                # judge whether an interaction changed the visible state.
                _capture_screenshot(page, shot_before, "before")
                # harness-5vn6t: optional scenario. Exercises
                # code paths that don't run on a bare load — conditionally
                # spawned entities (cops, bullets, peds) whose render/init
                # is a verify blind spot otherwise. Runs AFTER load so the
                # game's globals/functions exist, BEFORE the settle window
                # so the next rAF ticks drive the exercised state and any
                # throw surfaces as a pageerror/console.error. A setup that
                # throws is a real failure (the function the issue should
                # provide is missing/broken), so we record it as an error.
                if setup_source is not None:
                    try:
                        page.evaluate(f"() => {{ {setup_source} }}")
                    except Exception as exc:  # surface any setup throw as a failure
                        errors.append(f"setup-scenario error: {exc}")
                page.wait_for_timeout(_settle_ms())
                # harness-ke4hx.1: post-settle final-state shots. Captured
                # regardless of console/page errors so the advisory QA can
                # see a broken state too. Best-effort — never fails the
                # smoke. `--screenshot` and `--screenshot-after` are
                # aliases (symmetric before/after naming).
                _capture_screenshot(page, shot_path, "screenshot")
                _capture_screenshot(page, shot_after, "after")
                # harness-u1il5: behavioral assertions, post-settle. The
                # setup scenario drove input (synthetic KeyboardEvents,
                # state pokes) before the settle window; now the game loop
                # has had frames to process it, so the assert script can
                # check the resulting state. The body is wrapped in an
                # arrow fn and must RETURN an array of failure strings —
                # empty/falsy passes, a non-empty array or a throw fails.
                # Runs only on an otherwise-clean load (a page that threw
                # has a more actionable error than a cascaded assert miss)
                # and BEFORE the blank-canvas check so a behavioral failure
                # outranks the coarser "canvas is one color" signal.
                if assert_source is not None and not errors:
                    try:
                        verdict = page.evaluate(f"() => {{ {assert_source} }}")
                        if isinstance(verdict, list):
                            errors.extend(f"assert failure: {item}" for item in verdict)
                        elif verdict:
                            errors.append(
                                f"assert script returned a non-list truthy value "
                                f"({verdict!r}); expected an array of failure strings"
                            )
                    except Exception as exc:  # surface any assert throw as a failure
                        errors.append(f"assert-scenario error: {exc}")
                # Blank-canvas check runs only if the load was otherwise
                # clean — a page that already threw has a more actionable
                # error to report than "your canvas is one color."
                if not errors and _blank_canvas_check_enabled(cli_disabled=no_blank_canvas):
                    try:
                        result = page.evaluate(_BLANK_CANVAS_JS)
                        if isinstance(result, list):
                            blanks = result
                    except Exception as exc:
                        # Canvas inspection is best-effort. A failure to
                        # evaluate (page navigated away, eval disabled)
                        # must not turn a clean load red.
                        print(
                            f"smoke-execute: blank-canvas check skipped: {exc}",
                            file=sys.stderr,
                        )
            finally:
                browser.close()
    except Exception as exc:
        message = str(exc)
        if "Executable doesn't exist" in message or "playwright install" in message.lower():
            # Chromium binary not installed. Skip cleanly so the
            # operator gets one hint rather than a verify-fail on
            # every drive turn.
            print(
                "smoke-execute: chromium browser not installed "
                "(run `playwright install chromium`); skipping",
                file=sys.stderr,
            )
            return 0
        print(f"smoke-execute: Playwright error: {exc}", file=sys.stderr)
        return 1

    # harness-wngfm: while a render milestone is open, the not-yet-defined
    # entry symbol throws a ReferenceError that's expected, not a
    # regression — split those off as advisory so a pre-render bead isn't
    # structurally un-closable. Every other runtime error still gates.
    errors, tolerated = _partition_tolerated_errors(
        errors, tolerate_undefined_refs=tolerate_undefined_refs
    )
    for err in tolerated:
        print(
            f"smoke-execute: tolerated (render milestone open) — {err}",
            file=sys.stderr,
        )
    if errors:
        print("smoke-execute: runtime errors detected on load:", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    if blanks:
        print(
            "smoke-execute: canvas rendered nothing (entirely one color) after "
            f"{_settle_ms()}ms — draw loop likely not wired up:",
            file=sys.stderr,
        )
        for blank in blanks:
            idx = blank.get("index", "?") if isinstance(blank, dict) else "?"
            dims = (
                f"{blank.get('width', '?')}x{blank.get('height', '?')}"
                if isinstance(blank, dict)
                else "?"
            )
            rgba = blank.get("rgba", "?") if isinstance(blank, dict) else "?"
            print(f"blank-canvas #{idx} ({dims}) uniform rgba={rgba}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
