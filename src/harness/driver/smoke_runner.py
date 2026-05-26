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


def main(argv: list[str]) -> int:
    flags = {a for a in argv[1:] if a.startswith("--")}
    positionals = [a for a in argv[1:] if not a.startswith("--")]
    no_blank_canvas = "--no-blank-canvas" in flags
    if not positionals:
        print(
            "usage: python -m harness.driver.smoke_runner [--no-blank-canvas] <index.html>",
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
                page.wait_for_timeout(_settle_ms())
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
