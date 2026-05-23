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

import sys
from pathlib import Path
from typing import Any

# Window in ms the runner waits AFTER the load event for animation-
# frame and microtask-driven init bugs to surface. ~500 ms covers
# one requestAnimationFrame tick on a hot machine plus a few queued
# Promise resolutions; longer windows trade verify cost for marginal
# additional coverage.
_SETTLE_MS = 500


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
    if len(argv) < 2:
        print(
            "usage: python -m harness.driver.smoke_runner <index.html>",
            file=sys.stderr,
        )
        return 1
    raw = Path(argv[1])
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
                page.wait_for_timeout(_SETTLE_MS)
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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
