"""Shared probe for whether a launchable chromium is actually present.

`importlib.util.find_spec("playwright")` only proves the *python package*
is importable — `uv sync --extra all` installs that, but the browser
binary is a separate `uv run playwright install chromium` step. Gating on
the package alone lets a browser-less checkout RUN the smoke tests, and
`smoke_runner` skips cleanly on a missing binary (returns 0), so the
assertion sees a green smoke and fails on a None outcome instead of the
test being skipped (harness-le6f).

Probe once per session and share the marker across test modules.
"""

from __future__ import annotations

import functools
import importlib.util
from pathlib import Path

import pytest


@functools.cache
def chromium_available() -> bool:
    """True when playwright imports AND its chromium binary is on disk.

    `executable_path` resolves the per-platform browser cache
    (`~/Library/Caches/ms-playwright`, `~/.cache/ms-playwright`,
    `PLAYWRIGHT_BROWSERS_PATH`) without launching the browser. Any
    failure along the way means "no usable chromium" — the caller only
    ever needs the boolean.
    """
    if importlib.util.find_spec("playwright") is None:
        return False
    try:
        from playwright.sync_api import (  # pyright: ignore[reportMissingImports]
            sync_playwright,
        )

        with sync_playwright() as p:
            return Path(p.chromium.executable_path).is_file()
    except Exception:
        return False


requires_playwright = pytest.mark.skipif(
    not chromium_available(),
    reason="requires `playwright` AND an installed chromium (`uv run playwright install chromium`)",
)
