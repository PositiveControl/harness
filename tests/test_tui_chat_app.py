"""Phase 1 smoke tests for the Textual chat scaffold (harness-1o8).

These assert the layout + event plumbing: the three widgets compose,
the banner lands in the log on mount, an input submission echoes a
user line and an assistant line into the output log, and an empty
submission is a no-op. Future phases replace the echo in
`on_input_submitted` with a model-driven worker, at which point the
'echo' assertion here will change — the structural assertions
(widgets exist, no crashes on mount) are the parts worth keeping
long-term.

We use Textual's own `App.run_test()` harness (async) rather than
snapshot tests — this keeps the suite pure-Python and avoids
depending on pytest-textual-snapshot yet.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, RichLog

from harness.tui import ChatApp


@pytest.mark.asyncio
async def test_chat_app_mounts_with_expected_widgets() -> None:
    app = ChatApp(character_name="airton", speaker="mark")
    async with app.run_test() as pilot:
        # The three load-bearing widgets exist and the input has focus
        # so the user can start typing immediately.
        assert pilot.app.query_one("#output", RichLog) is not None
        assert pilot.app.query_one("#prompt", Input) is not None
        assert pilot.app.query_one("#metrics") is not None
        assert pilot.app.query_one("#prompt", Input).has_focus


@pytest.mark.asyncio
async def test_chat_app_echoes_user_input_back_to_log() -> None:
    """Phase 1 contract: a non-empty submission appends a user line
    AND an assistant echo line to the output log. When phase 2 swaps
    the echo for a worker, this test will be rewritten to assert the
    worker got invoked — not the echo text."""
    app = ChatApp(character_name="airton", speaker="mark")
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "hello"
        await pilot.press("enter")

        # RichLog exposes the written renderables as `.lines`. Both
        # the user line and the assistant echo should appear.
        log = pilot.app.query_one("#output", RichLog)
        rendered = "\n".join(str(line) for line in log.lines)
        assert "mark" in rendered
        assert "airton" in rendered
        assert "hello" in rendered
        # Input cleared for the next message.
        assert prompt.value == ""


@pytest.mark.asyncio
async def test_chat_app_ignores_empty_submission() -> None:
    """Whitespace-only input must not create empty turns in the log.
    The user hit Enter by accident; the scaffold should shrug it
    off, not splat a blank 'mark ›' line into the history."""
    app = ChatApp(character_name="airton", speaker="mark")
    async with app.run_test() as pilot:
        log = pilot.app.query_one("#output", RichLog)
        lines_before = len(log.lines)

        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "   "
        await pilot.press("enter")

        assert len(log.lines) == lines_before
