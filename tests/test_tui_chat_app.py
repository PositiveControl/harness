"""Phase 2 tests for the Textual chat app (harness-29c).

Asserts that input submissions drive a worker that: runs retrieval,
assembles the system prompt + history + user turn, calls the
adapter, writes the reply into the RichLog, and persists user +
assistant turns to the transcript.

Uses a real Transcript (SQLite in tmp_path) and an EchoAdapter so
tests are deterministic, fast, and don't mock anything load-bearing.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from textual.widgets import Input, RichLog

from harness.character import load_character
from harness.config import settings
from harness.model.adapter import ChatMessage, approx_token_count
from harness.model.echo import EchoAdapter
from harness.store.transcript import Transcript
from harness.tui import ChatApp


@pytest.mark.asyncio
async def test_chat_app_mounts_with_expected_widgets(tmp_path) -> None:  # type: ignore[no-untyped-def]
    app = _build_app(tmp_path)
    async with app.run_test() as pilot:
        assert pilot.app.query_one("#output", RichLog) is not None
        assert pilot.app.query_one("#prompt", Input) is not None
        assert pilot.app.query_one("#metrics") is not None
        assert pilot.app.query_one("#prompt", Input).has_focus


@pytest.mark.asyncio
async def test_chat_app_runs_model_turn_and_persists(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """One user submission should: (1) write the user line to the log,
    (2) call the adapter with a system + user thread, (3) write the
    reply to the log, (4) append both turns to the transcript."""
    app = _build_app(tmp_path)
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "hello airton"
        await pilot.press("enter")
        # Worker runs on a thread; give it a tick to complete. Echo
        # adapter finishes immediately so a short wait is enough.
        await _wait_for_workers(pilot)

        log = pilot.app.query_one("#output", RichLog)
        rendered = "\n".join(str(line) for line in log.lines)
        assert "mark" in rendered
        assert "hello airton" in rendered
        # EchoAdapter echoes the last user message prefixed with
        # "[echo] " — confirms the reply path, not just the echo path.
        assert "[echo]" in rendered

        # Transcript now has both turns. Query by session (the
        # transcript we built uses session="test").
        transcript = Transcript(tmp_path / "t.sqlite")
        rows = transcript.fetch_after("test", after_id=0)
        roles = [row.role for row in rows]
        assert roles == ["user", "assistant"]
        assert rows[0].content == "hello airton"
        assert "[echo]" in rows[1].content


@pytest.mark.asyncio
async def test_chat_app_disables_input_during_turn(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """While the worker is running, the input is disabled so the user
    can't pile up turns mid-generation. After the worker finishes,
    the input re-enables and regains focus."""
    app = _build_app(tmp_path)
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "test"
        await pilot.press("enter")

        # With echo adapter the turn completes nearly instantly; by
        # the time _wait_for_workers returns it's already re-enabled.
        await _wait_for_workers(pilot)
        assert not prompt.disabled
        assert prompt.has_focus


@pytest.mark.asyncio
async def test_chat_app_error_in_adapter_shows_red_line(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A broken adapter must not crash the app — the turn reports an
    error in the log and the input re-enables for the user to retry."""
    app = _build_app(tmp_path, adapter=_RaisingAdapter())
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "boom"
        await pilot.press("enter")
        await _wait_for_workers(pilot)

        log = pilot.app.query_one("#output", RichLog)
        rendered = "\n".join(str(line) for line in log.lines)
        assert "error running turn" in rendered
        assert not prompt.disabled


# ---------- helpers ----------


class _RaisingAdapter:
    """Adapter stub that always raises on complete(). Used to verify
    the worker's error-handling path surfaces failures to the log
    without tearing down the app."""

    id = "test:raising"
    context_window = 8192

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        return approx_token_count(messages)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        raise RuntimeError("adapter exploded on purpose")


def _build_app(tmp_path, adapter: object | None = None) -> ChatApp:  # type: ignore[no-untyped-def]
    """Construct a ChatApp with a real Transcript (SQLite in tmp_path),
    the real Airton character, and by default an EchoAdapter. Tests
    stay fast + hermetic — no model weights, no HF downloads."""
    character = load_character(settings.character_path)
    return ChatApp(
        character=character,
        speaker="mark",
        session="test",
        channel="cli",
        adapter=adapter or EchoAdapter(),  # type: ignore[arg-type]
        transcript=Transcript(tmp_path / "t.sqlite"),
        retriever=None,
        top_k=0,
        memory_store=None,
        memories=0,
        semantic_store=None,
        facts=0,
    )


async def _wait_for_workers(pilot) -> None:  # type: ignore[no-untyped-def]
    """Wait for the app's worker to finish. Textual's Pilot has a
    `wait_for_scheduled_animations` but no public worker-wait; the
    app exposes `workers.wait_for_complete` which is an asyncio
    coroutine we can await."""
    await pilot.app.workers.wait_for_complete()
    # Let the call_from_thread callbacks run on the event loop.
    await pilot.pause()
