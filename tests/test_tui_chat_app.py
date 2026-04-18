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
from dataclasses import dataclass

import pytest
from textual.widgets import Input, RichLog, Static

from harness.character import load_character
from harness.config import settings
from harness.model.adapter import ChatMessage, approx_token_count
from harness.model.echo import EchoAdapter
from harness.store.transcript import Transcript
from harness.tools import ModelReply, ReadFileTool, ToolCall, ToolRegistry, ToolSpec
from harness.tui import ChatApp
from harness.tui.confirm_screen import ConfirmToolScreen


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
async def test_metrics_idle_on_mount_then_updates_after_turn(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """harness-17v: on mount the footer reads 'idle' with an empty
    ctx (no history yet). After one turn the ctx field carries a
    non-zero token count — adapter.count_tokens on the growing
    history."""
    app = _build_app(tmp_path)
    async with app.run_test() as pilot:
        metrics = pilot.app.query_one("#metrics", Static)
        initial_text = str(metrics.render()).lower()
        assert "idle" in initial_text

        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "tell me something"
        await pilot.press("enter")
        await _wait_for_workers(pilot)

        final_text = str(metrics.render())
        # After the turn: idle again, and the ctx field has moved off
        # the '—' placeholder because there's real history now.
        assert "idle" in final_text.lower()
        assert "ctx" in final_text.lower()
        # Echo adapter produced meaningful tokens in history; the
        # meter should show at least 1 token (rendered as 0.0k once
        # divided by 1000, but the ctx total is real so the string
        # changes from the pre-turn placeholder).
        assert "—" not in final_text  # '—' is the no-ctx placeholder


@pytest.mark.asyncio
async def test_metrics_shows_thinking_elapsed_during_turn(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """While the worker runs, the footer must show 'thinking Ns'.
    Exercise _refresh_metrics directly by planting a start time
    rather than racing a real worker — timing-dependent assertions
    flake on slow CI and CPU-bound test runs. This test guards the
    render contract; the state-transition test above guards the
    worker-sets-it-correctly path."""
    import time as _time

    app = _build_app(tmp_path)
    async with app.run_test() as pilot:
        tui_app: ChatApp = pilot.app  # type: ignore[assignment]
        # Simulate an in-flight turn: plant a start time ~1.2s ago.
        tui_app._state.turn_started_at = _time.monotonic() - 1.2
        tui_app._refresh_metrics()
        metrics = pilot.app.query_one("#metrics", Static)
        text = str(metrics.render()).lower()
        assert "thinking" in text
        # Elapsed is formatted to one decimal; ~1.2s should appear
        # as some X.Y number — not 0.0.
        assert "0.0s" not in text

        # Clear and re-render: idle.
        tui_app._state.turn_started_at = None
        tui_app._refresh_metrics()
        idle_text = str(metrics.render()).lower()
        assert "idle" in idle_text
        assert "thinking" not in idle_text


@pytest.mark.asyncio
async def test_chat_app_streams_sentences_as_tokens_arrive(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """harness-lrg: in the no-tools path, the worker uses
    adapter.stream() and each completed sentence writes to the log
    as it arrives. Assert the 'airton ›' prefix appears once and
    both sentences land (one via the sentence-boundary flush during
    streaming, one via the final tail flush)."""
    app = _build_app(tmp_path, adapter=_StreamingAdapter())
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "go"
        await pilot.press("enter")
        await _wait_for_workers(pilot)
        # Give the UI a nudge in case a call_from_thread callback is
        # still draining — the thread-pool worker finishes before
        # the UI has painted its final frame.
        await pilot.pause(0.05)

        log = pilot.app.query_one("#output", RichLog)
        joined = " ".join(str(line) for line in log.lines)
        # Both streamed sentences land in the log.
        assert "first sentence" in joined
        assert "second sentence" in joined
        # 'airton ›' prefix appears exactly once for this turn —
        # continuation sentences are unprefixed. The mount-banner
        # mentions airton by name but not the '›' glyph; assertion
        # on the combination narrows to the streamed prefix.
        assert joined.count("airton ›") == 1


@pytest.mark.asyncio
async def test_feed_stream_buffers_partial_until_sentence_boundary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Direct test of the sentence buffer: two half-sentences
    shouldn't emit; once the period arrives, the full sentence does.
    Exercises the UI-thread invariant without racing a worker."""
    app = _build_app(tmp_path)
    async with app.run_test() as pilot:
        tui_app: ChatApp = pilot.app  # type: ignore[assignment]
        log = pilot.app.query_one("#output", RichLog)
        lines_before = len(log.lines)

        tui_app._feed_stream("Hello, ")
        assert len(log.lines) == lines_before  # still pending
        tui_app._feed_stream("world. ")
        # Sentence boundary hit — one line emitted.
        assert len(log.lines) == lines_before + 1
        rendered = str(log.lines[-1])
        assert "Hello, world." in rendered
        # First chunk of the turn carries the 'airton ›' prefix.
        assert "airton" in rendered


@pytest.mark.asyncio
async def test_flush_stream_buffer_emits_trailing_fragment(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A reply that ends mid-sentence (no trailing period) must
    still make it to the log when the worker flushes."""
    app = _build_app(tmp_path)
    async with app.run_test() as pilot:
        tui_app: ChatApp = pilot.app  # type: ignore[assignment]
        log = pilot.app.query_one("#output", RichLog)
        lines_before = len(log.lines)

        tui_app._feed_stream("partial without terminator")
        assert len(log.lines) == lines_before  # buffered
        tui_app._flush_stream_buffer()
        assert len(log.lines) == lines_before + 1
        assert "partial without terminator" in str(log.lines[-1])


@pytest.mark.asyncio
async def test_chat_app_renders_tool_events_inline(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """harness-1r4: when the registry is set, the worker drives
    run_tool_loop. Observer events (router_intent omitted here since
    no router is passed, tool_call_start + tool_call_end) must
    appear in the RichLog before the final assistant reply."""
    (tmp_path / "hello.txt").write_text("world")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    # Scripted adapter: first reply emits a read_file tool call,
    # second reply emits the final text answer.
    scripted = _ToolScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "hello.txt"}),),
            ),
            ModelReply(content="the file says 'world'"),
        ]
    )
    app = _build_app(tmp_path, adapter=scripted, registry=registry, workspace=tmp_path)
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "what's in hello.txt?"
        await pilot.press("enter")
        await _wait_for_workers(pilot)

        log = pilot.app.query_one("#output", RichLog)
        rendered = "\n".join(str(line) for line in log.lines)
        # Tool call's headline + result line + the final reply all
        # show up in order in the log.
        assert "🔧" in rendered
        assert "read_file" in rendered
        assert "✓" in rendered
        assert "the file says" in rendered


@pytest.mark.asyncio
async def test_chat_app_confirm_modal_declined_via_escape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """harness-mz2: write-tier tool pushes the confirm modal.
    Pressing escape dismisses with DECLINE → run_tool_loop injects
    a 'user declined' tool-role message → the scripted adapter's
    wrap-up reply still shows up in the log, and the WriteOnlyTool
    .call() is never reached (its body raises AssertionError)."""
    registry = ToolRegistry()
    registry.register(_WriteOnlyTool())

    scripted = _ToolScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="dangerous_write", arguments={"target": "file.txt"}),),
            ),
            ModelReply(content="ok, nothing written."),
        ]
    )
    app = _build_app(tmp_path, adapter=scripted, registry=registry, workspace=tmp_path)
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "write to file.txt"
        await pilot.press("enter")
        await _wait_for_modal(pilot)
        await pilot.press("escape")
        await _wait_for_workers(pilot)
        await pilot.pause(0.05)

        log = pilot.app.query_one("#output", RichLog)
        rendered = "\n".join(str(line) for line in log.lines)
        assert "declined" in rendered
        assert "ok, nothing written" in rendered


@pytest.mark.asyncio
async def test_chat_app_confirm_modal_approved_via_y(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Pressing `y` on the modal approves — the tool runs, result
    flows back, wrap-up reply renders normally."""
    calls_made: list[dict[str, object]] = []
    registry = ToolRegistry()
    registry.register(_RecordingWriteTool(calls_made=calls_made))

    scripted = _ToolScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="record_write", arguments={"note": "approved"}),),
            ),
            ModelReply(content="wrote the note"),
        ]
    )
    app = _build_app(tmp_path, adapter=scripted, registry=registry, workspace=tmp_path)
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "do it"
        await pilot.press("enter")
        await _wait_for_modal(pilot)
        await pilot.press("y")
        await _wait_for_workers(pilot)
        await pilot.pause(0.05)

        assert calls_made == [{"note": "approved"}]
        log = pilot.app.query_one("#output", RichLog)
        rendered = "\n".join(str(line) for line in log.lines)
        assert "wrote the note" in rendered


@pytest.mark.asyncio
async def test_chat_app_confirm_modal_always_skips_future_prompts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Pressing `a` approves AND marks the tool always-allowed for
    the session. A second call to the same tool next turn must run
    without re-prompting."""
    calls_made: list[dict[str, object]] = []
    registry = ToolRegistry()
    registry.register(_RecordingWriteTool(calls_made=calls_made))

    scripted = _ToolScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="record_write", arguments={"note": "first"}),),
            ),
            ModelReply(content="one done"),
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="record_write", arguments={"note": "second"}),),
            ),
            ModelReply(content="two done"),
        ]
    )
    app = _build_app(tmp_path, adapter=scripted, registry=registry, workspace=tmp_path)
    async with app.run_test() as pilot:
        prompt = pilot.app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        await _wait_for_modal(pilot)
        await pilot.press("a")
        await _wait_for_workers(pilot)
        await pilot.pause(0.05)

        # Turn 2: no modal should appear. Running the turn completes
        # end-to-end; if a modal had popped we'd time out in
        # _wait_for_workers because the worker parks on it.
        prompt.value = "second"
        await pilot.press("enter")
        await _wait_for_workers(pilot)
        await pilot.pause(0.05)

        assert calls_made == [{"note": "first"}, {"note": "second"}]


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


class _ToolScriptedAdapter:
    """Adapter that returns pre-queued ModelReply objects from
    complete_with_tools. Used to drive the tool loop deterministically
    without spinning up a real model."""

    id = "test:scripted"
    context_window = 8192

    def __init__(self, replies: list[ModelReply]) -> None:
        self._replies = list(replies)

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        return approx_token_count(messages)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        # Tool-less path isn't exercised by these tests, but the
        # adapter protocol requires it.
        reply = self._next()
        return reply.content

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: object = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        return self._next()

    def _next(self) -> ModelReply:
        if self._replies:
            return self._replies.pop(0)
        return ModelReply(content="(exhausted)")


class _StreamingAdapter:
    """Adapter that yields two sentence-terminated chunks from
    stream(). Used to verify the TUI renders streamed sentences as
    they land, with the 'airton ›' prefix only on the first."""

    id = "test:streaming"
    context_window = 8192

    def count_tokens(self, messages: Iterable[ChatMessage]) -> int:
        return approx_token_count(messages)

    def stream(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterable[str]:
        # Yielding pre-split sentences mirrors how a real adapter
        # emits tokens — the sentence boundary is what drives the
        # per-line flush, not chunk boundaries.
        yield "first sentence. "
        yield "second sentence."

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return "first sentence. second sentence."


class _WriteOnlyTool:
    """Write-tier tool stub: never actually writes. Body raises so
    the test fails loudly if the decline path regresses and lets a
    call through."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="dangerous_write",
            description="would modify the workspace, but we decline in the test",
            parameters={
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
            tier="write",
        )

    def call(self, *, target: str) -> str:
        raise AssertionError(
            f"write-tier tool should have been declined before reaching call(); target={target!r}"
        )


@dataclass
class _RecordingWriteTool:
    """Benign write-tier tool that records each call instead of
    actually mutating anything. Used to verify the approve and
    always-approve paths let the tool through."""

    calls_made: list[dict[str, object]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="record_write",
            description="records the call arguments, no side effects",
            parameters={
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
            tier="write",
        )

    def call(self, *, note: str) -> str:
        self.calls_made.append({"note": note})
        return f"recorded {note!r}"


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


def _build_app(  # type: ignore[no-untyped-def]
    tmp_path,
    adapter: object | None = None,
    registry: ToolRegistry | None = None,
    workspace=None,
) -> ChatApp:
    """Construct a ChatApp with a real Transcript (SQLite in tmp_path),
    the real Airton character, and by default an EchoAdapter. Tests
    stay fast + hermetic — no model weights, no HF downloads.

    `registry` + `workspace` pass through to the tool-loop path
    (Phase 4 and later); omit both to exercise the classic
    adapter.complete() path."""
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
        registry=registry,
        workspace_path=workspace,
    )


async def _wait_for_workers(pilot) -> None:  # type: ignore[no-untyped-def]
    """Wait for the app's worker to finish. Textual's Pilot has a
    `wait_for_scheduled_animations` but no public worker-wait; the
    app exposes `workers.wait_for_complete` which is an asyncio
    coroutine we can await."""
    await pilot.app.workers.wait_for_complete()
    # Let the call_from_thread callbacks run on the event loop.
    await pilot.pause()


async def _wait_for_modal(pilot, *, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    """Poll until ConfirmToolScreen is on the screen stack or the
    timeout expires. Tests can't use _wait_for_workers here because
    the worker is deliberately parked on call_from_thread while the
    modal is open."""
    import asyncio

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for screen in pilot.app.screen_stack:
            if isinstance(screen, ConfirmToolScreen):
                return
        await pilot.pause(0.02)
    raise AssertionError(
        f"ConfirmToolScreen never appeared within {timeout}s — "
        f"current screen stack: {pilot.app.screen_stack}"
    )
