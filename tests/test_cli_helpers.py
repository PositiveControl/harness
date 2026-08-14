from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from rich.console import Console

from harness.cli import (
    _decode_transcript_message,
    _encode_assistant_with_tool_calls,
    _format_ctx_meter,
    _render_tool_event,
    _RetrievalState,
    _retrieve_turn_context,
    _stream_or_complete,
    _StreamRenderer,
    _ThinkingSpinner,
    _topic_boundary_suffix,
)
from harness.model.adapter import ChatMessage, approx_token_count, count_tokens
from harness.model.echo import EchoAdapter
from harness.orchestrator import ToolLoopEvent
from harness.store.transcript import TranscriptMessage
from harness.tools import ToolCall, ToolResult


def _msg(role: str, content: str, speaker: str = "airton") -> TranscriptMessage:
    return TranscriptMessage(
        id=1,
        session="local",
        channel="cli",
        speaker=speaker,
        role=role,
        content=content,
        created_at=datetime.now(UTC),
    )


def test_encode_no_op_when_no_tool_calls() -> None:
    assert _encode_assistant_with_tool_calls("hello", ()) == "hello"


def test_encode_decode_round_trip() -> None:
    calls = (
        ToolCall(name="read_file", arguments={"path": "README.md"}),
        ToolCall(name="shell", arguments={"cmd": "ls -F"}),
    )
    encoded = _encode_assistant_with_tool_calls("", calls)
    decoded = _decode_transcript_message(_msg("assistant", encoded))
    assert decoded.role == "assistant"
    assert decoded.content == ""
    assert decoded.tool_calls == calls


def test_decode_plain_assistant_passes_through() -> None:
    decoded = _decode_transcript_message(_msg("assistant", "just prose"))
    assert decoded.tool_calls == ()
    assert decoded.content == "just prose"


def test_decode_tool_role_attaches_speaker_as_name() -> None:
    decoded = _decode_transcript_message(_msg("tool", "exit=0\n.", speaker="shell"))
    assert decoded.role == "tool"
    assert decoded.content == "exit=0\n."
    assert decoded.name == "shell"


def test_decode_user_role_unaffected() -> None:
    decoded = _decode_transcript_message(_msg("user", "hello", speaker="mark"))
    assert decoded.role == "user"
    assert decoded.content == "hello"
    assert decoded.tool_calls == ()
    assert decoded.name is None


# ---------- _retrieve_turn_context ----------


@dataclass
class _RaisingRetriever:
    def top_k(self, query: str, *, k: int = 6) -> list[object]:
        raise RuntimeError("embedder exploded")


@dataclass
class _RaisingStore:
    kind: str = "memory"

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        min_score: float = 0.0,
        user_id: str | None = None,
        allowed_sessions: tuple[str, ...] | None = None,
        recency_ranks: dict[str, int] | None = None,
        recency_weight: float = 0.0,
    ) -> list[tuple[object, float]]:
        raise RuntimeError(f"{self.kind} search blew up")


@dataclass
class _StaticMemoryStore:
    hits: list[tuple[object, float]] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        min_score: float = 0.0,
        user_id: str | None = None,
        allowed_sessions: tuple[str, ...] | None = None,
        recency_ranks: dict[str, int] | None = None,
        recency_weight: float = 0.0,
    ) -> list[tuple[object, float]]:
        return self.hits


def test_retrieve_turn_context_disables_failing_voice_and_warns_once() -> None:
    warnings: list[str] = []
    state = _RetrievalState()
    examples, recalled, facts = _retrieve_turn_context(
        user_input="hi",
        speaker="mark",
        retriever=_RaisingRetriever(),  # type: ignore[arg-type]
        memory_store=None,
        semantic_store=None,
        top_k=6,
        memories=0,
        memories_threshold=0.5,
        facts=0,
        facts_threshold=0.45,
        state=state,
        warn=warnings.append,
    )
    assert examples == []
    assert recalled == []
    assert facts == []
    assert state.voice_ok is False
    assert len(warnings) == 1
    assert "voice" in warnings[0].lower()

    # Second call: voice source is skipped entirely, no new warning fires.
    _retrieve_turn_context(
        user_input="hi again",
        speaker="mark",
        retriever=_RaisingRetriever(),  # type: ignore[arg-type]
        memory_store=None,
        semantic_store=None,
        top_k=6,
        memories=0,
        memories_threshold=0.5,
        facts=0,
        facts_threshold=0.45,
        state=state,
        warn=warnings.append,
    )
    assert len(warnings) == 1


def test_retrieve_turn_context_disables_each_source_independently() -> None:
    warnings: list[str] = []
    state = _RetrievalState()
    _retrieve_turn_context(
        user_input="hi",
        speaker="mark",
        retriever=None,
        memory_store=_RaisingStore(kind="memory"),  # type: ignore[arg-type]
        semantic_store=_RaisingStore(kind="semantic"),  # type: ignore[arg-type]
        top_k=0,
        memories=3,
        memories_threshold=0.5,
        facts=5,
        facts_threshold=0.45,
        state=state,
        warn=warnings.append,
    )
    assert state.voice_ok is True
    assert state.episodic_ok is False
    assert state.semantic_ok is False
    assert len(warnings) == 2


def test_retrieve_turn_context_returns_empty_when_muted() -> None:
    """/clear flips `state.muted=True` so prior-session memories can't
    leak into a fresh start via retrieval. The function must skip all
    three sources even when stores would happily return hits
    (harness-zpe). The raising stores prove no call is made at all."""
    warnings: list[str] = []
    state = _RetrievalState(muted=True)
    examples, recalled, facts = _retrieve_turn_context(
        user_input="My name is Mark",
        speaker="mark",
        retriever=_RaisingRetriever(),  # type: ignore[arg-type]
        memory_store=_RaisingStore(kind="memory"),  # type: ignore[arg-type]
        semantic_store=_RaisingStore(kind="semantic"),  # type: ignore[arg-type]
        top_k=6,
        memories=3,
        memories_threshold=0.5,
        facts=5,
        facts_threshold=0.45,
        state=state,
        warn=warnings.append,
    )
    assert examples == []
    assert recalled == []
    assert facts == []
    # No source was touched, so no 'disabled' warning fires either.
    assert warnings == []
    # Health flags unchanged — mute is a separate axis from error health.
    assert state.voice_ok is True
    assert state.episodic_ok is True
    assert state.semantic_ok is True


def test_retrieve_turn_context_returns_hits_when_healthy() -> None:
    fake_record = object()
    store = _StaticMemoryStore(hits=[(fake_record, 0.9)])
    state = _RetrievalState()
    examples, recalled, facts = _retrieve_turn_context(
        user_input="hi",
        speaker="mark",
        retriever=None,
        memory_store=store,  # type: ignore[arg-type]
        semantic_store=None,
        top_k=0,
        memories=3,
        memories_threshold=0.5,
        facts=0,
        facts_threshold=0.45,
        state=state,
        warn=lambda _msg: None,
    )
    assert examples == []
    assert recalled == [fake_record]
    assert facts == []
    assert state.episodic_ok is True


# ---------- context meter + count_tokens ----------


def test_count_tokens_uses_adapter_hook_when_available() -> None:
    class _FixedAdapter:
        id = "fixed"
        context_window = 1000

        def complete(
            self,
            messages: list[ChatMessage],
            *,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> str:
            return ""

        def count_tokens(self, messages: list[ChatMessage]) -> int:
            return 42

    msgs = [ChatMessage(role="user", content="anything")]
    assert count_tokens(_FixedAdapter(), msgs) == 42


def test_count_tokens_falls_back_when_adapter_has_no_hook() -> None:
    class _Plain:
        id = "plain"
        context_window = 100

        def complete(
            self,
            messages: list[ChatMessage],
            *,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> str:
            return ""

    msgs = [ChatMessage(role="user", content="a" * 40)]
    expected = approx_token_count(msgs)
    assert count_tokens(_Plain(), msgs) == expected


def test_count_tokens_swallows_adapter_failure() -> None:
    class _Broken:
        id = "broken"
        context_window = 100

        def complete(
            self,
            messages: list[ChatMessage],
            *,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> str:
            return ""

        def count_tokens(self, messages: list[ChatMessage]) -> int:
            raise RuntimeError("tokenizer corrupt")

    msgs = [ChatMessage(role="user", content="hello")]
    assert count_tokens(_Broken(), msgs) == approx_token_count(msgs)


def test_count_tokens_on_echo_uses_char_heuristic() -> None:
    msgs = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello world"),
    ]
    assert count_tokens(EchoAdapter(), msgs) == approx_token_count(msgs)


def test_format_ctx_meter_color_thresholds() -> None:
    # Well under 75% → dim
    low = _format_ctx_meter(used=1_000, total=10_000)
    assert "[dim]" in low
    # 80% → yellow
    mid = _format_ctx_meter(used=8_000, total=10_000)
    assert "[yellow]" in mid
    # 95% → red
    high = _format_ctx_meter(used=9_500, total=10_000)
    assert "[red]" in high


def test_format_ctx_meter_returns_empty_when_total_unknown() -> None:
    assert _format_ctx_meter(used=100, total=0) == ""


def test_decode_malformed_sentinel_payload_is_tolerated() -> None:
    """A row corrupted with bad JSON after the sentinel should not crash;
    tool_calls degrades to empty and we keep the head content."""
    bogus = "some text\n__TOOL_CALLS_V1__\n{not valid json"
    decoded = _decode_transcript_message(_msg("assistant", bogus))
    assert decoded.tool_calls == ()
    assert "some text" in decoded.content


# ---------- _render_tool_event ----------


def _build_render_deps() -> tuple[Console, _ThinkingSpinner, _StreamRenderer]:
    """Rich Console wired for in-memory capture. `record=True` lets us
    read back everything printed via `export_text()`. `force_terminal`
    + `color_system=None` keep output plain so assertions match on
    literal substrings without ANSI escapes."""
    console = Console(record=True, force_terminal=False, color_system=None, width=200)
    return console, _ThinkingSpinner(console), _StreamRenderer(console)


def _render_all(events: list[ToolLoopEvent]) -> str:
    console, thinking, stream_renderer = _build_render_deps()
    for event in events:
        _render_tool_event(
            event,
            console=console,
            thinking=thinking,
            stream_renderer=stream_renderer,
            tool_label=lambda name: name,
        )
    thinking.stop()
    return console.export_text()


def test_render_tool_event_prints_one_line_per_tool_call() -> None:
    """harness-cx2: the CLI renderer must render every tool_call_start
    event. Earlier report implied a 'first call only' guard; this locks
    in that two tool_call_start events produce two 🔧 lines and two
    tool_call_end events produce two ✓ lines."""
    call_a = ToolCall(name="read_file", arguments={"path": "a.txt"})
    call_b = ToolCall(name="search_web", arguments={"query": "q"})
    events = [
        ToolLoopEvent(kind="tool_call_start", call=call_a, round_index=0),
        ToolLoopEvent(
            kind="tool_call_end",
            call=call_a,
            result=ToolResult(tool_name="read_file", output="hello", success=True),
            round_index=0,
        ),
        ToolLoopEvent(kind="tool_call_start", call=call_b, round_index=1),
        ToolLoopEvent(
            kind="tool_call_end",
            call=call_b,
            result=ToolResult(tool_name="search_web", output="world", success=True),
            round_index=1,
        ),
    ]
    output = _render_all(events)
    assert output.count("🔧 read_file") == 1
    assert output.count("🔧 search_web") == 1
    # Two distinct ✓ lines — bugged renderers might print one and drop
    # the other on dedup; this guards against that regression.
    assert output.count("✓") == 2


def test_render_tool_event_renders_router_intent_then_tool_pair() -> None:
    """Router-prelude scenario from harness-cx2 repro: router fires a
    tool, then the main model fires another. All three headline lines
    (→ routed, 🔧 first, 🔧 second) must appear."""
    router_call = ToolCall(name="search_facts", arguments={"query": "x"})
    model_call = ToolCall(name="search_web", arguments={"query": "x"})
    events = [
        ToolLoopEvent(kind="router_intent", call=router_call, round_index=0),
        ToolLoopEvent(kind="tool_call_start", call=router_call, round_index=0),
        ToolLoopEvent(
            kind="tool_call_end",
            call=router_call,
            result=ToolResult(tool_name="search_facts", output="weak", success=True),
            round_index=0,
        ),
        ToolLoopEvent(kind="tool_call_start", call=model_call, round_index=1),
        ToolLoopEvent(
            kind="tool_call_end",
            call=model_call,
            result=ToolResult(tool_name="search_web", output="results", success=True),
            round_index=1,
        ),
    ]
    output = _render_all(events)
    assert "→ routed to search_facts" in output
    assert output.count("🔧") == 2
    assert "🔧 search_facts" in output
    assert "🔧 search_web" in output


def test_render_tool_event_marks_deduped_call() -> None:
    """harness-pun: when the orchestrator short-circuits a duplicate
    call, the CLI emits a dim one-liner so the user sees the tool was
    skipped rather than re-run. The nudge result itself isn't echoed —
    it's only there for the model's next round."""
    call = ToolCall(name="list_dir", arguments={"path": "", "recursive": True})
    events = [
        ToolLoopEvent(
            kind="tool_call_deduped",
            call=call,
            result=ToolResult(
                tool_name="list_dir",
                output="[duplicate call — identical arguments…]",
                success=True,
            ),
            round_index=1,
        ),
    ]
    output = _render_all(events)
    assert "duplicate call skipped" in output
    assert "list_dir" in output
    # 🔧 start-of-call glyph must NOT appear — that would imply the
    # call ran; deduped calls get a distinct arrow glyph instead.
    assert "🔧" not in output


def test_render_tool_event_marks_blocked_call() -> None:
    """A non-duplicate pre_tool guard (targeted_fix_no_overwrite,
    grounding, …) emits tool_call_blocked, not tool_call_deduped. The
    CLI surfaces the guard's reason — NOT a misleading 'duplicate' line."""
    call = ToolCall(name="write_file", arguments={"path": "game.js", "content": "…"})
    events = [
        ToolLoopEvent(
            kind="tool_call_blocked",
            call=call,
            result=ToolResult(
                tool_name="write_file",
                output="refusing write_file on existing game.js in TARGETED-FIX mode…",
                success=False,
                error="targeted_fix_no_overwrite",
            ),
            round_index=1,
        ),
    ]
    output = _render_all(events)
    assert "blocked" in output
    assert "targeted_fix_no_overwrite" in output
    assert "write_file" in output
    # Not labeled a duplicate — that was the bug this split fixed.
    assert "duplicate" not in output
    # No 🔧 — the call never ran.
    assert "🔧" not in output


def test_render_tool_event_marks_failed_and_declined() -> None:
    call = ToolCall(name="shell", arguments={"cmd": "ls"})
    events = [
        ToolLoopEvent(kind="tool_call_start", call=call, round_index=0),
        ToolLoopEvent(
            kind="tool_call_failed",
            call=call,
            result=ToolResult(
                tool_name="shell", output="permission denied", success=False, error="eacces"
            ),
            round_index=0,
        ),
        ToolLoopEvent(kind="tool_call_start", call=call, round_index=1),
        ToolLoopEvent(kind="tool_call_declined", call=call, round_index=1),
    ]
    output = _render_all(events)
    assert "✗" in output
    assert "declined" in output


def test_topic_boundary_suffix_silent_when_unmuted() -> None:
    """harness-eftf: a fresh / non-cleared session must not get the
    topic-boundary note. Default `_RetrievalState` has muted=False."""
    state = _RetrievalState()
    assert _topic_boundary_suffix(state) == ""


def test_topic_boundary_suffix_fires_when_muted() -> None:
    """When retrieval is muted (by /clear in this process or by the
    persistent watermark hydrating on init), the suffix carries a
    leading separator + the canonical NEW TOPIC marker so the model
    has a frame for ignoring spurious bleed."""
    state = _RetrievalState(muted=True)
    suffix = _topic_boundary_suffix(state)
    assert suffix.startswith("\n\n")
    assert "NEW TOPIC" in suffix
    assert "reset this conversation" in suffix


def test_topic_boundary_suffix_fires_when_scope_bounded() -> None:
    """harness-w3mo: when retrieval is unmuted but `--memory-scope` is
    bounding retrieval to a session subset, the suffix surfaces a
    different SESSION-SCOPED note explaining why earlier sessions
    don't apply."""
    state = _RetrievalState()
    suffix = _topic_boundary_suffix(state, allowed_sessions=("cli-2026-04-26",))
    assert "SESSION-SCOPED" in suffix
    assert "earlier sessions" in suffix


def test_topic_boundary_suffix_mute_wins_over_scope() -> None:
    """When both signals fire — /clear AND --memory-scope=
    current-session — the cleared-conversation note takes
    precedence; both can't render at once and 'just reset' is the
    stronger framing."""
    state = _RetrievalState(muted=True)
    suffix = _topic_boundary_suffix(state, allowed_sessions=("X",))
    assert "reset this conversation" in suffix
    assert "SESSION-SCOPED" not in suffix


# ---------- _make_write_file_redirect_hook (harness-hf4r) ----------


def test_write_file_redirect_hook_constructed_for_lazy_loaded_write_file(
    tmp_path: object,
) -> None:
    """harness-hf4r regression: the hook must be constructed even
    when write_file is NOT yet registered at session start. With
    --tool-set minimal the model discovers write_file via
    tool_search + load_tool mid-session; the hook's closures
    consult the LIVE registry at call time, so a hook that was
    None-ed out at construction time never fires.

    Mark's 2026-05-20T21:47 GTA session: minimal tool set →
    load_tool registers write_file mid-session → model emits
    write_file on an existing path → existing path errors
    instead of the redirect firing → loop exhausts."""
    from pathlib import Path as _Path

    from harness.cli_classic import _make_write_file_redirect_hook
    from harness.orchestrator.hooks import (
        BailContext,  # noqa: F401  # type pin for Skip return shape
        PreToolContext,
        Skip,
    )
    from harness.tools import EditFileTool, ToolCall, ToolRegistry, WriteFileTool

    workspace = _Path(tmp_path)  # type: ignore[arg-type]
    registry = ToolRegistry()
    # Registry starts EMPTY — no write_file, no edit_file. The hook
    # must still construct (returns a real WriteFileRedirectHook,
    # not None). This is the regression that harness-hf4r fixes.
    hook = _make_write_file_redirect_hook(registry=registry, workspace_path=workspace)
    assert hook is not None
    assert hook.name == "write_file_redirect"

    # Now lazy-load write_file + edit_file the way load_tool does
    # mid-session, plus pre-create the target so the redirect path
    # has something to read.
    registry.register(WriteFileTool(root=workspace))
    registry.register(EditFileTool(root=workspace))
    (workspace / "game.js").write_text(
        "// existing 50 bytes minimum content here for safe redirect"
    )

    # Fire the hook on a write_file call against the existing path.
    # The hook should Skip with the edit_file redirect result.
    call = ToolCall(
        name="write_file",
        arguments={
            "path": "game.js",
            "content": "// new content also at least 50 bytes minimum for safe redirect",
        },
    )
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Skip)
    assert "write_file → edit_file" in outcome.result.output
    assert outcome.result.success is True


def test_write_file_redirect_hook_no_op_on_unrelated_tool(tmp_path: object) -> None:
    """Always-constructed hook must be cheap on non-write_file
    calls — Continue is the only outcome for unrelated tools, so
    the always-on cost (harness-hf4r) is bounded to one is-eq check
    per pre_tool dispatch."""
    from pathlib import Path as _Path

    from harness.cli_classic import _make_write_file_redirect_hook
    from harness.orchestrator.hooks import Continue, PreToolContext
    from harness.tools import ToolCall, ToolRegistry

    workspace = _Path(tmp_path)  # type: ignore[arg-type]
    hook = _make_write_file_redirect_hook(registry=ToolRegistry(), workspace_path=workspace)
    assert hook is not None
    call = ToolCall(name="read_file", arguments={"path": "anything"})
    outcome = hook.check(PreToolContext(call=call, seen_calls={}))
    assert isinstance(outcome, Continue)


# ---------- _ThinkingSpinner ----------
#
# harness-z4k1.1 step 2: the spinner was 36% covered — start(), stop(),
# the ticker thread and the context manager were all unexercised. The
# chat loop leans on both calls being idempotent (the loop kicks it on
# at Enter, the tool-loop observer bounces it per model call, and every
# console.input() stops it), so that property is the contract worth
# pinning before the class moves file.


def _silent_console() -> Console:
    return Console(record=True, force_terminal=False, color_system=None, width=200)


def test_spinner_start_is_idempotent() -> None:
    spinner = _ThinkingSpinner(_silent_console())
    try:
        spinner.start()
        status_after_first = spinner._status
        thread_after_first = spinner._thread

        spinner.start()

        assert spinner._status is status_after_first
        assert spinner._thread is thread_after_first
    finally:
        spinner.stop()


def test_spinner_stop_is_idempotent_and_releases_the_thread() -> None:
    spinner = _ThinkingSpinner(_silent_console())
    spinner.start()
    thread = spinner._thread
    assert thread is not None

    spinner.stop()
    spinner.stop()  # second stop must be a no-op, not an error

    assert spinner._status is None
    assert spinner._thread is None
    assert not thread.is_alive()


def test_spinner_stop_without_start_is_a_no_op() -> None:
    """The chat loop stops the spinner before every console.input(),
    including paths where it was never started."""
    _ThinkingSpinner(_silent_console()).stop()


def test_spinner_context_manager_starts_and_stops() -> None:
    spinner = _ThinkingSpinner(_silent_console())

    # Read through locals: asserting on `spinner._running` directly
    # narrows the attribute for the rest of the function, and mypy has
    # no way to know __exit__ flipped it back.
    with spinner as entered:
        running_inside = spinner._running
        thread_inside = spinner._thread

    assert entered is spinner
    assert running_inside is True
    assert thread_inside is not None
    assert spinner._running is False
    assert spinner._thread is None


def test_spinner_tick_updates_the_elapsed_suffix() -> None:
    """The ticker rewrites the label with whole elapsed seconds. Driven
    through a stub event (one wait() returns False, then True) so the
    assertion doesn't depend on wall-clock timing."""

    class _OneShotEvent:
        def __init__(self) -> None:
            self.waits = 0

        def wait(self, timeout: float) -> bool:
            self.waits += 1
            return self.waits > 1

    class _RecordingStatus:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def update(self, label: str) -> None:
            self.labels.append(label)

    spinner = _ThinkingSpinner(_silent_console(), label="working")
    status = _RecordingStatus()
    spinner._status = status  # type: ignore[assignment]
    spinner._stop_event = _OneShotEvent()  # type: ignore[assignment]
    spinner._started_at = 0.0

    spinner._tick()

    assert len(status.labels) == 1
    assert "working…" in status.labels[0]
    assert status.labels[0].endswith("s[/dim]")


def test_spinner_tick_returns_when_the_status_is_torn_down() -> None:
    """stop() clears _status; a ticker mid-wait must notice and exit
    rather than update a dead Status."""

    class _AlwaysGoEvent:
        def wait(self, timeout: float) -> bool:
            return False

    spinner = _ThinkingSpinner(_silent_console())
    spinner._stop_event = _AlwaysGoEvent()  # type: ignore[assignment]
    spinner._status = None

    spinner._tick()  # returns instead of looping forever


# ---------- _render_tool_event: the model-call + retry branches ----------


def test_render_tool_event_model_call_start_raises_the_spinner() -> None:
    console, thinking, renderer = _build_render_deps()
    try:
        _render_tool_event(
            ToolLoopEvent(kind="model_call_start", round_index=0),
            console=console,
            thinking=thinking,
            stream_renderer=renderer,
            tool_label=lambda name: name,
        )

        assert thinking._running
    finally:
        thinking.stop()


def test_render_tool_event_first_token_drops_the_spinner_and_streams() -> None:
    """The spinner must die on the FIRST token, not at model_call_end —
    otherwise it animates on top of the streaming reply."""
    console, thinking, renderer = _build_render_deps()
    thinking.start()

    _render_tool_event(
        ToolLoopEvent(kind="token_delta", delta="hello ", round_index=0),
        console=console,
        thinking=thinking,
        stream_renderer=renderer,
        tool_label=lambda name: name,
    )

    assert not thinking._running
    assert renderer.active
    assert "hello" in renderer.stop()


def test_render_tool_event_empty_token_delta_still_stops_the_spinner() -> None:
    console, thinking, renderer = _build_render_deps()
    thinking.start()

    _render_tool_event(
        ToolLoopEvent(kind="token_delta", delta="", round_index=0),
        console=console,
        thinking=thinking,
        stream_renderer=renderer,
        tool_label=lambda name: name,
    )

    assert not thinking._running
    assert not renderer.active


def test_render_tool_event_model_call_end_closes_spinner_and_stream() -> None:
    console, thinking, renderer = _build_render_deps()
    thinking.start()
    renderer.start()
    renderer.append("partial")

    _render_tool_event(
        ToolLoopEvent(kind="model_call_end", round_index=0),
        console=console,
        thinking=thinking,
        stream_renderer=renderer,
        tool_label=lambda name: name,
    )

    assert not thinking._running
    assert not renderer.active


def test_render_tool_event_truncated_retry_reports_the_budget_change() -> None:
    """harness-738f: the user needs to see the partial was discarded AND
    what the budget went from/to, to tell runaway preamble from a
    healthy tail clip."""
    console, thinking, renderer = _build_render_deps()
    renderer.start()
    renderer.append("half a rep")

    _render_tool_event(
        ToolLoopEvent(kind="truncated_retry", budget_before=512, budget_after=1024, round_index=0),
        console=console,
        thinking=thinking,
        stream_renderer=renderer,
        tool_label=lambda name: name,
    )

    output = console.export_text()
    assert "truncated, retrying with wider budget" in output
    assert "512" in output
    assert "1024" in output
    assert not renderer.active


def test_render_tool_event_bail_retry_names_the_catcher_and_drops_the_draft() -> None:
    """harness-24xj: the fabricated draft must not stay stacked above
    the retry."""
    console, thinking, renderer = _build_render_deps()
    renderer.start()
    renderer.append("fabricated draft")

    _render_tool_event(
        ToolLoopEvent(kind="bail_retry", catcher="FabricatedSearch", round_index=0),
        console=console,
        thinking=thinking,
        stream_renderer=renderer,
        tool_label=lambda name: name,
    )

    output = console.export_text()
    assert "discarding draft, retrying" in output
    assert "FabricatedSearch" in output
    assert not renderer.active


def test_render_tool_event_bail_retry_without_a_catcher_omits_the_suffix() -> None:
    console, thinking, renderer = _build_render_deps()

    _render_tool_event(
        ToolLoopEvent(kind="bail_retry", round_index=0),
        console=console,
        thinking=thinking,
        stream_renderer=renderer,
        tool_label=lambda name: name,
    )

    assert "discarding draft, retrying…" in console.export_text()


# ---------- _stream_or_complete ----------


def test_stream_or_complete_streams_when_the_adapter_can() -> None:
    """Returns streamed=True so the caller skips the duplicate final
    Markdown print — Live already put the text on screen."""

    class _StreamingAdapter:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def stream(self, messages: list[ChatMessage], **kwargs: object) -> list[str]:
            self.kwargs = kwargs
            return ["one ", "two ", "three."]

        def complete(self, messages: list[ChatMessage], **kwargs: object) -> str:
            raise AssertionError("complete() must not be called when stream() exists")

    console = _silent_console()
    renderer = _StreamRenderer(console)
    adapter = _StreamingAdapter()

    text, streamed = _stream_or_complete(
        adapter,
        [ChatMessage(role="user", content="hi")],
        stream_renderer=renderer,
        max_tokens=64,
        temperature=0.1,
    )

    assert streamed
    assert "three." in text
    assert adapter.kwargs == {"max_tokens": 64, "temperature": 0.1}


def test_stream_or_complete_falls_back_to_complete() -> None:
    """No stream() on the adapter — return the blocking reply and
    streamed=False so the caller prints it itself."""

    class _BlockingAdapter:
        def complete(self, messages: list[ChatMessage], **kwargs: object) -> str:
            return "blocking reply"

    renderer = _StreamRenderer(_silent_console())

    text, streamed = _stream_or_complete(
        _BlockingAdapter(),
        [ChatMessage(role="user", content="hi")],
        stream_renderer=renderer,
    )

    assert text == "blocking reply"
    assert not streamed
    assert not renderer.active


def test_stream_or_complete_ignores_a_non_callable_stream_attribute() -> None:
    """`stream` present but not callable (a config flag, a stub) must
    not be mistaken for the streaming path."""

    class _OddAdapter:
        stream = "not callable"

        def complete(self, messages: list[ChatMessage], **kwargs: object) -> str:
            return "fallback"

    text, streamed = _stream_or_complete(
        _OddAdapter(),
        [ChatMessage(role="user", content="hi")],
        stream_renderer=_StreamRenderer(_silent_console()),
    )

    assert (text, streamed) == ("fallback", False)
