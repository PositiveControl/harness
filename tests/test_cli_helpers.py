from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from harness.cli import (
    _decode_transcript_message,
    _encode_assistant_with_tool_calls,
    _format_ctx_meter,
    _RetrievalState,
    _retrieve_turn_context,
)
from harness.model.adapter import ChatMessage, approx_token_count, count_tokens
from harness.model.echo import EchoAdapter
from harness.store.transcript import TranscriptMessage
from harness.tools import ToolCall


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
