from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from harness.compaction import (
    CompactionStore,
    run_compaction,
    should_compact,
)
from harness.compaction.summarizer import build_summarizer_messages, summarize_turns
from harness.model.adapter import ChatMessage
from harness.store.transcript import Transcript, TranscriptMessage


@dataclass
class _ScriptedAdapter:
    """Returns pre-queued replies in order; records prompts for assertion."""

    replies: list[str]
    id: str = "scripted"
    context_window: int = 8192
    calls: list[list[ChatMessage]] = field(default_factory=list)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(list(messages))
        return self.replies.pop(0) if self.replies else ""


def _add_turn(
    t: Transcript,
    *,
    session: str,
    role: str,
    speaker: str,
    content: str,
) -> TranscriptMessage:
    return t.append(session=session, channel="cli", speaker=speaker, role=role, content=content)


# ---------- should_compact ----------


def test_should_compact_fires_at_or_above_threshold() -> None:
    assert should_compact(used_tokens=8000, context_window=10000, threshold_pct=0.8)
    assert should_compact(used_tokens=8500, context_window=10000, threshold_pct=0.8)


def test_should_compact_does_not_fire_below_threshold() -> None:
    assert not should_compact(used_tokens=7000, context_window=10000, threshold_pct=0.8)


def test_should_compact_disabled_when_threshold_zero_or_one() -> None:
    assert not should_compact(used_tokens=100_000, context_window=10000, threshold_pct=0.0)
    assert not should_compact(used_tokens=100_000, context_window=10000, threshold_pct=1.0)


def test_should_compact_handles_zero_context_window() -> None:
    assert not should_compact(used_tokens=100, context_window=0, threshold_pct=0.8)


# ---------- build_summarizer_messages ----------


def test_build_summarizer_messages_includes_prior_summary(tmp_path: Path) -> None:
    t = Transcript(tmp_path / "t.sqlite")
    try:
        m = _add_turn(t, session="s", role="user", speaker="mark", content="hello")
        msgs = build_summarizer_messages([m], prior_summary="earlier discussion about X")
        assert msgs[0].role == "system"
        assert "compacting" in msgs[0].content.lower()
        assert "earlier discussion about X" in msgs[1].content
        assert "mark" in msgs[1].content
        assert "hello" in msgs[1].content
    finally:
        t.close()


# ---------- summarize_turns ----------


def test_summarize_turns_empty_returns_prior(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(replies=[])
    assert summarize_turns(adapter, [], prior_summary="keep this") == "keep this"
    assert summarize_turns(adapter, [], prior_summary=None) == ""


def test_summarize_turns_calls_adapter_and_strips(tmp_path: Path) -> None:
    t = Transcript(tmp_path / "t.sqlite")
    try:
        m = _add_turn(t, session="s", role="user", speaker="mark", content="hi")
        adapter = _ScriptedAdapter(replies=["  Summary line one.  "])
        out = summarize_turns(adapter, [m], prior_summary=None)
        assert out == "Summary line one."
        assert len(adapter.calls) == 1
    finally:
        t.close()


# ---------- CompactionStore ----------


def test_compaction_store_latest_returns_most_recent_row(tmp_path: Path) -> None:
    store = CompactionStore(tmp_path / "c.sqlite")
    try:
        assert store.latest_for_session("s") is None
        store.append(
            session_id="s",
            summary="first",
            up_to_turn_id=5,
            covered_turns=3,
            model_id="m",
        )
        store.append(
            session_id="s",
            summary="second",
            up_to_turn_id=10,
            covered_turns=5,
            model_id="m",
        )
        latest = store.latest_for_session("s")
        assert latest is not None
        assert latest.summary == "second"
        assert latest.up_to_turn_id == 10
    finally:
        store.close()


def test_compaction_store_sessions_are_isolated(tmp_path: Path) -> None:
    store = CompactionStore(tmp_path / "c.sqlite")
    try:
        store.append(
            session_id="a", summary="alpha", up_to_turn_id=1, covered_turns=1, model_id="m"
        )
        store.append(
            session_id="b", summary="beta", up_to_turn_id=1, covered_turns=1, model_id="m"
        )
        assert store.latest_for_session("a") is not None
        assert store.latest_for_session("a").summary == "alpha"  # type: ignore[union-attr]
        assert store.latest_for_session("b").summary == "beta"  # type: ignore[union-attr]
        assert store.latest_for_session("c") is None
    finally:
        store.close()


# ---------- run_compaction end-to-end ----------


def test_run_compaction_writes_summary_and_advances_pointer(tmp_path: Path) -> None:
    t = Transcript(tmp_path / "h.sqlite")
    store = CompactionStore(tmp_path / "h.sqlite")
    adapter = _ScriptedAdapter(replies=["folded summary"])
    try:
        for i in range(15):
            _add_turn(t, session="s", role="user", speaker="mark", content=f"msg {i}")

        outcome = run_compaction(
            adapter,
            t,
            store,
            session_id="s",
            keep_recent=5,
        )
        assert outcome.wrote is True
        # 15 total, keep 5 recent → 10 compacted
        assert outcome.covered_turns == 10
        assert outcome.summary == "folded summary"

        latest = store.latest_for_session("s")
        assert latest is not None
        assert latest.covered_turns == 10
        assert latest.up_to_turn_id == 10  # id of the last compacted turn
    finally:
        t.close()
        store.close()


def test_run_compaction_skips_when_not_enough_turns(tmp_path: Path) -> None:
    t = Transcript(tmp_path / "h.sqlite")
    store = CompactionStore(tmp_path / "h.sqlite")
    adapter = _ScriptedAdapter(replies=[])
    try:
        for i in range(3):
            _add_turn(t, session="s", role="user", speaker="mark", content=f"msg {i}")
        outcome = run_compaction(adapter, t, store, session_id="s", keep_recent=10)
        assert outcome.wrote is False
        assert outcome.covered_turns == 0
        # Adapter never called — nothing to summarize
        assert len(adapter.calls) == 0
    finally:
        t.close()
        store.close()


def test_run_compaction_extends_prior_summary(tmp_path: Path) -> None:
    t = Transcript(tmp_path / "h.sqlite")
    store = CompactionStore(tmp_path / "h.sqlite")
    adapter = _ScriptedAdapter(replies=["first-batch summary", "extended summary"])
    try:
        for i in range(20):
            _add_turn(t, session="s", role="user", speaker="mark", content=f"msg {i}")
        first = run_compaction(adapter, t, store, session_id="s", keep_recent=5)
        assert first.wrote

        # Add more turns and compact again — second call should see prior summary
        for i in range(20, 30):
            _add_turn(t, session="s", role="user", speaker="mark", content=f"msg {i}")
        second = run_compaction(adapter, t, store, session_id="s", keep_recent=5)
        assert second.wrote

        # Second summarizer call saw the prior summary in its user prompt
        user_msg_second = adapter.calls[1][1].content
        assert "first-batch summary" in user_msg_second
    finally:
        t.close()
        store.close()


def test_run_compaction_does_not_advance_on_empty_summary(tmp_path: Path) -> None:
    t = Transcript(tmp_path / "h.sqlite")
    store = CompactionStore(tmp_path / "h.sqlite")
    adapter = _ScriptedAdapter(replies=["   "])  # whitespace only → bail
    try:
        for i in range(15):
            _add_turn(t, session="s", role="user", speaker="mark", content=f"msg {i}")
        outcome = run_compaction(adapter, t, store, session_id="s", keep_recent=5)
        assert outcome.wrote is False
        assert store.latest_for_session("s") is None
    finally:
        t.close()
        store.close()
