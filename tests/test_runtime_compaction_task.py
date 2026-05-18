"""Tests for the periodic-compaction heartbeat task — harness-klwg."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from harness.compaction.store import CompactionStore
from harness.model.adapter import ChatMessage
from harness.runtime.tasks.compaction import (
    CompactionTaskOutcome,
    build_compaction_task,
)
from harness.store.transcript import Transcript


@dataclass
class _ScriptedAdapter:
    """Reuses the test_compaction pattern: hand back a queued summary
    string and record the prompts. The compaction task doesn't care
    about the prompt content — it just needs a non-empty reply so
    run_compaction advances the watermark."""

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


def _seed_session(t: Transcript, *, session: str, turns: int) -> None:
    """Append `turns` alternating user/assistant messages to `session`."""
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        speaker = "mark" if role == "user" else "airton"
        t.append(
            session=session,
            channel="cli",
            speaker=speaker,
            role=role,
            content=f"{role}-{i}",
        )


def _build(
    tmp_path: Path,
    *,
    sessions: dict[str, int],
    replies: int = 4,
    lookback_hours: float = 24.0,
    min_unprocessed_turns: int = 20,
    keep_recent: int = 10,
    clock: datetime | None = None,
) -> tuple[
    _ScriptedAdapter,
    Transcript,
    CompactionStore,
    list[CompactionTaskOutcome],
    object,
]:
    t = Transcript(tmp_path / "t.sqlite")
    cs = CompactionStore(tmp_path / "c.sqlite")
    for name, count in sessions.items():
        _seed_session(t, session=name, turns=count)
    adapter = _ScriptedAdapter(replies=[f"summary-{i}" for i in range(replies)])
    sink_records: list[CompactionTaskOutcome] = []
    fixed_now = clock or datetime.now(UTC)
    task = build_compaction_task(
        adapter=adapter,
        transcript=t,
        compaction_store=cs,
        lookback_hours=lookback_hours,
        keep_recent=keep_recent,
        min_unprocessed_turns=min_unprocessed_turns,
        sink=sink_records.append,
        clock=lambda: fixed_now,
    )
    return adapter, t, cs, sink_records, task


def test_compacts_session_over_threshold(tmp_path: Path) -> None:
    """Single session with 50 turns and no prior compaction: fire the
    task; the watermark advances; the sink records one compaction."""
    _adapter, _t, cs, sink, task = _build(tmp_path, sessions={"s": 50})
    task()  # type: ignore[operator]
    assert len(sink) == 1
    outcome = sink[0]
    assert outcome.sessions_compacted == ["s"]
    assert outcome.sessions_skipped == []
    assert outcome.errors == []
    record = cs.latest_for_session("s")
    assert record is not None
    assert record.covered_turns == 40  # 50 - keep_recent(10)


def test_idempotent_second_tick_no_new_work(tmp_path: Path) -> None:
    """Second tick on the same session with no new turns: nothing
    qualifies; session is skipped."""
    _adapter, _t, cs, sink, task = _build(tmp_path, sessions={"s": 50})
    task()  # type: ignore[operator]
    task()  # type: ignore[operator]
    assert len(sink) == 2
    # First tick compacted; second tick skipped.
    assert sink[0].sessions_compacted == ["s"]
    assert sink[1].sessions_compacted == []
    assert sink[1].sessions_skipped == ["s"]
    # Only one compaction row in the store.
    record = cs.latest_for_session("s")
    assert record is not None
    assert record.covered_turns == 40


def test_session_under_threshold_is_skipped(tmp_path: Path) -> None:
    """A session with min_unprocessed_turns(20) + keep_recent(10) - 1 = 29
    turns sits just under threshold; tick skips it."""
    _adapter, _t, cs, sink, task = _build(tmp_path, sessions={"s": 29})
    task()  # type: ignore[operator]
    assert sink[0].sessions_compacted == []
    assert sink[0].sessions_skipped == ["s"]
    assert cs.latest_for_session("s") is None


def test_cold_session_outside_lookback_is_not_inspected(tmp_path: Path) -> None:
    """A session whose last_at is older than the lookback window is
    excluded from inspection entirely (not just skipped). The outcome
    record reflects this via `sessions_inspected`."""
    t = Transcript(tmp_path / "t.sqlite")
    cs = CompactionStore(tmp_path / "c.sqlite")
    _seed_session(t, session="hot", turns=50)
    # Backdate the cold session by manually crafting its created_at via
    # SQLite. Each appended row carries the now() default, so we patch
    # the rows after insert.
    _seed_session(t, session="cold", turns=50)
    old = datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    # Test reaches into the store's private connection deliberately —
    # there's no public API for backdating transcript rows.
    t._conn.execute("UPDATE transcript SET created_at = ? WHERE session = ?", (old, "cold"))
    t._conn.commit()
    adapter = _ScriptedAdapter(replies=["summary"])
    sink_records: list[CompactionTaskOutcome] = []
    task = build_compaction_task(
        adapter=adapter,
        transcript=t,
        compaction_store=cs,
        lookback_hours=24.0,
        keep_recent=10,
        min_unprocessed_turns=20,
        sink=sink_records.append,
    )
    task()
    out = sink_records[0]
    assert out.sessions_inspected == 1  # only "hot"
    assert out.sessions_compacted == ["hot"]


def test_model_bail_lands_in_skipped_not_errors(tmp_path: Path) -> None:
    """An adapter that returns '' is a 'model bailed' signal — the
    runner doesn't advance the watermark and `wrote=False`. The task
    treats that as a skip, not an error (errors are exceptions).

    list_sessions() returns newest-first; with two equally-fresh
    sessions, the first one inserted is the newer one — exact ordering
    is opaque to the test but each session lands in exactly one bucket."""
    _adapter, t, cs, _sink, _task = _build(tmp_path, sessions={"a": 50, "b": 50})
    bailing_adapter = _ScriptedAdapter(replies=["summary-0", ""])
    sink_records: list[CompactionTaskOutcome] = []
    task = build_compaction_task(
        adapter=bailing_adapter,
        transcript=t,
        compaction_store=cs,
        sink=sink_records.append,
    )
    task()
    out = sink_records[0]
    assert len(out.sessions_compacted) == 1
    assert len(out.sessions_skipped) == 1
    assert set(out.sessions_compacted) | set(out.sessions_skipped) == {"a", "b"}
    assert out.errors == []  # bail is not an error


def test_raising_adapter_lands_in_errors(tmp_path: Path) -> None:
    """An adapter that raises mid-tick puts the failing session into
    `errors` and lets the loop continue."""

    @dataclass
    class _RaisingAdapter:
        id: str = "raising"
        context_window: int = 8192

        def complete(
            self,
            messages: Iterable[ChatMessage],
            *,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> str:
            raise RuntimeError("model dead")

    t = Transcript(tmp_path / "t.sqlite")
    cs = CompactionStore(tmp_path / "c.sqlite")
    _seed_session(t, session="a", turns=50)
    _seed_session(t, session="b", turns=50)
    sink_records: list[CompactionTaskOutcome] = []
    # _RaisingAdapter duck-types ModelAdapter — id + context_window +
    # complete is the structural surface compaction needs.
    task = build_compaction_task(
        adapter=_RaisingAdapter(),
        transcript=t,
        compaction_store=cs,
        sink=sink_records.append,
    )
    task()
    out = sink_records[0]
    assert out.sessions_inspected == 2
    # Both sessions error; neither compacted.
    assert out.sessions_compacted == []
    error_sessions = {sid for sid, _ in out.errors}
    assert error_sessions == {"a", "b"}
    for _, msg in out.errors:
        assert "RuntimeError" in msg


def test_no_sink_runs_silently(tmp_path: Path) -> None:
    """sink=None is a no-op — task still runs to completion."""
    t = Transcript(tmp_path / "t.sqlite")
    cs = CompactionStore(tmp_path / "c.sqlite")
    _seed_session(t, session="s", turns=50)
    adapter = _ScriptedAdapter(replies=["summary"])
    task = build_compaction_task(
        adapter=adapter,
        transcript=t,
        compaction_store=cs,
        sink=None,
    )
    task()
    # Store advanced even without a sink — sink is observability, not
    # control flow.
    record = cs.latest_for_session("s")
    assert record is not None


def test_inspects_count_includes_all_recent_sessions_including_skipped(tmp_path: Path) -> None:
    """sessions_inspected is the count of recent sessions the task
    looked at, regardless of whether each one passed the threshold."""
    _adapter, _t, _cs, sink, task = _build(
        tmp_path,
        sessions={"big_a": 50, "big_b": 50, "small": 10},
    )
    task()  # type: ignore[operator]
    out = sink[0]
    assert out.sessions_inspected == 3
    assert set(out.sessions_compacted) == {"big_a", "big_b"}
    assert out.sessions_skipped == ["small"]
