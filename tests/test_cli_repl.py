"""Classic-REPL slash-handler + ContextMeter tests (harness-c1r).

Scope:
- `/clear` — ContextMeter.clear pins a cutoff; load_history then
  returns only rows appended after the cutoff, and a prior
  compaction summary no longer leaks back into the next turn.
- `handle_clear_slash` — prints a confirmation banner and calls
  through to ctx_meter.clear.
- auto-scribe at compaction (harness-0kw) — when memory + semantic
  stores are wired and `auto_scribe=True`, `maybe_compact` scribes
  unprocessed turns into memory *before* the summarizer folds them.

Stores are real SQLite files under tmp_path per the project test
convention — no mocking of the transcript layer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from harness.character import load_character
from harness.cli_repl import ContextMeter, handle_clear_slash
from harness.compaction import CompactionStore
from harness.model.adapter import ChatMessage
from harness.model.echo import EchoAdapter
from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore
from harness.store.transcript import Transcript

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON = REPO_ROOT / "character" / "airton"


def _seed_transcript(transcript: Transcript, session: str, *, pairs: int) -> int:
    """Append `pairs` user+assistant exchanges; return the last row id."""
    for i in range(pairs):
        transcript.append(
            session=session,
            channel="cli",
            speaker="mark",
            role="user",
            content=f"u{i}",
        )
        transcript.append(
            session=session,
            channel="cli",
            speaker="airton",
            role="assistant",
            content=f"a{i}",
        )
    tail = transcript.tail(session, limit=1)
    return tail[-1].id if tail else 0


def _ctx_meter(
    tmp_path: Path,
    *,
    with_compaction: bool = False,
    session: str = "test",
) -> tuple[ContextMeter, Transcript, CompactionStore | None]:
    db = tmp_path / "harness.sqlite"
    transcript = Transcript(db)
    compaction = CompactionStore(db) if with_compaction else None
    meter = ContextMeter(
        adapter=EchoAdapter(),
        character=load_character(AIRTON),
        transcript=transcript,
        compaction_store=compaction,
        session=session,
        console=Console(file=open("/dev/null", "w")),  # noqa: SIM115 — lifetime = test
    )
    return meter, transcript, compaction


def test_clear_pins_cutoff_to_current_tip(tmp_path: Path) -> None:
    """After clear(), load_history must return only rows appended
    after the moment of the clear. Rows already in the transcript
    stay in the DB but never surface to the model again."""
    meter, transcript, _ = _ctx_meter(tmp_path)
    pre_id = _seed_transcript(transcript, meter.session, pairs=3)

    meter.clear()

    assert meter.clear_after_id == pre_id
    summary, history = meter.load_history()
    assert summary is None
    assert history == []


def test_clear_lets_new_turns_rebuild_history(tmp_path: Path) -> None:
    """Post-clear, a new turn's appended rows must be the only thing
    load_history returns. Verifies the cutoff moves-with-new-rows
    semantics, not a permanent blackout."""
    meter, transcript, _ = _ctx_meter(tmp_path)
    _seed_transcript(transcript, meter.session, pairs=2)
    meter.clear()

    transcript.append(
        session=meter.session,
        channel="cli",
        speaker="mark",
        role="user",
        content="after-clear prompt",
    )
    transcript.append(
        session=meter.session,
        channel="cli",
        speaker="airton",
        role="assistant",
        content="after-clear reply",
    )

    _, history = meter.load_history()
    assert [m.role for m in history] == ["user", "assistant"]
    assert [m.content for m in history] == ["after-clear prompt", "after-clear reply"]


def test_clear_suppresses_compaction_summary(tmp_path: Path) -> None:
    """A compaction summary is normally prepended to history so the
    model remembers folded turns. /clear must override that too —
    otherwise the 'fresh start' wouldn't actually be fresh."""
    meter, transcript, compaction = _ctx_meter(tmp_path, with_compaction=True)
    assert compaction is not None  # for mypy
    last_id = _seed_transcript(transcript, meter.session, pairs=3)
    compaction.append(
        session_id=meter.session,
        summary="summary-of-folded-turns",
        up_to_turn_id=last_id,
        covered_turns=6,
        model_id="echo",
    )

    # Sanity: without clear, load_history returns the summary.
    summary, _ = meter.load_history()
    assert summary is not None
    assert "summary-of-folded-turns" in summary.content

    meter.clear()

    summary_after, history_after = meter.load_history()
    assert summary_after is None
    assert history_after == []


def test_clear_is_a_noop_on_empty_transcript(tmp_path: Path) -> None:
    """A brand-new session with nothing in the transcript is a valid
    /clear target — set the cutoff to 0 so future rows (ids start at
    1) still surface."""
    meter, transcript, _ = _ctx_meter(tmp_path)
    meter.clear()
    assert meter.clear_after_id == 0

    transcript.append(
        session=meter.session,
        channel="cli",
        speaker="mark",
        role="user",
        content="first prompt",
    )
    _, history = meter.load_history()
    assert [m.content for m in history] == ["first prompt"]


def test_handle_clear_slash_calls_meter_and_prints_banner(tmp_path: Path) -> None:
    """The REPL's /clear handler is a thin shim: delegate to the
    ContextMeter and emit a visual separator so the user sees the
    reset happened. Verified with a spy on clear() + a captured
    Console."""
    from io import StringIO

    buf = StringIO()
    console = Console(file=buf, force_terminal=False)
    meter = MagicMock(spec=ContextMeter)

    handle_clear_slash(meter, console)

    meter.clear.assert_called_once()
    assert "context cleared" in buf.getvalue()


@pytest.mark.parametrize("cmd", ["/clear", "/CLEAR", "/Clear"])
def test_clear_command_set_is_case_insensitive(cmd: str) -> None:
    """Classic REPL lowercases user_input before matching, so every
    case variant of /clear must be in the command set."""
    from harness.cli_classic import _CLEAR_COMMANDS

    assert cmd.lower() in _CLEAR_COMMANDS


# ---------- auto-scribe at compaction (harness-0kw) ----------


@dataclass
class _FakeEmbedder:
    id: str = "fake"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            out.append(v / norm if norm > 0 else v)
        return np.stack(out)


@dataclass
class _ScriptedAdapter:
    """Adapter with a queued reply list. `complete` pops; calls the
    scribe and then the compaction summarizer in sequence when used
    from `maybe_compact`."""

    replies: list[str]
    id: str = "scripted"
    context_window: int = 4096
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


def _build_auto_scribe_meter(
    tmp_path: Path,
    *,
    adapter_replies: list[str],
    auto_scribe: bool = True,
    with_stores: bool = True,
) -> tuple[ContextMeter, Transcript, EpisodicStore | None, SemanticStore | None]:
    """Wire a ContextMeter with a scripted adapter + real stores so
    scribe + compact run for-real against SQLite. Stores are optional
    so tests can verify the no-stores path too."""
    db = tmp_path / "harness.sqlite"
    transcript = Transcript(db)
    compaction = CompactionStore(db)
    episodic: EpisodicStore | None = None
    semantic: SemanticStore | None = None
    if with_stores:
        episodic = EpisodicStore(db, embedder=_FakeEmbedder())
        semantic = SemanticStore(db, embedder=_FakeEmbedder())
    meter = ContextMeter(
        adapter=_ScriptedAdapter(replies=adapter_replies),
        character=load_character(AIRTON),
        transcript=transcript,
        compaction_store=compaction,
        session="auto-scribe",
        console=Console(file=open("/dev/null", "w")),  # noqa: SIM115 — test lifetime
        memory_store=episodic,
        semantic_store=semantic,
        scribe_user_id="mark",
        scribe_lock_dir=tmp_path / "locks",
        auto_scribe=auto_scribe,
    )
    return meter, transcript, episodic, semantic


def test_maybe_compact_scribes_before_folding(tmp_path: Path) -> None:
    """When auto-scribe is on and both stores are wired, the turns
    that are about to be compacted must land in episodic memory
    first. Without this hook the long-session failure mode is: a
    398-turn session produces zero working-tier memories and the
    agent answers 'what did we talk about' with 'I don't know'."""
    scribe_reply = (
        '{"episodic":[{"title":"chat with mark","body":"we discussed '
        'the harness internals and compaction.",'
        '"principle":null,"tags":["harness","compaction"]}],'
        '"semantic":[{"subject":"mark","predicate":"works_on",'
        '"object":"harness","confidence":0.9}]}'
    )
    # 10 pairs = 20 turns = exactly one scribe window, then 1 summarize call.
    meter, transcript, episodic, semantic = _build_auto_scribe_meter(
        tmp_path,
        adapter_replies=[scribe_reply, "folded summary"],
    )
    assert episodic is not None
    assert semantic is not None

    _seed_transcript(transcript, meter.session, pairs=10)

    # Force compaction regardless of measured tokens by stubbing.
    class _FakeSpinner:
        started = 0
        stopped = 0

        def start(self) -> None:
            self.started += 1

        def stop(self) -> None:
            self.stopped += 1

    meter.measure = lambda: 100_000  # type: ignore[method-assign]
    spinner = _FakeSpinner()
    meter.maybe_compact(
        compact_at=0.8,
        compact_keep_recent=5,
        thinking=spinner,
        ab_adapter=None,
    )

    ep_rows = episodic.all()
    sem_rows = semantic.all()
    # At least one working-tier memory row written from the pre-compaction
    # scribe pass (seed-tier rows from character load are also present).
    assert any(r.tier == "working" for r in ep_rows)
    assert any(r.tier == "working" for r in sem_rows)
    # Compaction still advanced (15 of 20 folded, keep_recent=5).
    latest = meter.compaction_store.latest_for_session(meter.session)  # type: ignore[union-attr]
    assert latest is not None
    assert latest.covered_turns == 15
    # Both passes ran — one scribe window + one summarize.
    assert spinner.started == 1
    assert spinner.stopped == 1


def test_maybe_compact_skips_scribe_when_disabled(tmp_path: Path) -> None:
    """auto_scribe=False is the escape hatch for the echo/test paths.
    Compaction still advances; no scribe rows land."""
    meter, transcript, episodic, semantic = _build_auto_scribe_meter(
        tmp_path,
        adapter_replies=["folded summary"],
        auto_scribe=False,
    )
    assert episodic is not None
    assert semantic is not None
    _seed_transcript(transcript, meter.session, pairs=10)

    class _NoopSpinner:
        def start(self) -> None: ...
        def stop(self) -> None: ...

    meter.measure = lambda: 100_000  # type: ignore[method-assign]
    meter.maybe_compact(
        compact_at=0.8,
        compact_keep_recent=5,
        thinking=_NoopSpinner(),
        ab_adapter=None,
    )
    assert not any(r.tier == "working" for r in episodic.all())
    assert not any(r.tier == "working" for r in semantic.all())


def test_maybe_compact_skips_scribe_when_stores_missing(tmp_path: Path) -> None:
    """Memory + semantic stores are optional. When either is absent,
    scribe silently skips and compaction proceeds unchanged —
    critical so the echo-adapter / memories=0 dry-run path stays
    working after this change."""
    meter, transcript, _, _ = _build_auto_scribe_meter(
        tmp_path,
        adapter_replies=["folded summary"],
        with_stores=False,
    )
    _seed_transcript(transcript, meter.session, pairs=10)

    class _NoopSpinner:
        def start(self) -> None: ...
        def stop(self) -> None: ...

    meter.measure = lambda: 100_000  # type: ignore[method-assign]
    # No exception even though scribe would need stores to run.
    meter.maybe_compact(
        compact_at=0.8,
        compact_keep_recent=5,
        thinking=_NoopSpinner(),
        ab_adapter=None,
    )
    latest = meter.compaction_store.latest_for_session(meter.session)  # type: ignore[union-attr]
    assert latest is not None
    assert latest.covered_turns == 15
