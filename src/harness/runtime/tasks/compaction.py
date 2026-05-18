"""Periodic-compaction heartbeat task — harness-klwg.

Wraps `harness.compaction.run_compaction` for use inside a
`runtime.Heartbeat`. Each tick scans recently-active sessions and runs
compaction on those whose unprocessed-turn count exceeds the
threshold. Skips sessions older than `lookback_hours` (no point folding
a session that's already cold) and sessions that don't have enough
fresh turns past the existing watermark.

Idempotent by virtue of `compaction.run_compaction`: the watermark
advances monotonically; a re-run on the same session finds nothing
new past the pointer and returns `wrote=False`.

Error isolation: one session that crashes (corrupt store, adapter
failure) doesn't take down the tick — other sessions still get
inspected. The error message is captured in the outcome record so
the daemon can surface it via the observability bead (m64i).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from harness.compaction import run_compaction

if TYPE_CHECKING:
    from harness.compaction.store import CompactionStore
    from harness.model.adapter import ModelAdapter
    from harness.store.transcript import Transcript


@dataclass(frozen=True)
class CompactionTaskOutcome:
    """One tick's audit record. Surfaced via the optional `sink`
    callback so the daemon (or a test) can log / inspect activity
    without poking at the compaction store directly.

    `sessions_compacted` and `sessions_skipped` are mutually exclusive
    — a session lands in exactly one bucket per tick. `errors` is
    additive: a session that errored is in `errors` and absent from
    the other two lists."""

    tick_at: datetime
    sessions_inspected: int
    sessions_compacted: list[str]
    sessions_skipped: list[str]
    errors: list[tuple[str, str]]


def build_compaction_task(
    *,
    adapter: ModelAdapter,
    transcript: Transcript,
    compaction_store: CompactionStore,
    lookback_hours: float = 24.0,
    keep_recent: int = 10,
    min_unprocessed_turns: int = 20,
    sink: Callable[[CompactionTaskOutcome], None] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Callable[[], None]:
    """Return a zero-arg callable for `Heartbeat.register()`.

    Args:
        adapter: model used to summarize. Loaded once at daemon startup
            and reused across ticks — instantiating an MLX adapter mid-
            tick would cost multiple seconds.
        transcript: source of session enumeration + raw turns.
        compaction_store: watermark + summary destination.
        lookback_hours: skip sessions whose last_at is older than now -
            this. Defaults to 24 h — a session that's been idle for a
            day either compacted itself during chat or is dormant.
        keep_recent: pass-through to `run_compaction`; how many recent
            turns to leave verbatim.
        min_unprocessed_turns: don't fire compaction unless this many
            new turns sit past the watermark. Avoids running the model
            on trivial deltas.
        sink: optional callback that receives a `CompactionTaskOutcome`
            after each tick. Daemons typically point it at a console
            logger; tests typically point it at a list-append.
        clock: injected for tests — defaults to `datetime.now(UTC)`.
    """

    def task() -> None:
        now = clock()
        cutoff = now - timedelta(hours=lookback_hours)
        recent = [s for s in transcript.list_sessions() if s.last_at >= cutoff]
        compacted: list[str] = []
        skipped: list[str] = []
        errors: list[tuple[str, str]] = []

        for summary in recent:
            try:
                prior = compaction_store.latest_for_session(summary.session)
                pointer = prior.up_to_turn_id if prior is not None else 0
                fresh = transcript.fetch_after(summary.session, after_id=pointer)
                if len(fresh) < min_unprocessed_turns + keep_recent:
                    skipped.append(summary.session)
                    continue
                outcome = run_compaction(
                    adapter,
                    transcript,
                    compaction_store,
                    session_id=summary.session,
                    keep_recent=keep_recent,
                )
                if outcome.wrote:
                    compacted.append(summary.session)
                else:
                    skipped.append(summary.session)
            except Exception as exc:
                # Heartbeat task must not crash the loop — every exception
                # in a per-session pass gets captured into the outcome,
                # then the loop moves to the next session.
                errors.append((summary.session, repr(exc)))

        if sink is not None:
            sink(
                CompactionTaskOutcome(
                    tick_at=now,
                    sessions_inspected=len(recent),
                    sessions_compacted=compacted,
                    sessions_skipped=skipped,
                    errors=errors,
                )
            )

    return task
