"""Periodic-consolidation heartbeat task — harness-srus.

Wraps `harness.consolidate.run_consolidation` for use inside a
`runtime.Heartbeat`. Each tick (typical interval: 1 hour) the task:

  * Checks the working-tier episodic-record count — if below
    `min_working_records`, skip the tick to avoid an empty scan.
  * Runs the full consolidator pass (episodic clustering + semantic
    fact grouping). Per-user partitions are preserved by the
    underlying consolidator — shared rows never merge with private
    rows; one user's relationship memory never merges with another's.
  * Captures the ConsolidationSummary into a per-tick audit record
    surfaced through an optional sink callback.

Watermark semantics: the consolidator only ever inspects `tier="working"`.
A successfully consolidated row leaves the working tier (gets promoted
to consolidated + the originals marked superseded), so a re-tick with
no new working rows naturally produces zero merges. No explicit
high-water mark needed — the tier transition is the watermark.

Error isolation: a raising consolidator (corrupt store, embedder
failure) captures the exception into outcome.error and lets the
heartbeat continue. The pass is atomic in practice — `run_consolidation`
writes each consolidated record + supersede mark in its own statement,
so a mid-pass crash leaves a coherent partial state that the next
tick can pick up where this one left off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from harness.consolidate import run_consolidation

if TYPE_CHECKING:
    from harness.consolidate import ConsolidationSummary
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore


@dataclass(frozen=True)
class ConsolidationTaskOutcome:
    """One tick's audit record.

    `ran=True` means the consolidator's `run_consolidation` was called
    and produced a `summary` (which may itself report zero merges if
    nothing clustered). `ran=False` means the task short-circuited
    before calling the consolidator — either because the threshold
    wasn't met (`skip_reason` set) or because of an early exception
    (`error` set).

    `ran=False` + `skip_reason is None` + `error is not None`: a fatal
    error before the pass could be attempted.
    """

    tick_at: datetime
    ran: bool
    summary: ConsolidationSummary | None
    skip_reason: str | None = None
    error: str | None = None


def build_consolidation_task(
    *,
    episodic_store: EpisodicStore,
    semantic_store: SemanticStore,
    min_working_records: int = 5,
    episodic_threshold: float = 0.80,
    sink: Callable[[ConsolidationTaskOutcome], None] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Callable[[], None]:
    """Return a zero-arg callable for `Heartbeat.register()`.

    Args:
        episodic_store: working-tier source for clustering + promotion.
        semantic_store: working-tier source for fact grouping +
            promotion.
        min_working_records: don't fire the consolidator unless the
            working tier has at least this many episodic rows. Default
            5 — clustering a single row is wasted work; the consolidator
            needs 2+ in a partition to merge, and shared/per-user
            partitions split further. 5 is a small safety margin
            (operator can tune via the daemon CLI flag).
        episodic_threshold: pass-through to the consolidator's
            similarity threshold. Default 0.80 matches the manual
            `harness memory consolidate` CLI default.
        sink: optional callback receiving a ConsolidationTaskOutcome
            after each tick. Daemons typically log to console; tests
            typically append to a list.
        clock: injected for tests — defaults to `datetime.now(UTC)`.
    """

    def task() -> None:
        now = clock()
        try:
            working = episodic_store.all(tier="working")
        except Exception as exc:
            if sink is not None:
                sink(
                    ConsolidationTaskOutcome(
                        tick_at=now,
                        ran=False,
                        summary=None,
                        skip_reason=None,
                        error=repr(exc),
                    )
                )
            return

        if len(working) < min_working_records:
            if sink is not None:
                sink(
                    ConsolidationTaskOutcome(
                        tick_at=now,
                        ran=False,
                        summary=None,
                        skip_reason=(
                            f"only {len(working)} working-tier episodic "
                            f"record(s); threshold {min_working_records}"
                        ),
                    )
                )
            return

        try:
            summary = run_consolidation(
                episodic_store,
                semantic_store,
                episodic_threshold=episodic_threshold,
            )
        except Exception as exc:
            if sink is not None:
                sink(
                    ConsolidationTaskOutcome(
                        tick_at=now,
                        ran=False,
                        summary=None,
                        skip_reason=None,
                        error=repr(exc),
                    )
                )
            return

        if sink is not None:
            sink(
                ConsolidationTaskOutcome(
                    tick_at=now,
                    ran=True,
                    summary=summary,
                )
            )

    return task
