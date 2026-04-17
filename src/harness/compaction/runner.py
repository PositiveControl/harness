from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.compaction.summarizer import summarize_turns

if TYPE_CHECKING:
    from harness.compaction.store import CompactionRecord, CompactionStore
    from harness.model.adapter import ModelAdapter
    from harness.store.transcript import Transcript


@dataclass(frozen=True)
class CompactionOutcome:
    """Returned from run_compaction so the caller can log / UI the
    result without re-reading the store. `wrote=False` means nothing
    qualified for compaction this call (not enough turns past the
    existing watermark, or `keep_recent` covers everything)."""

    wrote: bool
    covered_turns: int
    new_up_to_turn_id: int
    summary: str | None = None


def should_compact(*, used_tokens: int, context_window: int, threshold_pct: float) -> bool:
    """Trigger predicate for the chat loop. Compaction fires when
    `used_tokens >= threshold_pct * context_window`. A threshold of 0
    disables compaction; >=1 disables compaction too (the meter never
    exceeds capacity by enough to matter before the model itself fails)."""
    if context_window <= 0:
        return False
    if threshold_pct <= 0.0 or threshold_pct >= 1.0:
        return False
    return used_tokens >= int(context_window * threshold_pct)


def run_compaction(
    adapter: ModelAdapter,
    transcript: Transcript,
    compaction_store: CompactionStore,
    *,
    session_id: str,
    keep_recent: int = 10,
    max_summary_tokens: int = 1024,
) -> CompactionOutcome:
    """Fold everything in `session_id` between the last compaction
    pointer and the Nth-most-recent turn into a single summary. The
    most-recent `keep_recent` turns stay verbatim — they carry the
    conversational lead-in the model needs to respond well to the
    current question.

    Safe to call repeatedly; the pointer advances monotonically so the
    same turn is never summarized twice. If nothing qualifies this
    call, returns CompactionOutcome(wrote=False)."""
    prior: CompactionRecord | None = compaction_store.latest_for_session(session_id)
    pointer = prior.up_to_turn_id if prior is not None else 0

    fresh = transcript.fetch_after(session_id, after_id=pointer)
    if len(fresh) <= keep_recent:
        return CompactionOutcome(wrote=False, covered_turns=0, new_up_to_turn_id=pointer)

    to_compact = fresh[:-keep_recent] if keep_recent > 0 else list(fresh)
    if not to_compact:
        return CompactionOutcome(wrote=False, covered_turns=0, new_up_to_turn_id=pointer)

    new_summary = summarize_turns(
        adapter,
        to_compact,
        prior_summary=prior.summary if prior is not None else None,
        max_tokens=max_summary_tokens,
    )
    if not new_summary:
        # Model bailed — don't advance the pointer, we'll retry next turn.
        return CompactionOutcome(wrote=False, covered_turns=0, new_up_to_turn_id=pointer)

    new_up_to = to_compact[-1].id
    compaction_store.append(
        session_id=session_id,
        summary=new_summary,
        up_to_turn_id=new_up_to,
        covered_turns=len(to_compact),
        model_id=adapter.id,
    )
    return CompactionOutcome(
        wrote=True,
        covered_turns=len(to_compact),
        new_up_to_turn_id=new_up_to,
        summary=new_summary,
    )
