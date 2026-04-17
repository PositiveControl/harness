"""Compaction — summarize old transcript turns when the context window
fills up. Acts on the chat loop only; the canonical transcript stays
untouched. Compaction summaries live in their own table, keyed by
session, with a pointer to the last turn they cover. The chat loader
reassembles each turn as (summary-system-message, turns-after-pointer)."""

from harness.compaction.runner import (
    CompactionOutcome,
    run_compaction,
    should_compact,
)
from harness.compaction.store import CompactionRecord, CompactionStore
from harness.compaction.summarizer import summarize_turns

__all__ = [
    "CompactionOutcome",
    "CompactionRecord",
    "CompactionStore",
    "run_compaction",
    "should_compact",
    "summarize_turns",
]
