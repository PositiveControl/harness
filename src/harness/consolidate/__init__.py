"""Consolidator — periodic pass that promotes working-tier memories to
the consolidated tier, merges near-duplicates, and retires superseded
facts. Triggered manually via `harness memory consolidate`; automated
in Phase 5."""

from harness.consolidate.consolidator import (
    ConsolidationSummary,
    cluster_by_similarity,
    consolidate_episodic,
    consolidate_semantic,
    run_consolidation,
)

__all__ = [
    "ConsolidationSummary",
    "cluster_by_similarity",
    "consolidate_episodic",
    "consolidate_semantic",
    "run_consolidation",
]
