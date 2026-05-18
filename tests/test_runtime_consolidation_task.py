"""Tests for the periodic-consolidation heartbeat task — harness-srus."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.runtime.tasks.consolidation import (
    ConsolidationTaskOutcome,
    build_consolidation_task,
)
from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore


@dataclass
class _ControlledEmbedder:
    """Same shape as the consolidator-test embedder: table-driven so the
    test can dictate similarity. The episodic store prepends a context
    header to embed-text; `_lookup_key` strips it so fixtures match."""

    table: dict[str, np.ndarray]
    id: str = "controlled"
    dimension: int = 4

    @staticmethod
    def _lookup_key(text: str) -> str:
        if text.startswith("["):
            end = text.find("]")
            if end != -1 and text[end + 1 : end + 3] == "\n\n":
                return text[end + 3 :]
        return text

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for text in texts:
            key = self._lookup_key(text)
            vec = self.table.get(key)
            if vec is None:
                h = sum(ord(c) for c in key.lower())
                v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
                norm = float(np.linalg.norm(v))
                vec = v / norm if norm > 0 else v
            out.append(vec.astype(np.float32))
        return np.stack(out)


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 0 else arr


def _stores_with_clusterable_episodic(
    tmp_path: Path,
) -> tuple[EpisodicStore, SemanticStore]:
    """Seed two near-duplicate episodic working rows + one isolate.
    A run_consolidation pass should merge the first two and leave the
    third alone, matching the upstream consolidator test fixture."""
    dupe = _unit([1.0, 0.0, 0.0, 0.0])
    near = _unit([0.99, 0.05, 0.0, 0.0])
    lone = _unit([0.0, 1.0, 0.0, 0.0])
    embedder = _ControlledEmbedder(
        table={
            "alpha-1\n\ndoes foo\n\nbody one": dupe,
            "alpha-2\n\ndoes foo\n\nbody two": near,
            "beta\n\ndoes bar\n\nbody three": lone,
        }
    )
    ep = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    sem = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    ep.ingest(
        external_id="alpha-1",
        title="alpha-1",
        body="body one",
        principle="does foo",
        tier="working",
        source="test",
    )
    ep.ingest(
        external_id="alpha-2",
        title="alpha-2",
        body="body two",
        principle="does foo",
        tier="working",
        source="test",
    )
    ep.ingest(
        external_id="beta",
        title="beta",
        body="body three",
        principle="does bar",
        tier="working",
        source="test",
    )
    return ep, sem


def test_runs_consolidator_when_threshold_met(tmp_path: Path) -> None:
    """Three working rows (default threshold 5? — we lower to 2): the
    task fires the consolidator, sink records ran=True with a summary
    whose episodic_clusters_merged == 1."""
    ep, sem = _stores_with_clusterable_episodic(tmp_path)
    sink_records: list[ConsolidationTaskOutcome] = []
    task = build_consolidation_task(
        episodic_store=ep,
        semantic_store=sem,
        min_working_records=2,
        episodic_threshold=0.90,
        sink=sink_records.append,
    )
    task()
    assert len(sink_records) == 1
    out = sink_records[0]
    assert out.ran is True
    assert out.skip_reason is None
    assert out.error is None
    assert out.summary is not None
    assert out.summary.episodic_clusters_merged == 1
    assert out.summary.episodic_superseded == 2


def test_skips_when_working_tier_below_threshold(tmp_path: Path) -> None:
    """Fewer working rows than min_working_records: task short-circuits
    with skip_reason; consolidator is never called."""
    ep, sem = _stores_with_clusterable_episodic(tmp_path)
    sink_records: list[ConsolidationTaskOutcome] = []
    task = build_consolidation_task(
        episodic_store=ep,
        semantic_store=sem,
        min_working_records=10,  # 3 working rows < threshold
        sink=sink_records.append,
    )
    task()
    out = sink_records[0]
    assert out.ran is False
    assert out.skip_reason is not None
    assert "3 working-tier" in out.skip_reason
    assert out.summary is None
    assert out.error is None


def test_idempotent_second_tick_after_consolidation_has_zero_merges(
    tmp_path: Path,
) -> None:
    """After a successful consolidation the working-tier rows are now
    superseded (effectively gone from `tier='working'`). A second tick
    finds nothing to cluster and reports ran=True with 0 merges (or
    skips if the remaining count is below threshold)."""
    ep, sem = _stores_with_clusterable_episodic(tmp_path)
    sink_records: list[ConsolidationTaskOutcome] = []
    task = build_consolidation_task(
        episodic_store=ep,
        semantic_store=sem,
        min_working_records=1,  # always-fire so we see the idempotency directly
        episodic_threshold=0.90,
        sink=sink_records.append,
    )
    task()  # first pass merges
    task()  # second pass: no working clusters remain
    assert sink_records[0].summary is not None
    assert sink_records[0].summary.episodic_clusters_merged == 1
    assert sink_records[1].summary is not None
    # Only the isolate "beta" remains in working; clusters_merged == 0.
    assert sink_records[1].summary.episodic_clusters_merged == 0
    assert sink_records[1].summary.episodic_superseded == 0


def test_raising_store_lands_in_error(tmp_path: Path) -> None:
    """If the episodic store throws on `.all('working')` (corrupt
    sqlite, bad embedder), the task captures the exception into
    outcome.error and lets the heartbeat continue."""

    class _RaisingStore:
        def all(self, tier: str | None = None, *, include_superseded: bool = False) -> list[object]:
            raise RuntimeError("store dead")

    sink_records: list[ConsolidationTaskOutcome] = []
    task = build_consolidation_task(
        episodic_store=_RaisingStore(),  # type: ignore[arg-type]
        semantic_store=None,  # type: ignore[arg-type]
        sink=sink_records.append,
    )
    task()
    out = sink_records[0]
    assert out.ran is False
    assert out.error is not None
    assert "RuntimeError" in out.error
    assert out.summary is None
    assert out.skip_reason is None


def test_raising_consolidator_lands_in_error(tmp_path: Path) -> None:
    """An exception inside run_consolidation (e.g. embedder failure)
    is caught and reported via outcome.error after the working-rows
    check has already passed."""
    ep, sem = _stores_with_clusterable_episodic(tmp_path)

    # Break the fetch_embedding path so run_consolidation crashes after
    # the working-rows check succeeds.
    def _boom(_id: int) -> np.ndarray:
        raise RuntimeError("embed dead")

    ep.fetch_embedding = _boom  # type: ignore[assignment, method-assign]

    sink_records: list[ConsolidationTaskOutcome] = []
    task = build_consolidation_task(
        episodic_store=ep,
        semantic_store=sem,
        min_working_records=2,
        sink=sink_records.append,
    )
    task()
    out = sink_records[0]
    assert out.ran is False
    assert out.error is not None
    assert "embed dead" in out.error


def test_no_sink_runs_silently(tmp_path: Path) -> None:
    """sink=None: task completes without error, the store still
    consolidates."""
    ep, sem = _stores_with_clusterable_episodic(tmp_path)
    task = build_consolidation_task(
        episodic_store=ep,
        semantic_store=sem,
        min_working_records=2,
        episodic_threshold=0.90,
        sink=None,
    )
    task()
    # The alpha cluster should now be superseded.
    active = ep.all()
    assert any(r.tier == "consolidated" for r in active)


def test_preserves_per_user_partitioning(tmp_path: Path) -> None:
    """Two near-duplicate rows owned by different users must NOT merge.
    The task delegates partition logic to run_consolidation; this is a
    regression guard that the wrapping doesn't break that invariant."""
    a_vec = _unit([1.0, 0.0, 0.0, 0.0])
    b_vec = _unit([0.99, 0.05, 0.0, 0.0])
    embedder = _ControlledEmbedder(
        table={
            "u1-row\n\nprinciple p\n\nbody u1": a_vec,
            "u2-row\n\nprinciple p\n\nbody u2": b_vec,
        }
    )
    ep = EpisodicStore(tmp_path / "h.sqlite", embedder=embedder)
    sem = SemanticStore(tmp_path / "h.sqlite", embedder=embedder)
    ep.ingest(
        external_id="u1-row",
        title="u1-row",
        body="body u1",
        principle="principle p",
        tier="working",
        source="test",
        user_id="user_a",
    )
    ep.ingest(
        external_id="u2-row",
        title="u2-row",
        body="body u2",
        principle="principle p",
        tier="working",
        source="test",
        user_id="user_b",
    )
    sink_records: list[ConsolidationTaskOutcome] = []
    task = build_consolidation_task(
        episodic_store=ep,
        semantic_store=sem,
        min_working_records=1,
        episodic_threshold=0.90,
        sink=sink_records.append,
    )
    task()
    out = sink_records[0]
    assert out.ran is True
    assert out.summary is not None
    # Different users -> separate partitions -> no merge possible.
    assert out.summary.episodic_clusters_merged == 0
    assert out.summary.episodic_superseded == 0
