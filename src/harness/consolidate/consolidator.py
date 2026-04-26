from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from harness.store.episodic import EpisodicRecord, EpisodicStore
    from harness.store.semantic import SemanticFact, SemanticStore


@dataclass
class ConsolidationSummary:
    episodic_considered: int = 0
    episodic_clusters_merged: int = 0
    episodic_promoted: int = 0
    episodic_superseded: int = 0
    semantic_considered: int = 0
    semantic_groups_merged: int = 0
    semantic_promoted: int = 0
    semantic_superseded: int = 0
    notes: list[str] = field(default_factory=list)


# -------- Episodic clustering --------


def cluster_by_similarity(
    ids: list[int],
    embeddings: list[np.ndarray],
    *,
    threshold: float,
) -> list[list[int]]:
    """Single-link transitive-closure clustering: ids with pairwise
    cosine >= `threshold` land in the same cluster. O(N^2) — fine up to
    hundreds of records. Swap in proper clustering when that hurts."""
    n = len(ids)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i, record_id in enumerate(ids):
        clusters[find(i)].append(record_id)
    return list(clusters.values())


def consolidate_episodic(
    store: EpisodicStore,
    *,
    threshold: float = 0.80,
) -> tuple[int, int, int]:
    """Promote working-tier episodic records to consolidated. Clusters
    near-duplicates via embedding similarity; for each cluster of 2+,
    the most recent record is picked as the representative and a new
    consolidated copy is written; originals are marked superseded.

    Partitions by `user_id` before clustering — records belonging to
    different users (or shared vs private) are NEVER merged. The
    consolidated record inherits the partition's user_id, so shared
    stays shared and each user's relationship memory stays their own.

    Returns (considered, clusters_merged, superseded). A cluster with
    only one member is left alone (not enough signal to consolidate)."""
    working = store.all(tier="working")
    if len(working) < 2:
        return len(working), 0, 0

    # Partition by user scope. `None` (shared) is its own bucket.
    partitions: dict[str | None, list[EpisodicRecord]] = defaultdict(list)
    for rec in working:
        partitions[rec.user_id].append(rec)

    superseded = 0
    merged = 0
    for user_id, records in partitions.items():
        if len(records) < 2:
            continue
        ids = [r.id for r in records]
        records_by_id: dict[int, EpisodicRecord] = {r.id: r for r in records}
        embeddings = [store.fetch_embedding(r.id) for r in records]
        clusters = cluster_by_similarity(ids, embeddings, threshold=threshold)
        for cluster_ids in clusters:
            if len(cluster_ids) < 2:
                continue
            cluster_records = [records_by_id[i] for i in cluster_ids]
            # Pick the most recent as the representative. Recency is a
            # rough proxy for 'had the latest thinking'; if that turns
            # out wrong we can switch to longest principle or an LLM
            # judge.
            rep = max(cluster_records, key=lambda r: r.created_at)
            consolidated = store.ingest(
                external_id=None,
                title=rep.title,
                body=rep.body,
                principle=rep.principle,
                tags=tuple({*rep.tags, "consolidated"}),
                tier="consolidated",
                source=f"consolidator:merged={','.join(str(i) for i in sorted(cluster_ids))}",
                # Inherit the most-recent-session label across the
                # cluster (rep is `max(created_at)` so rep.session_id
                # picks the freshest contributor's session). Lets
                # `--memory-scope=current-session` retrieval still
                # surface a consolidated row when the user is back
                # in the session that produced the dominant entry.
                # Known limitation: a cluster spanning N sessions
                # collapses to one label — tracked in the C plan
                # as the open-questions item on multi-session arrays
                # (harness-w3mo).
                session_id=rep.session_id,
                user_id=user_id,
            )
            for cid in cluster_ids:
                store.mark_superseded(cid, by=consolidated.id)
            superseded += len(cluster_ids)
            merged += 1
    return len(working), merged, superseded


# -------- Semantic consolidation --------


def _fact_key(fact: SemanticFact) -> tuple[str | None, str, str]:
    """Group semantic facts by (user_id, subject, predicate), with the
    (subject, predicate) parts case-folded. Including user_id in the
    key means shared facts (user_id IS NULL) and per-user facts
    are never merged with each other, and one user's relationship
    memory is never merged with another's. Two facts sharing a key are
    candidates for consolidation — take the highest-confidence object,
    supersede the rest."""
    return (
        fact.user_id,
        fact.subject.strip().lower(),
        fact.predicate.strip().lower(),
    )


def consolidate_semantic(store: SemanticStore) -> tuple[int, int, int]:
    """Promote working-tier semantic facts to consolidated. Facts are
    grouped by (user_id, subject, predicate). If a group has 2+ members,
    the one with the highest confidence (tie-break: most recent) is
    picked as the representative and promoted; the rest are marked
    superseded. The consolidated fact inherits the group's user_id, so
    shared facts stay shared and each user's relationship memory stays
    their own.

    This does NOT handle the case where different objects under the
    same (subject, predicate) represent a real update — that needs
    temporal reasoning the MVP doesn't do yet."""
    working = store.all(tier="working")
    if len(working) < 2:
        return len(working), 0, 0

    groups: dict[tuple[str | None, str, str], list[SemanticFact]] = defaultdict(list)
    for fact in working:
        groups[_fact_key(fact)].append(fact)

    superseded = 0
    merged = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        rep = max(members, key=lambda f: (f.confidence, f.created_at))
        member_ids = ",".join(str(f.id) for f in sorted(members, key=lambda x: x.id))
        consolidated = store.add(
            subject=rep.subject,
            predicate=rep.predicate,
            object=rep.object,
            confidence=rep.confidence,
            source=f"consolidator:merged={member_ids}",
            tier="consolidated",
            # Inherit rep's session_id (rep is max(confidence, created_at))
            # for the same reason as the episodic consolidator above —
            # let `--memory-scope=current-session` still surface
            # consolidated facts for the session that produced the
            # dominant member (harness-w3mo).
            session_id=rep.session_id,
            user_id=rep.user_id,
        )
        for fact in members:
            store.mark_superseded(fact.id, by=consolidated.id)
        superseded += len(members)
        merged += 1
    return len(working), merged, superseded


# -------- Top-level runner --------


def run_consolidation(
    episodic_store: EpisodicStore,
    semantic_store: SemanticStore,
    *,
    episodic_threshold: float = 0.80,
) -> ConsolidationSummary:
    """Run both passes, return a summary. Safe to run repeatedly —
    consolidation is idempotent in the sense that consolidated tier is
    ignored (only working is processed)."""
    summary = ConsolidationSummary()

    ep_considered, ep_merged, ep_superseded = consolidate_episodic(
        episodic_store, threshold=episodic_threshold
    )
    summary.episodic_considered = ep_considered
    summary.episodic_clusters_merged = ep_merged
    summary.episodic_promoted = ep_merged  # one promotion per cluster
    summary.episodic_superseded = ep_superseded

    sem_considered, sem_merged, sem_superseded = consolidate_semantic(semantic_store)
    summary.semantic_considered = sem_considered
    summary.semantic_groups_merged = sem_merged
    summary.semantic_promoted = sem_merged
    summary.semantic_superseded = sem_superseded

    return summary
