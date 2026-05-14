"""Generic retrieval-shape eval (harness-kj2a / Phase 0 of the data-
retrieval-primitives spike).

Decouples "is the expected record in the top-K" from any specific
corpus type or store implementation. The eval here only knows three
things: a sequence of `ShapeCase` (query + expected record ids), a
`ShapeSearchFn` the caller supplies, and a depth `k`. Anything that
can return a ranked list of `(record_id, score)` for a query string
plugs in.

Why generic instead of reusing `evals/atc_retrieval.py`: that module
extracts ATC anchors out of `principle` strings (§N-N-N regex). Prose
and tabular corpora don't have anchors — the record id IS the answer.
By moving the "what counts as a hit" decision to the caller (it picks
which `record_id` to emit per hit), this eval handles every shape we
care about for Phases 0-3 without growing per-shape branches.

Phase 0 wiring: `scripts/bench_retrieval_shape.py` registers one
SearchFn per (corpus, retriever) pair. For the baseline run there's
exactly one retriever ("hybrid" via EpisodicStore.search). Later
phases plug in tree / table / contract retrievers without touching
this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeHit:
    """One returned record from a SearchFn.

    `record_id` is whatever stable identifier the corpus uses:
    `EpisodicRecord.external_id` for episodic-backed corpora, a row
    primary key like `orders:12847` for tabular, a section path like
    `jo71106.5:5-5-4` for tree-shaped docs. The eval doesn't care —
    it only checks set membership against `ShapeCase.expected_record_ids`.
    """

    record_id: str
    score: float


# (query, k) → ranked hits (most relevant first, length <= k).
ShapeSearchFn = Callable[[str, int], Sequence[ShapeHit]]


@dataclass(frozen=True)
class ShapeCase:
    """One fixture case. `expected_record_ids` is a tuple of acceptable
    answers — any of them appearing in the top-K counts as a hit. This
    mirrors the alternates-list pattern in `atc_eval.yaml` without
    embedding ATC-specific anchor parsing here."""

    id: str
    query: str
    expected_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class ShapeCaseResult:
    """Outcome of one scored case. `hit_record_ids` is the ranked list
    of returned ids (top-K, in order) — kept on the result so JSON
    output can show exactly what the retriever saw, which is the
    artifact most useful for diagnosing why a case missed."""

    id: str
    query: str
    expected_record_ids: tuple[str, ...]
    hit_record_ids: tuple[str, ...]
    rank_of_first_expected: int | None
    score_of_first_expected: float | None

    @property
    def found(self) -> bool:
        return self.rank_of_first_expected is not None

    def recall_at(self, depth: int) -> bool:
        """True when the first expected id appears at rank < depth.
        Depth 1 is top-1, depth 3 is top-3. A hard miss (rank=None)
        is False at every depth."""
        rank = self.rank_of_first_expected
        return rank is not None and rank < depth


@dataclass(frozen=True)
class ShapeResult:
    """Aggregate over scored cases. `k` is the depth the search was
    invoked with — the ceiling on any rank. recall@K equals
    (found cases) / (total cases) at that ceiling."""

    cases: tuple[ShapeCaseResult, ...]
    k: int

    @property
    def recall_at_1(self) -> float:
        return self._recall(1)

    @property
    def recall_at_3(self) -> float:
        return self._recall(3)

    @property
    def recall_at_5(self) -> float:
        return self._recall(5)

    @property
    def recall_at_k(self) -> float:
        return self._recall(self.k)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank over all cases. Hard-miss cases
        contribute 0. MRR rewards rank-1 hits more than rank-5 hits,
        complementing recall@K's binary present/absent view."""
        if not self.cases:
            return 0.0
        total = 0.0
        for case in self.cases:
            rank = case.rank_of_first_expected
            if rank is not None:
                total += 1.0 / (rank + 1)
        return total / len(self.cases)

    @property
    def median_rank(self) -> float | None:
        """Median 0-indexed rank across found cases. None when every
        case was a hard miss."""
        ranks = sorted(
            c.rank_of_first_expected for c in self.cases if c.rank_of_first_expected is not None
        )
        if not ranks:
            return None
        mid = len(ranks) // 2
        if len(ranks) % 2 == 1:
            return float(ranks[mid])
        return (ranks[mid - 1] + ranks[mid]) / 2.0

    def hard_misses(self) -> tuple[ShapeCaseResult, ...]:
        """Cases whose expected ids never appeared in the top-K — the
        retrieval-quality targets. Diagnosing these is where new
        primitives (tree / table / etc.) earn their keep."""
        return tuple(c for c in self.cases if not c.found)

    def _recall(self, depth: int) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.recall_at(depth)) / len(self.cases)


def score_case(case: ShapeCase, hits: Sequence[ShapeHit]) -> ShapeCaseResult:
    """Score one case against a pre-fetched hit list. Pure function;
    the caller decides K and passes pre-truncated hits."""
    expected = frozenset(case.expected_record_ids)
    rank: int | None = None
    first_score: float | None = None
    for idx, hit in enumerate(hits):
        if hit.record_id in expected:
            rank = idx
            first_score = hit.score
            break
    return ShapeCaseResult(
        id=case.id,
        query=case.query,
        expected_record_ids=case.expected_record_ids,
        hit_record_ids=tuple(h.record_id for h in hits),
        rank_of_first_expected=rank,
        score_of_first_expected=first_score,
    )


def run_retrieval_shape(
    cases: Sequence[ShapeCase],
    search_fn: ShapeSearchFn,
    *,
    k: int = 10,
) -> ShapeResult:
    """Run the eval over every case in `cases`. `search_fn` is invoked
    once per case with the user query and depth ceiling `k`. Returns
    an aggregated `ShapeResult` ready for human or JSON rendering."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    scored: list[ShapeCaseResult] = []
    for case in cases:
        hits = list(search_fn(case.query, k))[:k]
        scored.append(score_case(case, hits))
    return ShapeResult(cases=tuple(scored), k=k)
