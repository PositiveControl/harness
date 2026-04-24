"""atc retrieval eval — measures whether the episodic store surfaces
the expected section for each fixture case, without invoking the model.

Reuses the same YAML fixtures as `evals/atc.py` (one file, two scorers)
so a single source of truth drives both retrieval-quality measurement
and full-stack pass-rate measurement. For each case we pull the
section anchors out of `expected_citations` (e.g. "2-4-3", "91.155"),
run the case's `question` through a pluggable search callable, and
report the rank of the first hit whose principle carries any expected
anchor — along with aggregate recall@K (K=1, 3, 5, 10) across the
fixture.

Decouples retrieval quality from model quality. A chunker change, a
tokenizer swap, or an embedder upgrade can be measured in seconds
instead of minutes, and the delta is attributable: if recall@3 moves
and the full-stack pass rate doesn't, the retrieval fix didn't land
in the model's citation decisions — and vice versa.

The search callable (`SearchFn`) is a tiny protocol — (query, k) →
sequence of (principle, score). The CLI wires in the real
EpisodicStore.search via a hybrid-mode shim; tests inject a scripted
list so scoring stays deterministic (harness-dfa).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from harness.evals.atc import AtcFixtureRow


@dataclass(frozen=True)
class RetrievalHit:
    """Minimum shape the retrieval eval needs per hit — principle text
    (from which we parse the section anchor) and fused score. Callers
    with richer record types adapt down to this at the search-fn
    boundary so the eval stays decoupled from store internals."""

    principle: str
    score: float


# (query, k) → ranked hits (most relevant first).
SearchFn = Callable[[str, int], Sequence[RetrievalHit]]


# §N-N-N (JO 7110.65 / AIM), §N.M (CFR), bare N-N-N or N.M without the
# section glyph (the fixture YAML drops the § in expected_citations).
# We accept both sides: extraction matches both "§3-10-3" and "3-10-3"
# within principle strings, and expected anchors from the fixture
# retain whatever form the author used.
_ANCHOR_RE = re.compile(r"§?\s*(\d+(?:[-.]\d+){1,3}[a-z]?)")


def _extract_anchors(text: str) -> frozenset[str]:
    """Pull every N-N-N / N.M style anchor out of a principle or
    citation string. Returns a frozenset so case-level set-intersection
    is fast and order-insensitive."""
    if not text:
        return frozenset()
    return frozenset(_ANCHOR_RE.findall(text))


def _expected_anchor_set(row: AtcFixtureRow) -> frozenset[str]:
    """Flatten `expected_citations` (tuple of alternate-tuples) into a
    single set of acceptable section anchors. A case whose first entry
    lists ["91.155", "91.153", "91.173"] passes if ANY of those three
    is present in the retrieved results."""
    out: set[str] = set()
    for entry in row.expected_citations:
        for alt in entry:
            out.update(_ANCHOR_RE.findall(alt) or [alt.strip()])
    return frozenset(x for x in out if x)


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    audience: str
    query: str
    expected_anchors: tuple[str, ...]
    # Top-K hits (principle, score) captured for debug / JSON replay.
    hits: tuple[RetrievalHit, ...]
    # 0-indexed rank of the first hit whose principle carries ANY
    # expected anchor. None when no expected anchor appears in any hit
    # (a hard miss — the retrieval stack doesn't surface the right
    # section at all within K).
    rank_of_first_expected: int | None
    score_of_first_expected: float | None

    @property
    def found(self) -> bool:
        return self.rank_of_first_expected is not None

    def recall_at(self, depth: int) -> bool:
        """True when the first expected anchor appears at rank < depth.
        Depth 1 is top-1, depth 3 is top-3, etc. A case with rank=None
        (hard miss) returns False at every depth."""
        rank = self.rank_of_first_expected
        return rank is not None and rank < depth


def score_case(row: AtcFixtureRow, hits: Sequence[RetrievalHit]) -> RetrievalCase:
    """Score one case against a pre-fetched hit list. Pure function;
    the caller decides what K to pre-fetch."""
    expected = _expected_anchor_set(row)
    rank: int | None = None
    first_score: float | None = None
    for idx, hit in enumerate(hits):
        hit_anchors = _extract_anchors(hit.principle)
        if hit_anchors & expected:
            rank = idx
            first_score = hit.score
            break
    return RetrievalCase(
        id=row.id,
        audience=row.audience,
        query=row.question,
        expected_anchors=tuple(sorted(expected)),
        hits=tuple(hits),
        rank_of_first_expected=rank,
        score_of_first_expected=first_score,
    )


@dataclass(frozen=True)
class RetrievalResult:
    """Aggregate over all scored cases. K is the depth the search fn
    was queried with (upper bound on any rank). The recall_at_*
    properties compute standard retrieval-eval recall against that
    same depth, so recall@K == fraction of cases that found the
    expected section anywhere in the top-K."""

    cases: tuple[RetrievalCase, ...]
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
        """Recall at the ceiling the search was run with. Equals
        (found cases) / (total cases)."""
        return self._recall(self.k)

    def _recall(self, depth: int) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.recall_at(depth)) / len(self.cases)

    @property
    def median_rank(self) -> float | None:
        """Median 0-indexed rank of the first expected hit across cases
        that found something. None when every case was a hard miss."""
        ranks: list[int] = sorted(
            c.rank_of_first_expected for c in self.cases if c.rank_of_first_expected is not None
        )
        if not ranks:
            return None
        mid = len(ranks) // 2
        if len(ranks) % 2 == 1:
            return float(ranks[mid])
        return (ranks[mid - 1] + ranks[mid]) / 2.0

    def hard_misses(self) -> tuple[RetrievalCase, ...]:
        """Cases where the expected section never appeared in the
        top-K. These are the retrieval-quality targets — no amount of
        model tuning recovers a miss the retrieval stack didn't supply."""
        return tuple(c for c in self.cases if not c.found)


def run_atc_retrieval(
    fixture: Sequence[AtcFixtureRow],
    search_fn: SearchFn,
    *,
    k: int = 10,
) -> RetrievalResult:
    """Run the retrieval-only eval over every row in `fixture`. The
    `search_fn` is invoked once per row with the user-facing `question`
    string and the depth ceiling `k`. Returns an aggregated result
    ready for human or JSON rendering."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    scored: list[RetrievalCase] = []
    for row in fixture:
        hits = search_fn(row.question, k)
        scored.append(score_case(row, list(hits)[:k]))
    return RetrievalResult(cases=tuple(scored), k=k)


def default_baseline_path(character_path: Path) -> Path:
    """Where `harness eval atc-retrieval --save-baseline` writes the
    snapshot. Separate file from atc_baseline.json (which holds the
    full-stack eval) so retrieval-only deltas don't fight reply-side
    deltas in the same diff."""
    return character_path / "atc_retrieval_baseline.json"
