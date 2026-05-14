"""Tests for harness.evals.retrieval_shape (harness-kj2a / Phase 0 of
the data-retrieval-primitives spike).

Pure-Python scorer — scripted SearchFn returns a deterministic hit
list so the ranking / recall / MRR math is reproducible without any
embedder or store dependency."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from harness.evals.retrieval_shape import (
    ShapeCase,
    ShapeCaseResult,
    ShapeHit,
    ShapeResult,
    run_retrieval_shape,
    score_case,
)


def _hits(*ids_with_scores: tuple[str, float]) -> list[ShapeHit]:
    return [ShapeHit(record_id=rid, score=score) for rid, score in ids_with_scores]


# ---------- score_case ----------


def test_score_case_rank_zero_when_expected_is_first() -> None:
    case = ShapeCase(id="q1", query="?", expected_record_ids=("a",))
    result = score_case(case, _hits(("a", 0.9), ("b", 0.5)))
    assert result.rank_of_first_expected == 0
    assert result.score_of_first_expected == 0.9
    assert result.hit_record_ids == ("a", "b")
    assert result.found is True
    assert result.recall_at(1) is True
    assert result.recall_at(5) is True


def test_score_case_rank_two_when_expected_at_index_two() -> None:
    case = ShapeCase(id="q1", query="?", expected_record_ids=("c",))
    result = score_case(case, _hits(("a", 0.9), ("b", 0.5), ("c", 0.1)))
    assert result.rank_of_first_expected == 2
    assert result.score_of_first_expected == 0.1
    assert result.recall_at(1) is False
    assert result.recall_at(3) is True


def test_score_case_hard_miss_when_no_hit_matches() -> None:
    case = ShapeCase(id="q1", query="?", expected_record_ids=("c",))
    result = score_case(case, _hits(("a", 0.9), ("b", 0.5)))
    assert result.rank_of_first_expected is None
    assert result.score_of_first_expected is None
    assert result.found is False
    assert result.recall_at(10) is False


def test_score_case_picks_first_matching_alternate() -> None:
    """When expected_record_ids has multiple alternates, the FIRST hit
    that matches any alternate wins — not the lowest-ranked alternate."""
    case = ShapeCase(id="q1", query="?", expected_record_ids=("a", "c"))
    result = score_case(case, _hits(("b", 0.9), ("c", 0.7), ("a", 0.5)))
    assert result.rank_of_first_expected == 1
    assert result.score_of_first_expected == 0.7


# ---------- run_retrieval_shape ----------


def _scripted_search(
    table: dict[str, Sequence[ShapeHit]],
) -> Callable[[str, int], Sequence[ShapeHit]]:
    """Build a SearchFn that returns scripted hits keyed by query string.
    Unknown queries return an empty list — clean way to test 'no
    candidates returned' paths."""

    def _fn(query: str, k: int) -> Sequence[ShapeHit]:
        return list(table.get(query, []))[:k]

    return _fn


def test_run_retrieval_shape_recall_aggregates() -> None:
    cases = [
        ShapeCase(id="q1", query="alpha", expected_record_ids=("a",)),
        ShapeCase(id="q2", query="beta", expected_record_ids=("b",)),
        ShapeCase(id="q3", query="gamma", expected_record_ids=("c",)),
        ShapeCase(id="q4", query="delta", expected_record_ids=("d",)),
    ]
    search = _scripted_search(
        {
            # q1: hit at rank 0
            "alpha": _hits(("a", 0.9), ("z", 0.5)),
            # q2: hit at rank 2 (in top-3 but not top-1)
            "beta": _hits(("x", 0.8), ("y", 0.7), ("b", 0.3)),
            # q3: hit at rank 4 (in top-5 only)
            "gamma": _hits(("x", 0.8), ("y", 0.7), ("w", 0.6), ("v", 0.5), ("c", 0.1)),
            # q4: hard miss
            "delta": _hits(("x", 0.8), ("y", 0.7)),
        }
    )
    result = run_retrieval_shape(cases, search, k=10)

    assert result.k == 10
    assert len(result.cases) == 4
    # 1 of 4 found at rank 0
    assert result.recall_at_1 == 0.25
    # 2 of 4 found at rank < 3
    assert result.recall_at_3 == 0.5
    # 3 of 4 found at rank < 5
    assert result.recall_at_5 == 0.75
    # 3 of 4 found at rank < 10
    assert result.recall_at_k == 0.75
    # MRR: (1 + 1/3 + 1/5 + 0) / 4
    expected_mrr = (1.0 + 1.0 / 3.0 + 1.0 / 5.0 + 0.0) / 4.0
    assert abs(result.mrr - expected_mrr) < 1e-6
    # Median rank across [0, 2, 4]
    assert result.median_rank == 2.0


def test_run_retrieval_shape_hard_misses_only_returns_missed_cases() -> None:
    cases = [
        ShapeCase(id="q1", query="alpha", expected_record_ids=("a",)),
        ShapeCase(id="q2", query="beta", expected_record_ids=("b",)),
    ]
    search = _scripted_search(
        {
            "alpha": _hits(("a", 0.9)),
            "beta": _hits(("z", 0.9)),  # miss
        }
    )
    result = run_retrieval_shape(cases, search, k=5)
    misses = result.hard_misses()
    assert len(misses) == 1
    assert misses[0].id == "q2"


def test_run_retrieval_shape_empty_cases_returns_zero_recall() -> None:
    result = run_retrieval_shape([], _scripted_search({}), k=5)
    assert result.recall_at_1 == 0.0
    assert result.recall_at_k == 0.0
    assert result.mrr == 0.0
    assert result.median_rank is None
    assert result.hard_misses() == ()


def test_run_retrieval_shape_all_hard_misses_returns_no_median() -> None:
    cases = [
        ShapeCase(id="q1", query="a", expected_record_ids=("a",)),
        ShapeCase(id="q2", query="b", expected_record_ids=("b",)),
    ]
    search = _scripted_search({"a": _hits(("x", 0.9)), "b": _hits(("y", 0.9))})
    result = run_retrieval_shape(cases, search, k=5)
    assert result.median_rank is None
    assert result.recall_at_k == 0.0
    assert result.mrr == 0.0


def test_run_retrieval_shape_truncates_to_k() -> None:
    """SearchFn returning more than k hits should be truncated before
    scoring — guards against a misbehaving SearchFn lifting recall by
    over-returning."""
    case = ShapeCase(id="q1", query="alpha", expected_record_ids=("z",))
    # 'z' lives at rank 5 but k=3, so it should not be found.
    search = _scripted_search(
        {
            "alpha": _hits(
                ("a", 0.9),
                ("b", 0.8),
                ("c", 0.7),
                ("d", 0.6),
                ("e", 0.5),
                ("z", 0.1),
            ),
        }
    )
    result = run_retrieval_shape([case], search, k=3)
    assert result.cases[0].rank_of_first_expected is None
    assert result.cases[0].hit_record_ids == ("a", "b", "c")


def test_run_retrieval_shape_rejects_zero_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        run_retrieval_shape([], _scripted_search({}), k=0)


# ---------- ShapeResult typing sanity ----------


def test_shape_case_result_is_frozen() -> None:
    case = ShapeCaseResult(
        id="q1",
        query="?",
        expected_record_ids=("a",),
        hit_record_ids=("a",),
        rank_of_first_expected=0,
        score_of_first_expected=0.9,
    )
    with pytest.raises(AttributeError):
        case.id = "q2"  # type: ignore[misc]


def test_shape_result_is_frozen() -> None:
    result = ShapeResult(cases=(), k=10)
    with pytest.raises(AttributeError):
        result.k = 5  # type: ignore[misc]
