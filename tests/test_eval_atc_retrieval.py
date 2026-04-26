"""Tests for harness.evals.atc_retrieval (harness-dfa).

Pure-Python scorer — fake search callable returns a scripted hit list
so ranking and recall math are deterministic."""

from __future__ import annotations

from collections.abc import Sequence

from harness.evals.atc import AtcFixtureRow
from harness.evals.atc_retrieval import (
    AggregateDelta,
    BaselineComparison,
    CaseRankDelta,
    RetrievalHit,
    RetrievalResult,
    _expected_anchor_set,
    _extract_anchors,
    compare_baselines,
    run_atc_retrieval,
    score_case,
)


def _row(
    case_id: str,
    question: str,
    expected: tuple[tuple[str, ...], ...],
    audience: str = "controller",
) -> AtcFixtureRow:
    return AtcFixtureRow(
        id=case_id,
        audience=audience,
        question=question,
        expected_citations=expected,
        expected_keywords=(),
        min_keyword_hits=0,
    )


def _hit(principle: str, score: float = 0.03) -> RetrievalHit:
    return RetrievalHit(principle=principle, score=score)


# ---------- anchor extraction ----------


def test_extract_anchors_finds_jo_style_section() -> None:
    anchors = _extract_anchors("JO_7110.65 §3-10-3 (Arrival — SAME RUNWAY)")
    assert "3-10-3" in anchors


def test_extract_anchors_finds_cfr_style_section() -> None:
    anchors = _extract_anchors("14 CFR §91.155 — VFR weather minimums")
    assert "91.155" in anchors


def test_extract_anchors_ignores_bare_numbers_in_body() -> None:
    # Stray numbers like "3 miles" should not create spurious anchors —
    # the regex requires at least one dash or dot to split pieces.
    anchors = _extract_anchors("Body text: 3 miles, 5 miles, 40 miles.")
    assert anchors == frozenset()


def test_expected_anchor_set_flattens_alternates() -> None:
    row = _row(
        "cancel_ifr_imc",
        "Can I cancel IFR in IMC?",
        expected=(("91.155", "91.153", "91.173"),),
    )
    assert _expected_anchor_set(row) == frozenset({"91.155", "91.153", "91.173"})


def test_expected_anchor_set_handles_scalar_without_glyph() -> None:
    row = _row(
        "readback",
        "What must a controller verify on readback?",
        expected=(("2-4-3",),),
    )
    assert _expected_anchor_set(row) == frozenset({"2-4-3"})


# ---------- score_case ----------


def test_score_case_finds_expected_at_rank_0() -> None:
    row = _row("emerg", "Who may declare an emergency?", expected=(("10-2-5",),))
    hits = [
        _hit("JO_7110.65 §10-2-5 (Emergency Assistance — EMERGENCY SITUATIONS)", score=0.033),
        _hit("JO_7110.65 §10-1-1 (General — EMERGENCY DETERMINATIONS)", score=0.028),
    ]
    result = score_case(row, hits)
    assert result.found
    assert result.rank_of_first_expected == 0
    assert result.score_of_first_expected == 0.033


def test_score_case_finds_expected_at_later_rank() -> None:
    row = _row("emerg", "Who may declare an emergency?", expected=(("10-2-5",),))
    hits = [
        _hit("JO_7110.65 §10-1-1 (General — EMERGENCY DETERMINATIONS)", score=0.033),
        _hit("JO_7110.65 §2-1-8 (General — MINIMUM FUEL)", score=0.029),
        _hit("JO_7110.65 §10-2-5 (Emergency Assistance — EMERGENCY SITUATIONS)", score=0.025),
    ]
    result = score_case(row, hits)
    assert result.found
    assert result.rank_of_first_expected == 2
    assert result.score_of_first_expected == 0.025


def test_score_case_hard_miss() -> None:
    row = _row("emerg", "Who may declare an emergency?", expected=(("10-2-5",),))
    hits = [
        _hit("JO_7110.65 §5-7-2 (Speed Adjustment — METHODS)", score=0.030),
        _hit("JO_7110.65 §2-1-4 (General — OPERATIONAL PRIORITY)", score=0.028),
    ]
    result = score_case(row, hits)
    assert not result.found
    assert result.rank_of_first_expected is None
    assert result.score_of_first_expected is None


def test_score_case_accepts_any_alternate() -> None:
    """expected_citations lists alternates — if ANY appears in the
    hits, the case counts as found, and the rank is taken from the
    first hit carrying any alternate."""
    row = _row(
        "cancel_ifr_imc",
        "Can I cancel IFR in IMC?",
        expected=(("91.155", "91.153", "91.173"),),
    )
    hits = [
        _hit("JO_7110.65 §2-1-1 (General — ATC SERVICE)", score=0.033),
        _hit("14 CFR §91.173 — ATC clearance required", score=0.029),
    ]
    result = score_case(row, hits)
    assert result.found
    assert result.rank_of_first_expected == 1


# ---------- run_atc_retrieval aggregation ----------


def test_run_atc_retrieval_aggregates_recall() -> None:
    fixture = (
        _row("a", "q1", (("10-2-5",),)),  # rank 0
        _row("b", "q2", (("4-6-4",),)),  # rank 2
        _row("c", "q3", (("8-7-3",),)),  # hard miss
    )

    scripted: dict[str, list[RetrievalHit]] = {
        "q1": [
            _hit("§10-2-5"),
            _hit("§10-1-1"),
            _hit("§2-1-8"),
        ],
        "q2": [
            _hit("§2-1-4"),
            _hit("§5-7-2"),
            _hit("§4-6-4"),
        ],
        "q3": [
            _hit("§1-1-1"),
            _hit("§2-2-2"),
            _hit("§3-3-3"),
        ],
    }

    def search(query: str, k: int) -> Sequence[RetrievalHit]:
        return scripted[query][:k]

    result = run_atc_retrieval(fixture, search, k=5)
    # a hits at 0 → counts for @1/@3/@5. b hits at 2 → counts for @3/@5.
    # c misses → counts for none.
    assert result.recall_at_1 == 1 / 3
    assert result.recall_at_3 == 2 / 3
    assert result.recall_at_5 == 2 / 3
    # Median of [0, 2] == 1.0
    assert result.median_rank == 1.0
    assert len(result.hard_misses()) == 1
    assert result.hard_misses()[0].id == "c"


def test_run_atc_retrieval_truncates_to_k() -> None:
    """Hits beyond the requested k are ignored, even if the search fn
    returns more — pins the depth ceiling against scorer drift."""
    fixture = (_row("a", "q1", (("10-2-5",),)),)

    def search(query: str, k: int) -> Sequence[RetrievalHit]:
        # Return 10 hits; expected is at rank 7, past k=5.
        return [_hit(f"§{i}-{i}-{i}") for i in range(7)] + [
            _hit("§10-2-5"),
            _hit("§99-9-9"),
            _hit("§99-9-8"),
        ]

    result = run_atc_retrieval(fixture, search, k=5)
    assert result.recall_at_5 == 0.0
    # But the case captured the truncated top-k for debug.
    assert len(result.cases[0].hits) == 5


def test_run_atc_retrieval_empty_fixture_returns_zero() -> None:
    def search(query: str, k: int) -> Sequence[RetrievalHit]:
        return []

    result = run_atc_retrieval((), search, k=5)
    assert result.recall_at_5 == 0.0
    assert result.median_rank is None
    assert result.cases == ()


# ---------- baseline comparator (harness-sb6r) ----------


def _baseline(
    cases: tuple[tuple[str, int | None], ...],
    *,
    recall_at_1: float | None = None,
    recall_at_3: float | None = None,
    recall_at_5: float | None = None,
    recall_at_k: float | None = None,
) -> dict[str, object]:
    """Tiny baseline-JSON factory used by the comparator tests. Mirrors
    the on-disk shape (id + rank_of_first_expected per case + four
    recall aggregates)."""
    payload: dict[str, object] = {
        "cases": [{"id": case_id, "rank_of_first_expected": rank} for case_id, rank in cases],
    }
    for key, value in (
        ("recall_at_1", recall_at_1),
        ("recall_at_3", recall_at_3),
        ("recall_at_5", recall_at_5),
        ("recall_at_k", recall_at_k),
    ):
        if value is not None:
            payload[key] = value
    return payload


def _result_from_cases(
    cases: tuple[tuple[str, int | None], ...], *, k: int = 10
) -> RetrievalResult:
    """Build a synthetic RetrievalResult by feeding scripted hits to
    score_case, so the comparator tests don't reach into private
    construction. Each case_id becomes a fixture row whose expected
    anchor matches a scripted hit at the requested rank."""
    fixture_rows = []
    scripted: dict[str, list[RetrievalHit]] = {}
    for case_id, rank in cases:
        anchor = f"{case_id}-1-1"
        fixture_rows.append(_row(case_id, case_id, ((anchor,),)))
        if rank is None:
            scripted[case_id] = [_hit(f"§9-9-{i}") for i in range(k)]
        else:
            hits: list[RetrievalHit] = [_hit(f"§9-9-{i}") for i in range(rank)]
            hits.append(_hit(f"§{anchor}"))
            while len(hits) < k:
                hits.append(_hit(f"§9-9-{len(hits)}"))
            scripted[case_id] = hits

    def search(query: str, depth: int) -> Sequence[RetrievalHit]:
        return scripted[query][:depth]

    return run_atc_retrieval(tuple(fixture_rows), search, k=k)


def test_case_rank_delta_regression_when_rank_worsens() -> None:
    delta = CaseRankDelta(id="a", old_rank=0, new_rank=2)
    assert delta.is_regression is True
    assert delta.is_improvement is False


def test_case_rank_delta_improvement_when_rank_better() -> None:
    delta = CaseRankDelta(id="a", old_rank=4, new_rank=1)
    assert delta.is_improvement is True
    assert delta.is_regression is False


def test_case_rank_delta_regression_when_was_found_now_missed() -> None:
    delta = CaseRankDelta(id="a", old_rank=2, new_rank=None)
    assert delta.is_regression is True


def test_case_rank_delta_improvement_when_was_missed_now_found() -> None:
    delta = CaseRankDelta(id="a", old_rank=None, new_rank=4)
    assert delta.is_improvement is True
    assert delta.is_regression is False


def test_case_rank_delta_neither_when_both_missed() -> None:
    delta = CaseRankDelta(id="a", old_rank=None, new_rank=None)
    assert delta.is_regression is False
    assert delta.is_improvement is False


def test_aggregate_delta_regression_when_recall_drops() -> None:
    delta = AggregateDelta(metric="recall_at_1", old=1.0, new=0.8)
    assert delta.is_regression is True


def test_aggregate_delta_no_regression_when_equal() -> None:
    delta = AggregateDelta(metric="recall_at_1", old=0.8, new=0.8)
    assert delta.is_regression is False


def test_compare_baselines_no_change() -> None:
    """Identical baseline + result → zero regressions, zero improvements."""
    baseline = _baseline(
        (("a", 0), ("b", 2)),
        recall_at_1=0.5,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_k=1.0,
    )
    result = _result_from_cases((("a", 0), ("b", 2)))
    cmp = compare_baselines(baseline, result)
    assert cmp.case_regressions == ()
    assert cmp.case_improvements == ()
    assert cmp.aggregate_regressions == ()
    assert cmp.has_regression() is False


def test_compare_baselines_aggregate_regression_fails_gate() -> None:
    """Aggregate recall drop is unconditional fail — budget can't rescue."""
    baseline = _baseline(
        (("a", 0), ("b", 0)),
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_k=1.0,
    )
    # b regresses to a hard miss → recall@1 drops 1.0 → 0.5.
    result = _result_from_cases((("a", 0), ("b", None)))
    cmp = compare_baselines(baseline, result)
    assert any(d.metric == "recall_at_1" for d in cmp.aggregate_regressions)
    # Even with a generous budget, aggregate failure trumps.
    assert cmp.has_regression(regression_budget=10) is True


def test_compare_baselines_per_case_regression_within_budget() -> None:
    """Rank shuffle that keeps aggregate recall stable can pass under
    --regression-budget. b drops from rank 2 → rank 4, still in top-5."""
    baseline = _baseline(
        (("a", 0), ("b", 2)),
        recall_at_1=0.5,
        recall_at_3=1.0,
        recall_at_5=1.0,
        recall_at_k=1.0,
    )
    result = _result_from_cases((("a", 0), ("b", 4)))
    cmp = compare_baselines(baseline, result)
    # recall@3 dropped (b moved out of top-3), so aggregate fails — adjust:
    # use a baseline that has @3 already at the lower rate.
    baseline2 = _baseline(
        (("a", 0), ("b", 2)),
        recall_at_1=0.5,
        recall_at_5=1.0,
        recall_at_k=1.0,
    )
    cmp2 = compare_baselines(baseline2, result)
    assert len(cmp2.case_regressions) == 1
    assert cmp2.case_regressions[0].id == "b"
    assert cmp2.has_regression(regression_budget=0) is True
    assert cmp2.has_regression(regression_budget=1) is False
    # The original aggregate-tracking version still fails.
    assert cmp.has_regression(regression_budget=10) is True


def test_compare_baselines_improvements_listed_separately() -> None:
    baseline = _baseline((("a", 4), ("b", 2)))
    result = _result_from_cases((("a", 0), ("b", 0)))
    cmp = compare_baselines(baseline, result)
    assert {d.id for d in cmp.case_improvements} == {"a", "b"}
    assert cmp.case_regressions == ()
    assert cmp.has_regression() is False


def test_compare_baselines_tracks_new_and_dropped_cases() -> None:
    """Fixture growth (new case) and fixture shrink (dropped case) are
    surfaced separately from regressions — neither flips the gate."""
    baseline = _baseline((("a", 0), ("b", 0)))
    result = _result_from_cases((("a", 0), ("c", 0)))
    cmp = compare_baselines(baseline, result)
    assert cmp.new_cases == ("c",)
    assert cmp.dropped_cases == ("b",)
    assert cmp.case_regressions == ()
    assert cmp.has_regression() is False


def test_compare_baselines_tolerates_missing_aggregate_keys() -> None:
    """An older baseline that lacks recall_at_5 still compares cleanly
    on the metrics it does carry — no KeyError, just fewer aggregate
    rows."""
    baseline = _baseline(
        (("a", 0),),
        recall_at_1=1.0,  # only this metric present
    )
    result = _result_from_cases((("a", 0),))
    cmp = compare_baselines(baseline, result)
    metrics = {d.metric for d in cmp.aggregate_deltas}
    assert metrics == {"recall_at_1"}


def test_baseline_comparison_dataclass_is_frozen() -> None:
    """BaselineComparison is a frozen dataclass — mutation should fail."""
    cmp = BaselineComparison(
        case_deltas=(),
        aggregate_deltas=(),
        new_cases=(),
        dropped_cases=(),
    )
    try:
        cmp.case_deltas = ()  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("BaselineComparison should be frozen")
