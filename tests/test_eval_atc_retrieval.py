"""Tests for harness.evals.atc_retrieval (harness-dfa).

Pure-Python scorer — fake search callable returns a scripted hit list
so ranking and recall math are deterministic."""

from __future__ import annotations

from collections.abc import Sequence

from harness.evals.atc import AtcFixtureRow
from harness.evals.atc_retrieval import (
    RetrievalHit,
    _expected_anchor_set,
    _extract_anchors,
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
