"""Tests for the phraseology eval module (harness-15dy).

Covers:
- Fixture loader (required fields, oos null-citation invariant, scalar
  field validation, expected_verdict enum check).
- Pure scoring (verdict + citation pass flags, accuracy aggregates,
  per-scenario breakdown).
- Baseline comparator (per-case pass flips, aggregate-accuracy
  regression, regression budget, new/dropped cases).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.evals.phraseology import (
    AggregateAccuracyDelta,
    BaselineComparison,
    CasePassDelta,
    PhraseologyEvalResult,
    PhraseologyFixtureRow,
    compare_baselines,
    load_fixture,
    run_phraseology_eval,
)
from harness.tools.phraseology_lint import PhraseologyVerdict

# ---------- Fixture loader ----------


def _write_fixture(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "phraseology_eval.yaml"
    p.write_text(body)
    return p


def test_load_fixture_minimal_ok_row(tmp_path: Path) -> None:
    p = _write_fixture(
        tmp_path,
        """
cases:
  - id: dep_takeoff
    scenario: departure
    utterance: "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF."
    expected_verdict: ok
    expected_section: "3-9-10"
    expected_phraseology: "RUNWAY (number), CLEARED FOR TAKEOFF."
""",
    )
    rows = load_fixture(p)
    assert len(rows) == 1
    row = rows[0]
    assert row.id == "dep_takeoff"
    assert row.scenario == "departure"
    assert row.expected_verdict == "ok"
    assert row.expected_section == "3-9-10"


def test_load_fixture_oos_row(tmp_path: Path) -> None:
    p = _write_fixture(
        tmp_path,
        """
cases:
  - id: emerg_pilot_oos
    scenario: emergency
    utterance: "MAYDAY MAYDAY MAYDAY"
    expected_verdict: out_of_scope
""",
    )
    rows = load_fixture(p)
    assert rows[0].expected_section is None
    assert rows[0].expected_phraseology is None


def test_load_fixture_strips_section_glyph(tmp_path: Path) -> None:
    p = _write_fixture(
        tmp_path,
        """
cases:
  - id: x
    scenario: departure
    utterance: "X"
    expected_verdict: ok
    expected_section: "§3-9-10"
""",
    )
    assert load_fixture(p)[0].expected_section == "3-9-10"


def test_load_fixture_rejects_unknown_verdict(tmp_path: Path) -> None:
    p = _write_fixture(
        tmp_path,
        """
cases:
  - id: x
    scenario: departure
    utterance: "X"
    expected_verdict: maybe
    expected_section: "3-9-10"
""",
    )
    with pytest.raises(ValueError, match="expected_verdict"):
        load_fixture(p)


def test_load_fixture_rejects_oos_with_section(tmp_path: Path) -> None:
    p = _write_fixture(
        tmp_path,
        """
cases:
  - id: x
    scenario: emergency
    utterance: "X"
    expected_verdict: out_of_scope
    expected_section: "10-1-1"
""",
    )
    with pytest.raises(ValueError, match="out_of_scope"):
        load_fixture(p)


def test_load_fixture_rejects_non_oos_without_section(tmp_path: Path) -> None:
    p = _write_fixture(
        tmp_path,
        """
cases:
  - id: x
    scenario: departure
    utterance: "X"
    expected_verdict: ok
""",
    )
    with pytest.raises(ValueError, match="must carry"):
        load_fixture(p)


def test_load_fixture_empty_file(tmp_path: Path) -> None:
    p = _write_fixture(tmp_path, "")
    assert load_fixture(p) == ()


# ---------- Scoring ----------


def _row(
    *,
    case_id: str = "x",
    scenario: str = "departure",
    expected_verdict: str = "ok",
    expected_section: str | None = "3-9-10",
) -> PhraseologyFixtureRow:
    return PhraseologyFixtureRow(
        id=case_id,
        scenario=scenario,
        utterance="X",
        expected_verdict=expected_verdict,
        expected_section=expected_section,
        expected_phraseology=None,
        mismatch=None,
        notes=None,
    )


def _verdict(
    *,
    verdict: str = "ok",
    section: str | None = "3-9-10",
) -> PhraseologyVerdict:
    return PhraseologyVerdict(
        verdict=verdict,  # type: ignore[arg-type]
        expected_section=section,
        expected_phraseology=None,
        mismatch=None,
        citation_quote=None,
    )


def test_run_eval_passes_when_both_match() -> None:
    fixture = (_row(case_id="ok1"),)

    def lint_fn(utterance: str, hint: str | None) -> PhraseologyVerdict:
        return _verdict()

    result = run_phraseology_eval(fixture, lint_fn)
    assert result.combined_accuracy == 1.0
    assert result.cases[0].passed


def test_run_eval_verdict_mismatch_fails_combined() -> None:
    fixture = (_row(case_id="v"),)

    def lint_fn(utterance: str, hint: str | None) -> PhraseologyVerdict:
        # Same section, wrong verdict
        return _verdict(verdict="wrong")

    result = run_phraseology_eval(fixture, lint_fn)
    assert result.cases[0].verdict_pass is False
    assert result.cases[0].citation_pass is True
    assert result.cases[0].passed is False
    assert result.combined_accuracy == 0.0
    assert result.citation_accuracy == 1.0


def test_run_eval_citation_mismatch_fails_combined() -> None:
    fixture = (_row(case_id="c"),)

    def lint_fn(utterance: str, hint: str | None) -> PhraseologyVerdict:
        # Right verdict, wrong section
        return _verdict(section="3-10-5")

    result = run_phraseology_eval(fixture, lint_fn)
    assert result.cases[0].verdict_pass is True
    assert result.cases[0].citation_pass is False
    assert result.combined_accuracy == 0.0


def test_run_eval_oos_round_trip() -> None:
    fixture = (
        _row(
            case_id="oos1",
            scenario="emergency",
            expected_verdict="out_of_scope",
            expected_section=None,
        ),
    )

    def lint_fn(utterance: str, hint: str | None) -> PhraseologyVerdict:
        return _verdict(verdict="out_of_scope", section=None)

    result = run_phraseology_eval(fixture, lint_fn)
    assert result.cases[0].passed is True
    assert result.combined_accuracy == 1.0


def test_run_eval_per_scenario_breakdown() -> None:
    fixture = (
        _row(case_id="d1", scenario="departure"),
        _row(case_id="d2", scenario="departure"),
        _row(case_id="a1", scenario="arrival"),
    )
    # d1 passes, d2 fails verdict, a1 passes
    replies = {
        "d1": _verdict(),
        "d2": _verdict(verdict="wrong"),
        "a1": _verdict(),
    }
    cur = iter(("d1", "d2", "a1"))

    def lint_fn(utterance: str, hint: str | None) -> PhraseologyVerdict:
        return replies[next(cur)]

    result = run_phraseology_eval(fixture, lint_fn)
    by_scen = result.by_scenario()
    assert by_scen["departure"]["combined_accuracy"] == 0.5
    assert by_scen["departure"]["count"] == 2
    assert by_scen["arrival"]["combined_accuracy"] == 1.0
    assert by_scen["arrival"]["count"] == 1


# ---------- Baseline comparator ----------


def _baseline_for(
    *,
    cases: list[dict[str, object]] | None = None,
    verdict_accuracy: float = 1.0,
    citation_accuracy: float = 1.0,
    combined_accuracy: float = 1.0,
) -> dict[str, object]:
    return {
        "verdict_accuracy": verdict_accuracy,
        "citation_accuracy": citation_accuracy,
        "combined_accuracy": combined_accuracy,
        "cases": cases or [],
    }


def _eval_with(
    pairs: list[tuple[PhraseologyFixtureRow, PhraseologyVerdict]],
) -> PhraseologyEvalResult:
    fixture = tuple(p[0] for p in pairs)
    verdicts = iter(p[1] for p in pairs)

    def lint_fn(utterance: str, hint: str | None) -> PhraseologyVerdict:
        return next(verdicts)

    return run_phraseology_eval(fixture, lint_fn)


def test_compare_baselines_no_regression_when_identical() -> None:
    result = _eval_with([(_row(case_id="x"), _verdict())])
    baseline = _baseline_for(
        cases=[
            {"id": "x", "verdict_pass": True, "citation_pass": True},
        ]
    )
    cmp = compare_baselines(baseline, result)
    assert cmp.has_regression() is False
    assert cmp.case_regressions == ()
    assert cmp.aggregate_regressions == ()


def test_compare_baselines_detects_verdict_regression() -> None:
    # Was passing both, now verdict fails.
    result = _eval_with([(_row(case_id="x"), _verdict(verdict="wrong"))])
    baseline = _baseline_for(
        cases=[{"id": "x", "verdict_pass": True, "citation_pass": True}],
        verdict_accuracy=1.0,
        combined_accuracy=1.0,
    )
    cmp = compare_baselines(baseline, result)
    assert cmp.has_regression() is True
    assert len(cmp.case_regressions) == 1
    delta = cmp.case_regressions[0]
    assert delta.id == "x"
    assert delta.old_verdict_pass is True
    assert delta.new_verdict_pass is False


def test_compare_baselines_detects_citation_regression() -> None:
    result = _eval_with([(_row(case_id="x"), _verdict(section="9-9-9"))])
    baseline = _baseline_for(
        cases=[{"id": "x", "verdict_pass": True, "citation_pass": True}],
        citation_accuracy=1.0,
        combined_accuracy=1.0,
    )
    cmp = compare_baselines(baseline, result)
    assert cmp.has_regression() is True
    delta = cmp.case_regressions[0]
    assert delta.old_citation_pass is True
    assert delta.new_citation_pass is False


def test_compare_baselines_aggregate_regression_overrides_budget() -> None:
    """Aggregate-accuracy drop fails the gate even when regression
    budget would tolerate the per-case flip."""
    result = _eval_with([(_row(case_id="x"), _verdict(verdict="wrong"))])
    baseline = _baseline_for(
        cases=[{"id": "x", "verdict_pass": True, "citation_pass": True}],
        verdict_accuracy=1.0,
        combined_accuracy=1.0,
    )
    cmp = compare_baselines(baseline, result)
    # Even with a generous budget, aggregate accuracy dropped
    # (verdict_accuracy 1.0 → 0.0), so the gate fails.
    assert cmp.has_regression(regression_budget=10) is True


def test_compare_baselines_budget_tolerates_per_case_flip() -> None:
    """Per-case flip with no aggregate drop is tolerable under budget.

    Construct a 2-case run: one improved (verdict_pass False → True),
    one regressed (verdict_pass True → False). Aggregate verdict_accuracy
    holds at 0.5 → 0.5 (no aggregate regression), so budget=1 should
    let it pass."""
    rows = (_row(case_id="a"), _row(case_id="b"))
    verdicts = (_verdict(verdict="wrong"), _verdict())  # a fails, b passes
    fixture_iter = iter(verdicts)

    def lint_fn(utterance: str, hint: str | None) -> PhraseologyVerdict:
        return next(fixture_iter)

    result = run_phraseology_eval(rows, lint_fn)
    # Baseline: a was passing, b was failing — net 50% verdict accuracy.
    baseline = _baseline_for(
        cases=[
            {"id": "a", "verdict_pass": True, "citation_pass": True},
            {"id": "b", "verdict_pass": False, "citation_pass": True},
        ],
        verdict_accuracy=0.5,
        citation_accuracy=1.0,
        combined_accuracy=0.5,
    )
    cmp = compare_baselines(baseline, result)
    # One per-case regression (a) + one improvement (b).
    assert len(cmp.case_regressions) == 1
    assert len(cmp.case_improvements) == 1
    # No aggregate regression — verdict accuracy held at 0.5.
    assert cmp.aggregate_regressions == ()
    # budget=1 absorbs the single per-case flip.
    assert cmp.has_regression(regression_budget=1) is False
    # budget=0 doesn't.
    assert cmp.has_regression(regression_budget=0) is True


def test_compare_baselines_new_and_dropped_cases() -> None:
    result = _eval_with([(_row(case_id="new"), _verdict())])
    baseline = _baseline_for(cases=[{"id": "old", "verdict_pass": True, "citation_pass": True}])
    cmp = compare_baselines(baseline, result)
    assert cmp.new_cases == ("new",)
    assert cmp.dropped_cases == ("old",)


def test_compare_baselines_handles_missing_aggregate_field() -> None:
    """Older snapshot without `combined_accuracy` doesn't error out —
    just skips that aggregate comparison."""
    result = _eval_with([(_row(case_id="x"), _verdict())])
    baseline = {
        "verdict_accuracy": 1.0,
        # no citation_accuracy, no combined_accuracy
        "cases": [{"id": "x", "verdict_pass": True, "citation_pass": True}],
    }
    cmp = compare_baselines(baseline, result)
    metrics = {d.metric for d in cmp.aggregate_deltas}
    assert metrics == {"verdict_accuracy"}


# ---------- Dataclass shape ----------


def test_baseline_comparison_dataclass_has_expected_fields() -> None:
    cmp = BaselineComparison(
        case_deltas=(),
        aggregate_deltas=(),
        new_cases=(),
        dropped_cases=(),
    )
    assert cmp.case_regressions == ()
    assert cmp.aggregate_regressions == ()


def test_case_pass_delta_no_change_is_neither_regression_nor_improvement() -> None:
    delta = CasePassDelta(
        id="x",
        old_verdict_pass=True,
        new_verdict_pass=True,
        old_citation_pass=True,
        new_citation_pass=True,
    )
    assert delta.is_regression is False
    assert delta.is_improvement is False


def test_aggregate_delta_equality_is_not_regression() -> None:
    d = AggregateAccuracyDelta(metric="x", old=0.5, new=0.5)
    assert d.is_regression is False
