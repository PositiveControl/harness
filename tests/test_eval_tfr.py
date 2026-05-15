"""Tests for the TFR/NOTAM eval (harness-3jz1.2).

Deterministic — no model, no network. Exercises the scoring logic
against in-memory fixture rows plus the real fixture at
`character/airton_c_tfr/tfr_eval.yaml` to lock in baseline accuracy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.evals.tfr import (
    ExpectedGeometry,
    default_fixture_path,
    load_fixture,
    run_tfr_eval,
    score_case,
    score_citations,
    score_geometry,
)
from harness.notam import Altitude, Coordinate, parse_notam

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_FIXTURE = _REPO_ROOT / "character" / "airton_c_tfr" / "tfr_eval.yaml"


def _stadium_row(**overrides: object) -> dict[str, object]:
    """Build a fixture row matching the stadium-globe-life-field case
    so individual tests can perturb one field and assert the verdict
    flips."""
    base: dict[str, object] = {
        "id": "test-stadium",
        "raw_notam": (
            "PURSUANT TO 14 CFR SECTION 91.145 TFR FOR MAJOR SPORTING "
            "EVENT. WITHIN 3 NMR OF 324504N/0970454W SFC TO 3000FT AGL. "
            "EFFECTIVE 2605142305-2605150330 UTC."
        ),
        "expected_verdict": "Stadium TFR — 3 NM cylinder.",
        "expected_type": "stadium",
        "expected_citations": ["§91.145"],
        "expected_geometry": {
            "center": {"lat": 32.7511, "lon": -97.0817},
            "radius_nm": 3.0,
            "floor": {"value": 0, "reference": "SFC"},
            "ceiling": {"value": 3000, "reference": "AGL"},
        },
    }
    base.update(overrides)
    return base


def test_score_case_passes_when_parser_matches_expectations() -> None:
    case = score_case(_stadium_row())
    assert case.passed
    assert case.type_correct
    assert case.geometry_correct
    assert case.citations_correct


def test_score_case_fails_type_mismatch() -> None:
    case = score_case(_stadium_row(expected_type="vip"))
    assert case.type_correct is False
    assert case.geometry_correct
    assert case.citations_correct
    assert case.passed is False


def test_score_case_fails_citation_mismatch() -> None:
    # Expect a §-anchor the source text doesn't carry. Set-equality
    # scoring fails because the parsed set is {"§91.145"} not {"§91.141"}.
    case = score_case(_stadium_row(expected_citations=["§91.141"]))
    assert case.citations_correct is False
    assert case.passed is False


def test_score_case_fails_geometry_radius_out_of_tolerance() -> None:
    # Parsed radius is 3.0 NM; expected 5.0 is 2.0 outside the 0.1 NM
    # tolerance.
    case = score_case(
        _stadium_row(
            expected_geometry={
                "center": {"lat": 32.7511, "lon": -97.0817},
                "radius_nm": 5.0,
                "floor": {"value": 0, "reference": "SFC"},
                "ceiling": {"value": 3000, "reference": "AGL"},
            }
        )
    )
    assert case.geometry_correct is False
    assert case.passed is False


def test_score_case_passes_geometry_within_tolerance() -> None:
    # 0.003° off center is within the 0.005° tolerance.
    case = score_case(
        _stadium_row(
            expected_geometry={
                "center": {"lat": 32.7541, "lon": -97.0847},
                "radius_nm": 3.05,
                "floor": {"value": 0, "reference": "SFC"},
                "ceiling": {"value": 3000, "reference": "AGL"},
            }
        )
    )
    assert case.geometry_correct
    assert case.passed


def test_score_geometry_partial_expected_skips_unpinned_fields() -> None:
    # Pin only center + radius; floor/ceiling left unset on expected.
    # Parser produces values for all four — the unpinned fields should
    # not be scored against.
    parsed = parse_notam(
        "TFR PER §91.145 WITHIN 3 NMR OF 324504N/0970454W SFC TO 3000FT AGL. "
        "EFFECTIVE 2605142305-2605150330."
    )
    expected = ExpectedGeometry(
        center=Coordinate(lat=32.7511, lon=-97.0817),
        radius_nm=3.0,
    )
    assert score_geometry(parsed, expected)


def test_score_geometry_floor_mismatch_fails_when_pinned() -> None:
    parsed = parse_notam("TFR PER §91.145 WITHIN 3 NMR OF 324504N/0970454W SFC TO 3000FT AGL.")
    # Pin floor to MSL — parsed is SFC, so should fail.
    expected = ExpectedGeometry(
        center=Coordinate(lat=32.7511, lon=-97.0817),
        radius_nm=3.0,
        floor=Altitude(value=1000, reference="MSL"),
    )
    assert score_geometry(parsed, expected) is False


def test_score_citations_empty_expected_is_no_check() -> None:
    # Empty expected_citations = caller doesn't care; always pass.
    parsed = parse_notam("TFR PER §91.145 WITHIN 3 NMR OF 324504N/0970454W.")
    assert score_citations(parsed, ()) is True


def test_score_citations_order_independent() -> None:
    parsed = parse_notam("TFR PER §91.137(a)(1). SEE AIM 3-5-3. WITHIN 5 NMR OF 350000N/0900000W.")
    # Same set, different order — should pass.
    assert score_citations(parsed, ("AIM 3-5-3", "§91.137(a)(1)")) is True


def test_score_case_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="expected_type"):
        score_case(_stadium_row(expected_type="not-a-real-type"))


def test_score_case_rejects_non_list_citations() -> None:
    with pytest.raises(ValueError, match="expected_citations"):
        score_case(_stadium_row(expected_citations="just a string"))


def test_score_case_rejects_bad_altitude_reference() -> None:
    bad = _stadium_row(
        expected_geometry={
            "center": {"lat": 32.7511, "lon": -97.0817},
            "radius_nm": 3.0,
            "floor": {"value": 0, "reference": "QFE"},
            "ceiling": {"value": 3000, "reference": "AGL"},
        }
    )
    with pytest.raises(ValueError, match="SFC/AGL/MSL/FL"):
        score_case(bad)


def test_run_tfr_eval_aggregates_correctly() -> None:
    rows: tuple[dict[str, object], ...] = (
        _stadium_row(id="case-1"),
        _stadium_row(id="case-2", expected_type="vip"),  # type fails
        _stadium_row(id="case-3", expected_citations=["§91.141"]),  # cite fails
    )
    result = run_tfr_eval(rows)
    assert len(result.cases) == 3
    assert result.pass_rate == pytest.approx(1 / 3)
    assert result.type_accuracy == pytest.approx(2 / 3)
    assert result.citation_accuracy == pytest.approx(2 / 3)
    assert result.geometry_accuracy == 1.0  # all three have correct geometry
    failures = result.failures()
    assert {c.case_id for c in failures} == {"case-2", "case-3"}


def test_load_fixture_real_yaml_loads_without_error() -> None:
    if not _REAL_FIXTURE.exists():
        pytest.skip(f"fixture not present at {_REAL_FIXTURE}")
    rows = load_fixture(_REAL_FIXTURE)
    assert len(rows) >= 20, f"expected ≥20 fixture rows, got {len(rows)}"


def test_real_fixture_parser_accuracy_meets_bar() -> None:
    """Lock in the parser-axis pass rate against the curated fixture.
    Drops if the parser regresses on a known case OR if the fixture is
    edited in a way that breaks an existing expectation."""
    if not _REAL_FIXTURE.exists():
        pytest.skip(f"fixture not present at {_REAL_FIXTURE}")
    rows = load_fixture(_REAL_FIXTURE)
    result = run_tfr_eval(rows)
    assert result.pass_rate >= 0.85, (
        f"parser pass rate {result.pass_rate:.2%} below ≥0.85 bar. "
        f"Failures: {[c.case_id for c in result.failures()]}"
    )


def test_default_fixture_path_uses_character_dir(tmp_path: Path) -> None:
    expected = tmp_path / "tfr_eval.yaml"
    assert default_fixture_path(tmp_path) == expected
