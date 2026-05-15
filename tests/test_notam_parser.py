"""Tests for the deterministic NOTAM/TFR parser (harness-3jz1.3).

Deterministic — no LLM, no live FAA data. Fixtures are synthesized
NOTAM-shape strings exercising the regex pipeline. The actual 20-30
golden TFR cases land in `character/airton_c_tfr/tfr_eval.yaml` under
harness-3jz1.2 and drive the integration eval; these tests guard the
parser shape (coords, radius, altitudes, window, citations, type guess,
polygon math) and edge cases (decimal-coord fallback, SFC-only floor,
single-altitude ambiguity, missing fields).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from harness.notam import (
    ActiveWindow,
    Altitude,
    Coordinate,
    ParsedNOTAM,
    parse_notam,
)

# Stadium TFR shape (§91.145). Globe Life Field, Arlington TX.
_STADIUM_TFR = (
    "PURSUANT TO 14 CFR SECTION 91.145 TEMPORARY FLIGHT RESTRICTIONS "
    "FOR MAJOR SPORTING EVENT. AIRCRAFT FLIGHT OPERATIONS ARE "
    "PROHIBITED WITHIN AN AREA DEFINED AS 3 NMR OF 324504N/0970454W "
    "SFC TO 3000FT AGL. EFFECTIVE 2605142305-2605150330 UTC."
)

# VIP TFR shape (§91.141).
_VIP_TFR = (
    "PURSUANT TO 14 CFR SECTION 91.141 TEMPORARY FLIGHT RESTRICTIONS "
    "DUE TO VIP MOVEMENT. AIRCRAFT FLIGHT OPERATIONS ARE PROHIBITED "
    "WITHIN 10 NMR OF 384154N/0863912W SFC TO FL180. "
    "EFFECTIVE 2606010000-2606012359 UTC."
)

# Disaster TFR shape (§91.137(a)(1)) — wildfire scenario.
_DISASTER_TFR = (
    "PURSUANT TO 14 CFR SECTION 91.137(a)(1) TEMPORARY FLIGHT "
    "RESTRICTIONS FOR DISASTER RELIEF OPERATIONS DUE TO WILDFIRE. "
    "AIRCRAFT FLIGHT OPERATIONS PROHIBITED WITHIN 5 NM RADIUS OF "
    "390000N/1200000W. SFC TO 8500FT MSL. "
    "EFFECTIVE 2607010000-2607080000 UTC. SEE AIM 3-5-3."
)

# Space launch shape (§91.143).
_SPACE_TFR = (
    "PURSUANT TO 14 CFR SECTION 91.143 SPACE FLIGHT OPERATIONS. "
    "AIRCRAFT FLIGHT OPERATIONS PROHIBITED IN AREA WITHIN 30 NMR OF "
    "281245N/0803156W SFC TO UNLIMITED. ROCKET LAUNCH WINDOW "
    "2605160100-2605160300 UTC."
)


def test_parse_stadium_tfr_extracts_full_struct() -> None:
    parsed = parse_notam(_STADIUM_TFR)
    assert parsed.type_guess == "stadium"
    assert parsed.center is not None
    assert parsed.center.lat == pytest.approx(32.7511, abs=1e-4)
    assert parsed.center.lon == pytest.approx(-97.0817, abs=1e-4)
    assert parsed.radius_nm == 3.0
    assert parsed.floor == Altitude(value=0, reference="SFC")
    assert parsed.ceiling == Altitude(value=3000, reference="AGL")
    assert parsed.active is not None
    assert parsed.active.start == datetime(2026, 5, 14, 23, 5, tzinfo=UTC)
    assert parsed.active.end == datetime(2026, 5, 15, 3, 30, tzinfo=UTC)
    assert "§91.145" in parsed.cited_sections


def test_parse_vip_tfr_with_fl_ceiling() -> None:
    parsed = parse_notam(_VIP_TFR)
    assert parsed.type_guess == "vip"
    assert parsed.radius_nm == 10.0
    assert parsed.floor == Altitude(value=0, reference="SFC")
    assert parsed.ceiling == Altitude(value=18000, reference="FL")
    assert "§91.141" in parsed.cited_sections


def test_parse_disaster_tfr_with_msl_ceiling_and_aim_cite() -> None:
    parsed = parse_notam(_DISASTER_TFR)
    assert parsed.type_guess == "disaster"
    assert parsed.center is not None
    assert parsed.center.lat == pytest.approx(39.0, abs=1e-4)
    assert parsed.center.lon == pytest.approx(-120.0, abs=1e-4)
    assert parsed.radius_nm == 5.0
    assert parsed.floor == Altitude(value=0, reference="SFC")
    assert parsed.ceiling == Altitude(value=8500, reference="MSL")
    assert "§91.137(a)(1)" in parsed.cited_sections
    assert "AIM 3-5-3" in parsed.cited_sections


def test_parse_space_tfr_keyword_beats_citation() -> None:
    # Both ROCKET LAUNCH keywords and §91.143 cite — keyword takes
    # precedence because keywords are checked first in _TYPE_KEYWORDS.
    parsed = parse_notam(_SPACE_TFR)
    assert parsed.type_guess == "space"
    assert parsed.radius_nm == 30.0


def test_parse_decimal_coord_fallback() -> None:
    text = "TFR PER 14 CFR SECTION 91.139 CENTER 32.7511N 097.0825W 3 NMR SFC TO 3000FT AGL"
    parsed = parse_notam(text)
    assert parsed.center is not None
    assert parsed.center.lat == pytest.approx(32.7511, abs=1e-4)
    # 097.0825W → -97.0825 (W = negative). The parser preserves the
    # sign rule even with an explicit W suffix.
    assert parsed.center.lon == pytest.approx(-97.0825, abs=1e-4)
    assert parsed.type_guess == "security"  # from §91.139 cite hint


def test_type_guess_falls_back_to_citation_when_no_keywords() -> None:
    # No category keywords — only a §-cite to disambiguate.
    text = (
        "PURSUANT TO 14 CFR SECTION 91.144 TFR. WITHIN 5 NMR OF "
        "350000N/0900000W SFC TO 5000FT MSL. EFFECTIVE 2608010000-2608020000."
    )
    parsed = parse_notam(text)
    assert parsed.type_guess == "hazard"


def test_type_guess_other_when_nothing_matches() -> None:
    parsed = parse_notam("This is not a NOTAM. Just words.")
    assert parsed.type_guess == "other"
    assert parsed.center is None
    assert parsed.radius_nm is None
    assert parsed.floor is None
    assert parsed.ceiling is None
    assert parsed.active is None
    assert parsed.cited_sections == ()


def test_single_altitude_treated_as_floor_when_no_sfc() -> None:
    # When the NOTAM gives one numeric altitude and no SFC, treat that
    # altitude as the floor — the ceiling is unspecified (None).
    text = "TFR WITHIN 5 NMR OF 350000N/0900000W 2000FT AGL. CITES 91.137."
    parsed = parse_notam(text)
    assert parsed.floor == Altitude(value=2000, reference="AGL")
    assert parsed.ceiling is None


def test_dms_coord_southern_hemisphere() -> None:
    # Negative lat/lon resolution: 100000S/0700000W → -10.0, -70.0
    text = "TFR WITHIN 3 NMR OF 100000S/0700000W SFC TO 3000FT AGL."
    parsed = parse_notam(text)
    assert parsed.center is not None
    assert parsed.center.lat == pytest.approx(-10.0, abs=1e-4)
    assert parsed.center.lon == pytest.approx(-70.0, abs=1e-4)


def test_parsed_notam_to_polygon_has_n_points_and_radius() -> None:
    pyproj = pytest.importorskip("pyproj")  # noqa: F841 — guard, not used
    parsed = parse_notam(_STADIUM_TFR)
    polygon = parsed.to_polygon(n_points=32)
    assert len(polygon) == 32
    # Every vertex sits approximately radius_nm from the center.
    assert parsed.center is not None
    assert parsed.radius_nm is not None
    expected_m = parsed.radius_nm * 1852.0
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    for vertex in polygon:
        _, _, distance = geod.inv(parsed.center.lon, parsed.center.lat, vertex.lon, vertex.lat)
        assert distance == pytest.approx(expected_m, rel=1e-3)


def test_to_polygon_empty_without_center_or_radius() -> None:
    bare = ParsedNOTAM(
        raw_text="x",
        center=None,
        radius_nm=None,
        floor=None,
        ceiling=None,
        active=None,
        type_guess="other",
    )
    assert bare.to_polygon() == ()


def test_radius_nmr_and_nm_radius_both_match() -> None:
    nmr = parse_notam("WITHIN 4 NMR OF 350000N/0900000W SFC TO 3000FT AGL.")
    radius = parse_notam("WITHIN 4 NM RADIUS OF 350000N/0900000W SFC TO 3000FT AGL.")
    assert nmr.radius_nm == 4.0
    assert radius.radius_nm == 4.0


def test_multiple_citations_dedup_in_order() -> None:
    text = (
        "TFR PER §91.137(a)(1) AND §91.137(a)(1). "
        "SEE ALSO AIM 3-5-3 AND AIM 3-5-3. "
        "WITHIN 5 NMR OF 350000N/0900000W SFC TO 3000FT AGL. "
        "EFFECTIVE 2605142305-2605150330."
    )
    parsed = parse_notam(text)
    assert parsed.cited_sections == ("§91.137(a)(1)", "AIM 3-5-3")


def test_active_window_with_to_separator() -> None:
    # NOTAMs sometimes write "FROM ... TO ..." instead of a dash.
    text = (
        "TFR PER §91.145 WITHIN 3 NMR OF 324504N/0970454W SFC TO 3000FT AGL. "
        "EFFECTIVE 2605142305 TO 2605150330 UTC."
    )
    parsed = parse_notam(text)
    assert parsed.active == ActiveWindow(
        start=datetime(2026, 5, 14, 23, 5, tzinfo=UTC),
        end=datetime(2026, 5, 15, 3, 30, tzinfo=UTC),
    )


def test_invalid_date_returns_no_active_window() -> None:
    # 2613301200 is "year 26, month 13, day 30" — invalid month. Parser
    # should swallow the ValueError and emit active=None rather than
    # crash.
    text = (
        "TFR WITHIN 3 NMR OF 324504N/0970454W SFC TO 3000FT AGL. EFFECTIVE 2613301200-2613302359."
    )
    parsed = parse_notam(text)
    assert parsed.active is None


def test_coordinate_is_immutable() -> None:
    coord = Coordinate(lat=32.7511, lon=-97.0825)
    with pytest.raises(AttributeError):
        coord.lat = 0.0  # type: ignore[misc]
