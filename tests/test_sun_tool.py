"""Tests for the `sun` reckon-profile tool — harness-jqep."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from harness.tools.sun import SunTool


def _extract(out: str, key: str) -> str:
    """Pull the value off a '<key> : <value>' line."""
    for line in out.splitlines():
        if line.startswith(key):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"missing {key!r} in output:\n{out}")


def test_sun_spec_shape() -> None:
    spec = SunTool().spec
    assert spec.name == "sun"
    assert spec.tier == "read"
    props = spec.parameters["properties"]
    assert set(props) == {"date", "lat", "lon", "tz"}
    assert sorted(spec.parameters["required"]) == ["date", "lat", "lon"]


def test_sun_phoenix_summer_solstice_sunset_within_one_minute() -> None:
    # Phoenix (33.4484, -112.0740), 2026-06-20 (summer solstice). NOAA-
    # style authoritative value: sunset is ~19:41 MST. Allow 60 s slack
    # for atmospheric-refraction model differences.
    out = SunTool().call(
        date="2026-06-20",
        lat=33.4484,
        lon=-112.0740,
        tz="America/Phoenix",
    )
    sunset_iso = _extract(out, "sunset")
    sunset = datetime.fromisoformat(sunset_iso)
    expected = datetime(2026, 6, 20, 19, 41, 0, tzinfo=ZoneInfo("America/Phoenix"))
    delta = abs((sunset - expected).total_seconds())
    assert delta <= 60, f"sunset off by {delta}s — got {sunset}, expected ~{expected}"


def test_sun_polar_night_raises_clear_error() -> None:
    # Tromsø, Norway in deep polar night. astral raises 'sun always below
    # horizon'; the tool re-raises a clearer ValueError that names the
    # location and date so the model can report instead of fabricating.
    with pytest.raises(ValueError, match="no sun events available"):
        SunTool().call(
            date="2026-12-21",
            lat=69.6492,
            lon=18.9553,
            tz="Europe/Oslo",
        )


def test_sun_southern_hemisphere_winter_short_day() -> None:
    # Sydney winter solstice 2026-06-21. Daylight should be ~9h50m,
    # noticeably shorter than the 12h equinox baseline.
    out = SunTool().call(
        date="2026-06-21",
        lat=-33.8688,
        lon=151.2093,
        tz="Australia/Sydney",
    )
    day_length = _extract(out, "day_length")
    assert day_length.startswith("9h"), f"expected 9-hour day, got {day_length}"
    # Sanity: sunrise wall time should be a Sydney winter morning (06:30-07:30).
    sunrise = datetime.fromisoformat(_extract(out, "sunrise"))
    assert 6 <= sunrise.hour <= 7


def test_sun_rejects_bad_latitude() -> None:
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        SunTool().call(date="2026-06-20", lat=120.0, lon=0.0)


def test_sun_rejects_bad_longitude() -> None:
    with pytest.raises(ValueError, match=r"lon .* out of range"):
        SunTool().call(date="2026-06-20", lat=0.0, lon=200.0)


def test_sun_rejects_non_numeric_lat() -> None:
    with pytest.raises(ValueError, match="lat must be a number"):
        SunTool().call(date="2026-06-20", lat="33", lon=-112.0)  # type: ignore[arg-type]


def test_sun_rejects_bool_as_lat() -> None:
    # bool is a subclass of int — silently coercing True->1.0 would mask
    # a model bug.
    with pytest.raises(ValueError, match="lat must be a number"):
        SunTool().call(date="2026-06-20", lat=True, lon=0.0)


def test_sun_rejects_unknown_tz() -> None:
    with pytest.raises(ValueError, match="unknown timezone"):
        SunTool().call(
            date="2026-06-20",
            lat=33.4484,
            lon=-112.0740,
            tz="Not/A/Zone",
        )


def test_sun_rejects_malformed_date() -> None:
    with pytest.raises(ValueError, match="not a valid ISO date"):
        SunTool().call(date="next Tuesday", lat=0.0, lon=0.0)


def test_sun_default_tz_when_omitted() -> None:
    # default_tz field defaults to America/Phoenix; tool emits times in
    # that zone when tz is omitted.
    out = SunTool().call(date="2026-06-20", lat=33.4484, lon=-112.0740)
    assert "tz                   : America/Phoenix" in out


def test_sun_headline_names_target_timezone() -> None:
    out = SunTool().call(
        date="2026-06-20",
        lat=51.5072,
        lon=-0.1276,
        tz="Europe/London",
    )
    last_line = out.strip().splitlines()[-1]
    assert "Europe/London" in last_line
    assert "Sunset" in last_line
    assert "day length" in last_line


def test_sun_output_includes_all_six_events() -> None:
    out = SunTool().call(
        date="2026-06-20",
        lat=33.4484,
        lon=-112.0740,
        tz="America/Phoenix",
    )
    for key in (
        "civil_twilight_begin",
        "sunrise",
        "solar_noon",
        "sunset",
        "civil_twilight_end",
        "day_length",
    ):
        assert key in out, f"missing key {key!r}"
