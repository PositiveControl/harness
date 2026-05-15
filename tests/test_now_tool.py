"""Tests for the `now` reckon-profile tool — harness-6zi9."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from harness.tools.now import NowTool


def _frozen_clock(when: datetime) -> Callable[[ZoneInfo], datetime]:
    """Returns a clock function that always reports `when`, in whatever
    target zone the tool requests. The frozen instant is converted to
    that zone so day-of-week / wall time still vary with tz."""

    def _clock(tz: ZoneInfo) -> datetime:
        return when.astimezone(tz)

    return _clock


def test_now_default_tz_is_phoenix() -> None:
    tool = NowTool()
    spec = tool.spec
    assert spec.name == "now"
    assert spec.tier == "read"
    assert "tz" in spec.parameters["properties"]
    assert tool.default_tz == "America/Phoenix"


def test_now_returns_structured_record() -> None:
    frozen = datetime(2026, 5, 15, 14, 32, 0, tzinfo=ZoneInfo("America/Phoenix"))
    tool = NowTool(clock=_frozen_clock(frozen))
    out = tool.call()
    assert "iso         : 2026-05-15T14:32:00-07:00" in out
    assert "day_of_week : Friday" in out
    assert "tz          : America/Phoenix" in out
    assert "utc_offset  : -07:00" in out


def test_now_honors_explicit_tz_override() -> None:
    frozen = datetime(2026, 5, 15, 21, 32, 0, tzinfo=ZoneInfo("UTC"))
    tool = NowTool(clock=_frozen_clock(frozen))
    out = tool.call(tz="UTC")
    assert "tz          : UTC" in out
    assert "21:32:00" in out


def test_now_rejects_unknown_tz() -> None:
    tool = NowTool()
    with pytest.raises(ValueError, match="unknown timezone"):
        tool.call(tz="Not/A/Zone")


def test_now_rejects_unknown_format() -> None:
    tool = NowTool()
    with pytest.raises(ValueError, match="format must be one of"):
        tool.call(format="hex")


def test_now_human_format_names_timezone() -> None:
    frozen = datetime(2026, 5, 15, 14, 32, 0, tzinfo=ZoneInfo("America/Phoenix"))
    tool = NowTool(clock=_frozen_clock(frozen))
    out = tool.call(format="human")
    # Headline (last line) must name the zone — dates_with_timezones value.
    last_line = out.strip().splitlines()[-1]
    assert "America/Phoenix" in last_line
    assert "Friday" in last_line


def test_now_iso_format_emphasis_matches_headline() -> None:
    frozen = datetime(2026, 5, 15, 14, 32, 0, tzinfo=ZoneInfo("America/Phoenix"))
    tool = NowTool(clock=_frozen_clock(frozen))
    out = tool.call(format="iso")
    assert out.strip().splitlines()[-1] == "2026-05-15T14:32:00-07:00"


def test_now_rfc3339_includes_milliseconds() -> None:
    frozen = datetime(2026, 5, 15, 14, 32, 0, tzinfo=ZoneInfo("America/Phoenix"))
    tool = NowTool(clock=_frozen_clock(frozen))
    out = tool.call(format="rfc3339")
    last_line = out.strip().splitlines()[-1]
    assert ".000" in last_line


def test_now_week_of_year_is_iso() -> None:
    # Jan 5 2026 lands in ISO week 2 (Mon-start, week with 4+ Jan days).
    frozen = datetime(2026, 1, 5, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    tool = NowTool(clock=_frozen_clock(frozen))
    out = tool.call(tz="UTC")
    assert "week_of_year: 02" in out
