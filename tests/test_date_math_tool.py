"""Tests for the `date_math` reckon-profile tool — harness-6mh8."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from harness.tools.date_math import DateMathTool


def _frozen_clock(when: datetime) -> Callable[[ZoneInfo], datetime]:
    def _clock(tz: ZoneInfo) -> datetime:
        return when.astimezone(tz)

    return _clock


# 2026-05-15 was a Friday; the tool's "today" comes from the frozen clock.
_FRIDAY = datetime(2026, 5, 15, 14, 32, 0, tzinfo=ZoneInfo("America/Phoenix"))


def _tool() -> DateMathTool:
    return DateMathTool(clock=_frozen_clock(_FRIDAY))


def test_spec_shape() -> None:
    t = _tool()
    spec = t.spec
    assert spec.name == "date_math"
    assert spec.tier == "read"
    assert "op" in spec.parameters["properties"]
    assert "args" in spec.parameters["properties"]


def test_parse_today_tomorrow_yesterday() -> None:
    t = _tool()
    assert t.call(op="parse", args={"expr": "today"}).startswith("2026-05-15")
    assert t.call(op="parse", args={"expr": "tomorrow"}).startswith("2026-05-16")
    assert t.call(op="parse", args={"expr": "yesterday"}).startswith("2026-05-14")


def test_parse_next_weekday() -> None:
    t = _tool()
    # next Thursday from Friday 2026-05-15 = 2026-05-21
    assert t.call(op="parse", args={"expr": "next Thursday"}).startswith("2026-05-21")
    # next Friday (same weekday) should skip a full week, not stay.
    assert t.call(op="parse", args={"expr": "next Friday"}).startswith("2026-05-22")


def test_parse_last_weekday() -> None:
    t = _tool()
    # last Friday from Friday 2026-05-15 = 2026-05-08
    assert t.call(op="parse", args={"expr": "last Friday"}).startswith("2026-05-08")
    # last Thursday = 2026-05-14
    assert t.call(op="parse", args={"expr": "last Thursday"}).startswith("2026-05-14")


def test_parse_in_n_days() -> None:
    t = _tool()
    assert t.call(op="parse", args={"expr": "in 3 days"}).startswith("2026-05-18")
    assert t.call(op="parse", args={"expr": "in 2 weeks"}).startswith("2026-05-29")
    assert t.call(op="parse", args={"expr": "in 1 month"}).startswith("2026-06-15")


def test_parse_n_ago() -> None:
    t = _tool()
    assert t.call(op="parse", args={"expr": "3 days ago"}).startswith("2026-05-12")
    assert t.call(op="parse", args={"expr": "1 week ago"}).startswith("2026-05-08")


def test_parse_iso() -> None:
    t = _tool()
    assert t.call(op="parse", args={"expr": "2026-12-25"}).startswith("2026-12-25")


def test_parse_natural_long_form() -> None:
    t = _tool()
    out = t.call(op="parse", args={"expr": "May 15 2026"})
    assert out.startswith("2026-05-15")


def test_parse_rejects_garbage() -> None:
    t = _tool()
    with pytest.raises(ValueError, match="could not parse"):
        t.call(op="parse", args={"expr": "not a date string at all asdfjkl"})


def test_parse_rejects_unknown_weekday() -> None:
    t = _tool()
    with pytest.raises(ValueError, match="unknown weekday"):
        t.call(op="parse", args={"expr": "next Frunday"})


def test_add_simple_delta() -> None:
    t = _tool()
    plus3w = t.call(op="add", args={"base": "2026-05-15", "delta": "+3 weeks"})
    minus1m = t.call(op="add", args={"base": "2026-05-15", "delta": "-1 month"})
    assert plus3w.startswith("2026-06-05")
    assert minus1m.startswith("2026-04-15")


def test_add_compound_delta() -> None:
    t = _tool()
    out = t.call(op="add", args={"base": "2026-05-15", "delta": "+3 weeks 2 days"})
    assert out.startswith("2026-06-07")


def test_add_unsigned_delta_means_positive() -> None:
    t = _tool()
    out = t.call(op="add", args={"base": "2026-05-15", "delta": "10 days"})
    assert out.startswith("2026-05-25")


def test_add_rejects_unknown_delta() -> None:
    t = _tool()
    # "two weeks" is rejected because the numeric word doesn't match the
    # `<int> <unit>` grammar — surfaces as a trailing-fragment error.
    with pytest.raises(ValueError, match="unrecognized"):
        t.call(op="add", args={"base": "2026-05-15", "delta": "two weeks"})


def test_add_rejects_empty_delta() -> None:
    t = _tool()
    with pytest.raises(ValueError, match="no recognized tokens"):
        t.call(op="add", args={"base": "2026-05-15", "delta": ""})


def test_diff_forward() -> None:
    t = _tool()
    out = t.call(op="diff", args={"a": "2026-05-15", "b": "2026-06-20"})
    assert "36 days" in out
    assert "5.14 weeks" in out


def test_diff_backward_signed() -> None:
    t = _tool()
    out = t.call(op="diff", args={"a": "2026-06-20", "b": "2026-05-15"})
    assert "-36 days" in out


def test_format_emits_strftime() -> None:
    t = _tool()
    out = t.call(op="format", args={"iso": "2026-05-15", "format": "%A, %B %d %Y"})
    assert out == "Friday, May 15 2026"


def test_format_accepts_datetime_iso() -> None:
    t = _tool()
    out = t.call(op="format", args={"iso": "2026-05-15T14:32:00", "format": "%H:%M"})
    assert out == "14:32"


def test_unknown_op_rejected() -> None:
    t = _tool()
    with pytest.raises(ValueError, match="op must be one of"):
        t.call(op="evaluate", args={})


def test_missing_args_rejected() -> None:
    t = _tool()
    with pytest.raises(ValueError, match="non-empty"):
        t.call(op="parse", args={})  # expr missing
    with pytest.raises(ValueError, match="must be strings"):
        t.call(op="add", args={"base": "2026-05-15"})  # delta missing
