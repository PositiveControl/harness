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


# --- business_days op (harness-kncj) -------------------------------------


def test_business_days_count_across_a_weekend() -> None:
    # 2026-05-15 (Fri) to 2026-05-18 (Mon). Inclusive of both endpoints,
    # weekends-only -> Fri + Mon = 2 business days, Sat+Sun = 2 weekend
    # skips, 0 holidays.
    t = _tool()
    out = t.call(
        op="business_days",
        args={"a": "2026-05-15", "b": "2026-05-18", "holidays": "none"},
    )
    assert out.startswith("2 business days")
    assert "weekend_skips=2" in out
    assert "holiday_skips=0" in out
    assert "holidays=none" in out


def test_business_days_count_skips_memorial_day() -> None:
    # 2026-05-22 (Fri) to 2026-05-26 (Tue), us_federal:
    # Fri ✓, Sat ✗, Sun ✗, Mon (Memorial Day) ✗, Tue ✓ -> 2 business days,
    # 2 weekend skips, 1 holiday skip.
    t = _tool()
    out = t.call(
        op="business_days",
        args={"a": "2026-05-22", "b": "2026-05-26", "holidays": "us_federal"},
    )
    assert out.startswith("2 business days")
    assert "weekend_skips=2" in out
    assert "holiday_skips=1" in out
    assert "holidays=us_federal" in out


def test_business_days_count_no_holidays_includes_memorial_day() -> None:
    # Same window, holidays='none' -> Memorial Day counts as a workday.
    t = _tool()
    out = t.call(
        op="business_days",
        args={"a": "2026-05-22", "b": "2026-05-26", "holidays": "none"},
    )
    assert out.startswith("3 business days")
    assert "holiday_skips=0" in out


def test_business_days_count_negative_when_a_after_b() -> None:
    # Reverse the window: a=Mon b=Fri previous week. Signed count is
    # negative; skip counts stay positive (real calendar days).
    t = _tool()
    out = t.call(
        op="business_days",
        args={"a": "2026-05-18", "b": "2026-05-15", "holidays": "none"},
    )
    assert out.startswith("-2 business days")
    assert "weekend_skips=2" in out


def test_business_days_count_same_day_business() -> None:
    t = _tool()
    out = t.call(
        op="business_days",
        args={"a": "2026-05-18", "b": "2026-05-18", "holidays": "none"},
    )
    assert out.startswith("1 business days")
    assert "weekend_skips=0" in out


def test_business_days_count_same_day_weekend() -> None:
    t = _tool()
    out = t.call(
        op="business_days",
        args={"a": "2026-05-16", "b": "2026-05-16", "holidays": "none"},
    )
    assert out.startswith("0 business days")
    assert "weekend_skips=1" in out


def test_business_days_advance_from_saturday_lands_on_monday() -> None:
    # 2026-05-16 is a Saturday. Snap-to-business semantics: delta=0
    # rolls forward to the next workday (Mon 2026-05-18).
    t = _tool()
    out = t.call(
        op="business_days",
        args={"base": "2026-05-16", "delta_business_days": 0, "holidays": "none"},
    )
    assert out.startswith("2026-05-18 (Monday)")
    assert "delta_business_days" not in out  # echo uses a humanized phrasing
    assert "+0 business days from 2026-05-16" in out


def test_business_days_advance_skips_holiday() -> None:
    # Step-forward semantics: base itself doesn't consume a step.
    # +10 business days from 2026-05-18 (Mon), us_federal: skips
    # Memorial Day 2026-05-25. Without us_federal +10 -> 2026-06-01;
    # with the holiday skipped, the count rolls one day later.
    t = _tool()
    with_federal = t.call(
        op="business_days",
        args={
            "base": "2026-05-18",
            "delta_business_days": 10,
            "holidays": "us_federal",
        },
    )
    without_federal = t.call(
        op="business_days",
        args={
            "base": "2026-05-18",
            "delta_business_days": 10,
            "holidays": "none",
        },
    )
    assert with_federal.startswith("2026-06-02 (Tuesday)")
    assert without_federal.startswith("2026-06-01 (Monday)")


def test_business_days_advance_negative() -> None:
    # -3 business days from Fri 2026-05-15: Thu, Wed, Tue -> 2026-05-12.
    t = _tool()
    out = t.call(
        op="business_days",
        args={"base": "2026-05-15", "delta_business_days": -3, "holidays": "none"},
    )
    assert out.startswith("2026-05-12 (Tuesday)")


def test_business_days_advance_from_saturday_with_negative_delta() -> None:
    # base=Sat 2026-05-16, delta=-1, holidays=none: snap back to Fri
    # 2026-05-15, then -1 business day = Thu 2026-05-14.
    t = _tool()
    out = t.call(
        op="business_days",
        args={"base": "2026-05-16", "delta_business_days": -1, "holidays": "none"},
    )
    assert out.startswith("2026-05-14 (Thursday)")


def test_business_days_unknown_holiday_set_raises() -> None:
    t = _tool()
    with pytest.raises(ValueError, match="holidays must be one of"):
        t.call(
            op="business_days",
            args={"a": "2026-05-15", "b": "2026-05-18", "holidays": "uk_bank"},
        )


def test_business_days_rejects_count_and_advance_in_same_call() -> None:
    t = _tool()
    with pytest.raises(ValueError, match=r"either \{a, b\} for count"):
        t.call(
            op="business_days",
            args={
                "a": "2026-05-15",
                "b": "2026-05-18",
                "delta_business_days": 3,
            },
        )


def test_business_days_advance_rejects_non_int_delta() -> None:
    t = _tool()
    with pytest.raises(ValueError, match="must be an integer"):
        t.call(
            op="business_days",
            args={"base": "2026-05-18", "delta_business_days": "10"},
        )


def test_business_days_advance_rejects_bool_delta() -> None:
    # bool is a subclass of int in Python, but using True for "advance by
    # True days" is almost certainly a bug — reject it explicitly.
    t = _tool()
    with pytest.raises(ValueError, match="must be an integer"):
        t.call(
            op="business_days",
            args={"base": "2026-05-18", "delta_business_days": True},
        )


def test_business_days_defaults_holidays_to_none_when_omitted() -> None:
    t = _tool()
    out = t.call(
        op="business_days",
        args={"a": "2026-05-22", "b": "2026-05-26"},
    )
    # Without us_federal, Memorial Day counts; we get 3 business days,
    # not 2.
    assert out.startswith("3 business days")
    assert "holidays=none" in out


def test_business_days_op_listed_in_spec_enum() -> None:
    t = _tool()
    enum = t.spec.parameters["properties"]["op"]["enum"]
    assert "business_days" in enum
