"""Tests for the `tz_convert` reckon-profile tool — harness-2668."""

from __future__ import annotations

import pytest

from harness.tools.tz_convert import TzConvertTool


def test_tz_convert_spec_shape() -> None:
    tool = TzConvertTool()
    spec = tool.spec
    assert spec.name == "tz_convert"
    assert spec.tier == "read"
    props = spec.parameters["properties"]
    assert set(props) == {"when", "from_tz", "to_tz"}
    assert spec.parameters["required"] == ["when", "to_tz"]


def test_phoenix_to_london_no_dst_shift() -> None:
    # 2026-05-18 15:00 Phoenix (MST, -07:00, no DST) -> London (BST, +01:00).
    tool = TzConvertTool()
    out = tool.call(
        when="2026-05-18T15:00:00",
        from_tz="America/Phoenix",
        to_tz="Europe/London",
    )
    assert "input_iso   : 2026-05-18T15:00:00-07:00" in out
    assert "output_iso  : 2026-05-18T23:00:00+01:00" in out
    assert "to_tz       : Europe/London" in out
    assert "to_offset   : +01:00" in out
    assert "day_of_week : Monday" in out
    assert "day_shift   : no shift" in out


def test_tokyo_to_nyc_same_day_when_tokyo_evening() -> None:
    # 22:00 Tokyo Monday is 09:00 NYC Monday — same calendar day in both
    # zones because the time difference (-13 in summer) lands within
    # the same date window.
    tool = TzConvertTool()
    out = tool.call(
        when="2026-05-18T22:00:00",
        from_tz="Asia/Tokyo",
        to_tz="America/New_York",
    )
    assert "output_iso  : 2026-05-18T09:00:00-04:00" in out
    assert "day_shift   : no shift" in out
    assert "day_of_week : Monday" in out


def test_nyc_evening_to_tokyo_rolls_forward_a_day() -> None:
    # 22:00 NYC Sunday is 11:00 Tokyo Monday — calendar date advances.
    tool = TzConvertTool()
    out = tool.call(
        when="2026-05-17T22:00:00",  # Sunday in NYC
        from_tz="America/New_York",
        to_tz="Asia/Tokyo",
    )
    assert "output_iso  : 2026-05-18T11:00:00+09:00" in out
    assert "day_shift   : +1 day (Sun -> Mon)" in out
    assert "day_of_week : Monday" in out


def test_negative_day_shift_when_crossing_back() -> None:
    # 01:00 Tokyo Monday = 12:00 NYC Sunday — date rolls back.
    tool = TzConvertTool()
    out = tool.call(
        when="2026-05-18T01:00:00",
        from_tz="Asia/Tokyo",
        to_tz="America/New_York",
    )
    assert "output_iso  : 2026-05-17T12:00:00-04:00" in out
    assert "day_shift   : -1 day (Mon -> Sun)" in out


def test_dst_spring_forward_us_eastern() -> None:
    # 2026-03-08 is the US DST start. 02:30 ET doesn't exist locally;
    # we pass an explicit-offset input to make the conversion well-defined
    # and verify the target zone math matches.
    # 06:30 UTC on the spring-forward morning = 02:30 EST (pre-shift,
    # -05:00) but DST started at 02:00 -> jump to 03:00 EDT (-04:00).
    tool = TzConvertTool()
    out = tool.call(
        when="2026-03-08T07:30:00+00:00",
        from_tz="UTC",
        to_tz="America/New_York",
    )
    # 07:30 UTC after the DST transition is 03:30 EDT.
    assert "output_iso  : 2026-03-08T03:30:00-04:00" in out
    assert "to_offset   : -04:00" in out


def test_dst_fall_back_uk() -> None:
    # 2026-10-25 02:30 UTC: the UK falls back at 01:00 UTC (= 02:00 BST
    # -> 01:00 GMT). After the transition, BST -> GMT (offset +00:00).
    tool = TzConvertTool()
    out = tool.call(
        when="2026-10-25T02:30:00+00:00",
        to_tz="Europe/London",
    )
    assert "output_iso  : 2026-10-25T02:30:00+00:00" in out
    assert "to_offset   : +00:00" in out


def test_input_with_offset_ignores_from_tz_for_math() -> None:
    # The input already carries -07:00; the converter trusts it. We
    # still accept (and validate) from_tz as informational so the
    # output's `from_tz` label is human-readable.
    tool = TzConvertTool()
    out = tool.call(
        when="2026-05-18T15:00:00-07:00",
        from_tz="America/Phoenix",
        to_tz="UTC",
    )
    assert "input_iso   : 2026-05-18T15:00:00-07:00" in out
    assert "output_iso  : 2026-05-18T22:00:00+00:00" in out
    assert "from_tz     : America/Phoenix" in out


def test_input_with_offset_works_without_from_tz() -> None:
    tool = TzConvertTool()
    out = tool.call(
        when="2026-05-18T15:00:00-07:00",
        to_tz="UTC",
    )
    assert "output_iso  : 2026-05-18T22:00:00+00:00" in out
    # No from_tz supplied — label falls back to the offset.
    assert "from_tz     : UTC-07:00" in out


def test_naive_datetime_without_from_tz_raises() -> None:
    tool = TzConvertTool()
    with pytest.raises(ValueError, match="naive"):
        tool.call(when="2026-05-18T15:00:00", to_tz="UTC")


def test_unknown_to_tz_raises_with_field_label() -> None:
    tool = TzConvertTool()
    with pytest.raises(ValueError, match="'to_tz'"):
        tool.call(
            when="2026-05-18T15:00:00",
            from_tz="UTC",
            to_tz="Not/A/Zone",
        )


def test_unknown_from_tz_raises_with_field_label() -> None:
    tool = TzConvertTool()
    with pytest.raises(ValueError, match="'from_tz'"):
        tool.call(
            when="2026-05-18T15:00:00",
            from_tz="Not/A/Zone",
            to_tz="UTC",
        )


def test_malformed_datetime_string_raises() -> None:
    tool = TzConvertTool()
    with pytest.raises(ValueError, match="not a valid ISO datetime"):
        tool.call(when="next Tuesday", from_tz="UTC", to_tz="UTC")


def test_headline_names_the_target_timezone() -> None:
    tool = TzConvertTool()
    out = tool.call(
        when="2026-05-18T15:00:00",
        from_tz="America/Phoenix",
        to_tz="Europe/London",
    )
    last_line = out.strip().splitlines()[-1]
    assert "Europe/London" in last_line
    assert "Monday" in last_line
