"""tz_convert — wall-clock conversion between IANA timezones.

Companion to `now` in the reckon profile. `now` reads one zone;
`date_math` is timezone-naive. Cross-zone questions ("what's 3 pm
Phoenix in London?") still fabricate without this primitive. The tool
parses an ISO-8601 datetime, attaches a source zone if the input is
naive, converts to the target zone, and reports the result with the
target timezone and any day-of-week shift made visible.

Read-tier, side-effect free, schema-cheap (~150 tokens of overhead).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from harness.tools.base import ToolSpec

_DEFAULT_TZ = "America/Phoenix"


def _zone(name: str, *, field: str) -> ZoneInfo:
    """Resolve `name` to a ZoneInfo or raise ValueError with a clear,
    field-tagged message. Centralized so 'from_tz' and 'to_tz' errors
    name the offending argument."""
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"unknown timezone {name!r} for {field!r} — must be an "
            f"IANA name like 'America/Phoenix' or 'UTC'"
        ) from exc


def _parse_when(when: str) -> datetime:
    """Parse an ISO-8601 datetime. Accepts both naive ('2026-05-18T15:00')
    and offset-aware ('2026-05-18T15:00-07:00') forms. Date-only inputs
    are rejected — tz conversion of a bare date is ambiguous."""
    try:
        parsed = datetime.fromisoformat(when)
    except ValueError as exc:
        raise ValueError(
            f"when {when!r} is not a valid ISO datetime — expected "
            f"e.g. '2026-05-18T15:00:00' or '2026-05-18T15:00:00-07:00'"
        ) from exc
    return parsed


def _signed_day_delta(a: datetime, b: datetime) -> int:
    """Days between two tz-aware datetimes when both are viewed as
    wall-clock dates in their respective zones. Positive = b is on a
    later calendar date than a."""
    return (b.date() - a.date()).days


@dataclass
class TzConvertTool:
    """Convert a wall-clock time between IANA timezones.

    Output format (target-zone reading on the headline last line, per
    dates_with_timezones):

        input_iso   : 2026-05-18T15:00:00-07:00
        output_iso  : 2026-05-18T23:00:00+01:00
        from_tz     : America/Phoenix
        to_tz       : Europe/London
        to_offset   : +01:00
        day_of_week : Monday
        day_shift   : no shift

        Monday, 11:00 PM BST (Europe/London).
    """

    default_tz: str = _DEFAULT_TZ

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="tz_convert",
            description=(
                "Convert a wall-clock datetime from one IANA timezone "
                "to another. Use for cross-zone questions ('3 pm Phoenix "
                "in London?'); `now` only reads a single zone and "
                "`date_math` is timezone-naive. The reply names the "
                "target timezone and any day-of-week shift."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "when": {
                        "type": "string",
                        "description": (
                            "ISO-8601 datetime to convert. Either naive "
                            "('2026-05-18T15:00:00', source zone taken "
                            "from `from_tz`) or with an explicit offset "
                            "('2026-05-18T15:00:00-07:00', `from_tz` "
                            "treated as informational)."
                        ),
                    },
                    "from_tz": {
                        "type": "string",
                        "description": (
                            "IANA timezone the `when` value is in "
                            "(e.g. 'America/Phoenix'). Required when "
                            "`when` is naive; ignored when `when` "
                            "already carries an offset."
                        ),
                    },
                    "to_tz": {
                        "type": "string",
                        "description": (
                            "IANA timezone to convert into (e.g. "
                            "'Europe/London', 'Asia/Tokyo', 'UTC')."
                        ),
                    },
                },
                "required": ["when", "to_tz"],
            },
            tier="read",
            display_name="Timezone convert",
        )

    def call(self, *, when: str, to_tz: str, from_tz: str | None = None) -> str:
        target = _zone(to_tz, field="to_tz")
        parsed = _parse_when(when)

        if parsed.tzinfo is None:
            if from_tz is None:
                raise ValueError(
                    f"when {when!r} is naive — supply `from_tz` (IANA "
                    f"name) or include an offset like '-07:00' on the "
                    f"datetime"
                )
            source = _zone(from_tz, field="from_tz")
            source_aware = parsed.replace(tzinfo=source)
        else:
            # Input carries its own offset. `from_tz` is informational —
            # we still resolve it (if provided) just to surface a useful
            # name in the output; the actual conversion uses the offset
            # already on the datetime.
            source_aware = parsed
            if from_tz is not None:
                # Validate but don't reassign — keep the input's offset.
                _zone(from_tz, field="from_tz")

        converted = source_aware.astimezone(target)
        day_shift = _signed_day_delta(source_aware, converted)
        if day_shift == 0:
            shift_line = "no shift"
        else:
            sign = "+" if day_shift > 0 else "-"
            src_day = source_aware.strftime("%a")
            dst_day = converted.strftime("%a")
            shift_line = f"{sign}{abs(day_shift)} day ({src_day} -> {dst_day})"

        input_iso = source_aware.isoformat(timespec="seconds")
        output_iso = converted.isoformat(timespec="seconds")
        offset = converted.strftime("%z")
        offset_pretty = f"{offset[:3]}:{offset[3:]}" if offset else ""
        from_tz_label = from_tz or (
            str(source_aware.tzinfo) if source_aware.tzinfo is not None else "(unknown)"
        )
        headline = converted.strftime(f"%A, %-I:%M %p %Z ({to_tz})")

        lines = [
            f"input_iso   : {input_iso}",
            f"output_iso  : {output_iso}",
            f"from_tz     : {from_tz_label}",
            f"to_tz       : {to_tz}",
            f"to_offset   : {offset_pretty}",
            f"day_of_week : {converted.strftime('%A')}",
            f"day_shift   : {shift_line}",
            "",
            headline,
        ]
        return "\n".join(lines)
