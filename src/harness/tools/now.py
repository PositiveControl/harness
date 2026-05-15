"""now — deterministic current-time tool for the reckon profile.

Replaces "what's today's date?" / "what day of the week is it?" model
guesswork. The model emitting `now()` is a tool intent the orchestrator
can audit; the model paraphrasing the system-prompt date stamp is not.

Read-tier, side-effect free, schema-cheap (~120 tokens of overhead).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from harness.tools.base import ToolSpec

# Default timezone when the caller doesn't name one. Hardcoded to
# Mark's local; the per-character `core.yaml` could override this in a
# future iteration. For now, the constitution names it explicitly.
_DEFAULT_TZ = "America/Phoenix"

_FORMATS = ("iso", "rfc3339", "human")

# Indirection so tests can inject a fixed clock. Production wiring uses
# `datetime.now`; tests pass a frozen-time callable.
_Clock = Callable[[ZoneInfo], datetime]


def _wall_clock(tz: ZoneInfo) -> datetime:
    return datetime.now(tz=tz)


@dataclass
class NowTool:
    """Return the current wall-clock time as a structured record.

    Output format (always include unit / timezone in the human-readable
    last line; harness-1u2h enforces no bare scalars):

        iso         : 2026-05-15T14:32:00-07:00
        rfc3339     : 2026-05-15T14:32:00.000-07:00
        day_of_week : Friday
        week_of_year: 20
        tz          : America/Phoenix
        utc_offset  : -07:00
        unix_ts     : 1781636320

    Friday, 2:32 PM MST (America/Phoenix).
    """

    default_tz: str = _DEFAULT_TZ
    clock: _Clock = _wall_clock

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="now",
            description=(
                "Return the current wall-clock time. Use this for any "
                "question about the current date, day of week, or "
                "timezone — never paraphrase the system prompt's date "
                "stamp. The reply names the timezone."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tz": {
                        "type": "string",
                        "description": (
                            "IANA timezone name (e.g. 'America/Phoenix', "
                            "'UTC', 'Europe/London'). Default is the "
                            "character's configured local zone."
                        ),
                    },
                    "format": {
                        "type": "string",
                        "enum": list(_FORMATS),
                        "description": (
                            "Which formatted line to emphasize: 'iso' "
                            "(default), 'rfc3339', or 'human'."
                        ),
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Current time",
        )

    def call(self, *, tz: str | None = None, format: str = "iso") -> str:
        zone_name = tz or self.default_tz
        if format not in _FORMATS:
            raise ValueError(f"format must be one of {_FORMATS!r}, got {format!r}")
        try:
            zone = ZoneInfo(zone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"unknown timezone {zone_name!r} — must be an IANA name "
                f"like 'America/Phoenix' or 'UTC'"
            ) from exc

        wall = self.clock(zone)
        iso = wall.isoformat(timespec="seconds")
        rfc3339 = wall.isoformat(timespec="milliseconds")
        day_of_week = wall.strftime("%A")
        week_of_year = wall.isocalendar().week
        utc_offset = wall.strftime("%z")
        utc_offset_pretty = f"{utc_offset[:3]}:{utc_offset[3:]}" if utc_offset else ""
        unix_ts = int(wall.timestamp())

        # Build the "human" line in three forms; the model picks which
        # one to surface based on the `format` arg. All three carry
        # the timezone tag, per the dates_with_timezones value.
        if format == "human":
            headline = wall.strftime(f"%A, %-I:%M %p %Z ({zone_name})")
        elif format == "rfc3339":
            headline = rfc3339
        else:
            headline = iso

        lines = [
            f"iso         : {iso}",
            f"rfc3339     : {rfc3339}",
            f"day_of_week : {day_of_week}",
            f"week_of_year: {week_of_year:02d}",
            f"tz          : {zone_name}",
            f"utc_offset  : {utc_offset_pretty}",
            f"unix_ts     : {unix_ts}",
            "",
            headline,
        ]
        return "\n".join(lines)
