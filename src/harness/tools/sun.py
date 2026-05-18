"""sun — sunrise / sunset / civil twilight / solar noon at lat/lon/date.

Tier 2 reckon-profile companion to `now` + `date_math`. Deterministic
astronomical reckoning for daily-life questions ("when does it get
dark?") and aviation-currency math (FAA night-flight rules anchor on
civil twilight + 1 hour). astral does the trig; this tool unifies the
output shape with `now` (structured record + headline) and the
dates_with_timezones value (every emitted time names its zone).

Read-tier, schema-cheap (~200 tokens). astral is pure-Python (~50 KB),
no network, no compiled deps — fits the local-first invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astral import Observer
from astral.sun import sun as _astral_sun

from harness.tools.base import ToolSpec

_DEFAULT_TZ = "America/Phoenix"


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"unknown timezone {name!r} — must be an IANA name like 'America/Phoenix' or 'UTC'"
        ) from exc


def _parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"date {s!r} is not a valid ISO date — expected 'YYYY-MM-DD'") from exc


def _format_hm(seconds: float) -> str:
    """Render a duration in seconds as 'Hh Mm' (e.g. '14h 22m')."""
    total = round(seconds)
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    return f"{hours}h {minutes:02d}m"


def _format_time(dt: datetime) -> str:
    """Trim microseconds and emit ISO-8601 with offset, dates_with_timezones-style."""
    return dt.replace(microsecond=0).isoformat()


@dataclass
class SunTool:
    """Sunrise / sunset / civil twilight / solar noon for a lat/lon/date.

    Output format (structured record + headline last line; the headline
    names the target timezone per dates_with_timezones):

        date                   : 2026-06-20
        lat                    : 33.4484
        lon                    : -112.074
        tz                     : America/Phoenix
        civil_twilight_begin   : 2026-06-20T04:49:31-07:00
        sunrise                : 2026-06-20T05:19:01-07:00
        solar_noon             : 2026-06-20T12:29:47-07:00
        sunset                 : 2026-06-20T19:40:55-07:00
        civil_twilight_end     : 2026-06-20T20:10:25-07:00
        day_length             : 14h 21m

        Sunset 7:40 PM MST (America/Phoenix).
    """

    default_tz: str = _DEFAULT_TZ

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="sun",
            description=(
                "Return sunrise, sunset, civil twilight (begin/end), "
                "solar noon, and day length for a given date at a "
                "lat/lon. Use for 'when does it get dark in X?' or "
                "FAA night-currency questions (civil twilight + 1 hr). "
                "Coordinates use signed decimal degrees; negative lon "
                "is west. Returns ISO-8601 times in the named timezone."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "ISO date 'YYYY-MM-DD' in the target timezone.",
                    },
                    "lat": {
                        "type": "number",
                        "description": ("Latitude in signed decimal degrees, range [-90, 90]."),
                    },
                    "lon": {
                        "type": "number",
                        "description": (
                            "Longitude in signed decimal degrees, range "
                            "[-180, 180]. Negative is west."
                        ),
                    },
                    "tz": {
                        "type": "string",
                        "description": (
                            "IANA timezone name (e.g. 'America/Phoenix', "
                            "'Europe/Oslo'). Default is the character's "
                            "configured local zone."
                        ),
                    },
                },
                "required": ["date", "lat", "lon"],
            },
            tier="read",
            display_name="Sun position",
        )

    def call(
        self,
        *,
        date: str,
        lat: float,
        lon: float,
        tz: str | None = None,
    ) -> str:
        if not isinstance(lat, (int, float)) or isinstance(lat, bool):
            raise ValueError(f"lat must be a number, got {type(lat).__name__}")
        if not isinstance(lon, (int, float)) or isinstance(lon, bool):
            raise ValueError(f"lon must be a number, got {type(lon).__name__}")
        if not -90.0 <= float(lat) <= 90.0:
            raise ValueError(f"lat {lat!r} out of range — must be in [-90, 90]")
        if not -180.0 <= float(lon) <= 180.0:
            raise ValueError(f"lon {lon!r} out of range — must be in [-180, 180]")

        zone_name = tz or self.default_tz
        zone = _zone(zone_name)
        target = _parse_date(date)
        observer = Observer(latitude=float(lat), longitude=float(lon))

        try:
            events = _astral_sun(observer, target, tzinfo=zone)
        except ValueError as exc:
            # astral raises ValueError for polar day/night and other
            # 'no sunrise today' conditions. Surface the message
            # verbatim so the model can report it instead of fabricating.
            raise ValueError(
                f"no sun events available at lat={lat}, lon={lon} on "
                f"{target.isoformat()} ({zone_name}): {exc}"
            ) from exc

        sunrise = events["sunrise"]
        sunset = events["sunset"]
        day_length = (sunset - sunrise).total_seconds()
        # Phoenix midwinter night when sun is up briefly — guard against
        # negative durations from edge cases astral didn't reject.
        if day_length < 0:
            day_length += 24 * 3600

        lines = [
            f"date                 : {target.isoformat()}",
            f"lat                  : {float(lat):g}",
            f"lon                  : {float(lon):g}",
            f"tz                   : {zone_name}",
            f"civil_twilight_begin : {_format_time(events['dawn'])}",
            f"sunrise              : {_format_time(sunrise)}",
            f"solar_noon           : {_format_time(events['noon'])}",
            f"sunset               : {_format_time(sunset)}",
            f"civil_twilight_end   : {_format_time(events['dusk'])}",
            f"day_length           : {_format_hm(day_length)}",
            "",
            sunset.strftime(f"Sunset %-I:%M %p %Z ({zone_name}); ")
            + f"day length {_format_hm(day_length)}.",
        ]
        return "\n".join(lines)
