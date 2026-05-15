"""date_math — parse, add, diff, format dates and durations.

Companion to `now` in the reckon profile. The model emits `date_math`
when it needs to reason about future or past dates, time-between, or
day-of-week math. The tool handles four ops:

    parse  : "next Thursday" / "tomorrow" / "2026-05-15" -> iso date
    add    : base="2026-05-15", delta="+3 weeks 2 days" -> iso date
    diff   : a, b -> {days, weeks, months_approx, hours, minutes, seconds}
    format : iso, format="%A, %B %d %Y" -> formatted string

All ops return strings with explicit units / timezones tagged.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as _dt_parser
from dateutil.relativedelta import relativedelta

from harness.tools.base import ToolSpec

_DEFAULT_TZ = "America/Phoenix"
_OPS = ("parse", "add", "diff", "format")
# Re-exported indirection so a future per-character `default_tz`
# override can `from harness.tools.date_math import ZoneInfoNotFound`
# at validation time. Kept as a type alias rather than a new exception.
ZoneInfoNotFound = ZoneInfoNotFoundError

_WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

# delta token grammar: "<int> <unit>" repeated, optional leading +/-.
_DELTA_TOKEN = re.compile(
    r"(?P<n>\d+)\s*(?P<unit>year|month|week|day|hour|minute|second)s?\b",
    re.IGNORECASE,
)

_Clock = Callable[[ZoneInfo], datetime]


def _wall_clock(tz: ZoneInfo) -> datetime:
    return datetime.now(tz=tz)


def _parse_delta(delta: str) -> relativedelta:
    """Turn '+3 weeks 2 days' into a relativedelta. Raises ValueError
    if the string doesn't fully consume into known tokens."""
    text = delta.strip()
    sign = 1
    if text.startswith("+"):
        text = text[1:].strip()
    elif text.startswith("-"):
        sign = -1
        text = text[1:].strip()

    consumed: list[tuple[int, int, str]] = []
    pos = 0
    for match in _DELTA_TOKEN.finditer(text):
        if match.start() != pos and text[pos : match.start()].strip():
            raise ValueError(f"unrecognized fragment in delta: {text[pos : match.start()]!r}")
        n = int(match.group("n"))
        unit = match.group("unit").lower()
        consumed.append((match.start(), n, unit))
        pos = match.end()
    if pos != len(text) and text[pos:].strip():
        raise ValueError(f"unrecognized trailing in delta: {text[pos:]!r}")
    if not consumed:
        raise ValueError(
            f"delta {delta!r} has no recognized tokens — expected "
            f"e.g. '+3 weeks 2 days', '-1 hour 30 minutes', '1 year'"
        )

    kwargs: dict[str, int] = {}
    for _, n, unit in consumed:
        key = f"{unit}s"  # relativedelta uses plurals (years/months/...)
        kwargs[key] = kwargs.get(key, 0) + sign * n
    return relativedelta(**kwargs)  # type: ignore[arg-type]


def _parse_natural(expr: str, *, today: date) -> date:
    """Handle 'today' / 'tomorrow' / 'yesterday' / 'next <weekday>' /
    'last <weekday>' / 'in N days'. Falls back to dateutil.parser for
    iso / slashes / 'May 15 2026' etc. Returns a date (not datetime)."""
    text = expr.strip().lower()
    if text in {"today", "now"}:
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    if text == "yesterday":
        return today - timedelta(days=1)

    m = re.fullmatch(r"in\s+(\d+)\s+(day|week|month|year)s?", text)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "day":
            return today + timedelta(days=n)
        if unit == "week":
            return today + timedelta(weeks=n)
        return today + relativedelta(**{f"{unit}s": n})  # type: ignore[arg-type]

    m = re.fullmatch(r"(\d+)\s+(day|week|month|year)s?\s+ago", text)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "day":
            return today - timedelta(days=n)
        if unit == "week":
            return today - timedelta(weeks=n)
        return today - relativedelta(**{f"{unit}s": n})  # type: ignore[arg-type]

    m = re.fullmatch(r"(next|last)\s+([a-z]+)", text)
    if m:
        direction = m.group(1)
        wd_name = m.group(2)
        if wd_name not in _WEEKDAYS:
            raise ValueError(
                f"unknown weekday {wd_name!r} — expected one of {sorted(set(_WEEKDAYS))!r}"
            )
        target = _WEEKDAYS[wd_name]
        delta_days = (target - today.weekday()) % 7
        if direction == "next" and delta_days == 0:
            delta_days = 7
        if direction == "last":
            delta_days = -((today.weekday() - target) % 7)
            if delta_days == 0:
                delta_days = -7
        return today + timedelta(days=delta_days)

    # Fallback: dateutil parser. Returns a datetime; we want a date.
    try:
        parsed = _dt_parser.parse(expr, default=datetime(today.year, today.month, today.day))
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"could not parse date expression {expr!r}: {exc}") from exc
    return parsed.date()


@dataclass
class DateMathTool:
    """Date arithmetic + parsing tool.

    Args: `op` (parse/add/diff/format) + `args` (dict, shape varies):
        parse  : {"expr": "next Thursday"} -> "2026-05-21 (Thursday)"
        add    : {"base": "2026-05-15", "delta": "+3 weeks 2 days"}
        diff   : {"a": "2026-05-15", "b": "2026-06-20"}
        format : {"iso": "2026-05-15", "format": "%A, %B %d %Y"}
    """

    default_tz: str = _DEFAULT_TZ
    clock: _Clock = _wall_clock

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="date_math",
            description=(
                "Parse natural-language dates, add/subtract durations, "
                "diff two dates, or reformat. Use for any date math "
                "beyond a single read of 'today' (which goes to `now`)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": list(_OPS),
                        "description": (
                            "parse: natural-language -> iso date. "
                            "add: base + delta -> iso date. "
                            "diff: returns {days, weeks, months_approx, "
                            "hours, minutes, seconds}. "
                            "format: iso + strftime -> string."
                        ),
                    },
                    "args": {
                        "type": "object",
                        "description": (
                            "Shape depends on op. "
                            "parse: {expr}. "
                            "add: {base, delta}. "
                            "diff: {a, b}. "
                            "format: {iso, format}."
                        ),
                    },
                },
                "required": ["op", "args"],
            },
            tier="read",
            display_name="Date math",
        )

    def call(self, *, op: str, args: dict[str, Any]) -> str:
        if op not in _OPS:
            raise ValueError(f"op must be one of {_OPS!r}, got {op!r}")
        if not isinstance(args, dict):
            raise ValueError(f"args must be a dict, got {type(args).__name__}")

        if op == "parse":
            return self._op_parse(args)
        if op == "add":
            return self._op_add(args)
        if op == "diff":
            return self._op_diff(args)
        return self._op_format(args)

    def _today(self) -> date:
        return self.clock(ZoneInfo(self.default_tz)).date()

    def _op_parse(self, args: dict[str, Any]) -> str:
        expr = args.get("expr")
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("parse: args.expr must be a non-empty string")
        result = _parse_natural(expr, today=self._today())
        return f"{result.isoformat()} ({result.strftime('%A')})"

    def _op_add(self, args: dict[str, Any]) -> str:
        base = args.get("base")
        delta = args.get("delta")
        if not isinstance(base, str) or not isinstance(delta, str):
            raise ValueError("add: args.base and args.delta must be strings")
        base_date = _parse_natural(base, today=self._today())
        rd = _parse_delta(delta)
        result = base_date + rd
        if isinstance(result, datetime):
            result = result.date()
        return f"{result.isoformat()} ({result.strftime('%A')})"

    def _op_diff(self, args: dict[str, Any]) -> str:
        a = args.get("a")
        b = args.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError("diff: args.a and args.b must be strings")
        today = self._today()
        a_d = _parse_natural(a, today=today)
        b_d = _parse_natural(b, today=today)
        delta = b_d - a_d
        total_seconds = int(delta.total_seconds())
        sign = "" if total_seconds >= 0 else "-"
        days = abs(delta.days)
        weeks_exact = days / 7
        months_approx = days / 30.4375  # average Gregorian month
        # Months_approx is what users actually want for diff in months;
        # `relativedelta(b_d, a_d).months` underreports across year boundaries.
        return (
            f"{sign}{days} days "
            f"({sign}{weeks_exact:.2f} weeks, "
            f"{sign}{months_approx:.2f} months_approx, "
            f"{sign}{abs(total_seconds)} seconds)"
        )

    def _op_format(self, args: dict[str, Any]) -> str:
        iso = args.get("iso")
        fmt = args.get("format")
        if not isinstance(iso, str) or not isinstance(fmt, str):
            raise ValueError("format: args.iso and args.format must be strings")
        try:
            parsed = date.fromisoformat(iso)
        except ValueError:
            # Try datetime then narrow to date if user passed a full timestamp.
            try:
                parsed_dt = datetime.fromisoformat(iso)
            except ValueError as exc:
                raise ValueError(
                    f"format: iso must be a valid ISO date or datetime, got {iso!r}"
                ) from exc
            return parsed_dt.strftime(fmt)
        return parsed.strftime(fmt)


__all__ = ["DateMathTool", "ZoneInfoNotFound"]
