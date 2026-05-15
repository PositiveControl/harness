"""Deterministic NOTAM/TFR text parser.

Takes raw published NOTAM/TFR text and emits a structured `ParsedNOTAM`
without invoking an LLM. The two-stage pipeline rationale: LLMs hallucinate
digits when asked to parse coordinates, radii, and altitudes — those are
regex/arithmetic territory. The explainer (LLM) reasons over the parsed
struct; the parser handles the deterministic part.

Stdlib only at import time. `pyproj` is imported lazily inside
`ParsedNOTAM.to_polygon` so a base install (without the `notam` extra) can
import the module and use the regex pipeline; only callers of `to_polygon`
need pyproj available.

No `harness.*` imports — the module is vendoring-clean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

NOTAMType = Literal[
    "stadium",
    "vip",
    "disaster",
    "space",
    "security",
    "hazard",
    "other",
]


@dataclass(frozen=True)
class Coordinate:
    """Decimal-degree coordinate. North/East positive."""

    lat: float
    lon: float


@dataclass(frozen=True)
class Altitude:
    """Altitude with explicit reference frame.

    `value` is feet (FL is converted to feet on parse). `reference` names
    the frame: SFC (surface), AGL, MSL, or FL (flight level — value is
    feet but the surface dataset reads as e.g. FL180 = 18,000 ft MSL).
    """

    value: int
    reference: Literal["SFC", "AGL", "MSL", "FL"]


@dataclass(frozen=True)
class ActiveWindow:
    """Time window, UTC, tz-aware."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class ParsedNOTAM:
    """Deterministic parse of a published NOTAM/TFR text.

    Every field is optional except `raw_text` and `type_guess` — a NOTAM
    that is missing geometry or a window simply yields `None` for that
    field, and the explainer downstream is expected to surface that as a
    caveat ("ceiling not explicit in source — confirm with a current
    briefing").
    """

    raw_text: str
    center: Coordinate | None
    radius_nm: float | None
    floor: Altitude | None
    ceiling: Altitude | None
    active: ActiveWindow | None
    type_guess: NOTAMType
    cited_sections: tuple[str, ...] = ()

    def to_polygon(self, n_points: int = 64) -> tuple[Coordinate, ...]:
        """Approximate the cylinder footprint as an `n_points`-vertex polygon
        using WGS84 great-circle geodesy. Empty tuple when center or radius
        is missing.

        Lazy pyproj import keeps the regex pipeline usable without the
        `notam` extra installed.
        """
        if self.center is None or self.radius_nm is None:
            return ()
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        radius_m = self.radius_nm * 1852.0
        out: list[Coordinate] = []
        for i in range(n_points):
            bearing = (360.0 * i) / n_points
            lon2, lat2, _ = geod.fwd(self.center.lon, self.center.lat, bearing, radius_m)
            out.append(Coordinate(lat=lat2, lon=lon2))
        return tuple(out)


# DDMMSSN / DDDMMSSW — FAA NOTAM coordinate encoding.
# Example: 384154N/0863912W = 38°41'54"N, 86°39'12"W.
_DMS_COORD_RE = re.compile(
    r"(?P<lat_deg>\d{2})(?P<lat_min>\d{2})(?P<lat_sec>\d{2})(?P<lat_hemi>[NS])"
    r"\s*[/\s]\s*"
    r"(?P<lon_deg>\d{3})(?P<lon_min>\d{2})(?P<lon_sec>\d{2})(?P<lon_hemi>[EW])"
)

# Decimal-degree fallback. Allows N/S/E/W suffix or signed value, comma/
# slash/whitespace separator. Conservative: requires at least one decimal.
_DECIMAL_COORD_RE = re.compile(
    r"(?P<lat>-?\d{1,2}\.\d+)\s*(?P<lat_hemi>[NS])?\s*[,/\s]\s*"
    r"(?P<lon>-?\d{1,3}\.\d+)\s*(?P<lon_hemi>[EW])?"
)

# Radius: "6 NMR" / "3NM" / "6 NM RADIUS". NMR is FAA shorthand for
# "nautical mile radius." Plain "NM" without "RADIUS" is matched as a
# fallback so NOTAMs that drop the R still parse.
_RADIUS_RE = re.compile(
    r"(?P<radius>\d+(?:\.\d+)?)\s*(?:NMR\b|NM\s*RADIUS\b|NM\b)",
    re.IGNORECASE,
)

# Flight level: "FL180" / "FL 180" — three-digit count of hundreds of feet.
_FL_RE = re.compile(r"\bFL\s*(?P<fl>\d{3})\b", re.IGNORECASE)

# Feet-MSL / feet-AGL altitudes. The optional "FT"/"FEET" eats common
# formatting variation. AGL/MSL is required so we don't false-match raw
# numbers in the NOTAM body.
_ALT_FT_RE = re.compile(
    r"(?P<value>\d{1,5})\s*(?:FT|FEET)?\s*(?P<ref>AGL|MSL)\b",
    re.IGNORECASE,
)

# Surface tokens — different NOTAMs use SFC, GND, or "SURFACE."
_SFC_RE = re.compile(r"\b(?:SFC|GND|SURFACE)\b", re.IGNORECASE)

# Time window. FAA NOTAM format is YYMMDDHHMM, all UTC.
# Matches a hyphen, en-dash (U+2013 — sometimes appears in NOTAMs that
# pass through publishing pipelines that auto-substitute punctuation),
# or "TO" separator.
_TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{10})\s*(?:-|–|TO)\s*(?P<end>\d{10})",
    re.IGNORECASE,
)

# 14 CFR section citation. Matches "14 CFR §91.137(a)(1)", "§91.145",
# "SECTION 91.139", etc. We capture the bare section number; the caller
# can normalize back to §-prefix form.
_CFR_CITE_RE = re.compile(
    r"(?:14\s+CFR\s+)?(?:SECTION\s+|§\s*)"
    r"(?P<sec>\d+\.\d+(?:\([a-z]\)(?:\(\d+\))?)?)",
    re.IGNORECASE,
)

# AIM citation: "AIM 3-5-3", "AIM 5-6-1".
_AIM_CITE_RE = re.compile(r"\bAIM\s+(\d+-\d+(?:-\d+)?)\b", re.IGNORECASE)

# Type-guess keyword roster. Order matters — first match wins, and the
# more specific categories (space, stadium) are checked before broader
# ones (security, hazard) so e.g. "ROCKET LAUNCH" doesn't get caught
# as "hazard" before "space" sees it.
_TYPE_KEYWORDS: tuple[tuple[NOTAMType, tuple[str, ...]], ...] = (
    ("space", ("SPACE FLIGHT", "LAUNCH", "REENTRY", "ROCKET")),
    ("stadium", ("STADIUM", "SPORTING EVENT", "MAJOR SPORTING")),
    ("vip", ("VIP", "POTUS", "PRESIDENT", "VICE PRESIDENT", "DIGNITARY")),
    ("disaster", ("DISASTER", "WILDFIRE", "WILD FIRE", "HURRICANE", "TORNADO", "FLOOD")),
    ("security", ("PROHIBITED AREA", "NATIONAL SECURITY", "SECURITY")),
    ("hazard", ("HAZARDOUS", "HAZARD", "EXPLOSIVE", "TOXIC")),
)

# Citation-based fallback for type-guess when keywords don't fire.
# Maps the base §-number to the canonical TFR category.
_TYPE_CITE_HINTS: dict[str, NOTAMType] = {
    "91.137": "disaster",
    "91.138": "disaster",
    "91.139": "security",
    "91.141": "vip",
    "91.143": "space",
    "91.144": "hazard",
    "91.145": "stadium",
}


def parse_notam(text: str) -> ParsedNOTAM:
    """Parse a raw NOTAM/TFR text into a deterministic struct.

    Always returns a `ParsedNOTAM`; missing fields are `None`. Never
    raises — the parser is intended to be defensive against the
    ill-formed NOTAM text the FAA actually publishes.
    """
    upper = text.upper()
    center = _parse_center(text)
    radius_nm = _parse_radius(upper)
    floor = _parse_floor(upper)
    ceiling = _parse_ceiling(upper)
    active = _parse_active_window(upper)
    cited_sections = _parse_cited_sections(text)
    type_guess = _guess_type(upper, cited_sections)
    return ParsedNOTAM(
        raw_text=text,
        center=center,
        radius_nm=radius_nm,
        floor=floor,
        ceiling=ceiling,
        active=active,
        type_guess=type_guess,
        cited_sections=cited_sections,
    )


def _parse_center(text: str) -> Coordinate | None:
    dms = _DMS_COORD_RE.search(text)
    if dms is not None:
        lat = _dms_to_decimal(
            int(dms["lat_deg"]),
            int(dms["lat_min"]),
            int(dms["lat_sec"]),
            dms["lat_hemi"],
        )
        lon = _dms_to_decimal(
            int(dms["lon_deg"]),
            int(dms["lon_min"]),
            int(dms["lon_sec"]),
            dms["lon_hemi"],
        )
        return Coordinate(lat=lat, lon=lon)
    dec = _DECIMAL_COORD_RE.search(text)
    if dec is not None:
        lat = float(dec["lat"])
        lon = float(dec["lon"])
        if dec["lat_hemi"] == "S":
            lat = -abs(lat)
        if dec["lon_hemi"] == "W":
            lon = -abs(lon)
        return Coordinate(lat=round(lat, 4), lon=round(lon, 4))
    return None


def _dms_to_decimal(deg: int, minutes: int, seconds: int, hemi: str) -> float:
    val = deg + minutes / 60.0 + seconds / 3600.0
    if hemi in ("S", "W"):
        val = -val
    return round(val, 4)


def _parse_radius(text: str) -> float | None:
    match = _RADIUS_RE.search(text)
    if match is None:
        return None
    return float(match["radius"])


def _parse_floor(text: str) -> Altitude | None:
    # SFC takes precedence over numeric altitudes — when both appear, the
    # surface token names the floor and the numeric value is the ceiling.
    if _SFC_RE.search(text):
        return Altitude(value=0, reference="SFC")
    matches = list(_ALT_FT_RE.finditer(text))
    if not matches:
        return None
    first = matches[0]
    return Altitude(value=int(first["value"]), reference=_norm_ref(first["ref"]))


def _parse_ceiling(text: str) -> Altitude | None:
    # FL takes precedence — TFRs that use FL never mix in raw feet for the
    # ceiling. If FL is absent, use the last numeric AGL/MSL altitude.
    fl = _FL_RE.search(text)
    if fl is not None:
        return Altitude(value=int(fl["fl"]) * 100, reference="FL")
    matches = list(_ALT_FT_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    # If only one numeric altitude is present and SFC named the floor,
    # that single altitude is the ceiling. If SFC is absent and there's
    # only one numeric altitude, treat it as the floor and return None
    # here (caller's `_parse_floor` will have picked it up).
    if len(matches) == 1 and not _SFC_RE.search(text):
        return None
    return Altitude(value=int(last["value"]), reference=_norm_ref(last["ref"]))


def _norm_ref(raw: str) -> Literal["AGL", "MSL"]:
    upper = raw.upper()
    if upper == "AGL":
        return "AGL"
    return "MSL"


def _parse_active_window(text: str) -> ActiveWindow | None:
    match = _TIME_RANGE_RE.search(text)
    if match is None:
        return None
    start = _parse_yymmddhhmm(match["start"])
    end = _parse_yymmddhhmm(match["end"])
    if start is None or end is None:
        return None
    return ActiveWindow(start=start, end=end)


def _parse_yymmddhhmm(raw: str) -> datetime | None:
    try:
        year = 2000 + int(raw[0:2])
        month = int(raw[2:4])
        day = int(raw[4:6])
        hour = int(raw[6:8])
        minute = int(raw[8:10])
        return datetime(year, month, day, hour, minute, tzinfo=UTC)
    except ValueError:
        return None


def _parse_cited_sections(text: str) -> tuple[str, ...]:
    cites: list[str] = []
    for match in _CFR_CITE_RE.finditer(text):
        cites.append(f"§{match['sec']}")
    for match in _AIM_CITE_RE.finditer(text):
        cites.append(f"AIM {match.group(1)}")
    seen: set[str] = set()
    out: list[str] = []
    for cite in cites:
        if cite in seen:
            continue
        seen.add(cite)
        out.append(cite)
    return tuple(out)


def _guess_type(upper: str, cited_sections: tuple[str, ...]) -> NOTAMType:
    for type_name, keywords in _TYPE_KEYWORDS:
        for keyword in keywords:
            if keyword in upper:
                return type_name
    for cite in cited_sections:
        base = cite.lstrip("§").split("(")[0]
        if base in _TYPE_CITE_HINTS:
            return _TYPE_CITE_HINTS[base]
    return "other"
