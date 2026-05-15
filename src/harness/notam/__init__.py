"""Deterministic NOTAM/TFR text parser (harness-3jz1.3).

Pure-Python module: stdlib + (optionally) pyproj. No `harness.*` imports —
the module is import-clean so it can be vendored or used standalone if the
TFR explainer ever splits off as its own service. pyproj is imported
lazily inside `ParsedNOTAM.to_polygon` so a base install (without the
`notam` extra) can still import and use the regex pipeline; only callers
of `to_polygon` need pyproj on the path.
"""

from .parser import (
    ActiveWindow,
    Altitude,
    Coordinate,
    NOTAMType,
    ParsedNOTAM,
    parse_notam,
)

__all__ = [
    "ActiveWindow",
    "Altitude",
    "Coordinate",
    "NOTAMType",
    "ParsedNOTAM",
    "parse_notam",
]
