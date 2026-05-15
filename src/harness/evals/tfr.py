"""TFR/NOTAM eval (harness-3jz1.2).

Replays a fixture of golden TFR/NOTAM cases through the deterministic
parser (`harness.notam.parse_notam`) and scores each case on three
deterministic axes:

  - **geometry**: center lat/lon within ±0.005° (~500 m), radius within
    ±0.1 NM, floor/ceiling exact match (value + reference frame).
  - **type**: parser type-guess matches expected category exactly.
  - **citations**: parser-extracted §-cites and AIM-cites match expected
    sets exactly (set-equality, order-independent).

This is the **parser-axis** half of the hybrid scoring spec from the
TFR explainer refinement pass (harness-3jz1 epic notes). The
**model-axis** half — citation grounded in retrieved chunks + LLM
verdict judge gated behind det≥0.9 — lands when the FastAPI gateway
(harness-3jz1.4) wires the request-scoped `read_parsed_notam` tool and
the airton_c_tfr corpus is ingested. Until then, this eval validates
that the parser handles realistic NOTAM shapes across all six TFR
categories and gives the downstream eval a known-good substrate.

No model invocation, no filesystem writes, no network. Pure beyond the
parser call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.notam import (
    ActiveWindow,
    Altitude,
    Coordinate,
    NOTAMType,
    ParsedNOTAM,
    parse_notam,
)

# Geometry tolerances. Decision recorded in harness-3jz1 epic notes
# (2026-05-14 refinement). Lat/lon are decimal-degree comparison;
# radius is NM; altitude requires exact match on both value and frame.
LAT_LON_TOLERANCE_DEG: float = 0.005
RADIUS_TOLERANCE_NM: float = 0.1


@dataclass(frozen=True)
class ExpectedGeometry:
    """Expected geometry struct for a fixture case. Every field is
    optional — a NOTAM case may pin only what it cares about (e.g. a
    parser stress test that only asserts `center` and `radius_nm`)."""

    center: Coordinate | None = None
    radius_nm: float | None = None
    floor: Altitude | None = None
    ceiling: Altitude | None = None


@dataclass(frozen=True)
class TfrEvalCase:
    """One case in the fixture, plus the parser's actual output and
    per-axis scoring verdicts."""

    case_id: str
    raw_notam: str
    expected_verdict: str
    expected_type: NOTAMType
    expected_citations: tuple[str, ...]
    expected_geometry: ExpectedGeometry
    expected_active: ActiveWindow | None
    caveats: tuple[str, ...]

    parsed: ParsedNOTAM
    geometry_correct: bool
    type_correct: bool
    citations_correct: bool

    @property
    def passed(self) -> bool:
        return self.geometry_correct and self.type_correct and self.citations_correct


@dataclass(frozen=True)
class TfrEvalResult:
    """Aggregate result of running the eval across the fixture."""

    cases: tuple[TfrEvalCase, ...]

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    @property
    def geometry_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.geometry_correct) / len(self.cases)

    @property
    def type_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.type_correct) / len(self.cases)

    @property
    def citation_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.citations_correct) / len(self.cases)

    def failures(self) -> tuple[TfrEvalCase, ...]:
        return tuple(c for c in self.cases if not c.passed)


def load_fixture(path: Path) -> tuple[dict[str, Any], ...]:
    """Parse a tfr_eval YAML file into raw row dicts. Public so the CLI
    can surface a row count before running.

    YAML shape per row:
      - id: str  (case identifier)
      - raw_notam: str  (the NOTAM text to parse)
      - expected_verdict: str  (judge-axis target; unused in parser-only mode)
      - expected_type: str  (one of stadium/vip/disaster/space/security/hazard/other)
      - expected_citations: list[str]  (e.g. ["§91.145", "AIM 3-5-3"])
      - expected_geometry: mapping (optional sub-keys; see _coerce_geometry)
      - expected_active: mapping {from: iso8601, to: iso8601}  (optional)
      - caveats: list[str]  (optional)
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"tfr_eval fixture {path} is not a YAML list")
    out: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{idx}] is not a mapping")
        for required in ("id", "raw_notam", "expected_type"):
            if required not in entry:
                raise ValueError(f"{path}[{idx}] missing required key {required!r}")
        out.append(entry)
    return tuple(out)


def _coerce_coordinate(raw: Any, *, ctx: str) -> Coordinate | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx} must be a mapping with 'lat'/'lon'")
    try:
        return Coordinate(lat=float(raw["lat"]), lon=float(raw["lon"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{ctx} requires numeric 'lat' + 'lon'") from exc


def _coerce_altitude(raw: Any, *, ctx: str) -> Altitude | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx} must be a mapping with 'value'/'reference'")
    try:
        ref = raw["reference"]
    except KeyError as exc:
        raise ValueError(f"{ctx} requires 'reference' (SFC/AGL/MSL/FL)") from exc
    if ref not in ("SFC", "AGL", "MSL", "FL"):
        raise ValueError(f"{ctx}.reference must be one of SFC/AGL/MSL/FL; got {ref!r}")
    try:
        value = int(raw["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{ctx} requires integer 'value'") from exc
    return Altitude(value=value, reference=ref)


def _coerce_geometry(raw: Any) -> ExpectedGeometry:
    if raw is None:
        return ExpectedGeometry()
    if not isinstance(raw, dict):
        raise ValueError("expected_geometry must be a mapping")
    return ExpectedGeometry(
        center=_coerce_coordinate(raw.get("center"), ctx="expected_geometry.center"),
        radius_nm=float(raw["radius_nm"]) if raw.get("radius_nm") is not None else None,
        floor=_coerce_altitude(raw.get("floor"), ctx="expected_geometry.floor"),
        ceiling=_coerce_altitude(raw.get("ceiling"), ctx="expected_geometry.ceiling"),
    )


def _coerce_active_window(raw: Any) -> ActiveWindow | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("expected_active must be a mapping with 'from'/'to'")
    from datetime import datetime

    try:
        start = datetime.fromisoformat(str(raw["from"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(raw["to"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise ValueError("expected_active.from / .to must be ISO-8601") from exc
    return ActiveWindow(start=start, end=end)


def score_geometry(parsed: ParsedNOTAM, expected: ExpectedGeometry) -> bool:
    """All expected fields match within tolerance. Fields the fixture
    left as None are skipped — pinning a partial geometry is allowed."""
    if expected.center is not None:
        if parsed.center is None:
            return False
        if abs(parsed.center.lat - expected.center.lat) > LAT_LON_TOLERANCE_DEG:
            return False
        if abs(parsed.center.lon - expected.center.lon) > LAT_LON_TOLERANCE_DEG:
            return False
    if expected.radius_nm is not None:
        if parsed.radius_nm is None:
            return False
        if abs(parsed.radius_nm - expected.radius_nm) > RADIUS_TOLERANCE_NM:
            return False
    if expected.floor is not None and parsed.floor != expected.floor:
        return False
    return not (expected.ceiling is not None and parsed.ceiling != expected.ceiling)


def score_citations(parsed: ParsedNOTAM, expected: tuple[str, ...]) -> bool:
    """Set-equality between parsed cites and expected cites. Order-
    independent. Empty expected = no citation check on this case."""
    if not expected:
        return True
    return set(parsed.cited_sections) == set(expected)


def score_case(row: dict[str, Any]) -> TfrEvalCase:
    """Parse one fixture row, run the parser, and return the scored
    case. Raises ValueError on malformed row shape."""
    case_id = str(row["id"])
    raw_notam = str(row["raw_notam"])
    expected_verdict = str(row.get("expected_verdict", ""))
    expected_type_raw = str(row["expected_type"])
    if expected_type_raw not in (
        "stadium",
        "vip",
        "disaster",
        "space",
        "security",
        "hazard",
        "other",
    ):
        raise ValueError(
            f"case {case_id!r}: expected_type must be one of stadium/vip/disaster/"
            f"space/security/hazard/other; got {expected_type_raw!r}"
        )
    expected_type: NOTAMType = expected_type_raw  # type: ignore[assignment]
    expected_citations_raw = row.get("expected_citations", []) or []
    if not isinstance(expected_citations_raw, list):
        raise ValueError(f"case {case_id!r}: expected_citations must be a list")
    expected_citations = tuple(str(c) for c in expected_citations_raw)
    expected_geometry = _coerce_geometry(row.get("expected_geometry"))
    expected_active = _coerce_active_window(row.get("expected_active"))
    caveats_raw = row.get("caveats", []) or []
    if not isinstance(caveats_raw, list):
        raise ValueError(f"case {case_id!r}: caveats must be a list")
    caveats = tuple(str(c) for c in caveats_raw)

    parsed = parse_notam(raw_notam)
    geometry_correct = score_geometry(parsed, expected_geometry)
    type_correct = parsed.type_guess == expected_type
    citations_correct = score_citations(parsed, expected_citations)

    return TfrEvalCase(
        case_id=case_id,
        raw_notam=raw_notam,
        expected_verdict=expected_verdict,
        expected_type=expected_type,
        expected_citations=expected_citations,
        expected_geometry=expected_geometry,
        expected_active=expected_active,
        caveats=caveats,
        parsed=parsed,
        geometry_correct=geometry_correct,
        type_correct=type_correct,
        citations_correct=citations_correct,
    )


def run_tfr_eval(rows: tuple[dict[str, Any], ...]) -> TfrEvalResult:
    """Score every row and return the aggregate result.

    Pure — no model invocation, no I/O beyond the parser. The CLI wraps
    this; tests exercise it directly with in-memory fixture rows.
    """
    cases = tuple(score_case(row) for row in rows)
    return TfrEvalResult(cases=cases)


def default_fixture_path(character_path: Path) -> Path:
    """Conventional fixture location: `<character>/tfr_eval.yaml`.
    Mirrors the router-eval / session-resume-eval convention."""
    return character_path / "tfr_eval.yaml"
