"""stats — descriptive statistics on a list of numbers.

Companion to `calc` + `python_eval` in the reckon profile. python_eval
can do everything stats does, but small models write broken
`statistics` snippets often (missing imports, wrong percentile method,
edge-case crashes). A one-step deterministic primitive trades the
round-trip cost for reliability.

Read-tier, no new dep, schema-cheap (~250 tokens of overhead). The
percentile method is numpy-default ('linear' interpolation) so a
model that pipes stats into a downstream comparison sees the same
numbers it would from numpy.percentile.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from harness.tools.base import ToolSpec

_METRICS: tuple[str, ...] = (
    "mean",
    "median",
    "stdev",
    "variance",
    "min",
    "max",
    "count",
    "sum",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
)

_DEFAULT_METRICS: tuple[str, ...] = (
    "count",
    "sum",
    "mean",
    "median",
    "stdev",
    "min",
    "max",
)

# Metrics that need at least 2 data points (sample variance / stdev).
_NEEDS_TWO: frozenset[str] = frozenset({"stdev", "variance"})


def _percentile_linear(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy default, method='linear').
    `sorted_data` must already be sorted ascending; `p` is in [0, 100]."""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    pos = (n - 1) * (p / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_data[lower]
    return sorted_data[lower] + (pos - lower) * (sorted_data[upper] - sorted_data[lower])


def _compute_metric(name: str, data: list[float], sorted_data: list[float]) -> str:
    """Return a one-line string for the named metric: '<value>' or
    'undefined (need n>=2)' for sample-stat metrics on n=1 inputs."""
    n = len(data)
    if name in _NEEDS_TWO and n < 2:
        return "undefined (need n>=2)"
    if name == "mean":
        return _fmt(statistics.fmean(data))
    if name == "median":
        return _fmt(statistics.median(data))
    if name == "stdev":
        return _fmt(statistics.stdev(data))
    if name == "variance":
        return _fmt(statistics.variance(data))
    if name == "min":
        return _fmt(min(data))
    if name == "max":
        return _fmt(max(data))
    if name == "count":
        return f"{n} (count)"
    if name == "sum":
        return _fmt(sum(data))
    if name.startswith("p") and name[1:].isdigit():
        return _fmt(_percentile_linear(sorted_data, float(name[1:])))
    raise ValueError(f"unknown metric {name!r} — must be one of {_METRICS!r}")


def _fmt(value: float) -> str:
    """Stable float repr — ten significant digits so the printed value
    survives a round-trip vs. a numpy reference within 1e-6 absolute
    tolerance (the contract documented in the tool spec). Whole-number
    values render as '<n>.0' so the reader sees the decimal point and
    knows it's a float, not a count."""
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{value:.1f}"
    return f"{value:.10g}"


@dataclass
class StatsTool:
    """Descriptive statistics on a list of numbers.

    Output format (structured record + headline; the model's reply
    should attach the data's source unit, per units_attached):

        count    : 10 (count)
        sum      : 55.0
        mean     : 5.5
        median   : 5.5
        stdev    : 3.02765
        min      : 1.0
        max      : 10.0

        7 metrics over 10 values.
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="stats",
            description=(
                "Compute descriptive statistics on a list of numbers. "
                "Use this for mean / median / stdev / percentile "
                "questions rather than writing `statistics` snippets "
                "in python_eval — it's a one-step deterministic call. "
                "Percentiles use linear interpolation (numpy default)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": ("Non-empty list of numbers to summarize."),
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(_METRICS),
                        },
                        "description": (
                            "Which statistics to report. Defaults to "
                            "[count, sum, mean, median, stdev, min, max]. "
                            "Available: mean, median, stdev, variance, "
                            "min, max, count, sum, p25, p50, p75, p90, "
                            "p95, p99."
                        ),
                    },
                },
                "required": ["data"],
            },
            tier="read",
            display_name="Statistics",
        )

    def call(
        self,
        *,
        data: list[float] | list[int],
        metrics: list[str] | None = None,
    ) -> str:
        if not isinstance(data, list):
            raise ValueError(f"data must be a list, got {type(data).__name__}")
        if not data:
            raise ValueError("data must be a non-empty list of numbers")
        cleaned: list[float] = []
        for i, x in enumerate(data):
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                raise ValueError(f"data[{i}] must be a number, got {type(x).__name__}: {x!r}")
            cleaned.append(float(x))

        chosen = tuple(metrics) if metrics is not None else _DEFAULT_METRICS
        if not chosen:
            raise ValueError("metrics must be a non-empty list (or omitted to use the default set)")
        for m in chosen:
            if m not in _METRICS:
                raise ValueError(f"unknown metric {m!r} — must be one of {_METRICS!r}")

        sorted_data = sorted(cleaned)
        rows = [(m, _compute_metric(m, cleaned, sorted_data)) for m in chosen]
        label_width = max(len(name) for name, _ in rows)
        lines = [f"{name:<{label_width}} : {value}" for name, value in rows]
        lines.append("")
        lines.append(
            f"{len(chosen)} metric{'s' if len(chosen) != 1 else ''} over {len(cleaned)} values."
        )
        return "\n".join(lines)
