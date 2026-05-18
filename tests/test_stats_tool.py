"""Tests for the `stats` reckon-profile tool — harness-dsps."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from harness.tools.stats import _METRICS, StatsTool


def _parse_rows(out: str) -> dict[str, str]:
    """Parse '<name>  : <value>' lines into {name: value} so tests don't
    depend on the right-padding width of the label column."""
    parsed: dict[str, str] = {}
    for line in out.splitlines():
        if " : " not in line:
            continue
        name, _, value = line.partition(" : ")
        parsed[name.rstrip()] = value
    return parsed


def test_stats_spec_shape() -> None:
    spec = StatsTool().spec
    assert spec.name == "stats"
    assert spec.tier == "read"
    props = spec.parameters["properties"]
    assert "data" in props
    assert "metrics" in props
    assert spec.parameters["required"] == ["data"]
    enum = props["metrics"]["items"]["enum"]
    assert set(enum) == set(_METRICS)


def test_stats_default_metric_set_on_basic_input() -> None:
    out = StatsTool().call(data=[1, 2, 3, 4, 5])
    # Default metrics: count, sum, mean, median, stdev, min, max.
    # The label column is right-padded to the longest metric's width,
    # so we parse the lines instead of asserting exact whitespace.
    parsed = _parse_rows(out)
    assert parsed["count"] == "5 (count)"
    assert parsed["sum"] == "15.0"
    assert parsed["mean"] == "3.0"
    assert parsed["median"] == "3.0"
    assert parsed["min"] == "1.0"
    assert parsed["max"] == "5.0"
    assert "stdev" in parsed
    assert "variance" not in parsed  # not in default set


def test_stats_all_metrics_on_ten_element_list() -> None:
    out = StatsTool().call(data=list(range(1, 11)), metrics=list(_METRICS))
    parsed = _parse_rows(out)
    for m in _METRICS:
        assert m in parsed, f"missing metric {m!r}: {parsed!r}"
    # Footer reports the metric + value counts.
    assert "14 metrics over 10 values." in out


def test_stats_p95_matches_numpy_within_tolerance() -> None:
    rng = np.random.default_rng(seed=0xDEAD_BEEF)
    data = rng.standard_normal(1000).tolist()
    out = StatsTool().call(data=data, metrics=["p95"])
    ours = float(_parse_rows(out)["p95"])
    expected = float(np.percentile(np.array(data), 95, method="linear"))
    assert ours == pytest.approx(expected, abs=1e-6)


def test_stats_percentile_endpoints() -> None:
    out = StatsTool().call(data=[1, 2, 3, 4, 5], metrics=["p25", "p50", "p75", "p99"])
    parsed = _parse_rows(out)
    arr = np.array([1, 2, 3, 4, 5], dtype=float)
    for p in (25, 50, 75, 99):
        ours = float(parsed[f"p{p}"])
        expected = float(np.percentile(arr, p, method="linear"))
        assert ours == pytest.approx(expected, abs=1e-6), f"p{p}: ours={ours} expected={expected}"


def test_stats_count_carries_unit_hint() -> None:
    out = StatsTool().call(data=[10.0, 20.0, 30.0], metrics=["count"])
    assert "count" in out
    assert "(count)" in out  # units_attached value enforces the explicit hint.


def test_stats_stdev_undefined_for_n_eq_1() -> None:
    out = StatsTool().call(data=[42.0], metrics=["stdev", "variance", "mean"])
    parsed = _parse_rows(out)
    assert parsed["stdev"] == "undefined (need n>=2)"
    assert parsed["variance"] == "undefined (need n>=2)"
    # mean is defined for n=1.
    assert parsed["mean"] == "42.0"


def test_stats_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        StatsTool().call(data=[])


def test_stats_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        StatsTool().call(data=[1.0, 2.0], metrics=["mode"])


def test_stats_empty_metrics_list_raises() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        StatsTool().call(data=[1.0, 2.0], metrics=[])


def test_stats_rejects_non_numeric_data() -> None:
    # The model can emit any JSON; the tool defends at the boundary.
    # Cast at the call site so the test exercises the runtime check
    # without lying about the annotated parameter type.
    with pytest.raises(ValueError, match="must be a number"):
        StatsTool().call(data=cast("list[float]", [1.0, "two", 3.0]))


def test_stats_rejects_bool_as_data_element() -> None:
    # bool is a subclass of int in Python — silently coercing True->1
    # would mask a model bug, so reject explicitly.
    with pytest.raises(ValueError, match="must be a number"):
        StatsTool().call(data=cast("list[int]", [1, True, 3]))


def test_stats_handles_negative_and_mixed_floats() -> None:
    out = StatsTool().call(data=[-1.5, 0.0, 1.5, 2.5], metrics=["min", "max", "mean"])
    parsed = _parse_rows(out)
    assert parsed["min"] == "-1.5"
    assert parsed["max"] == "2.5"
    assert parsed["mean"] == "0.625"


def test_stats_p50_equals_median_for_odd_length() -> None:
    out = StatsTool().call(data=[1, 2, 3, 4, 5], metrics=["p50", "median"])
    parsed = _parse_rows(out)
    assert parsed["p50"] == parsed["median"]
