"""Tests for the tool-loop failure-corpus eval (harness-cfm7).

Pins fixture loading, scoring, the disable_catchers context manager, and
the shipped `character/airton/tool_loop_eval.yaml` fixture so a
regression in the orchestrator's catcher stack shows up as a failing
test rather than silent drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.evals.tool_loop import (
    CATCHER_NAMES,
    default_fixture_path,
    disable_catchers,
    load_fixture,
    run_attribution,
    run_tool_loop_eval,
)
from harness.orchestrator.tool_loop import _DISABLED_CATCHERS

# ---------- load_fixture ----------


def test_load_fixture_requires_list(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("id: x\n")
    with pytest.raises(ValueError, match="not a YAML list"):
        load_fixture(f)


def test_load_fixture_requires_id(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("- label: teaser\n")
    with pytest.raises(ValueError, match="missing 'id'"):
        load_fixture(f)


def test_load_fixture_rejects_unknown_label(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("- id: x\n  label: bogus\n")
    with pytest.raises(ValueError, match="not a known catcher"):
        load_fixture(f)


def test_load_fixture_rejects_duplicate_ids(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("- id: x\n  label: control\n- id: x\n  label: control\n")
    with pytest.raises(ValueError, match="duplicate id"):
        load_fixture(f)


def test_load_fixture_accepts_control_label(tmp_path: Path) -> None:
    f = tmp_path / "ok.yaml"
    f.write_text("- id: clean\n  label: control\n")
    rows = load_fixture(f)
    assert len(rows) == 1
    assert rows[0]["label"] == "control"


# ---------- disable_catchers ----------


def test_disable_catchers_mutates_and_restores() -> None:
    assert "teaser" not in _DISABLED_CATCHERS
    with disable_catchers(["teaser", "tool_intent"]):
        assert "teaser" in _DISABLED_CATCHERS
        assert "tool_intent" in _DISABLED_CATCHERS
    assert "teaser" not in _DISABLED_CATCHERS
    assert "tool_intent" not in _DISABLED_CATCHERS


def test_disable_catchers_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown catcher"), disable_catchers(["bogus"]):
        pass


def test_disable_catchers_is_nesting_safe() -> None:
    with disable_catchers(["teaser"]):
        with disable_catchers(["teaser", "truncated"]):
            assert "teaser" in _DISABLED_CATCHERS
            assert "truncated" in _DISABLED_CATCHERS
        # inner context only removes what it added — teaser stays.
        assert "teaser" in _DISABLED_CATCHERS
        assert "truncated" not in _DISABLED_CATCHERS
    assert "teaser" not in _DISABLED_CATCHERS


# ---------- shipped fixture pin ----------


AIRTON_FIXTURE = (
    Path(__file__).resolve().parents[1] / "character" / "airton" / "tool_loop_eval.yaml"
)


def test_default_fixture_path_points_at_shipped_file() -> None:
    char_root = AIRTON_FIXTURE.parent
    assert default_fixture_path(char_root) == AIRTON_FIXTURE


def test_shipped_fixture_covers_every_catcher() -> None:
    """Every orchestrator catcher must have at least one scenario. A
    new catcher without a fixture entry would land uncovered and the
    attribution harness would silently report it as dead code."""
    rows = load_fixture(AIRTON_FIXTURE)
    labels = {r.get("label") for r in rows}
    missing = [name for name in CATCHER_NAMES if name not in labels]
    assert not missing, f"catchers without fixture coverage: {missing}"


def test_shipped_fixture_all_scenarios_pass_baseline() -> None:
    """With all catchers enabled, every shipped scenario must pass.
    This is the baseline any rearchitect PR has to beat."""
    rows = load_fixture(AIRTON_FIXTURE)
    result = run_tool_loop_eval(rows)
    failures = result.failures()
    assert not failures, [
        (
            f.id,
            f.label,
            f.missing_contains,
            f.unexpected_contains,
            f.missing_events,
            f.unexpected_events,
            f.missing_message_substrings,
            f.unexpected_message_substrings,
            f.expected_fallback,
            f.fallback_triggered,
        )
        for f in failures
    ]


def test_attribution_every_catcher_has_effect() -> None:
    """Every orchestrator catcher must change at least one scenario's
    pass/fail when disabled. A no-effect catcher is either dead code
    or uncovered — either way, the fixture needs updating."""
    rows = load_fixture(AIRTON_FIXTURE)
    attr = run_attribution(rows)
    dead = [a.catcher for a in attr.attributions if a.no_effect]
    assert not dead, f"catchers with no fixture coverage: {dead}"


def test_attribution_baseline_is_clean() -> None:
    """Attribution baseline is the same as the plain eval — no
    catchers disabled → every scenario passes."""
    rows = load_fixture(AIRTON_FIXTURE)
    attr = run_attribution(rows)
    assert not attr.baseline.failures()
