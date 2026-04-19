"""Tests for harness.evals.session_resume — fixture loader, eval
runner, and scoring shape. The eval has no network / subprocess
dependencies; these tests hit the full code path with synthetic
fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness.evals.session_resume import (
    default_fixture_path,
    load_fixture,
    run_session_resume_eval,
)


def _write_fixture(tmp_path: Path, data: list[dict[str, object]]) -> Path:
    path = tmp_path / "fx.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_default_fixture_path_convention(tmp_path: Path) -> None:
    assert default_fixture_path(tmp_path).name == "session_resume_eval.yaml"


def test_load_fixture_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"id": "oops"}))
    with pytest.raises(ValueError, match="not a YAML list"):
        load_fixture(path)


def test_load_fixture_rejects_missing_id(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, [{"description": "no id"}])
    with pytest.raises(ValueError, match="missing 'id'"):
        load_fixture(path)


def test_eval_passes_when_contains_match(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path,
        [
            {
                "id": "happy",
                "focus": {"id": "h-1", "title": "ship it", "scope": "personal", "priority": 1},
                "in_progress": [],
                "memories": ["keep bd queries fast"],
                "drift": [],
                "expected_contains": ["h-1", "ship it", "keep bd queries"],
                "expected_not_contains": [],
            }
        ],
    )
    result = run_session_resume_eval(load_fixture(path))
    assert result.pass_rate == 1.0
    assert len(result.cases) == 1
    assert result.cases[0].passed


def test_eval_fails_when_expected_missing(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path,
        [
            {
                "id": "sad",
                "focus": None,
                "in_progress": [],
                "memories": [],
                "drift": [],
                "expected_contains": ["this string absolutely does not appear"],
                "expected_not_contains": [],
            }
        ],
    )
    result = run_session_resume_eval(load_fixture(path))
    assert result.pass_rate == 0.0
    failures = result.failures()
    assert len(failures) == 1
    assert "this string absolutely does not appear" in failures[0].missing_contains


def test_eval_fails_when_unexpected_present(tmp_path: Path) -> None:
    path = _write_fixture(
        tmp_path,
        [
            {
                "id": "leak",
                "focus": {"id": "hush", "title": "secret", "scope": "personal", "priority": 2},
                "in_progress": [],
                "memories": [],
                "drift": [],
                "expected_contains": [],
                "expected_not_contains": ["secret"],
            }
        ],
    )
    result = run_session_resume_eval(load_fixture(path))
    assert result.pass_rate == 0.0
    assert "secret" in result.failures()[0].unexpected_contains


def test_shipped_fixtures_pass_end_to_end() -> None:
    """The character's actual session-resume fixtures must all pass —
    this is the gate that catches regressions in build_resume_summary."""
    shipped = Path(__file__).parent.parent / "character" / "airton_b" / "session_resume_eval.yaml"
    if not shipped.exists():  # sanity skip if repo layout shifts
        pytest.skip("shipped fixture missing; skipping end-to-end")
    result = run_session_resume_eval(load_fixture(shipped))
    assert result.pass_rate == 1.0, (
        f"shipped fixtures regressed: {[c.id for c in result.failures()]}"
    )
