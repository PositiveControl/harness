"""Tests for the per-request debug log capture (harness-3jz1.9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.web.logging import RequestDebugLog, debug_dir


def test_debug_dir_returns_none_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARNESS_WEB_DEBUG_DIR", raising=False)
    assert debug_dir() is None


def test_debug_dir_creates_path_when_env_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "captures"
    monkeypatch.setenv("HARNESS_WEB_DEBUG_DIR", str(target))
    resolved = debug_dir()
    assert resolved == target
    assert target.exists()
    assert target.is_dir()


def test_request_debug_log_no_op_when_capture_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HARNESS_WEB_DEBUG_DIR", raising=False)
    with RequestDebugLog.start(character_name="t", endpoint="/x") as log:
        log.raw_input = {"text": "hi"}
    # No file should have been created.
    assert list(tmp_path.iterdir()) == []


def test_request_debug_log_writes_json_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_WEB_DEBUG_DIR", str(tmp_path))
    with RequestDebugLog.start(character_name="airton_c_tfr", endpoint="/explain") as log:
        log.raw_input = {"text": "TFR text"}
        log.raw_model_completion = "model reply"
        log.parsed_json_reply = {"verdict": "Stadium TFR"}
        log.citations = ["§91.145"]

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["character"] == "airton_c_tfr"
    assert payload["endpoint"] == "/explain"
    assert payload["raw_input"] == {"text": "TFR text"}
    assert payload["raw_model_completion"] == "model reply"
    assert payload["parsed_json_reply"] == {"verdict": "Stadium TFR"}
    assert payload["citations"] == ["§91.145"]
    assert payload["error"] is None
    assert payload["duration_ms"] is not None


def test_request_debug_log_captures_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_WEB_DEBUG_DIR", str(tmp_path))

    def _raise() -> None:
        with RequestDebugLog.start(character_name="t", endpoint="/x") as log:
            log.raw_input = {"text": "in"}
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _raise()

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["error"].startswith("ValueError: boom")
    assert payload["raw_input"] == {"text": "in"}


def test_filename_includes_endpoint_slug(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HARNESS_WEB_DEBUG_DIR", str(tmp_path))
    with RequestDebugLog.start(character_name="t", endpoint="/debug/parse"):
        pass
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    # Slashes become dashes so the slug stays a single filename segment.
    assert "debug-parse" in files[0].name
