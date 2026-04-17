"""Tests for the shared `_write_voice_capture` helper and the
`_open_in_editor` wrapper. Covers the core of the in-chat /edit
slash command without spinning up the full chat loop."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from harness.cli import _open_in_editor, _write_voice_capture
from harness.config import settings

# ---------- _write_voice_capture ----------


def test_write_voice_capture_creates_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """First call against a fresh character dir writes version+samples."""
    character_dir = tmp_path / "character" / "airton"
    (character_dir / "voice").mkdir(parents=True)
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "character_name", "airton")

    captured_path, sid, total = _write_voice_capture(
        prompt="what's the move?",
        gold="ship small, inspect often.",
        session="local",
        original="Let's ship incrementally and verify each step.",
    )
    assert captured_path == character_dir / "voice" / "captured.yaml"
    assert total == 1
    assert sid.startswith("captured-")

    doc = yaml.safe_load(captured_path.read_text())
    assert doc["samples"][0]["prompt"] == "what's the move?"
    assert doc["samples"][0]["gold"] == "ship small, inspect often."
    assert doc["samples"][0]["original"].startswith("Let's ship")
    assert doc["samples"][0]["captured_from"] == "session=local"


def test_write_voice_capture_appends_to_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    character_dir = tmp_path / "character" / "airton"
    (character_dir / "voice").mkdir(parents=True)
    existing = character_dir / "voice" / "captured.yaml"
    existing.write_text(
        textwrap.dedent(
            """\
            version: 1
            samples:
              - id: captured-old
                prompt: what?
                gold: first sample.
                captured_at: '2026-04-01T00:00:00+00:00'
                captured_from: session=local
            """
        )
    )
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "character_name", "airton")

    _, _, total = _write_voice_capture(
        prompt="new prompt",
        gold="new gold.",
        session="local",
        original=None,
    )
    assert total == 2
    doc = yaml.safe_load(existing.read_text())
    assert next(s["id"] for s in doc["samples"]) == "captured-old"
    assert doc["samples"][1]["gold"] == "new gold."


def test_write_voice_capture_strips_gold_whitespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "character" / "airton" / "voice").mkdir(parents=True)
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "character_name", "airton")

    path, _, _ = _write_voice_capture(
        prompt="q",
        gold="   trimmed gold.  \n\n",
        session="local",
        original=None,
    )
    doc = yaml.safe_load(path.read_text())
    assert doc["samples"][0]["gold"] == "trimmed gold."


def test_write_voice_capture_respects_custom_sample_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "character" / "airton" / "voice").mkdir(parents=True)
    monkeypatch.setattr(settings, "root", tmp_path)
    monkeypatch.setattr(settings, "character_name", "airton")

    _, sid, _ = _write_voice_capture(
        prompt="q",
        gold="g",
        session="local",
        original=None,
        sample_id="my-custom-id",
    )
    assert sid == "my-custom-id"


# ---------- _open_in_editor ----------


def test_open_in_editor_captures_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub EDITOR with a script that appends ' EDITED' to the temp file
    so we can assert the round-trip without needing a real editor."""
    editor = tmp_path / "fake_editor.sh"
    editor.write_text("#!/bin/bash\necho ' EDITED' >> \"$1\"\n")
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))

    out = _open_in_editor("original text")
    assert out is not None
    assert "original text" in out
    assert "EDITED" in out


def test_open_in_editor_returns_none_on_no_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editor that exits without writing anything → no capture."""
    editor = tmp_path / "noop_editor.sh"
    editor.write_text("#!/bin/bash\nexit 0\n")
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(editor))

    assert _open_in_editor("unchanged text") is None


def test_open_in_editor_missing_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    """No $EDITOR, no $VISUAL, no vi on PATH → return None instead of raising."""
    monkeypatch.setenv("EDITOR", "definitely-not-a-real-binary-xyz")
    monkeypatch.delenv("VISUAL", raising=False)
    # Force shutil.which to fail for the phony editor AND for `vi`.
    with patch("shutil.which", return_value=None):
        assert _open_in_editor("text") is None


def test_open_in_editor_passes_editor_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editors like `code --wait` include args. Split on whitespace and
    pass through. Verifies by having the 'editor' record its argv."""
    marker = tmp_path / "argv.txt"
    editor = tmp_path / "arg_echo.sh"
    editor.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {marker}\necho extra >> "${{!#}}"\n')
    editor.chmod(0o755)
    monkeypatch.setenv("EDITOR", f"{editor} --flag")

    out = _open_in_editor("text")
    # Argv should contain the --flag plus the tmp file path.
    args = marker.read_text().splitlines()
    assert "--flag" in args
    assert out is not None
