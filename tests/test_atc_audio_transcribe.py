"""Tests for scripts/atc_audio_transcribe.py (harness-xko1).

Pin the per-clip envelope shape, the idempotent skip path, the
clip-id discovery via meta/*.json, and the dry-run preview. The real
mlx-whisper call is monkey-patched (its model is 1.5 GB and Apple-
Silicon only); a separate manual smoke run exercises the live
transcribe pipeline before each push that touches it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
_SCRIPT = REPO / "scripts" / "atc_audio_transcribe.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("atc_audio_transcribe", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atc_audio_transcribe"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def transcribe() -> types.ModuleType:
    return _load_module()


# ---------- fixture builder ----------


def _write_meta(
    target_dir: Path,
    clip_id: str,
    *,
    source_filename: str,
    recorded_at: str | None = None,
    norm_bytes: bytes = b"FAKE-FLAC-bytes",
) -> Path:
    """Create a tiny ingest-style layout: norm/<cid>.flac + meta/<cid>.json."""
    meta_dir = target_dir / "meta"
    norm_dir = target_dir / "norm"
    meta_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_path = norm_dir / f"{clip_id}.flac"
    norm_path.write_bytes(norm_bytes)
    meta = {
        "clip_id": clip_id,
        "source_filename": source_filename,
        "norm_relpath": f"norm/{clip_id}.flac",
        "parsed": {"recorded_at": recorded_at},
    }
    meta_path = meta_dir / f"{clip_id}.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path


def _fake_transcribe_result(text: str = "hello atc") -> dict[str, Any]:
    """Mimic the mlx_whisper.transcribe return shape."""
    return {
        "text": text,
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": text,
                "tokens": [50364, 1, 2, 3],
                "temperature": 0.0,
                "avg_logprob": -0.4,
                "compression_ratio": 1.6,
                "no_speech_prob": 0.1,
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.9},
                    {"word": "atc", "start": 0.5, "end": 1.0, "probability": 0.85},
                ],
            }
        ],
    }


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch, transcribe: Any) -> None:
    def fake(audio_path: Path, model_repo: str) -> dict[str, Any]:
        # Echo the model in the text so tests can assert it was threaded through.
        return _fake_transcribe_result(text=f"[{model_repo}] {audio_path.name}")

    monkeypatch.setattr(transcribe, "transcribe_clip", fake)


# ---------- transcribe_one envelope ----------


def test_transcribe_one_writes_envelope(
    transcribe: Any, tmp_path: Path, fake_whisper: None
) -> None:
    target = tmp_path / "atc_audio"
    _write_meta(target, "abc123", source_filename="crashKAVQ2-CTAF-Apr-09-2026-0000Z.mp3")

    envelope, was_new = transcribe.transcribe_one("abc123", target, "mlx-community/whisper-tiny")

    assert was_new is True
    assert envelope["clip_id"] == "abc123"
    assert envelope["source_filename"] == "crashKAVQ2-CTAF-Apr-09-2026-0000Z.mp3"
    assert envelope["model_repo"] == "mlx-community/whisper-tiny"
    assert envelope["language"] == "en"
    assert "[mlx-community/whisper-tiny]" in envelope["text"]
    assert isinstance(envelope["segments"], list)
    assert envelope["segments"][0]["words"][0]["word"] == "hello"

    # Persisted to stt/<clip_id>.json with the same shape.
    stt_path = target / "stt" / "abc123.json"
    assert stt_path.exists()
    persisted = json.loads(stt_path.read_text(encoding="utf-8"))
    assert persisted["clip_id"] == "abc123"
    assert persisted["text"] == envelope["text"]


def test_transcribe_one_is_idempotent(transcribe: Any, tmp_path: Path, fake_whisper: None) -> None:
    target = tmp_path / "atc_audio"
    _write_meta(target, "abc123", source_filename="x.mp3")

    transcribe.transcribe_one("abc123", target, "mlx-community/whisper-tiny")
    _, was_new = transcribe.transcribe_one("abc123", target, "mlx-community/whisper-tiny")
    assert was_new is False


def test_transcribe_one_force_re_runs(transcribe: Any, tmp_path: Path, fake_whisper: None) -> None:
    target = tmp_path / "atc_audio"
    _write_meta(target, "abc123", source_filename="x.mp3")

    transcribe.transcribe_one("abc123", target, "mlx-community/whisper-tiny")
    envelope, was_new = transcribe.transcribe_one(
        "abc123", target, "mlx-community/whisper-tiny", force=True
    )
    assert was_new is True
    assert envelope["model_repo"] == "mlx-community/whisper-tiny"


def test_transcribe_one_force_can_swap_model(
    transcribe: Any, tmp_path: Path, fake_whisper: None
) -> None:
    target = tmp_path / "atc_audio"
    _write_meta(target, "abc123", source_filename="x.mp3")

    transcribe.transcribe_one("abc123", target, "mlx-community/whisper-tiny")
    new_envelope, _ = transcribe.transcribe_one(
        "abc123", target, "mlx-community/whisper-large-v3-mlx", force=True
    )
    assert new_envelope["model_repo"] == "mlx-community/whisper-large-v3-mlx"


def test_transcribe_one_missing_meta_raises(transcribe: Any, tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    target.mkdir()
    with pytest.raises(FileNotFoundError):
        transcribe.transcribe_one("nope", target, "mlx-community/whisper-tiny")


def test_transcribe_one_missing_norm_audio_raises(transcribe: Any, tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    meta_path = _write_meta(target, "abc123", source_filename="x.mp3")
    # Remove the norm file the meta points at.
    (target / "norm" / "abc123.flac").unlink()
    assert meta_path.exists()
    with pytest.raises(FileNotFoundError):
        transcribe.transcribe_one("abc123", target, "mlx-community/whisper-tiny")


# ---------- clip-id discovery ----------


def test_list_clip_ids_sorted_by_recorded_at(transcribe: Any, tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    _write_meta(target, "later", source_filename="b.mp3", recorded_at="2026-04-15T00:00:00+00:00")
    _write_meta(target, "earlier", source_filename="a.mp3", recorded_at="2026-03-10T00:00:00+00:00")
    _write_meta(target, "messy", source_filename="c.mp3", recorded_at=None)
    ids = transcribe._list_clip_ids(target)
    # Empty-string recorded_at sorts first; then chronological.
    assert ids[0] == "messy"
    assert ids[1] == "earlier"
    assert ids[2] == "later"


def test_list_clip_ids_only_filter(transcribe: Any, tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    _write_meta(target, "a", source_filename="a.mp3")
    _write_meta(target, "b", source_filename="b.mp3")
    _write_meta(target, "c", source_filename="c.mp3")
    ids = transcribe._list_clip_ids(target, only=("a", "c"))
    assert set(ids) == {"a", "c"}


def test_list_clip_ids_empty_when_no_meta(transcribe: Any, tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    assert transcribe._list_clip_ids(target) == []


# ---------- main: dry-run and limit ----------


def test_main_dry_run_does_not_load_model(
    transcribe: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run must never call into mlx_whisper, so it works on
    machines without `--extra asr` installed."""

    def boom(*_args: object, **_kw: object) -> dict[str, Any]:
        raise AssertionError("model loaded on dry-run")

    monkeypatch.setattr(transcribe, "transcribe_clip", boom)

    target = tmp_path / "atc_audio"
    _write_meta(target, "a", source_filename="a.mp3")
    _write_meta(target, "b", source_filename="b.mp3")

    rc = transcribe.main(["--target", str(target), "--dry-run"])
    assert rc == 0
    assert not (target / "stt").exists()


def test_main_limit_caps_clips(transcribe: Any, tmp_path: Path, fake_whisper: None) -> None:
    target = tmp_path / "atc_audio"
    _write_meta(target, "a", source_filename="a.mp3", recorded_at="2026-03-01T00:00:00+00:00")
    _write_meta(target, "b", source_filename="b.mp3", recorded_at="2026-03-02T00:00:00+00:00")
    _write_meta(target, "c", source_filename="c.mp3", recorded_at="2026-03-03T00:00:00+00:00")

    rc = transcribe.main(["--target", str(target), "--limit", "2"])
    assert rc == 0
    written = sorted(p.stem for p in (target / "stt").glob("*.json"))
    assert written == ["a", "b"]


def test_main_no_meta_returns_error(transcribe: Any, tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    target.mkdir()
    rc = transcribe.main(["--target", str(target)])
    assert rc == 1
