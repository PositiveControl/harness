"""Tests for scripts/atc_audio_ingest.py (harness-d8o2).

Pin the filename parser, the clip-id derivation, the meta/index layout,
and the idempotent re-run path. ffmpeg is monkey-patched in the
end-to-end test so the suite doesn't depend on ffmpeg being installed
on the test runner — there's a separate pre-push smoke run that
exercises the real binary against real audio.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
_SCRIPT = REPO / "scripts" / "atc_audio_ingest.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("atc_audio_ingest", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atc_audio_ingest"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ingest() -> types.ModuleType:
    return _load_module()


# ---------- parse_filename: dominant shape ----------


def test_parse_canonical_event_icao_position_date(ingest: Any) -> None:
    parsed = ingest.parse_filename("crashKAVQ2-CTAF-Apr-09-2026-0000Z.mp3")
    assert parsed.parse_failed is False
    assert parsed.event_tag == "crash"
    assert parsed.icao == "KAVQ"
    assert parsed.icao_source == "2"
    assert parsed.position == "CTAF"
    assert parsed.extras == []
    assert parsed.recorded_at == "2026-04-09T00:00:00+00:00"
    assert parsed.comment is None
    assert parsed.duplicate_index == 0


def test_parse_no_event_tag_starts_with_icao(ingest: Any) -> None:
    parsed = ingest.parse_filename("KMRB1-Guard-Apr-12-2026-1900Z meow.mp3")
    assert parsed.parse_failed is False
    assert parsed.event_tag is None
    assert parsed.icao == "KMRB"
    assert parsed.icao_source == "1"
    assert parsed.position == "Guard"
    assert parsed.comment == "meow"


def test_parse_no_position_section(ingest: Any) -> None:
    """Some files go straight from `<event><ICAO>` to the date tail."""
    parsed = ingest.parse_filename("electricfailKTTA-Apr-24-2026-1530Z.mp3")
    assert parsed.parse_failed is False
    assert parsed.event_tag == "electricfail"
    assert parsed.icao == "KTTA"
    assert parsed.icao_source is None
    assert parsed.position is None
    assert parsed.extras == []


def test_parse_frequency_in_position_block(ingest: Any) -> None:
    """Numeric tokens between ICAO and date split off into `extras`."""
    parsed = ingest.parse_filename("7700KIND9-App-121100-Apr-09-2026-0030Z.mp3")
    assert parsed.event_tag == "7700"
    assert parsed.icao == "KIND"
    assert parsed.icao_source == "9"
    assert parsed.position == "App"
    assert parsed.extras == ["121100"]


def test_parse_compound_position(ingest: Any) -> None:
    parsed = ingest.parse_filename("blowntireKEWR-App-North-Arrival-Mar-26-2026-0030Z.mp3")
    assert parsed.event_tag == "blowntire"
    assert parsed.icao == "KEWR"
    assert parsed.position == "App-North-Arrival"


def test_parse_two_icao_keeps_secondary_in_position(ingest: Any) -> None:
    """A secondary airport in the head stays opaque in `position`; the
    labelling step splits it out if needed. v0 just has to be stable."""
    parsed = ingest.parse_filename("wingclipKSAN-KMYF-Twr-Pri-Mar-31-2026-1700Z.mp3")
    assert parsed.event_tag == "wingclip"
    assert parsed.icao == "KSAN"
    assert parsed.position == "KMYF-Twr-Pri"


def test_parse_caribbean_icao_t_prefix(ingest: Any) -> None:
    parsed = ingest.parse_filename("waterTJIG2-Twr-Apr-15-2026-1400Z.mp3")
    assert parsed.icao == "TJIG"
    assert parsed.icao_source == "2"
    assert parsed.event_tag == "water"


def test_parse_livatc_alphanumeric_source_suffix(ingest: Any) -> None:
    """LiveATC sometimes mints suffixes like `KORD1N2` — keep them as
    a single opaque source string rather than dropping the trailing
    chars."""
    parsed = ingest.parse_filename("hydrofailKORD1N2-App-119000-Apr-16-2026-0130Z.mp3")
    assert parsed.icao == "KORD"
    assert parsed.icao_source == "1N2"
    assert parsed.position == "App"
    assert parsed.extras == ["119000"]


def test_parse_dup_suffix_marker(ingest: Any) -> None:
    parsed = ingest.parse_filename("TFRKPBI2-Guard-Mar-29-2026-1730Z (1).mp3")
    assert parsed.parse_failed is False
    assert parsed.duplicate_index == 1
    assert parsed.icao == "KPBI"
    assert parsed.event_tag == "TFR"


# ---------- parse_filename: messy fallbacks ----------


def test_parse_failed_when_no_date_tail(ingest: Any) -> None:
    """Plain callsign-only filenames fall to the messy bucket — they're
    still ingested, just flagged for the labelling step to triage."""
    parsed = ingest.parse_filename("DAL1082.mp3")
    assert parsed.parse_failed is True
    assert parsed.icao is None
    assert parsed.recorded_at is None
    assert parsed.raw_stem == "DAL1082"


def test_parse_failed_with_freeform_text(ingest: Any) -> None:
    parsed = ingest.parse_filename("UAL 2384 RTO-odor-emergency.mp3")
    assert parsed.parse_failed is True
    assert parsed.raw_stem == "UAL 2384 RTO-odor-emergency"


def test_parse_starts_with_icao_only_trailing_comment(ingest: Any) -> None:
    """File starts directly with `KDCA1-` then date + free comment."""
    parsed = ingest.parse_filename("KDCA1-Apr-22-2026-1930Z PAT26 to Fort Meyer INBOUND.mp3")
    assert parsed.parse_failed is False
    assert parsed.event_tag is None
    assert parsed.icao == "KDCA"
    assert parsed.icao_source == "1"
    assert parsed.position is None
    assert parsed.comment == "PAT26 to Fort Meyer INBOUND"


# ---------- clip_id ----------


def test_clip_id_is_sha256_prefix(ingest: Any, tmp_path: Path) -> None:
    src = tmp_path / "x.mp3"
    src.write_bytes(b"hello atc")
    cid = ingest.clip_id_for(src)
    assert len(cid) == 12
    assert all(c in "0123456789abcdef" for c in cid)
    # Same bytes → same id (idempotency anchor).
    src2 = tmp_path / "y.mp3"
    src2.write_bytes(b"hello atc")
    assert ingest.clip_id_for(src2) == cid


def test_clip_id_differs_for_different_bytes(ingest: Any, tmp_path: Path) -> None:
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(b"clip-A")
    b.write_bytes(b"clip-B")
    assert ingest.clip_id_for(a) != ingest.clip_id_for(b)


# ---------- ingest_clip end-to-end (ffmpeg monkey-patched) ----------


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch, ingest: Any) -> None:
    """Replace ffmpeg + ffprobe with no-op writers so the suite doesn't
    require ffmpeg on the test runner. The real binary gets exercised
    in the pre-push smoke run."""

    def fake_normalize(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"FAKE-FLAC-" + src.read_bytes())

    def fake_duration(_path: Path) -> float | None:
        return 1.25

    monkeypatch.setattr(ingest, "ffmpeg_normalize", fake_normalize)
    monkeypatch.setattr(ingest, "ffprobe_duration", fake_duration)


def test_ingest_clip_writes_artefacts(ingest: Any, tmp_path: Path, fake_ffmpeg: None) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "crashKAVQ2-CTAF-Apr-09-2026-0000Z.mp3"
    src.write_bytes(b"audio-bytes-1")

    target = tmp_path / "out"
    meta, was_new = ingest.ingest_clip(src, target)

    assert was_new is True
    cid = meta["clip_id"]
    assert isinstance(cid, str)
    assert len(cid) == 12
    assert (target / "raw" / src.name).exists()
    assert (target / "norm" / f"{cid}.flac").exists()
    assert (target / "meta" / f"{cid}.json").exists()
    assert meta["norm_sample_rate"] == 16_000
    assert meta["norm_channels"] == 1
    assert meta["duration_s"] == pytest.approx(1.25)

    parsed = meta["parsed"]
    assert isinstance(parsed, dict)
    assert parsed["icao"] == "KAVQ"
    assert parsed["event_tag"] == "crash"


def test_ingest_clip_is_idempotent(ingest: Any, tmp_path: Path, fake_ffmpeg: None) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "evacKEWR-Twr-Mar-23-2026-1100Z.mp3"
    src.write_bytes(b"audio-bytes-2")
    target = tmp_path / "out"

    _, first_new = ingest.ingest_clip(src, target)
    _, second_new = ingest.ingest_clip(src, target)

    assert first_new is True
    assert second_new is False


def test_ingest_clip_force_re_runs(ingest: Any, tmp_path: Path, fake_ffmpeg: None) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "evacKEWR-Twr-Mar-23-2026-1100Z.mp3"
    src.write_bytes(b"audio-bytes-3")
    target = tmp_path / "out"

    ingest.ingest_clip(src, target)
    _, was_new = ingest.ingest_clip(src, target, force=True)
    assert was_new is True


def test_ingest_clip_no_copy_raw(ingest: Any, tmp_path: Path, fake_ffmpeg: None) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "fireKMKE3-App-Dep-Apr-27-2026-0030Z.mp3"
    src.write_bytes(b"audio-bytes-4")
    target = tmp_path / "out"

    meta, _ = ingest.ingest_clip(src, target, copy_raw=False)
    assert meta["raw_relpath"] is None
    assert not (target / "raw" / src.name).exists()


def test_rebuild_index_sorts_by_recorded_at(ingest: Any, tmp_path: Path, fake_ffmpeg: None) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    later = src_dir / "fireKMKE3-App-Dep-Apr-27-2026-0030Z.mp3"
    earlier = src_dir / "evacKEWR-Twr-Mar-23-2026-1100Z.mp3"
    later.write_bytes(b"audio-late")
    earlier.write_bytes(b"audio-early")
    target = tmp_path / "out"

    ingest.ingest_clip(later, target)
    ingest.ingest_clip(earlier, target)

    n = ingest.rebuild_index(target)
    assert n == 2

    rows = [
        json.loads(line)
        for line in (target / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["parsed"]["recorded_at"].startswith("2026-03-23")
    assert rows[1]["parsed"]["recorded_at"].startswith("2026-04-27")


def test_main_dry_run_does_not_invoke_ffmpeg(
    ingest: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run should never call ffmpeg even when present, so the
    parser preview can run on a cold checkout."""

    def boom(*_args: object, **_kw: object) -> None:
        raise AssertionError("ffmpeg invoked on dry-run")

    monkeypatch.setattr(ingest, "ffmpeg_normalize", boom)
    monkeypatch.setattr(ingest, "ffprobe_duration", boom)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "crashKAVQ2-CTAF-Apr-09-2026-0000Z.mp3").write_bytes(b"x")
    (src_dir / "DAL1082.mp3").write_bytes(b"y")
    target = tmp_path / "out"

    rc = ingest.main(["--source", str(src_dir), "--target", str(target), "--dry-run"])
    assert rc == 0
    # No artefacts written.
    assert not (target / "meta").exists()
    assert not (target / "norm").exists()
