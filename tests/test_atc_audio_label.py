"""Tests for scripts/atc_audio_label.py (harness-hp8k).

Pin the row-template defaults, the cross-field validation rules, the
resume-via-existing-indices behaviour, and one end-to-end label run
with Rich Prompt monkey-patched. Audio playback is monkey-patched out
so the suite doesn't depend on ffplay.
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
_SCRIPT = REPO / "scripts" / "atc_audio_label.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("atc_audio_label", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atc_audio_label"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def label() -> types.ModuleType:
    return _load_module()


# ---------- default_speaker heuristic ----------


def test_default_speaker_ctaf_is_pilot(label: Any) -> None:
    assert label.default_speaker("CTAF", "crash") == "pilot"


def test_default_speaker_tower_is_controller(label: Any) -> None:
    assert label.default_speaker("Twr", "crash") == "controller"
    assert label.default_speaker("App-North-Arrival", "blowntire") == "controller"
    assert label.default_speaker("Gnd-Twr", "park") == "controller"


def test_default_speaker_guard_is_unknown(label: Any) -> None:
    """121.5 is mostly pilots-calling-out but not exclusively — force the
    human to pick rather than seed a wrong default."""
    assert label.default_speaker("Guard", "TFR") == "unknown"


def test_default_speaker_no_position_is_unknown(label: Any) -> None:
    assert label.default_speaker(None, None) == "unknown"


# ---------- row template ----------


def _meta(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "clip_id": "abc123",
        "source_filename": "crashKAVQ2-CTAF-Apr-09-2026-0000Z.mp3",
        "norm_relpath": "norm/abc123.flac",
        "duration_s": 30.0,
        "parsed": {
            "icao": "KAVQ",
            "position": "CTAF",
            "event_tag": "crash",
            "recorded_at": "2026-04-09T00:00:00+00:00",
            "extras": [],
        },
    }
    base.update(overrides)
    return base


def _segment(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "start": 1.5,
        "end": 4.25,
        "text": " November 478 Alpha Tango",
        "avg_logprob": -0.32,
    }
    base.update(overrides)
    return base


def test_row_template_seed_defaults(label: Any) -> None:
    row = label.row_template(
        _meta(),
        _segment(),
        clip_id="abc123",
        utt_index=2,
        model_repo="mlx-community/whisper-large-v3-mlx",
    )
    assert row["clip_id"] == "abc123"
    assert row["utt_index"] == 2
    assert row["start_s"] == pytest.approx(1.5)
    assert row["end_s"] == pytest.approx(4.25)
    assert row["speaker_role"] == "pilot"  # CTAF heuristic
    assert row["facility"] == "KAVQ"
    assert row["position"] == "CTAF"
    assert row["frequency"] is None
    assert row["event_tag"] == "crash"
    assert row["transcript_text"] == "November 478 Alpha Tango"
    assert row["transcript_seed"] == "November 478 Alpha Tango"
    assert row["transcript_confidence"] == pytest.approx(-0.32)
    assert row["expected_verdict"] == "out_of_scope"
    assert row["expected_section"] is None
    assert row["expected_phraseology"] is None
    assert row["mismatch"] is None
    assert row["human_verified"] is False
    assert row["labeled_at"] is None


def test_row_template_picks_frequency_from_extras(label: Any) -> None:
    meta = _meta(parsed={**_meta()["parsed"], "extras": ["121100"]})
    row = label.row_template(meta, _segment(), clip_id="abc123", utt_index=0, model_repo="m")
    assert row["frequency"] == "121100"


def test_row_template_skips_non_digit_extras(label: Any) -> None:
    meta = _meta(parsed={**_meta()["parsed"], "extras": ["odd"]})
    row = label.row_template(meta, _segment(), clip_id="abc123", utt_index=0, model_repo="m")
    assert row["frequency"] is None


def test_row_template_default_verdict_is_oos(label: Any) -> None:
    """OOS is the most common verdict on radio audio (chatter, pilot
    side, ground noise) — so the default should be OOS to keep the
    one-key fast-path productive."""
    row = label.row_template(_meta(), _segment(), clip_id="abc123", utt_index=0, model_repo="m")
    assert row["expected_verdict"] == "out_of_scope"


# ---------- validate_row ----------


def _ok_row(label: Any) -> dict[str, Any]:
    row: dict[str, Any] = label.row_template(
        _meta(), _segment(), clip_id="abc123", utt_index=0, model_repo="m"
    )
    return row


def test_validate_oos_row_passes(label: Any) -> None:
    row = _ok_row(label)
    label.validate_row(row)


def test_validate_oos_with_section_fails(label: Any) -> None:
    row = _ok_row(label)
    row["expected_section"] = "3-9-10"
    with pytest.raises(ValueError, match="out_of_scope"):
        label.validate_row(row)


def test_validate_ok_requires_section(label: Any) -> None:
    row = _ok_row(label)
    row["expected_verdict"] = "ok"
    with pytest.raises(ValueError, match="expected_section"):
        label.validate_row(row)


def test_validate_wrong_requires_section_phraseology_mismatch(label: Any) -> None:
    row = _ok_row(label)
    row["expected_verdict"] = "wrong"
    row["expected_section"] = "3-9-10"
    with pytest.raises(ValueError, match="mismatch"):
        label.validate_row(row)
    row["mismatch"] = "wrong altitude"
    with pytest.raises(ValueError, match="phraseology"):
        label.validate_row(row)
    row["expected_phraseology"] = "CLEARED FOR TAKEOFF RUNWAY {N}"
    label.validate_row(row)


def test_validate_incomplete_requires_section_and_mismatch(label: Any) -> None:
    row = _ok_row(label)
    row["expected_verdict"] = "incomplete"
    row["expected_section"] = "3-9-10"
    with pytest.raises(ValueError, match="mismatch"):
        label.validate_row(row)
    row["mismatch"] = "missing runway number"
    label.validate_row(row)


def test_validate_rejects_bad_verdict(label: Any) -> None:
    row = _ok_row(label)
    row["expected_verdict"] = "maybe"
    with pytest.raises(ValueError, match="expected_verdict"):
        label.validate_row(row)


def test_validate_rejects_bad_speaker(label: Any) -> None:
    row = _ok_row(label)
    row["speaker_role"] = "bot"
    with pytest.raises(ValueError, match="speaker_role"):
        label.validate_row(row)


# ---------- existing-indices / append ----------


def test_load_existing_indices_returns_set(label: Any, tmp_path: Path) -> None:
    utt_path = tmp_path / "x.jsonl"
    utt_path.write_text(
        json.dumps({"utt_index": 0, "clip_id": "x"})
        + "\n"
        + json.dumps({"utt_index": 3, "clip_id": "x"})
        + "\n",
        encoding="utf-8",
    )
    assert label.load_existing_indices(utt_path) == {0, 3}


def test_load_existing_indices_tolerates_corrupt_lines(label: Any, tmp_path: Path) -> None:
    utt_path = tmp_path / "x.jsonl"
    utt_path.write_text(
        json.dumps({"utt_index": 0, "clip_id": "x"})
        + "\n"
        + "{not json\n"
        + json.dumps({"clip_id": "x"})
        + "\n"  # missing utt_index
        + json.dumps({"utt_index": 5, "clip_id": "x"})
        + "\n",
        encoding="utf-8",
    )
    assert label.load_existing_indices(utt_path) == {0, 5}


def test_load_existing_indices_missing_file(label: Any, tmp_path: Path) -> None:
    assert label.load_existing_indices(tmp_path / "nope.jsonl") == set()


def test_append_row_creates_parent_dir(label: Any, tmp_path: Path) -> None:
    utt_path = tmp_path / "utt" / "abc123.jsonl"
    label.append_row(utt_path, {"utt_index": 0, "x": 1})
    label.append_row(utt_path, {"utt_index": 1, "x": 2})
    rows = [
        json.loads(line)
        for line in utt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["utt_index"] for r in rows] == [0, 1]


def test_iter_unlabeled_skips_indices(label: Any) -> None:
    segments = [{"start": i} for i in range(5)]
    pending = list(label.iter_unlabeled(segments, {1, 3}))
    assert [idx for idx, _ in pending] == [0, 2, 4]


# ---------- end-to-end label loop with patched Prompt ----------


def _write_clip(target: Path, clip_id: str, segments: list[dict[str, Any]]) -> None:
    meta = _meta(clip_id=clip_id, source_filename=f"{clip_id}.mp3")
    meta["norm_relpath"] = f"norm/{clip_id}.flac"
    meta["clip_id"] = clip_id
    (target / "meta").mkdir(parents=True, exist_ok=True)
    (target / "stt").mkdir(parents=True, exist_ok=True)
    (target / "norm").mkdir(parents=True, exist_ok=True)
    (target / "meta" / f"{clip_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    (target / "stt" / f"{clip_id}.json").write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "model_repo": "mlx-community/whisper-tiny",
                "language": "en",
                "segments": segments,
            }
        ),
        encoding="utf-8",
    )
    (target / "norm" / f"{clip_id}.flac").write_bytes(b"FAKE")


@pytest.fixture
def quiet_audio(monkeypatch: pytest.MonkeyPatch, label: Any) -> None:
    monkeypatch.setattr(label, "play_segment", lambda *_a, **_k: None)
    monkeypatch.setattr(label, "stop_player", lambda _proc: None)


def _make_prompt_stub(answers: list[str]) -> Any:
    iterator = iter(answers)

    def stub(_prompt: str, **_kwargs: Any) -> str:
        return next(iterator)

    return stub


def test_label_loop_one_key_oos_path(
    label: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_audio: None
) -> None:
    target = tmp_path / "atc_audio"
    _write_clip(
        target,
        "cid1",
        [
            _segment(start=0.0, end=1.0, text=" alpha"),
            _segment(start=1.0, end=2.0, text=" bravo"),
            _segment(start=2.0, end=3.0, text=" charlie"),
        ],
    )

    # Three 'o' presses: one per utterance, all OOS fast-path.
    monkeypatch.setattr(label.Prompt, "ask", _make_prompt_stub(["o", "o", "o"]))

    rc = label.main(["--target", str(target), "--only", "cid1"])
    assert rc == 0

    rows = [
        json.loads(line)
        for line in (target / "utt" / "cid1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 3
    assert all(r["expected_verdict"] == "out_of_scope" for r in rows)
    assert all(r["human_verified"] is True for r in rows)
    assert [r["transcript_text"] for r in rows] == ["alpha", "bravo", "charlie"]


def test_label_loop_full_label_path(
    label: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_audio: None
) -> None:
    target = tmp_path / "atc_audio"
    # Single utterance, label it fully as a controller-issued 'wrong'.
    _write_clip(
        target,
        "cid1",
        [_segment(start=0.0, end=2.0, text=" delta one cleared takeoff")],
    )

    answers = [
        "l",  # top-level: full label
        "Delta One cleared for takeoff runway 9",  # transcript
        "c",  # speaker = controller
        "w",  # verdict = wrong
        "3-9-10",  # section
        "{N} CLEARED FOR TAKEOFF RUNWAY {RWY}",  # phraseology
        "missing 'for'; verb form drift",  # mismatch
        "",  # notes
    ]
    monkeypatch.setattr(label.Prompt, "ask", _make_prompt_stub(answers))

    rc = label.main(["--target", str(target), "--only", "cid1"])
    assert rc == 0

    rows = [
        json.loads(line)
        for line in (target / "utt" / "cid1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["expected_verdict"] == "wrong"
    assert row["expected_section"] == "3-9-10"
    assert row["mismatch"].startswith("missing 'for'")
    assert row["speaker_role"] == "controller"
    assert row["transcript_text"] == "Delta One cleared for takeoff runway 9"
    assert row["transcript_seed"] == "delta one cleared takeoff"
    assert row["human_verified"] is True


def test_label_loop_quit_saves_progress(
    label: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_audio: None
) -> None:
    target = tmp_path / "atc_audio"
    _write_clip(
        target,
        "cid1",
        [
            _segment(start=0.0, end=1.0, text=" alpha"),
            _segment(start=1.0, end=2.0, text=" bravo"),
            _segment(start=2.0, end=3.0, text=" charlie"),
        ],
    )

    # OOS first one, quit on the second. Third should not be reached.
    monkeypatch.setattr(label.Prompt, "ask", _make_prompt_stub(["o", "q"]))
    rc = label.main(["--target", str(target), "--only", "cid1"])
    assert rc == 0

    rows = [
        json.loads(line)
        for line in (target / "utt" / "cid1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["utt_index"] == 0


def test_label_loop_resume_skips_existing(
    label: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_audio: None
) -> None:
    target = tmp_path / "atc_audio"
    _write_clip(
        target,
        "cid1",
        [
            _segment(start=0.0, end=1.0, text=" alpha"),
            _segment(start=1.0, end=2.0, text=" bravo"),
        ],
    )
    # Pre-seed utt 0 as already labelled.
    label.append_row(
        target / "utt" / "cid1.jsonl",
        {"utt_index": 0, "clip_id": "cid1", "human_verified": True},
    )

    # Only one new prompt should be needed (utt 1).
    monkeypatch.setattr(label.Prompt, "ask", _make_prompt_stub(["o"]))
    rc = label.main(["--target", str(target), "--only", "cid1"])
    assert rc == 0

    rows = [
        json.loads(line)
        for line in (target / "utt" / "cid1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {r["utt_index"] for r in rows} == {0, 1}


def test_label_loop_skip_clip_advances(
    label: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_audio: None
) -> None:
    target = tmp_path / "atc_audio"
    _write_clip(target, "cid1", [_segment(text=" a"), _segment(text=" b")])
    _write_clip(target, "cid2", [_segment(text=" c")])

    # 's' on cid1 (skip rest of clip), 'o' on cid2.
    # Sort by recorded_at puts both at the same value since _meta returns
    # a constant — fall back to clip_id alphabetical → cid1, cid2.
    monkeypatch.setattr(label.Prompt, "ask", _make_prompt_stub(["s", "o"]))

    rc = label.main(["--target", str(target), "--only", "cid1", "cid2"])
    assert rc == 0

    cid1_path = target / "utt" / "cid1.jsonl"
    cid2_path = target / "utt" / "cid2.jsonl"
    assert not cid1_path.exists()  # nothing labelled before skip
    assert cid2_path.exists()
    rows = [
        json.loads(line)
        for line in cid2_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
