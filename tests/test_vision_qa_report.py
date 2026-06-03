"""Pins the precision/recall math in scripts/vision_qa_report.py
(harness-ke4hx.5). The Phase-2 gating decision rides on these numbers, so
the aggregation is tested even though the file lives under scripts/ (loaded
via importlib — scripts/ is not an importable package)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vision_qa_report.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("vision_qa_report", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: `from __future__ import annotations` + @dataclass
    # makes dataclasses resolve annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


vqr = _load()


def test_summarize_empty() -> None:
    rep = vqr.summarize([])
    assert rep.total == 0
    assert rep.agreement == 0.0
    assert rep.precision == 0.0
    assert rep.recall == 0.0


def test_summarize_confusion_and_metrics() -> None:
    records = [
        {"verdict": "fail", "gate_passed": False, "issue_id": "a", "reason": "blank"},  # TP
        {"verdict": "fail", "gate_passed": True, "issue_id": "b", "reason": "looks off"},  # FP
        {"verdict": "pass", "gate_passed": False, "issue_id": "c", "reason": "ok"},  # FN
        {"verdict": "pass", "gate_passed": True, "issue_id": "d", "reason": "ok"},  # TN
        {"verdict": "unknown", "gate_passed": True, "issue_id": "e"},  # excluded
        {"verdict": "fail"},  # parse error — no gate_passed
    ]
    rep = vqr.summarize(records)

    assert rep.total == 5
    assert rep.parse_errors == 1
    assert (rep.tp, rep.fp, rep.fn, rep.tn) == (1, 1, 1, 1)
    assert rep.unknown_excluded == 1
    assert rep.gate_pass == 3
    assert rep.gate_fail == 2
    assert rep.agreement == 0.5
    assert rep.precision == 0.5
    assert rep.recall == 0.5
    assert [r["issue_id"] for r in rep.flags_gate_passed] == ["b"]
    assert [r["issue_id"] for r in rep.blind_spots] == ["c"]


def test_summarize_all_agree_perfect() -> None:
    records = [
        {"verdict": "pass", "gate_passed": True, "issue_id": "a"},
        {"verdict": "fail", "gate_passed": False, "issue_id": "b"},
    ]
    rep = vqr.summarize(records)
    assert rep.agreement == 1.0
    assert rep.recall == 1.0
    assert rep.precision == 1.0
    assert rep.blind_spots == []
    assert rep.flags_gate_passed == []


def test_load_records_reads_jsonl_and_warns_on_missing(tmp_path: Path) -> None:
    log = tmp_path / "vision_qa.jsonl"
    log.write_text(
        '{"verdict": "pass", "gate_passed": true, "issue_id": "a"}\n'
        "\n"  # blank line ignored
        "not json\n"  # unparseable → warning
        '{"verdict": "fail", "gate_passed": false, "issue_id": "b"}\n'
    )
    records, warnings = vqr.load_records([log, tmp_path / "missing.jsonl"])
    assert len(records) == 2
    assert any("unparseable" in w for w in warnings)
    assert any("no such file" in w for w in warnings)


def test_render_text_empty_is_friendly() -> None:
    out = vqr.render_text(vqr.summarize([]))
    assert "no advisory-QA data yet" in out


def test_render_text_includes_metrics() -> None:
    records = [
        {"verdict": "fail", "gate_passed": False, "issue_id": "a", "reason": "blank canvas"},
        {"verdict": "pass", "gate_passed": False, "issue_id": "c", "reason": "missed it"},
    ]
    out = vqr.render_text(vqr.summarize(records))
    assert "agreement:" in out
    assert "BLIND SPOTS" in out
    assert "missed it" in out


def test_main_json_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "vision_qa.jsonl"
    log.write_text('{"verdict": "fail", "gate_passed": false, "issue_id": "a"}\n')
    rc = vqr.main([str(log), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"recall"' in out
    assert '"tp": 1' in out
