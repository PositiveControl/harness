"""Tests for scripts/atc_ingest.py (harness-xbk.6).

Covers the pure helpers — noise filter, dedup, external-id / principle
/ tags derivation, load_rows_for_slug — without opening the real
EpisodicStore. The store path is exercised end-to-end via the ingest
run itself; these tests pin the filter + metadata shape so downstream
changes don't drift.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SCRIPT = REPO / "scripts" / "atc_ingest.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("atc_ingest", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atc_ingest"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ingest() -> object:
    return _load_module()


# ---------- noise filter ----------


def test_is_noise_rejects_short_body(ingest: object) -> None:
    row = {"body": "tiny", "source": "PCG", "section": "X"}
    assert ingest.is_noise(row) is True  # type: ignore[attr-defined]


def test_is_noise_accepts_long_body(ingest: object) -> None:
    row = {"body": "x" * 60, "source": "PCG", "section": "FOO"}
    assert ingest.is_noise(row) is False  # type: ignore[attr-defined]


# ---------- dedup by anchor ----------


def test_dedup_by_anchor_keeps_longest_body(ingest: object) -> None:
    """JO / AIM ship an 'Explanation of Changes' block that re-states
    every touched anchor with a short change-description body, AND
    the real-content body later. Dedup must keep the richer row."""
    short = {"source": "JO_7110.65", "section": "2-6-4", "chunk_index": 0, "body": "short"}
    long_body = {"source": "JO_7110.65", "section": "2-6-4", "chunk_index": 0, "body": "x" * 400}
    other = {"source": "AIM", "section": "3-5-5", "chunk_index": 0, "body": "y" * 100}
    out = ingest.dedup_by_anchor([short, long_body, other])  # type: ignore[attr-defined]
    by_key = {(r["source"], r["section"]): r for r in out}
    assert len(out) == 2
    assert by_key[("JO_7110.65", "2-6-4")]["body"] == "x" * 400


def test_dedup_preserves_different_chunk_indices(ingest: object) -> None:
    """A section split across multiple rows (chunk_index 0, 1, 2) must
    keep every piece — size-shaping is upstream, and each piece has
    its own content."""
    a = {"source": "CFR_14_vol2", "section": "91.155", "chunk_index": 0, "body": "a"}
    b = {"source": "CFR_14_vol2", "section": "91.155", "chunk_index": 1, "body": "b"}
    c = {"source": "CFR_14_vol2", "section": "91.155", "chunk_index": 2, "body": "c"}
    out = ingest.dedup_by_anchor([a, b, c])  # type: ignore[attr-defined]
    assert len(out) == 3


# ---------- external id + metadata derivation ----------


def test_external_id_format(ingest: object) -> None:
    row = {"source": "JO_7110.65", "section": "2-6-4", "chunk_index": 0}
    assert ingest.external_id_for(row) == "atc-corpus:JO_7110.65:2-6-4:0"  # type: ignore[attr-defined]


def test_external_id_includes_chunk_index_for_splits(ingest: object) -> None:
    row = {"source": "AIM", "section": "4-7-1", "chunk_index": 3}
    assert ingest.external_id_for(row) == "atc-corpus:AIM:4-7-1:3"  # type: ignore[attr-defined]


def test_principle_encodes_section_anchor(ingest: object) -> None:
    """The principle string goes into _build_embed_text's tag header
    AND as a standalone line — both lift retrieval on section-id
    queries. Shape must stay stable across ingests."""
    row = {"source": "JO_7110.65", "section": "2-6-4"}
    assert ingest.principle_for(row) == "JO_7110.65 §2-6-4"  # type: ignore[attr-defined]


def test_tags_extend_with_section_and_chapter(ingest: object) -> None:
    row = {
        "source": "AIM",
        "section": "3-5-5",
        "chapter": "3",
        "tags": ["atc", "corpus", "AIM"],
    }
    out = ingest.tags_for(row)  # type: ignore[attr-defined]
    assert "atc" in out
    assert "corpus" in out
    assert "AIM" in out
    assert "section:3-5-5" in out
    assert "chapter:3" in out


def test_tags_skip_empty_metadata(ingest: object) -> None:
    row = {"source": "PCG", "section": "WAKE TURBULENCE", "chapter": "", "tags": ["atc"]}
    out = ingest.tags_for(row)  # type: ignore[attr-defined]
    assert "section:WAKE TURBULENCE" in out
    # No chapter tag when chapter is empty.
    assert not any(t.startswith("chapter:") for t in out)


# ---------- CFR scope filter (harness-74n) ----------


def test_cfr_in_scope_accepts_part_91_standard_sections(ingest: object) -> None:
    """Part 91 subparts A-J (§91.1-§91.999) are the pilot-facing rules
    — must pass scope."""
    for section in ("91.1", "91.103", "91.155", "91.175", "91.185", "91.215", "91.999"):
        assert ingest._cfr_in_scope(section), section  # type: ignore[attr-defined]


def test_cfr_in_scope_rejects_part_91_subpart_k(ingest: object) -> None:
    """Part 91 Subpart K (fractional ownership, §§91.1001-91.1099) is
    commercial-specific — must drop. The baseline run 1 showed
    §91.1031 diluting 'preflight action' retrieval against the real
    §91.103."""
    for section in ("91.1001", "91.1031", "91.1039", "91.1099", "91.1101"):
        assert not ingest._cfr_in_scope(section), section  # type: ignore[attr-defined]


def test_cfr_in_scope_accepts_pilot_certification(ingest: object) -> None:
    """Part 61 (airman cert) and Part 67 (medical) cover PPL/IFR
    prerequisites — must pass scope."""
    for section in ("61.3", "61.57", "61.95", "67.101", "67.305"):
        assert ingest._cfr_in_scope(section), section  # type: ignore[attr-defined]


def test_cfr_in_scope_rejects_out_of_scope_parts(ingest: object) -> None:
    """Parts outside the Phase-1 PPL/IFR allowlist drop — airworthiness
    standards (Parts 23/25/39), UAS (Part 107), ultralights (103),
    parachuting (105), commercial ops (121/135)."""
    for section in ("23.1", "25.101", "39.11", "103.23", "105.3", "107.1", "121.365", "135.1"):
        assert not ingest._cfr_in_scope(section), section  # type: ignore[attr-defined]


def test_cfr_in_scope_empty_section_rejects(ingest: object) -> None:
    assert not ingest._cfr_in_scope("")  # type: ignore[attr-defined]


def test_is_noise_drops_out_of_scope_cfr_rows(ingest: object) -> None:
    """End-to-end: is_noise treats §103.23 as noise even when the body
    is substantial, because ultralights aren't in atc's scope."""
    row = {
        "source": "CFR_14_vol2",
        "section": "103.23",
        "body": "x" * 400,  # plenty long — not a length-filter drop
    }
    assert ingest.is_noise(row) is True  # type: ignore[attr-defined]


def test_is_noise_keeps_in_scope_cfr_rows(ingest: object) -> None:
    row = {
        "source": "CFR_14_vol2",
        "section": "91.155",
        "body": "x" * 400,
    }
    assert ingest.is_noise(row) is False  # type: ignore[attr-defined]


def test_is_noise_keeps_non_cfr_rows_regardless(ingest: object) -> None:
    """AIM / PCG / JO / PHAK rows don't hit the CFR filter — only
    the body-length check."""
    row = {
        "source": "AIM",
        "section": "4-4-7",
        "body": "x" * 400,
    }
    assert ingest.is_noise(row) is False  # type: ignore[attr-defined]


# ---------- load_rows_for_slug ----------


def test_load_rows_for_slug_missing_file_returns_empty(
    ingest: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ingest, "CORPUS_CHUNKS", tmp_path)
    assert ingest.load_rows_for_slug("does_not_exist") == []  # type: ignore[attr-defined]


def test_load_rows_for_slug_applies_filter_and_dedup(
    ingest: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-trip: write a jsonl fixture that mixes short-body noise,
    duplicate anchors, and good rows. load_rows_for_slug must drop the
    first and dedup the second."""
    monkeypatch.setattr(ingest, "CORPUS_CHUNKS", tmp_path)
    path = tmp_path / "aim.jsonl"
    rows = [
        # Too-short body — dropped.
        {"source": "AIM", "section": "1-1-1", "chunk_index": 0, "body": "x"},
        # Duplicate anchor — longest wins.
        {"source": "AIM", "section": "3-5-5", "chunk_index": 0, "body": "short version"},
        {"source": "AIM", "section": "3-5-5", "chunk_index": 0, "body": "y" * 200},
        # Unique good row.
        {"source": "AIM", "section": "4-7-1", "chunk_index": 0, "body": "z" * 200},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = ingest.load_rows_for_slug("aim")  # type: ignore[attr-defined]
    sections = sorted({r["section"] for r in out})
    assert sections == ["3-5-5", "4-7-1"]
    kept = next(r for r in out if r["section"] == "3-5-5")
    assert kept["body"] == "y" * 200
