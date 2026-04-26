"""Tests for harness.orchestrator.section_index (harness-aise).

The index reads chunks JSONL files and returns the canonical §-anchor
set + §N-N parents. Pure-function — no embedder, no SQLite. Tests
build a synthetic chunks dir under tmp_path so the assertions don't
depend on the real airton_c1 corpus state."""

from __future__ import annotations

import json
from pathlib import Path

from harness.orchestrator.section_index import collect_valid_anchors


def _write_chunks(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def test_missing_dir_returns_empty_frozenset(tmp_path: Path) -> None:
    """Non-corpus characters get an empty set — hook stays silent."""
    assert collect_valid_anchors(tmp_path / "does-not-exist") == frozenset()


def test_dir_exists_but_no_jsonl_returns_empty(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    assert collect_valid_anchors(chunks_dir) == frozenset()


def test_section_anchors_emit_canonical_form(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    _write_chunks(
        chunks_dir / "doc.jsonl",
        [
            {"section": "3-10-3", "title": "SAME RUNWAY"},
            {"section": "4-1-1", "title": "ALTITUDE"},
        ],
    )
    anchors = collect_valid_anchors(chunks_dir)
    assert "§3-10-3" in anchors
    assert "§4-1-1" in anchors


def test_n_n_n_anchors_also_emit_n_n_parent(tmp_path: Path) -> None:
    """Reply citing the parent section without paragraph passes."""
    chunks_dir = tmp_path / "chunks"
    _write_chunks(
        chunks_dir / "doc.jsonl",
        [{"section": "3-10-3", "title": "X"}],
    )
    anchors = collect_valid_anchors(chunks_dir)
    assert "§3-10-3" in anchors
    assert "§3-10" in anchors


def test_n_n_anchors_do_not_synthesize_paragraph(tmp_path: Path) -> None:
    """A row stamped with `section=3-10` only contributes §3-10. We
    don't fabricate §3-10-1 / §3-10-2 from an absent paragraph stamp
    — that would over-accept invented paragraphs of a real section."""
    chunks_dir = tmp_path / "chunks"
    _write_chunks(
        chunks_dir / "doc.jsonl",
        [{"section": "3-10", "title": "PARENT"}],
    )
    anchors = collect_valid_anchors(chunks_dir)
    assert anchors == frozenset({"§3-10"})


def test_multiple_jsonl_files_unioned(tmp_path: Path) -> None:
    """Characters with multiple corpus files get the union — covers
    the future airton_c case (JO + AIM + CFR all under one chunks dir)."""
    chunks_dir = tmp_path / "chunks"
    _write_chunks(
        chunks_dir / "jo.jsonl",
        [{"section": "3-10-3", "title": "X"}],
    )
    _write_chunks(
        chunks_dir / "aim.jsonl",
        [{"section": "5-2-1", "title": "Y"}],
    )
    anchors = collect_valid_anchors(chunks_dir)
    assert "§3-10-3" in anchors
    assert "§5-2-1" in anchors


def test_malformed_lines_skipped_not_raised(tmp_path: Path) -> None:
    """A single bad row shouldn't disable the whole index — surrounding
    valid rows still surface their anchors."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "doc.jsonl").write_text(
        '{"section": "3-10-3"}\nnot json at all\n{"section": "4-1-1"}\n'
    )
    anchors = collect_valid_anchors(chunks_dir)
    assert "§3-10-3" in anchors
    assert "§4-1-1" in anchors


def test_rows_without_section_field_skipped(tmp_path: Path) -> None:
    """A row that legitimately doesn't carry a section (some chunkers
    emit doc-level metadata rows) doesn't pollute the anchor set."""
    chunks_dir = tmp_path / "chunks"
    _write_chunks(
        chunks_dir / "doc.jsonl",
        [
            {"section": "3-10-3"},
            {"title": "metadata-only", "kind": "header"},
            {"section": "", "title": "empty-section"},
            {"section": None, "title": "null-section"},
        ],
    )
    anchors = collect_valid_anchors(chunks_dir)
    # Only the first row contributed.
    assert anchors == frozenset({"§3-10-3", "§3-10"})


def test_blank_lines_in_jsonl_skipped(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "doc.jsonl").write_text('\n\n{"section": "3-10-3"}\n\n{"section": "4-1-1"}\n\n')
    anchors = collect_valid_anchors(chunks_dir)
    assert "§3-10-3" in anchors
    assert "§4-1-1" in anchors


def test_returns_frozenset_not_mutable(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    _write_chunks(
        chunks_dir / "doc.jsonl",
        [{"section": "3-10-3"}],
    )
    anchors = collect_valid_anchors(chunks_dir)
    assert isinstance(anchors, frozenset)
