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


# ---------- collect_valid_anchors_from_markdown ----------


class _StubTreeSpec:
    """Minimal shape that satisfies `DocumentTreeSpec`-shaped duck-typing
    inside `collect_valid_anchors_from_markdown` (it reads `source_path`
    + `source_format` via getattr). Avoids importing the real Character
    / DocumentTreeSpec just to stand up a test fixture."""

    def __init__(self, source_path: Path, source_format: str = "markdown") -> None:
        self.source_path = source_path
        self.source_format = source_format


def test_markdown_walker_extracts_numeric_headings(tmp_path: Path) -> None:
    """Numeric section headings at any depth (## or ###) contribute
    `§N` / `§N.N` anchors. Section title text is irrelevant."""
    from harness.orchestrator.section_index import collect_valid_anchors_from_markdown

    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Example: A Short RFC-Style Document\n"
        "\n"
        "## 1. Scope\n"
        "Some body text.\n"
        "\n"
        "## 2. Terminology\n"
        "\n"
        "### 2.1 Greeter\n"
        "Body.\n"
        "### 2.2 Greetee\n"
        "More body.\n"
        "\n"
        "## 3. Frame Format\n"
        "### 3.1 HELLO\n"
        "### 3.2 HELLO-ACK\n"
        "## 4. Timeouts\n"
        "## 5. Security Considerations\n"
    )
    anchors = collect_valid_anchors_from_markdown((_StubTreeSpec(source_path=doc),))  # type: ignore[arg-type]
    assert anchors == frozenset({"§1", "§2", "§2.1", "§2.2", "§3", "§3.1", "§3.2", "§4", "§5"})


def test_markdown_walker_skips_non_numeric_headings(tmp_path: Path) -> None:
    """Headings without a numeric prefix are not section anchors — the
    document title (`# Title`), reference subheads (`## References`)
    and prose H2 (`## Implementation Notes`) all skipped."""
    from harness.orchestrator.section_index import collect_valid_anchors_from_markdown

    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Document Title\n"
        "## 1. Real Section\n"
        "## References\n"
        "## Implementation Notes\n"
        "## 2. Another Real Section\n"
    )
    anchors = collect_valid_anchors_from_markdown((_StubTreeSpec(source_path=doc),))  # type: ignore[arg-type]
    assert anchors == frozenset({"§1", "§2"})


def test_markdown_walker_skips_non_markdown_format(tmp_path: Path) -> None:
    """`source_format != 'markdown'` trees are ignored — they go
    through the JSONL chunks walker instead."""
    from harness.orchestrator.section_index import collect_valid_anchors_from_markdown

    doc = tmp_path / "doc.md"
    doc.write_text("## 1. Section\n")
    anchors = collect_valid_anchors_from_markdown(
        (_StubTreeSpec(source_path=doc, source_format="jsonl"),)  # type: ignore[arg-type]
    )
    assert anchors == frozenset()


def test_markdown_walker_skips_missing_source(tmp_path: Path) -> None:
    """Missing source file → silent skip. Same resilience contract
    as the JSONL walker."""
    from harness.orchestrator.section_index import collect_valid_anchors_from_markdown

    anchors = collect_valid_anchors_from_markdown(
        (_StubTreeSpec(source_path=tmp_path / "does-not-exist.md"),)  # type: ignore[arg-type]
    )
    assert anchors == frozenset()


def test_markdown_walker_unions_multiple_docs(tmp_path: Path) -> None:
    """Multiple doc trees contribute their anchors to a single union."""
    from harness.orchestrator.section_index import collect_valid_anchors_from_markdown

    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("## 1. A\n## 2. B\n")
    b.write_text("## 3. C\n### 3.1 D\n")
    anchors = collect_valid_anchors_from_markdown(
        (_StubTreeSpec(source_path=a), _StubTreeSpec(source_path=b)),  # type: ignore[arg-type]
    )
    assert anchors == frozenset({"§1", "§2", "§3", "§3.1"})


def test_markdown_walker_picks_up_airton_f_seed_doc() -> None:
    """End-to-end pin against the real airton_f seed document. Catches
    regressions if either the markdown layout drifts or the heading
    regex starts rejecting the canonical shape."""
    from harness.character import load_character
    from harness.orchestrator.section_index import collect_valid_anchors_from_markdown

    repo = Path(__file__).resolve().parents[1]
    character = load_character(repo / "character" / "airton_f")
    anchors = collect_valid_anchors_from_markdown(character.document_trees)
    # All the headings in 01-example-rfc-style.md. If the seed doc
    # changes, update this expectation alongside the doc.
    assert anchors == frozenset({"§1", "§2", "§2.1", "§2.2", "§3", "§3.1", "§3.2", "§4", "§5"})
