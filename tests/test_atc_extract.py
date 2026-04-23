"""Tests for scripts/atc_extract.py (harness-xbk.4).

Covers the pure helpers — CorpusDoc shape, idempotency, slug
narrowing, injected extractor — without actually importing
pymupdf4llm. The production path imports it lazily inside
extract_one() so `--help` works without the `atc` extra; these tests
pass a fake to_markdown callable to exercise the write path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SCRIPT = REPO / "scripts" / "atc_extract.py"


def _load_module() -> object:
    """Load scripts/atc_extract.py as a module. scripts/ isn't a
    package so we go through importlib rather than `import scripts…`."""
    spec = importlib.util.spec_from_file_location("atc_extract", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atc_extract"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def atc() -> object:
    return _load_module()


# ---------- corpus shape ----------


def test_phase_1_corpus_has_expected_slugs(atc: object) -> None:
    """Phase-1 priority: PCG, AIM, 7110.65, PHAK, CFR Vol 1, CFR Vol 2.
    Slug stability is load-bearing — it becomes the `source` metadata
    key in the chunker's output, and retrieval keys on it."""
    slugs = [doc.slug for doc in atc.PHASE_1_CORPUS]  # type: ignore[attr-defined]
    assert slugs == ["pcg", "aim", "jo_7110_65", "phak", "cfr_14_vol1", "cfr_14_vol2"]


def test_phase_1_corpus_source_filenames_look_like_pdfs(atc: object) -> None:
    for doc in atc.PHASE_1_CORPUS:  # type: ignore[attr-defined]
        assert doc.source_filename.endswith(".pdf"), doc
        assert doc.title, doc  # non-empty human-readable label


# ---------- resolve_docs ----------


def test_resolve_docs_empty_only_returns_everything(atc: object) -> None:
    out = atc.resolve_docs(atc.PHASE_1_CORPUS)  # type: ignore[attr-defined]
    assert out == atc.PHASE_1_CORPUS  # type: ignore[attr-defined]


def test_resolve_docs_narrows_by_slug(atc: object) -> None:
    out = atc.resolve_docs(atc.PHASE_1_CORPUS, only=("pcg", "aim"))  # type: ignore[attr-defined]
    assert [doc.slug for doc in out] == ["pcg", "aim"]


def test_resolve_docs_raises_on_unknown_slug(atc: object) -> None:
    """Typo shouldn't silently skip work."""
    with pytest.raises(ValueError, match="unknown doc slug"):
        atc.resolve_docs(atc.PHASE_1_CORPUS, only=("not_a_real_doc",))  # type: ignore[attr-defined]


# ---------- idempotency ----------


def test_is_up_to_date_false_when_target_missing(atc: object, tmp_path: Path) -> None:
    src = tmp_path / "src.pdf"
    src.write_bytes(b"pdf")
    dst = tmp_path / "out.md"
    assert atc.is_up_to_date(src, dst) is False  # type: ignore[attr-defined]


def test_is_up_to_date_true_when_target_fresher(atc: object, tmp_path: Path) -> None:
    src = tmp_path / "src.pdf"
    src.write_bytes(b"pdf")
    dst = tmp_path / "out.md"
    dst.write_text("# already extracted")
    # bump dst mtime above src
    import os

    os.utime(dst, (src.stat().st_atime + 10, src.stat().st_mtime + 10))
    assert atc.is_up_to_date(src, dst) is True  # type: ignore[attr-defined]


def test_is_up_to_date_false_when_source_fresher(atc: object, tmp_path: Path) -> None:
    src = tmp_path / "src.pdf"
    dst = tmp_path / "out.md"
    dst.write_text("# stale")
    src.write_bytes(b"pdf")
    import os

    os.utime(src, (dst.stat().st_atime + 10, dst.stat().st_mtime + 10))
    assert atc.is_up_to_date(src, dst) is False  # type: ignore[attr-defined]


# ---------- extract_one with injected fake ----------


def _make_src(tmp_path: Path, name: str = "pcg.pdf") -> Path:
    src_dir = tmp_path / "pdfs"
    src_dir.mkdir()
    src = src_dir / name
    src.write_bytes(b"%PDF-fake")
    return src


def test_extract_one_missing_source_reports_without_writing(atc: object, tmp_path: Path) -> None:
    doc = atc.CorpusDoc(slug="pcg", source_filename="missing.pdf", title="PCG")  # type: ignore[attr-defined]
    out_dir = tmp_path / "out"
    line = atc.extract_one(  # type: ignore[attr-defined]
        doc,
        source_root=tmp_path,
        output_root=out_dir,
        force=False,
        to_markdown=lambda _: "should not be called",
    )
    assert "SOURCE MISSING" in line
    assert not out_dir.exists(), "output dir should not be created when source is missing"


def test_extract_one_writes_markdown_when_target_missing(atc: object, tmp_path: Path) -> None:
    src = _make_src(tmp_path)
    doc = atc.CorpusDoc(slug="pcg", source_filename=src.name, title="PCG")  # type: ignore[attr-defined]
    out_dir = tmp_path / "out"
    captured: list[str] = []

    def fake(path: str) -> str:
        captured.append(path)
        return "# PCG\n\n## Section A\n\nbody\n"

    line = atc.extract_one(  # type: ignore[attr-defined]
        doc,
        source_root=src.parent,
        output_root=out_dir,
        force=False,
        to_markdown=fake,
    )
    assert "extracted" in line
    assert captured == [str(src)]
    out_md = out_dir / "pcg.md"
    assert out_md.exists()
    text = out_md.read_text()
    assert text.startswith("# PCG")


def test_extract_one_skips_when_up_to_date(atc: object, tmp_path: Path) -> None:
    src = _make_src(tmp_path)
    doc = atc.CorpusDoc(slug="pcg", source_filename=src.name, title="PCG")  # type: ignore[attr-defined]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    dst = out_dir / "pcg.md"
    dst.write_text("# already here")
    import os

    os.utime(dst, (src.stat().st_atime + 10, src.stat().st_mtime + 10))

    def must_not_call(_: str) -> str:
        raise AssertionError("should not re-extract when up-to-date")

    line = atc.extract_one(  # type: ignore[attr-defined]
        doc,
        source_root=src.parent,
        output_root=out_dir,
        force=False,
        to_markdown=must_not_call,
    )
    assert "up-to-date" in line
    assert dst.read_text() == "# already here"


def test_extract_one_force_rebuilds_even_when_fresh(atc: object, tmp_path: Path) -> None:
    src = _make_src(tmp_path)
    doc = atc.CorpusDoc(slug="pcg", source_filename=src.name, title="PCG")  # type: ignore[attr-defined]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    dst = out_dir / "pcg.md"
    dst.write_text("# stale content")
    import os

    os.utime(dst, (src.stat().st_atime + 10, src.stat().st_mtime + 10))
    atc.extract_one(  # type: ignore[attr-defined]
        doc,
        source_root=src.parent,
        output_root=out_dir,
        force=True,
        to_markdown=lambda _: "# fresh content",
    )
    assert dst.read_text() == "# fresh content"
