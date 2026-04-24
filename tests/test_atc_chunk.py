"""Tests for scripts/atc_chunk.py (harness-xbk.5).

Covers the pure parsing helpers with small in-memory markdown fixtures
— no real PDFs or extracted files needed. The chunker itself is a
pure function (string → list[Chunk]) so this stays tight.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SCRIPT = REPO / "scripts" / "atc_chunk.py"


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("atc_chunk", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["atc_chunk"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def chunker() -> object:
    return _load_module()


# ---------- heading segmentation ----------


def test_iter_blocks_separates_by_heading(chunker: object) -> None:
    md = (
        "## **Chapter 2. General Control**\n"
        "\n"
        "## **2-6-4. ISSUING WEATHER AND CHAFF AREAS**\n"
        "\n"
        "Weather and chaff areas body text.\n"
        "\n"
        "## **2-6-5. NEXT SECTION**\n"
        "\n"
        "Next section body.\n"
    )
    blocks = list(chunker._iter_blocks(md))  # type: ignore[attr-defined]
    assert [b.heading for b in blocks] == [
        "Chapter 2. General Control",
        "2-6-4. ISSUING WEATHER AND CHAFF AREAS",
        "2-6-5. NEXT SECTION",
    ]
    # First section's body lands with its heading, not the chapter's.
    ws_block = blocks[1]
    joined = "\n".join(ws_block.body_lines).strip()
    assert "Weather and chaff areas body text" in joined


def test_strip_wrappers_cleans_markdown_emphasis(chunker: object) -> None:
    assert chunker._strip_wrappers("**BOLD**") == "BOLD"  # type: ignore[attr-defined]
    assert chunker._strip_wrappers("_**ITALIC BOLD**_") == "ITALIC BOLD"  # type: ignore[attr-defined]
    assert chunker._strip_wrappers("plain") == "plain"  # type: ignore[attr-defined]


# ---------- body cleaner ----------


def test_clean_body_drops_page_noise(chunker: object) -> None:
    """Page-header fragments pymupdf4llm interleaves into the text
    (date stamps, short doc names, solo section numbers) must not
    reach the chunk body."""
    lines = [
        "Real content paragraph one.",
        "",
        "3-5-5",
        "",
        "2/20/25",
        "AIM",
        "",
        "Real content paragraph two.",
    ]
    cleaned = chunker._clean_body(lines)  # type: ignore[attr-defined]
    assert "Real content paragraph one." in cleaned
    assert "Real content paragraph two." in cleaned
    assert "2/20/25" not in cleaned
    assert "AIM\n" not in cleaned + "\n"
    # Solo section-number orphan dropped:
    assert "3-5-5\n" not in cleaned + "\n"


def test_clean_body_collapses_blank_runs(chunker: object) -> None:
    lines = ["para A", "", "", "", "para B"]
    out = chunker._clean_body(lines)  # type: ignore[attr-defined]
    assert out == "para A\n\npara B"


# ---------- numbered parser (JO / AIM) ----------


def test_numbered_parser_handles_unicode_minus(chunker: object) -> None:
    """pymupdf4llm emits U+2212 in JO/AIM anchors — the parser must
    normalise to ASCII hyphen in the row output."""
    md = (
        "## **2−6−4. ISSUING WEATHER AND CHAFF AREAS**\n"
        "\n"
        "Weather rule body. Controllers issue chaff areas when…\n"
    )
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section == "2-6-4"
    assert c.parent_section == "2-6"
    assert c.chapter == "2"
    assert c.title == "ISSUING WEATHER AND CHAFF AREAS"
    assert "Weather rule body" in c.body


def test_numbered_parser_strips_change_prefix(chunker: object) -> None:
    """The 'Explanation of Changes' block prefixes anchors with 'a. ',
    'b. ' etc. Those must not end up in the section or title."""
    md = "## **a. 3-5-5. PUBLISHED VFR ROUTES**\n\nThis change updates the VFR product names.\n"
    cfg = chunker.PARSERS["aim"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    assert chunks[0].section == "3-5-5"
    assert chunks[0].title == "PUBLISHED VFR ROUTES"


def test_numbered_parser_tracks_chapter_from_headers(chunker: object) -> None:
    """A standalone 'Chapter N' heading (without inline anchor) must
    update the chapter for subsequent sections that omit it
    explicitly."""
    md = (
        "## Chapter 4. Air Traffic Control\n"
        "\n"
        "## Section 1. Whatever\n"
        "\n"
        "## **4-1-1. RADIO COMMUNICATIONS**\n"
        "\n"
        "Radio comms rule body.\n"
    )
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    assert chunks[0].chapter == "4"


def test_numbered_parser_skips_section_with_empty_body(chunker: object) -> None:
    """A section anchor with no body lines (table-of-contents leakage)
    shouldn't emit a chunk — nothing to retrieve on."""
    md = "## **2-6-4. ISSUING WEATHER AND CHAFF AREAS**\n\n"
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert chunks == []


def test_numbered_parser_folds_note_subsection(chunker: object) -> None:
    """NOTE sub-blocks are part of the parent §X-Y-Z section, not a new
    section. pymupdf4llm emits them as italic-bold `##` headings so the
    raw block walker sees them as siblings of the anchor heading. Parser
    must fold them into the parent's body (harness-1s4)."""
    md = (
        "## **4-6-4. HOLDING INSTRUCTIONS**\n\n"
        "When issuing holding instructions, specify:\n\n"
        "- **a.** Direction of holding from the fix/waypoint.\n\n"
        "- **b.** Holding fix or waypoint.\n\n"
        "## _**NOTE−**_\n\n"
        "_The holding fix may be omitted if included at the beginning of the transmission "
        "as the clearance limit._\n\n"
        "- **c.** Radial, course, bearing, track, azimuth, airway, or route.\n"
    )
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    body = chunks[0].body
    assert "Direction of holding from the fix/waypoint" in body
    assert "holding fix may be omitted" in body, "NOTE content must survive the fold"
    assert "Radial, course, bearing" in body, "paragraph after the NOTE must also survive"


def test_numbered_parser_folds_phraseology_and_reference(chunker: object) -> None:
    """PHRASEOLOGY and REFERENCE sub-blocks fold the same way — FAA
    uses them for exact-wording phraseology and cross-refs, both of
    which are load-bearing content for controller-side retrieval."""
    md = (
        "## **3-1-4. EXAMPLE INSTRUCTION**\n\n"
        "Body before the sub-blocks.\n\n"
        "## _**PHRASEOLOGY−**_\n\n"
        "_EXACT WORDS TO SAY._\n\n"
        "## _**REFERENCE−**_\n\n"
        "_JO 7110.65, Some Other Section._\n"
    )
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    body = chunks[0].body
    assert "EXACT WORDS TO SAY" in body
    assert "JO 7110.65, Some Other Section" in body


def test_numbered_parser_folds_phraseology_term_headings(chunker: object) -> None:
    """Inside a PHRASEOLOGY block, individual phraseology terms (e.g.
    MAINTAIN, AFFIRMATIVE, HEAVY) are emitted by pymupdf4llm as
    italic-bold `##` sub-headings. They must also fold into the parent
    — not drop content between them."""
    md = (
        "## **2-1-18. OPERATIONAL REQUESTS**\n\n"
        "Opening body text.\n\n"
        "## _**MAINTAIN−**_\n\n"
        "_Remain at the specified altitude._\n\n"
        "## _**AFFIRMATIVE−**_\n\n"
        "_Yes._\n"
    )
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    body = chunks[0].body
    assert "Remain at the specified altitude" in body
    assert "Yes" in body


def test_numbered_parser_does_not_fold_across_anchors(chunker: object) -> None:
    """The fold must stop at the next §X-Y-Z anchor — otherwise two
    sections merge into one chunk."""
    md = (
        "## **4-6-4. FIRST SECTION**\n\n"
        "First body.\n\n"
        "## _**NOTE−**_\n\n"
        "_First note._\n\n"
        "## **4-6-5. SECOND SECTION**\n\n"
        "Second body.\n"
    )
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert [c.section for c in chunks] == ["4-6-4", "4-6-5"]
    assert "First note" in chunks[0].body
    assert "First note" not in chunks[1].body
    assert "Second body" in chunks[1].body
    assert "Second body" not in chunks[0].body


def test_numbered_parser_drops_frontmatter_before_first_anchor(chunker: object) -> None:
    """Non-anchor headings BEFORE any §X-Y-Z anchor are frontmatter
    (Table of Contents, RECORD OF CHANGES, etc.). They have no parent
    to fold into — must be dropped, not fused into the first real
    section."""
    md = (
        "## **Table of Contents**\n\n"
        "irrelevant frontmatter body\n\n"
        "## _**NOTE−**_\n\n"
        "orphan note before any anchor\n\n"
        "## **2-1-1. REAL FIRST SECTION**\n\n"
        "Real body content.\n"
    )
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    assert chunks[0].section == "2-1-1"
    assert "irrelevant frontmatter" not in chunks[0].body
    assert "orphan note before any anchor" not in chunks[0].body
    assert "Real body content" in chunks[0].body


def test_numbered_parser_does_not_fold_across_section_groups(chunker: object) -> None:
    """`Section N. Title` headers are a flush boundary — a NOTE between
    Section-group headers doesn't fold backwards across the boundary."""
    md = (
        "## **Chapter 4. IFR**\n\n"
        "## **Section 6. Holding**\n\n"
        "## **4-6-4. HOLDING INSTRUCTIONS**\n\n"
        "Body of 4-6-4.\n\n"
        "## **Section 7. Arrival**\n\n"
        "## **4-7-1. CLEARANCE INFORMATION**\n\n"
        "Body of 4-7-1.\n"
    )
    cfg = chunker.PARSERS["jo_7110_65"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert [c.section for c in chunks] == ["4-6-4", "4-7-1"]
    assert chunks[0].parent_section_title == "Holding"
    assert chunks[1].parent_section_title == "Arrival"


# ---------- CFR parser ----------


def test_cfr_parser_captures_section_number(chunker: object) -> None:
    md = (
        "## PART 91 — GENERAL OPERATING AND FLIGHT RULES\n"
        "\n"
        "## **§ 91.155 Basic VFR weather minimums.**\n"
        "\n"
        "Except as provided in §91.157, no person may operate an aircraft\n"
        "under VFR when the flight visibility is less, or at a distance\n"
        "from clouds that is less, than that prescribed for the corresponding\n"
        "altitude and class of airspace in the table in paragraph (a) of\n"
        "this section.\n"
    )
    cfg = chunker.PARSERS["cfr_14_vol2"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section == "91.155"
    assert c.parent_section == "91"
    assert c.chapter == "91"
    assert "Basic VFR weather minimums" in c.title


# ---------- PCG parser ----------


def test_glossary_parser_uses_letter_as_chapter(chunker: object) -> None:
    md = (
        "## **A**\n"
        "\n"
        "## **AAM-**\n"
        "\n"
        "(See ADVANCED AIR MOBILITY.)\n"
        "\n"
        "## ACTIVE RUNWAY-\n"
        "\n"
        "Any runway or runways currently being used for takeoff or landing.\n"
    )
    cfg = chunker.PARSERS["pcg"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    sections = [(c.chapter, c.section) for c in chunks]
    # Entry body must include the inline definition (Any runway…)
    assert any(sec == "ACTIVE RUNWAY" and chap == "A" for chap, sec in sections)
    # Cross-reference-only entries (body = "(See ...)") still chunked.
    active_chunk = next(c for c in chunks if c.section == "ACTIVE RUNWAY")
    assert "Any runway or runways" in active_chunk.body


def test_glossary_parser_skips_frontmatter(chunker: object) -> None:
    """'PURPOSE' / 'EXPLANATION OF CHANGES' sit above the first letter
    heading and shouldn't be emitted as glossary entries."""
    md = "## **PURPOSE**\n\nSome preface text.\n\n## **A**\n\n## **AAM-**\n\nBody.\n"
    cfg = chunker.PARSERS["pcg"]  # type: ignore[attr-defined]
    chunks = chunker.chunk_markdown(md, cfg)  # type: ignore[attr-defined]
    sections = [c.section for c in chunks]
    assert "PURPOSE" not in sections
    assert "AAM" in sections


# ---------- size shaping ----------


def test_split_oversized_returns_single_when_under_cap(chunker: object) -> None:
    short = "one short paragraph."
    out = chunker._split_oversized(short, max_chars=1200, overlap=80)  # type: ignore[attr-defined]
    assert out == [short]


def test_split_oversized_splits_on_paragraphs_with_overlap(chunker: object) -> None:
    # Build a body with many 200-char paragraphs so the splitter has
    # paragraph boundaries to use and overflows 1200.
    para = "x" * 200
    body = "\n\n".join([para] * 10)  # ~2000+ chars
    out = chunker._split_oversized(body, max_chars=600, overlap=20)  # type: ignore[attr-defined]
    assert len(out) > 1
    for chunk in out:
        # Each split chunk under the cap (overlap may nudge above by a
        # small margin, but not by multiples of the cap).
        assert len(chunk) <= 600 + 20


def test_expand_by_size_indexes_splits(chunker: object) -> None:
    """When a section body splits, each piece gets its own chunk_index
    so downstream ingest can build stable external_ids per split."""
    # Manufacture a chunk whose body exceeds the cap.
    huge = "paragraph-a " * 200 + "\n\n" + "paragraph-b " * 200
    chunk = chunker.Chunk(  # type: ignore[attr-defined]
        source="JO_7110.65",
        chapter="2",
        section="2-6-4",
        parent_section="2-6",
        title="HUGE",
        body=huge,
        tags=("atc", "corpus", "JO_7110.65"),
    )
    expanded = list(chunker._expand_by_size([chunk]))  # type: ignore[attr-defined]
    assert len(expanded) >= 2
    assert [c.chunk_index for c in expanded] == list(range(len(expanded)))
    # Section metadata stable across splits.
    for c in expanded:
        assert c.section == "2-6-4"
        assert c.parent_section == "2-6"


# ---------- Chunk.text contextual header ----------


def test_chunk_text_prepends_contextual_header(chunker: object) -> None:
    chunk = chunker.Chunk(  # type: ignore[attr-defined]
        source="JO_7110.65",
        chapter="2",
        section="2-6-4",
        parent_section="2-6",
        title="T",
        body="rule body",
    )
    assert chunk.text == "[source: JO_7110.65; chapter: 2; section: 2-6-4]\nrule body"


def test_chunk_text_omits_empty_metadata(chunker: object) -> None:
    chunk = chunker.Chunk(  # type: ignore[attr-defined]
        source="PCG",
        chapter="",
        section="WAKE TURBULENCE",
        parent_section="W",
        title="T",
        body="definition",
    )
    assert chunk.text == "[source: PCG; section: WAKE TURBULENCE]\ndefinition"


# ---------- JSONL write roundtrip ----------


def test_write_jsonl_produces_one_row_per_chunk(chunker: object, tmp_path: Path) -> None:
    chunks = [
        chunker.Chunk(  # type: ignore[attr-defined]
            source="AIM",
            chapter="3",
            section="3-5-5",
            parent_section="3-5",
            title="VFR Routes",
            body="body one",
        ),
        chunker.Chunk(  # type: ignore[attr-defined]
            source="AIM",
            chapter="4",
            section="4-7-1",
            parent_section="4-7",
            title="Intro",
            body="body two",
        ),
    ]
    dst = tmp_path / "aim.jsonl"
    count = chunker.write_jsonl(chunks, dst)  # type: ignore[attr-defined]
    assert count == 2
    rows = [json.loads(line) for line in dst.read_text().splitlines()]
    assert rows[0]["section"] == "3-5-5"
    assert rows[0]["text"].startswith("[source: AIM;")
    assert rows[1]["section"] == "4-7-1"
