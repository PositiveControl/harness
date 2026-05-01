"""Section-aware chunker for atc's FAA markdown corpus (harness-xbk.5).

Reads character/airton_c/corpus/markdown/<slug>.md (output of
harness-xbk.4's extractor) and produces retrieval-ready JSONL rows at
character/airton_c/corpus/chunks/<slug>.jsonl. One row per addressable
section; long sections split on paragraph boundaries with an
80-character overlap for continuity.

Row shape (one JSON object per line):
    {
      "source":         "JO_7110.65" | "AIM" | "PCG" | "PHAK" | "CFR_14_vol1" | ...,
      "chapter":        "2" | "1" | ... | "" (empty when N/A for this source),
      "section":        "2-6-4" | "91.155" | "WAKE TURBULENCE" | ...,
      "parent_section": "2-6" | "91" | "W" | "" (derived),
      "title":          "ISSUING WEATHER AND CHAFF AREAS",
      "body":           "<cleaned paragraph text>",
      "text":           "[source: JO_7110.65; chapter: 2; section: 2-6-4]\\n<body>",
      "principle":      "" (reserved for downstream atc taxonomy mapping),
      "tags":           ["atc", "corpus", "JO_7110.65"],
      "chunk_index":    0 | 1 | ... (non-zero when a section split across rows),
    }

Section anchors in the extractor's markdown use U+2212 (unicode minus)
for JO 7110.65 / AIM (e.g. "2−6−4"). This chunker normalises them to
ASCII hyphen in the output so downstream tooling doesn't have to handle
both. The extractor-side header format (#+ **N−N−N. TITLE**) is
preserved as the match target.

Usage (from repo root):
    uv run python scripts/atc_chunk.py
    uv run python scripts/atc_chunk.py --only pcg aim
    uv run python scripts/atc_chunk.py --force

Acceptance sampling:
    uv run python scripts/atc_chunk.py --sample 10

Not intended for direct ingest — atc-6 owns the episodic store ingest
path and constructs stable external_ids from the chunk metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Default character. HARNESS_CHARACTER_NAME overrides — the chunker
# reads markdown from / writes chunks to that character's corpus dir.
_CHARACTER = os.environ.get("HARNESS_CHARACTER_NAME", "airton_c")

DEFAULT_INPUT = REPO / "character" / _CHARACTER / "corpus" / "markdown"
DEFAULT_OUTPUT = REPO / "character" / _CHARACTER / "corpus" / "chunks"

# Chunk-size knobs. Short bodies stay as-is (one row). Long bodies split
# on paragraph boundaries with overlap for retrieval continuity.
MAX_CHARS = 1200
OVERLAP_CHARS = 80

# Accept both U+2212 (unicode minus, what pymupdf4llm emits for JO /
# AIM anchors) and ASCII hyphen. Normalised to '-' in row output.
_DASH_CLASS = r"[−–—\-]"


# ---------- row type ----------


@dataclass(frozen=True)
class Chunk:
    source: str
    chapter: str
    section: str
    parent_section: str
    title: str
    body: str
    # Title of the parent-section group (e.g. "Departure Procedures and
    # Separation" for §3-9-6). Captured from '## **Section N. Title**'
    # markdown headers. Populated for JO 7110.65 + AIM (both use the
    # numbered kind). Empty string for sources without a matching
    # header or for headings the chunker sees before the first Section
    # header in a chapter. Feeds into the ingest script's principle
    # enrichment (harness-8zx6) so embeds disambiguate §3-9-6
    # (Departure) from §3-10-3 (Arrival) — both titled "SAME RUNWAY
    # SEPARATION" in the raw JO 7110.65 markdown.
    parent_section_title: str = ""
    principle: str = ""
    tags: tuple[str, ...] = ()
    chunk_index: int = 0

    @property
    def text(self) -> str:
        """The retrieval-facing text: contextual header + body."""
        bits = [f"source: {self.source}"]
        if self.chapter:
            bits.append(f"chapter: {self.chapter}")
        if self.section:
            bits.append(f"section: {self.section}")
        header = "[" + "; ".join(bits) + "]"
        return f"{header}\n{self.body}"

    def to_json(self) -> dict[str, object]:
        row = asdict(self)
        row["text"] = self.text
        row["tags"] = list(self.tags)
        return row


# ---------- per-source parser config ----------


@dataclass(frozen=True)
class ParserConfig:
    slug: str
    source_key: str
    # Which family of anchor patterns to try. Determines the parser
    # dispatch below.
    kind: str  # "numbered" | "cfr" | "glossary" | "phak"
    tags: tuple[str, ...] = field(default=())


PARSERS: dict[str, ParserConfig] = {
    "pcg": ParserConfig("pcg", "PCG", "glossary", tags=("atc", "corpus", "PCG")),
    "aim": ParserConfig("aim", "AIM", "numbered", tags=("atc", "corpus", "AIM")),
    "jo_7110_65": ParserConfig(
        "jo_7110_65",
        "JO_7110.65",
        "numbered",
        tags=("atc", "corpus", "JO_7110.65"),
    ),
    "phak": ParserConfig("phak", "PHAK", "phak", tags=("atc", "corpus", "PHAK")),
    "cfr_14_vol1": ParserConfig(
        "cfr_14_vol1",
        "CFR_14_vol1",
        "cfr",
        tags=("atc", "corpus", "CFR_14", "CFR_14_vol1"),
    ),
    "cfr_14_vol2": ParserConfig(
        "cfr_14_vol2",
        "CFR_14_vol2",
        "cfr",
        tags=("atc", "corpus", "CFR_14", "CFR_14_vol2"),
    ),
}


# ---------- regex banks ----------
# Heading strip helpers: pymupdf4llm wraps many headings in markdown
# bold / italic (`## **TEXT**` / `## _**TEXT**_`) and some have an "a."
# explanation-of-changes prefix. Strip those before anchor matching so
# the patterns stay readable.

_LEADING_WRAPPERS = re.compile(r"^\s*(?:[*_]+\s*)+")
_TRAILING_WRAPPERS = re.compile(r"(?:\s*[*_]+)+\s*$")
_HEADING_LINE = re.compile(r"^(#+)\s+(.+?)\s*$")
_CHANGE_PREFIX = re.compile(r"^[a-z]\.\s+")

# AIM / JO 7110.65 anchor: N-N-N or N-N, with any dash.
_NUMBERED_ANCHOR = re.compile(
    r"^(\d+)" + _DASH_CLASS + r"(\d+)"
    r"(?:" + _DASH_CLASS + r"(\d+))?"
    r"\.?\s+(.+)$"
)
_CHAPTER_HDR = re.compile(r"^Chapter\s+(\d+)\.?\s*(.*)$", re.I)

# JO 7110.65 + AIM: '## **Section N. Title**' headers group a family of
# N-N-N anchors that share the same parent-section topic. Within a
# chapter, Section N groups all §<chapter>-<N>-<sub> anchors. Feeds the
# parent-section-title enrichment on emitted chunks (harness-8zx6).
_SECTION_HDR = re.compile(r"^Section\s+(\d+)\.?\s+(.+?)\s*$", re.I)

# CFR: "§ 1.1 General definitions." (may drop the trailing period).
_CFR_SECTION = re.compile(r"^§\s*(\d+)\.(\d+[a-z]*)(?:\s+(.*))?$")
_CFR_PART = re.compile(r"^PART\s+(\d+)" + _DASH_CLASS + r"*\s*(.*)$", re.I)

# PCG: bare letter heading ("A", "B") marks a group; TERM with a
# required trailing dash marks a glossary entry (headings without the
# dash are preface / frontmatter and don't get chunked).
_PCG_LETTER = re.compile(r"^([A-Z])$")
# Term chars: uppercase letters, digits, space, /, parens, dot, but NOT
# the hyphen — the PCG uses the hyphen as the "end-of-term" marker
# rather than as part of the term. Optional [ICAO]/[FSS] suffix.
# Inline definitions may follow after the dash on the same line; the
# parser lifts that text into the body.
_PCG_TERM = re.compile(
    r"^([A-Z][A-Z0-9 /()'.]*?(?:\s*\[[A-Z]+\])?)"
    r"\s*" + _DASH_CLASS + r"\s*(.*)$"
)

# PHAK: "Chapter N" is the only reliable anchor the extractor produces
# (subheadings are semantic but not numbered). Fall back to heading
# text as section.
_PHAK_CHAPTER = re.compile(r"^Chapter\s+(\d+)\s*(.*)$", re.I)


# ---------- page-noise filter ----------
# Lines the chunker strips from section bodies before emit. Each entry
# is a line-by-line regex — short boilerplate fragments that pymupdf4llm
# pulls from page headers / footers.

_NOISE_LINES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*\d+/\d+/\d+\s*$"),  # "2/20/25"
    re.compile(r"^\s*\d+" + _DASH_CLASS + r"\d+(?:" + _DASH_CLASS + r"\d+)?\s*$"),  # "3-5-5" solo
    re.compile(r"^\s*AIM\s*$", re.I),
    re.compile(r"^\s*PCG" + _DASH_CLASS + r"\d+\s*$", re.I),
    re.compile(r"^\s*JO\s+7110\.65[A-Z]*\s*$", re.I),
    re.compile(r"^\s*Pilot/Controller\s+Glossary\s*$", re.I),
    re.compile(r"^\s*Aeronautical\s+Information\s+Manual\s*$", re.I),
    re.compile(r"^\s*Pilot.?s\s+Handbook.*$", re.I),
    re.compile(r"^\s*14\s+CFR\s+Ch\.\s+[IVX]+.*Edition.*$", re.I),
    re.compile(r"^\s*\d+\s*$"),  # orphan page number
)


# ---------- markdown segmentation ----------


@dataclass(frozen=True)
class _Block:
    """A heading + its body lines, before anchor classification."""

    level: int  # number of # chars (2, 3, ...)
    heading: str
    body_lines: tuple[str, ...]


def _is_page_break_heading(text: str) -> bool:
    """True when a markdown heading matches one of the per-page running-
    header patterns pymupdf4llm wraps as ``## …`` mid-document (e.g.
    ``## 14 CFR Ch. I (1-1-25 Edition)`` between rows of a CFR table,
    ``## Pilot/Controller Glossary`` between PCG terms). These split
    real sections in half — left side keeps the heading, right side
    becomes a phantom section with no anchor — so retrieval misses
    chunks that span a page boundary (harness-wkom: §91.155(a) table
    truncated to the Class A/B/C row only)."""
    return any(pat.match(text) for pat in _NOISE_LINES)


def _iter_blocks(md: str) -> Iterator[_Block]:
    """Walk the markdown linewise and emit (heading, body) pairs. Body
    is every non-heading line between this heading and the next.

    Page-break headings (``## 14 CFR Ch. I…``, etc.) are detected and
    silently merged into the current block's body so a real section
    that straddles a page break stays intact."""
    current_heading: str | None = None
    current_level = 0
    buffer: list[str] = []
    for line in md.splitlines():
        m = _HEADING_LINE.match(line)
        if m is not None:
            heading_text = _strip_wrappers(m.group(2))
            if _is_page_break_heading(heading_text):
                # Drop the page-break heading entirely — accumulating
                # its text would only inject the running-header string
                # into the body, which the noise filter scrubs anyway.
                continue
            if current_heading is not None:
                yield _Block(current_level, current_heading, tuple(buffer))
            current_heading = heading_text
            current_level = len(m.group(1))
            buffer = []
        else:
            buffer.append(line)
    if current_heading is not None:
        yield _Block(current_level, current_heading, tuple(buffer))


def _strip_wrappers(text: str) -> str:
    """Drop leading/trailing markdown bold/italic wrappers from a
    heading so the anchor regexes can match the content directly."""
    text = _LEADING_WRAPPERS.sub("", text)
    text = _TRAILING_WRAPPERS.sub("", text)
    return text.strip()


def _clean_body(lines: Iterable[str]) -> str:
    """Drop page-noise lines, then collapse consecutive blank lines to
    a single blank and join. Returns the final body string (no
    header)."""
    kept: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if any(p.match(line) for p in _NOISE_LINES):
            continue
        kept.append(line)
    # Collapse consecutive blank runs.
    out: list[str] = []
    last_blank = True
    for line in kept:
        is_blank = not line.strip()
        if is_blank and last_blank:
            continue
        out.append(line)
        last_blank = is_blank
    while out and not out[-1].strip():
        out.pop()
    while out and not out[0].strip():
        out.pop(0)
    return "\n".join(out).strip()


# ---------- per-kind parsers ----------


def _parse_numbered(blocks: Iterable[_Block], cfg: ParserConfig) -> Iterator[Chunk]:
    """AIM and JO 7110.65 share the N-N-N anchor convention.

    Tracks parent-section titles (from 'Section N. Title' headers) per
    chapter, so emitted chunks carry the parent-section context. Within
    a chapter, each `Section N` group may repeat across chapters (e.g.
    Chapter 3 and Chapter 7 both have a `Section 9`), so the lookup key
    is `(chapter, section_num)` — reset implicitly when the chapter
    changes (new chapter starts a fresh section-title scope).

    Sub-headings that aren't themselves N-N-N anchors (NOTE,
    PHRASEOLOGY, REFERENCE, EXAMPLE, FIG, TBL, and phraseology-term
    headings like 'MAINTAIN' or 'AFFIRMATIVE' which pymupdf4llm emits
    as italic-bold `##` lines) are folded into the previous anchor's
    body rather than dropped. The fold keeps the sub-heading text
    preceding its body so the sub-block structure survives in the
    embedded content. Without this, ~26% of §X-Y-Z sections lost their
    NOTE / PHRASEOLOGY blocks — which is where the 'when' conditions
    and exact phraseology live (harness-1s4)."""
    chapter = ""
    section_titles: dict[tuple[str, str], str] = {}

    # Buffered current anchor: we defer yielding until we've seen the
    # next anchor, so non-anchor headings between them can be folded in.
    pending_chunk: Chunk | None = None
    pending_body_parts: list[str] = []

    def finalize() -> Chunk | None:
        nonlocal pending_chunk, pending_body_parts
        if pending_chunk is None:
            return None
        body = _clean_body(pending_body_parts)
        if not body:
            pending_chunk = None
            pending_body_parts = []
            return None
        out = Chunk(
            source=pending_chunk.source,
            chapter=pending_chunk.chapter,
            section=pending_chunk.section,
            parent_section=pending_chunk.parent_section,
            parent_section_title=pending_chunk.parent_section_title,
            title=pending_chunk.title,
            body=body,
            tags=pending_chunk.tags,
        )
        pending_chunk = None
        pending_body_parts = []
        return out

    for block in blocks:
        stripped = _CHANGE_PREFIX.sub("", block.heading)

        # Chapter header (non-anchor). Flush any pending anchor first —
        # sections don't span chapters.
        m_ch = _CHAPTER_HDR.match(block.heading)
        if m_ch is not None and not _NUMBERED_ANCHOR.match(stripped):
            flushed = finalize()
            if flushed is not None:
                yield flushed
            chapter = m_ch.group(1)
            continue

        # Section header (e.g. "Section 9. Departure Procedures and
        # Separation"). Also a flush boundary — sub-blocks never span
        # across Section N groups.
        m_sec = _SECTION_HDR.match(block.heading)
        if m_sec is not None and not _NUMBERED_ANCHOR.match(stripped):
            flushed = finalize()
            if flushed is not None:
                yield flushed
            sec_num = m_sec.group(1)
            sec_title = m_sec.group(2).strip()
            if chapter:
                section_titles[(chapter, sec_num)] = sec_title
            continue

        m = _NUMBERED_ANCHOR.match(stripped)
        if m is None:
            # Non-anchor heading between two anchors. Fold it into the
            # pending anchor's body so NOTE / PHRASEOLOGY / REFERENCE /
            # phraseology-term content is preserved. If there's no
            # pending anchor (frontmatter before the first real
            # section), drop it as before.
            if pending_chunk is not None:
                pending_body_parts.append("")
                pending_body_parts.append(block.heading)
                pending_body_parts.extend(block.body_lines)
            continue

        # New anchor — flush the previous one.
        flushed = finalize()
        if flushed is not None:
            yield flushed

        chap, sec, sub, title = m.group(1), m.group(2), m.group(3), m.group(4)
        parts = [chap, sec] + ([sub] if sub else [])
        section = "-".join(parts)
        parent = "-".join(parts[:-1]) if len(parts) > 1 else ""
        effective_chapter = chap or chapter
        parent_title = section_titles.get((effective_chapter, sec), "")
        pending_chunk = Chunk(
            source=cfg.source_key,
            chapter=effective_chapter,
            section=section,
            parent_section=parent,
            parent_section_title=parent_title,
            title=title.strip(),
            body="",  # filled in by finalize()
            tags=cfg.tags,
        )
        pending_body_parts = list(block.body_lines)

    flushed = finalize()
    if flushed is not None:
        yield flushed


def _parse_cfr(blocks: Iterable[_Block], cfg: ParserConfig) -> Iterator[Chunk]:
    """CFR Title 14: PART N headers track chapter-level context; §N.M
    headers are the addressable sections."""
    part = ""
    for block in blocks:
        m_part = _CFR_PART.match(block.heading)
        if m_part is not None:
            part = m_part.group(1)
            continue
        m = _CFR_SECTION.match(block.heading)
        if m is None:
            continue
        section_part, section_num, inline_title = m.group(1), m.group(2), (m.group(3) or "")
        section = f"{section_part}.{section_num}"
        body = _clean_body(block.body_lines)
        if not body:
            continue
        yield Chunk(
            source=cfg.source_key,
            chapter=part or section_part,
            section=section,
            parent_section=section_part,
            title=inline_title.strip() or section,
            body=body,
            tags=cfg.tags,
        )


def _parse_glossary(blocks: Iterable[_Block], cfg: ParserConfig) -> Iterator[Chunk]:
    """PCG: alphabetic letter headings partition the glossary; TERM
    entries are the addressable units. Inline definitions after the
    dash are folded into the body."""
    letter = ""
    for block in blocks:
        ml = _PCG_LETTER.match(block.heading)
        if ml is not None:
            letter = ml.group(1)
            continue
        mt = _PCG_TERM.match(block.heading)
        if mt is None:
            continue
        term, inline = mt.group(1).strip(), (mt.group(2) or "").strip()
        # Skip front-matter headings (PURPOSE / EXPLANATION OF CHANGES
        # etc.) — they have letter==None at document top.
        if not letter:
            continue
        # Some "terms" are actually cross-refs like "(See FOO.)" — those
        # still have semantic value as aliases, but their heading starts
        # with "(" which the term regex won't match anyway.
        body_lines = (inline, *block.body_lines) if inline else block.body_lines
        body = _clean_body(body_lines)
        if not body:
            continue
        yield Chunk(
            source=cfg.source_key,
            chapter=letter,
            section=term,
            parent_section=letter,
            title=term,
            body=body,
            tags=cfg.tags,
        )


def _parse_phak(blocks: Iterable[_Block], cfg: ParserConfig) -> Iterator[Chunk]:
    """PHAK: chapter-granular sections. The PDF's subheadings aren't
    numbered so we use heading text as section within the current
    chapter. Best-effort — Phase 1 accepts lower precision here."""
    chapter = ""
    chapter_title = ""
    for block in blocks:
        m = _PHAK_CHAPTER.match(block.heading)
        if m is not None:
            chapter = m.group(1)
            chapter_title = (m.group(2) or "").strip()
            continue
        if block.level > 2:
            # Sub-heading within a chapter — emit as a section.
            body = _clean_body(block.body_lines)
            if not body:
                continue
            title = block.heading.strip()
            yield Chunk(
                source=cfg.source_key,
                chapter=chapter,
                section=title,
                parent_section=f"Chapter {chapter}" if chapter else "",
                title=title,
                body=body,
                tags=cfg.tags,
            )
        elif chapter:
            # Top-level chapter body (before any subheading).
            body = _clean_body(block.body_lines)
            if body:
                yield Chunk(
                    source=cfg.source_key,
                    chapter=chapter,
                    section=f"Chapter {chapter}",
                    parent_section="",
                    title=chapter_title or f"Chapter {chapter}",
                    body=body,
                    tags=cfg.tags,
                )


_PARSERS_BY_KIND = {
    "numbered": _parse_numbered,
    "cfr": _parse_cfr,
    "glossary": _parse_glossary,
    "phak": _parse_phak,
}


# ---------- size shaping ----------


def _split_oversized(body: str, max_chars: int, overlap: int) -> list[str]:
    """Split `body` into chunks of at most `max_chars`. Prefer paragraph
    (double-newline) boundaries; fall back to sentence (period + space)
    boundaries; last resort is a hard character cut. Each split after
    the first overlaps the previous by `overlap` chars so retrieval
    continuity isn't broken mid-sentence."""
    if len(body) <= max_chars:
        return [body]
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    out: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
            continue
        candidate = current + "\n\n" + para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            out.append(current)
            # Carry the tail of the previous chunk as overlap.
            tail = current[-overlap:] if overlap < len(current) else current
            current = (tail + "\n\n" + para).strip()
    if current:
        out.append(current)
    # Any remaining oversized members get a hard character split.
    final: list[str] = []
    for block in out:
        while len(block) > max_chars:
            final.append(block[:max_chars])
            block = block[max_chars - overlap :]
        final.append(block)
    return final


def _expand_by_size(chunks: Iterable[Chunk]) -> Iterator[Chunk]:
    """Yield chunks, splitting oversized bodies on paragraph / sentence
    boundaries. The split-off rows share section metadata and get a
    non-zero chunk_index."""
    for chunk in chunks:
        pieces = _split_oversized(chunk.body, MAX_CHARS, OVERLAP_CHARS)
        if len(pieces) == 1:
            yield chunk
            continue
        for idx, piece in enumerate(pieces):
            yield Chunk(
                source=chunk.source,
                chapter=chunk.chapter,
                section=chunk.section,
                parent_section=chunk.parent_section,
                parent_section_title=chunk.parent_section_title,
                title=chunk.title,
                body=piece,
                principle=chunk.principle,
                tags=chunk.tags,
                chunk_index=idx,
            )


def _renumber_chunk_indexes(chunks: Iterable[Chunk]) -> Iterator[Chunk]:
    """Monotonically renumber `chunk_index` per (source, section) so the
    ingest dedup key `(source, section, chunk_index)` stays unique
    across distinct parser emissions that share a coarse section id.

    Same shape of fix as the JO dedup tiebreaker (harness-8zx6) — JO
    had different parser-emissions colliding under the same dedup key
    because the tiebreaker wasn't enrichment-aware; here PHAK has them
    colliding because the section id itself is coarse.

    No-op for fine-grained sources (JO 7110.65, AIM, 14 CFR) where each
    parser emission already has a unique (source, section): the counter
    starts at 0, expansion-derived chunk_indexes 0..N pass through
    unchanged because they're the only emission for that section.

    Lifesaving for PHAK: the source extractor produces blocks that
    `_parse_phak` files under `section="Chapter N"` (subheadings aren't
    reliably detected in PHAK's PDF→MD output). Without renumbering,
    909 distinct Chapter 16 parser emissions all collide under
    (PHAK, Chapter 16, 0..few) and dedup keeps only ~3 of 909.
    """
    counters: dict[tuple[str, str], int] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.section)
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        yield Chunk(
            source=chunk.source,
            chapter=chunk.chapter,
            section=chunk.section,
            parent_section=chunk.parent_section,
            parent_section_title=chunk.parent_section_title,
            title=chunk.title,
            body=chunk.body,
            principle=chunk.principle,
            tags=chunk.tags,
            chunk_index=idx,
        )


# ---------- top-level ----------


def chunk_markdown(md: str, cfg: ParserConfig) -> list[Chunk]:
    """Full pipeline for one document: segment → parse → size-shape →
    renumber. The renumber pass keeps `chunk_index` unique per
    `(source, section)` so coarse-section sources (PHAK) don't collapse
    under ingest-time dedup. No-op for fine-grained sources."""
    blocks = list(_iter_blocks(md))
    parser = _PARSERS_BY_KIND[cfg.kind]
    parsed = parser(blocks, cfg)
    return list(_renumber_chunk_indexes(_expand_by_size(parsed)))


def write_jsonl(chunks: Iterable[Chunk], dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dst.open("w", encoding="utf-8") as fp:
        for chunk in chunks:
            fp.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def chunk_one(
    cfg: ParserConfig,
    *,
    input_dir: Path,
    output_dir: Path,
    force: bool,
) -> str:
    src = input_dir / f"{cfg.slug}.md"
    if not src.exists():
        return f"  ! {cfg.slug}: SOURCE MISSING ({src})"
    dst = output_dir / f"{cfg.slug}.jsonl"
    if not force and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        with dst.open() as fp:
            rows = sum(1 for _ in fp)
        return f"  · {cfg.slug}: up-to-date ({rows:,} rows)"
    md = src.read_text(encoding="utf-8")
    chunks = chunk_markdown(md, cfg)
    count = write_jsonl(chunks, dst)
    return f"  + {cfg.slug}: {count:,} rows → {_rel(dst)}"


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def sample_rows(output_dir: Path, n: int, rng: random.Random) -> list[dict[str, object]]:
    """Collect every row across every jsonl under `output_dir` and
    return a uniform random sample of size `n`."""
    all_rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*.jsonl")):
        with path.open() as fp:
            for line in fp:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))
    if n >= len(all_rows):
        return all_rows
    return rng.sample(all_rows, n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", nargs="*", default=(), metavar="SLUG")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="After chunking, print a random sample of N rows for acceptance review.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --sample.")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input directory does not exist: {args.input}", file=sys.stderr)
        return 1

    only = tuple(args.only)
    if only:
        unknown = sorted(set(only) - set(PARSERS))
        if unknown:
            print(
                f"error: unknown doc slug(s): {unknown}. Known: {sorted(PARSERS)}",
                file=sys.stderr,
            )
            return 1

    cfgs = [PARSERS[slug] for slug in (only or tuple(PARSERS.keys()))]

    print(f"chunking {len(cfgs)} doc(s) from {_rel(args.input)}")
    print(f"  → {_rel(args.output)}")
    exit_code = 0
    for cfg in cfgs:
        line = chunk_one(cfg, input_dir=args.input, output_dir=args.output, force=args.force)
        print(line)
        if line.lstrip().startswith("!"):
            exit_code = 2

    if args.sample > 0:
        rng = random.Random(args.seed)  # noqa: S311 — acceptance sampling, not security
        sample = sample_rows(args.output, args.sample, rng)
        print(f"\n--- sample {len(sample)} rows ---")
        for row in sample:
            header = (
                f"[{row['source']} · chapter {row['chapter'] or '-'} · section {row['section']!r}]"
            )
            print(header)
            body = str(row["body"])
            preview = body[:300] + ("…" if len(body) > 300 else "")
            print(preview)
            print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
