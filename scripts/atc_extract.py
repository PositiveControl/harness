"""PDF → markdown batch extractor for atc's FAA corpus (harness-xbk.4).

Uses pymupdf4llm.to_markdown() — preserves headings, lists, and tables
in the markdown output so the section-aware chunker (harness-xbk.5)
has anchors to parse. Idempotent on source mtime so re-runs are cheap.

Requires the `atc` optional extra (run additively with `all` so the rest of
the env isn't pruned):
    uv sync --extra all --extra atc

Usage (from repo root):
    uv run python scripts/atc_extract.py
    uv run python scripts/atc_extract.py --source /path/to/pdfs
    uv run python scripts/atc_extract.py --force     # rebuild all
    uv run python scripts/atc_extract.py --only pcg aim

Phase-1 priority corpus (harness-xbk epic):
    - PCG — Pilot/Controller Glossary
    - AIM — Aeronautical Information Manual
    - JO 7110.65 — Air Traffic Control procedures
    - PHAK — Pilot's Handbook of Aeronautical Knowledge (8083-25C)
    - 14 CFR Title 14 Vol 1 — Parts 1-59 (incl. Part 61 airman cert)
    - 14 CFR Title 14 Vol 2 — Parts 60-109 (incl. Part 91 general ops)

Extending the corpus (Phase-1.5, harness-5zl): add entries to
PHASE_1_CORPUS and re-run. Existing outputs are preserved per
idempotency; only new/changed sources are re-extracted.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Shape of pymupdf4llm.to_markdown: (path_str) -> markdown_str. Wider
# than the upstream stub (which takes extra kwargs) but all we need.
ToMarkdown = Callable[[str], str]

REPO = Path(__file__).resolve().parents[1]

# Default character. HARNESS_CHARACTER_NAME overrides — the extractor
# writes markdown under that character's corpus dir so airton_c1,
# airton_c2, etc. can share the pipeline.
_CHARACTER = os.environ.get("HARNESS_CHARACTER_NAME", "airton_c")

# Default source dir: Mark's syber_vision_llm training-data corpus of
# FAA PDFs. Overridable via --source so a CI mirror or a stripped-down
# test fixture can land elsewhere.
DEFAULT_SOURCE = Path("/Users/mevans/dev/aishiteru/syber_vision_llm/training-data/0_pdfs")
DEFAULT_OUTPUT = REPO / "character" / _CHARACTER / "corpus" / "markdown"


@dataclass(frozen=True)
class CorpusDoc:
    """One source→output mapping. `slug` is the stable filename stem
    used for the output markdown (and, downstream, as the `source`
    metadata key in the chunker's JSONL rows). Change slugs at your
    own peril — retrieval keyed by source metadata will miss until
    ingest re-runs."""

    slug: str
    source_filename: str
    title: str


PHASE_1_CORPUS: tuple[CorpusDoc, ...] = (
    CorpusDoc(
        slug="pcg",
        source_filename="PCG_Bsc_dtd_2-20-25_POST.pdf",
        title="Pilot/Controller Glossary",
    ),
    CorpusDoc(
        slug="aim",
        source_filename="AIM_Basic_dtd_2-20-25_post.pdf",
        title="Aeronautical Information Manual",
    ),
    CorpusDoc(
        slug="jo_7110_65",
        source_filename="7110.65BB_Bsc_w_Chg_1_and_2_dtd_1-22-26_Final.pdf",
        title="JO 7110.65 — Air Traffic Control (Basic w/ Changes 1 & 2, 2026-01-22)",
    ),
    CorpusDoc(
        slug="phak",
        source_filename="faa-h-8083-25c.pdf",
        title="Pilot's Handbook of Aeronautical Knowledge (FAA-H-8083-25C)",
    ),
    CorpusDoc(
        slug="cfr_14_vol1",
        source_filename="CFR-2025-title14-vol1.pdf",
        title="14 CFR Title 14 Vol 1 (Parts 1-59)",
    ),
    CorpusDoc(
        slug="cfr_14_vol2",
        source_filename="CFR-2025-title14-vol2.pdf",
        title="14 CFR Title 14 Vol 2 (Parts 60-109)",
    ),
)


def is_up_to_date(src: Path, dst: Path) -> bool:
    """Idempotency check: target markdown exists and is at least as
    fresh as the source PDF. Uses >= so a rebuild triggered by
    touching the source to exactly match also re-extracts."""
    if not dst.exists():
        return False
    return dst.stat().st_mtime >= src.stat().st_mtime


def resolve_docs(
    corpus: tuple[CorpusDoc, ...],
    *,
    only: tuple[str, ...] = (),
) -> tuple[CorpusDoc, ...]:
    """Narrow `corpus` to the supplied `only` slugs (empty = all).
    Unknown slugs raise so a typo doesn't silently skip work."""
    if not only:
        return corpus
    by_slug = {doc.slug: doc for doc in corpus}
    unknown = sorted(set(only) - set(by_slug))
    if unknown:
        raise ValueError(f"unknown doc slug(s): {unknown}. Known: {sorted(by_slug)}")
    return tuple(by_slug[s] for s in only)


def extract_one(
    doc: CorpusDoc,
    *,
    source_root: Path,
    output_root: Path,
    force: bool,
    to_markdown: ToMarkdown | None = None,
) -> str:
    """Extract one doc. Returns a one-line status for logging.

    `to_markdown` is injectable for testing — production passes None
    and we import pymupdf4llm lazily so `--help` doesn't require the
    `atc` extra to be installed."""
    src = source_root / doc.source_filename
    if not src.exists():
        return f"  ! {doc.slug}: SOURCE MISSING ({src})"

    dst = output_root / f"{doc.slug}.md"
    if not force and is_up_to_date(src, dst):
        return f"  · {doc.slug}: up-to-date ({dst.stat().st_size:,}B)"

    extractor: ToMarkdown
    if to_markdown is None:
        try:
            import pymupdf4llm
        except ImportError:
            sys.exit(
                "pymupdf4llm is not installed. Run "
                "`uv sync --extra all --extra atc` to install atc's corpus-extraction deps."
            )
        extractor = pymupdf4llm.to_markdown
    else:
        extractor = to_markdown

    md = extractor(str(src))
    output_root.mkdir(parents=True, exist_ok=True)
    dst.write_text(md, encoding="utf-8")
    # Prefer a repo-relative path for the log line when the output is
    # inside the repo (the production case); fall back to the absolute
    # path when a caller overrides --output elsewhere (tests, mirrors).
    display = dst.relative_to(REPO) if dst.is_relative_to(REPO) else dst
    return f"  + {doc.slug}: extracted ({len(md):,}B → {display})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Directory containing source PDFs (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(f"Output directory for markdown (default: {DEFAULT_OUTPUT.relative_to(REPO)})"),
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=(),
        metavar="SLUG",
        help="Extract only these slugs (space-separated). Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even when the target is fresher than the source.",
    )
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"source directory does not exist: {args.source}", file=sys.stderr)
        return 1

    try:
        docs = resolve_docs(PHASE_1_CORPUS, only=tuple(args.only))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"extracting {len(docs)} doc(s) from {args.source}")
    print(
        f"  → {args.output.relative_to(REPO) if args.output.is_relative_to(REPO) else args.output}"
    )

    exit_code = 0
    for doc in docs:
        line = extract_one(
            doc,
            source_root=args.source,
            output_root=args.output,
            force=args.force,
        )
        print(line)
        if line.lstrip().startswith("!"):
            exit_code = 2  # source missing — non-fatal, but flag at exit
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
