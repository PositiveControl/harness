"""Ingest atc's chunked JSONL into the episodic store (harness-xbk.6).

Reads character/airton_c/corpus/chunks/*.jsonl (output of
harness-xbk.5's chunker) and writes rows to atc's per-character
episodic store as tier='seed', user_id=None (shared persona-wide).

Idempotency:
  external_id = "atc-corpus:<source>:<section>:<chunk_index>"
  Re-running is a no-op for already-present rows — EpisodicStore.ingest
  returns the existing record unchanged on external_id match.

Filters applied before ingest:
  - Skip chunks whose cleaned body is shorter than MIN_BODY_CHARS —
    typically PCG cross-reference stubs like "(See FOO.)" that carry
    no independent semantic signal.
  - Dedup (source, section, chunk_index): when the chunker emitted
    multiple rows for the same anchor (common for JO 7110.65 / AIM
    where the "Explanation of Changes" block duplicates every touched
    anchor), keep the longest body.

Metadata mapping to EpisodicStore columns / embed text:
  - title:     chunker title
  - body:      chunker body (the raw text; store builds embed text)
  - principle: "<source> §<section>" — so the section anchor lands in
               the structured-tag header AND as a standalone embed
               line, boosting retrieval on section-id queries.
  - tags:      chunker tags + "section:<section>", "chapter:<chapter>"
  - tier:      "seed"
  - source:    chunker source_key (JO_7110.65, AIM, PCG, ...)
  - user_id:   None (shared persona-wide)

Usage (from repo root):
    uv run python scripts/atc_ingest.py
    uv run python scripts/atc_ingest.py --only pcg aim
    uv run python scripts/atc_ingest.py --dry-run

Requires atc's character set (loader picks HARNESS_CHARACTER_NAME):
    HARNESS_CHARACTER_NAME=airton_c uv run python scripts/atc_ingest.py

The script sets HARNESS_CHARACTER_NAME=airton_c when unset — keeps
one-liner runs working without per-shell exports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MIN_BODY_CHARS = 50

# Set default character BEFORE importing harness modules — pydantic
# settings reads env at import time. Ingest targets the character's
# own corpus/chunks dir and DB (config.py's character_db_path), so
# HARNESS_CHARACTER_NAME is the single knob for ab, airton_c,
# airton_c1, etc.
os.environ.setdefault("HARNESS_CHARACTER_NAME", "airton_c")

CORPUS_CHUNKS = REPO / "character" / os.environ["HARNESS_CHARACTER_NAME"] / "corpus" / "chunks"

from harness.character import load_character  # noqa: E402
from harness.config import settings  # noqa: E402


def iter_rows(jsonl_path: Path) -> Iterator[dict[str, object]]:
    with jsonl_path.open(encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _body_of(row: dict[str, object]) -> str:
    return str(row.get("body", "")).strip()


# Phase-1 CFR scope filter (harness-74n). atc's corpus spans CFR Title
# 14 Vol 1 + Vol 2 — but most of Title 14 is out of PPL/IFR scope
# (airworthiness standards, UAS, ultralights, parachuting, commercial
# operations, fractional ownership). Baseline run 1 showed those
# competing for retrieval slots against Part 91 on queries where
# pilot-facing sections should win — e.g. §103.23 (ultralight "Flight
# visibility and cloud clearance requirements") ranked higher than
# §91.155 for "VFR cloud clearance" because its title was a closer
# BM25 match. Scoping CFR at ingest prevents the dilution.
_CFR_ALLOW_PARTS: frozenset[str] = frozenset(
    {
        "1",  # Definitions and abbreviations
        "3",  # General requirements
        "61",  # Airman certification
        "67",  # Medical standards
        "71",  # Airspace designations
        "91",  # General operating and flight rules (filtered below)
        "93",  # Special air traffic rules
        "95",  # IFR altitudes
        "97",  # Standard instrument approach procedures
    }
)


def _cfr_in_scope(section: str) -> bool:
    """Return True if `section` (e.g. '91.155', '103.23', '91.1031')
    is in atc's Phase-1 pilot-facing scope. Parts outside the
    allowlist drop. Part 91 is kept through Subpart J (§91.999);
    Subpart K (fractional ownership, §§91.1001-91.1099) and Subpart L
    (continued airworthiness, §91.1101+) are commercial/fractional-
    specific and drop."""
    if not section:
        return False
    part, _, suffix = section.partition(".")
    if part not in _CFR_ALLOW_PARTS:
        return False
    if part == "91":
        # Extract the leading numeric run of the suffix (handles things
        # like "155", "1031", "1001a"). A suffix ≥ 1000 is Subpart K or
        # later — out of scope.
        digits = "".join(ch for ch in suffix if ch.isdigit())
        try:
            if digits and int(digits) >= 1000:
                return False
        except ValueError:
            pass
    return True


def is_noise(row: dict[str, object]) -> bool:
    """Drop low-signal chunks so ingest doesn't swamp the store with
    rows that won't help retrieval. Two filters:

    - body shorter than MIN_BODY_CHARS (PCG cross-ref stubs, TOC
      leakage).
    - CFR rows outside the Phase-1 scope (see `_cfr_in_scope`).
    """
    body = _body_of(row)
    if len(body) < MIN_BODY_CHARS:
        return True
    source = str(row.get("source", ""))
    return source.startswith("CFR_14_") and not _cfr_in_scope(str(row.get("section", "")))


def dedup_by_anchor(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Keep the longest body per (source, section, chunk_index). JO /
    AIM both ship an explanation-of-changes block that re-states every
    touched anchor; the real-content row is typically longer, so
    'longest wins' biases toward content."""
    by_key: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["source"]),
            str(row["section"]),
            int(row.get("chunk_index", 0) or 0),
        )
        existing = by_key.get(key)
        if existing is None or len(_body_of(row)) > len(_body_of(existing)):
            by_key[key] = row
    return list(by_key.values())


def external_id_for(row: dict[str, object]) -> str:
    return f"atc-corpus:{row['source']}:{row['section']}:{row.get('chunk_index', 0)}"


def principle_for(row: dict[str, object]) -> str:
    return f"{row['source']} §{row['section']}"


def tags_for(row: dict[str, object]) -> list[str]:
    base = list(row.get("tags") or [])
    section = str(row.get("section", ""))
    chapter = str(row.get("chapter", ""))
    if section:
        base.append(f"section:{section}")
    if chapter:
        base.append(f"chapter:{chapter}")
    return base


def load_rows_for_slug(slug: str) -> list[dict[str, object]]:
    """Read one doc's chunks + apply filters + dedup. Returns the rows
    ready for ingest."""
    path = CORPUS_CHUNKS / f"{slug}.jsonl"
    if not path.exists():
        return []
    rows = [row for row in iter_rows(path) if not is_noise(row)]
    return dedup_by_anchor(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--only",
        nargs="*",
        default=(),
        metavar="SLUG",
        help="Only ingest these doc slugs (default: every *.jsonl under chunks/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report row counts per slug without opening the store.",
    )
    args = parser.parse_args(argv)

    if not CORPUS_CHUNKS.exists():
        print(f"corpus/chunks not found: {CORPUS_CHUNKS}", file=sys.stderr)
        return 1

    only = tuple(args.only)
    slugs = only or tuple(sorted(p.stem for p in CORPUS_CHUNKS.glob("*.jsonl")))

    if not slugs:
        print(f"no .jsonl files under {CORPUS_CHUNKS}", file=sys.stderr)
        return 1

    # Dry-run: filter counts only. Useful for tuning before paying the
    # embedder cost.
    if args.dry_run:
        print(f"[dry-run] character={settings.character_name}")
        print(f"[dry-run] target DB: {settings.character_db_path}")
        for slug in slugs:
            rows = load_rows_for_slug(slug)
            print(f"  {slug}: {len(rows):,} rows after filter + dedup")
        total = sum(len(load_rows_for_slug(s)) for s in slugs)
        print(f"[dry-run] total: {total:,} rows")
        return 0

    # Real run: load the character, open the episodic store, ingest
    # every filtered row idempotently. Imports are lazy so --dry-run
    # paths don't spin up sentence-transformers.
    try:
        from harness.retrieval.st_embedder import SentenceTransformersEmbedder
        from harness.store.episodic import EpisodicStore
    except ImportError as exc:
        print(f"retrieval extras not installed: {exc}", file=sys.stderr)
        return 1

    character = load_character(settings.character_path)
    print(f"character: {character.name}")
    print(f"db:        {settings.character_db_path}")
    print(f"embedder:  {settings.embedder_repo}")

    embedder = SentenceTransformersEmbedder(settings.embedder_repo)
    store = EpisodicStore(settings.character_db_path, embedder=embedder)

    total_new = 0
    total_existing = 0
    try:
        for slug in slugs:
            rows = load_rows_for_slug(slug)
            if not rows:
                print(f"  · {slug}: no rows")
                continue
            new_here = 0
            existing_here = 0
            for row in rows:
                ext_id = external_id_for(row)
                if store.has(ext_id):
                    existing_here += 1
                    continue
                store.ingest(
                    external_id=ext_id,
                    title=str(row.get("title") or row.get("section") or "(untitled)"),
                    body=str(row.get("body", "")),
                    principle=principle_for(row),
                    tags=tags_for(row),
                    tier="seed",
                    source=str(row["source"]),
                    user_id=None,
                )
                new_here += 1
            total_new += new_here
            total_existing += existing_here
            print(
                f"  + {slug}: {new_here:,} new, {existing_here:,} already present "
                f"({len(rows):,} considered)"
            )
    finally:
        store.close()

    print(f"\ntotal: {total_new:,} new, {total_existing:,} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
