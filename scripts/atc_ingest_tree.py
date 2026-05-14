"""Ingest JO 7110.65 into a hierarchical `DocumentTreeStore`
(harness-h5ly / Phase 1).

Reads the same chunk JSONL the flat ingest uses
(`character/airton_c1/corpus/chunks/jo_7110_65.jsonl`) but writes one
SECTION node per (chapter, parent_section, section) tuple — aggregating
the section's chunks into a single body — plus structural-only
chapter / parent_section nodes so the hierarchy is queryable. The
deliberate departure from `scripts/atc_ingest.py` is granularity:
section nodes carry a richer body and ONE embedding per section,
675 nodes total versus 1,854 leaf chunks under the flat path.

The output store is a separate SQLite under
`retrieval_eval/data/tree_atc.sqlite` so Phase 1 bench cells can open
it side-by-side with the live airton_c1 flat store without conflict.
The path is intentionally inside `retrieval_eval/data/` (already
git-tracked via the .gitignore exception) so the bench can recreate
the snapshot when the source JSONL drifts.

Usage:
    uv run python scripts/atc_ingest_tree.py
    uv run python scripts/atc_ingest_tree.py --rebuild     # wipe + redo
    uv run python scripts/atc_ingest_tree.py \\
        --source-jsonl path/to/chunks.jsonl \\
        --output path/to/tree.sqlite
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from harness.retrieval.st_embedder import SentenceTransformersEmbedder  # noqa: E402
from harness.store.document_tree import DocumentTreeStore  # noqa: E402

_DEFAULT_SOURCE = REPO / "character" / "airton_c1" / "corpus" / "chunks" / "jo_7110_65.jsonl"
_DEFAULT_OUTPUT = REPO / "retrieval_eval" / "data" / "tree_atc.sqlite"
_DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"


def _iter_rows(jsonl_path: Path) -> Iterator[dict[str, Any]]:
    with jsonl_path.open(encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _section_body(chunks: list[dict[str, Any]]) -> str:
    """Concatenate the chunks of a section into one body. Strips the
    per-chunk header line (which the flat ingest adds for context) so
    repeated headers don't dominate BM25 statistics inside one section
    body. Preserves order via `chunk_index` so paragraphs read in
    source order rather than ingest order."""
    ordered = sorted(chunks, key=lambda c: int(c.get("chunk_index", 0)))
    bodies: list[str] = []
    for chunk in ordered:
        body = str(chunk.get("body", "")).strip()
        if body:
            bodies.append(body)
    return "\n\n".join(bodies)


def _chapter_heading(chunks: list[dict[str, Any]]) -> str:
    """Chapter heading — picked from the first chunk's title that lives
    at chapter granularity. Fallback to a stub when no chapter-level
    chunk exists in the corpus (none do today; we synthesize)."""
    chapter = str(chunks[0].get("chapter", "?"))
    return f"Chapter {chapter}"


def _parent_section_heading(chunks: list[dict[str, Any]]) -> str:
    parent = str(chunks[0].get("parent_section", "?"))
    return f"§{parent}"


def ingest(
    *,
    source_jsonl: Path,
    output_db: Path,
    embedder_repo: str,
    rebuild: bool,
    verbose: bool = True,
) -> dict[str, int]:
    """Ingest one source JSONL into a tree store. Returns row counts
    keyed by node_type so callers can assert / log."""
    if rebuild and output_db.exists():
        output_db.unlink()
        # Also drop the WAL / SHM sidecars if they hang around.
        for ext in (".sqlite-wal", ".sqlite-shm"):
            sidecar = output_db.with_suffix(ext)
            if sidecar.exists():
                sidecar.unlink()

    embedder = SentenceTransformersEmbedder(model_name=embedder_repo)
    store = DocumentTreeStore(db_path=output_db, embedder=embedder)
    doc = store.upsert_document(name="JO_7110.65", source_uri=str(source_jsonl))

    # Group chunks by (chapter, parent_section, section). Hierarchy
    # walks from this single grouping — no second pass needed.
    by_section: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_rows(source_jsonl):
        key = (
            str(row.get("chapter", "")),
            str(row.get("parent_section", "")),
            str(row.get("section", "")),
        )
        if not all(key):
            continue  # skip malformed rows
        by_section[key].append(row)

    # Cache chapter / parent_section node ids so we don't re-create
    # them inside the inner loop. Ordinal is the order we first saw
    # the chapter / parent (stable across runs because dict iteration
    # is insertion-ordered and we read JSONL top-to-bottom).
    chapter_node_id: dict[str, int] = {}
    parent_node_id: dict[tuple[str, str], int] = {}

    counts = {"chapter": 0, "section_group": 0, "section": 0}
    ordinals = {"chapter": 0, "section_group": 0, "section": 0}

    for (chapter, parent_section, section), chunks in by_section.items():
        # Chapter — structural only.
        if chapter not in chapter_node_id:
            node = store.ingest_node(
                document_id=doc.id,
                parent_id=None,
                path=chapter,
                ordinal=ordinals["chapter"],
                depth=1,
                node_type="chapter",
                heading=_chapter_heading(chunks),
                embed=False,
            )
            chapter_node_id[chapter] = node.id
            ordinals["chapter"] += 1
            counts["chapter"] += 1

        # Parent section — structural only.
        pkey = (chapter, parent_section)
        if pkey not in parent_node_id:
            node = store.ingest_node(
                document_id=doc.id,
                parent_id=chapter_node_id[chapter],
                path=parent_section,
                ordinal=ordinals["section_group"],
                depth=2,
                node_type="section_group",
                heading=_parent_section_heading(chunks),
                embed=False,
            )
            parent_node_id[pkey] = node.id
            ordinals["section_group"] += 1
            counts["section_group"] += 1

        # Section — embedded leaf. Body is the concatenated chunk
        # bodies; heading is the section title.
        heading = str(chunks[0].get("title", "")) or section
        body = _section_body(chunks)
        store.ingest_node(
            document_id=doc.id,
            parent_id=parent_node_id[pkey],
            path=section,
            ordinal=ordinals["section"],
            depth=3,
            node_type="section",
            heading=heading,
            body=body,
            embed=True,
        )
        ordinals["section"] += 1
        counts["section"] += 1

        if verbose and counts["section"] % 100 == 0:
            sys.stderr.write(f"  · {counts['section']} sections embedded\n")

    if verbose:
        sys.stderr.write(
            f"done: {counts['chapter']} chapters, "
            f"{counts['section_group']} section groups, "
            f"{counts['section']} sections (embedded). "
            f"Total embedded: {store.count_embedded()}\n"
        )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=_DEFAULT_SOURCE,
        help=f"Input JSONL. Default: {_DEFAULT_SOURCE.relative_to(REPO)}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output SQLite tree store. Default: {_DEFAULT_OUTPUT.relative_to(REPO)}",
    )
    parser.add_argument(
        "--embedder",
        default=_DEFAULT_EMBEDDER,
        help=f"HF embedder repo. Default: {_DEFAULT_EMBEDDER}",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the output DB before ingest (otherwise append idempotently).",
    )
    args = parser.parse_args(argv)

    if not args.source_jsonl.exists():
        sys.stderr.write(f"source JSONL not found: {args.source_jsonl}\n")
        return 2

    ingest(
        source_jsonl=args.source_jsonl,
        output_db=args.output,
        embedder_repo=args.embedder,
        rebuild=args.rebuild,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
