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
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from harness.retrieval.st_embedder import SentenceTransformersEmbedder  # noqa: E402
from harness.store.document_tree import DocumentTreeStore  # noqa: E402
from harness.store.document_tree_ingest import (  # noqa: E402
    ingest_blueprints,
    iter_jsonl_nodes,
)

_DEFAULT_SOURCE = REPO / "character" / "airton_c1" / "corpus" / "chunks" / "jo_7110_65.jsonl"
_DEFAULT_OUTPUT = REPO / "retrieval_eval" / "data" / "tree_atc.sqlite"
_DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"


def ingest(
    *,
    source_jsonl: Path,
    output_db: Path,
    embedder_repo: str,
    rebuild: bool,
    verbose: bool = True,
) -> dict[str, int]:
    """Ingest one source JSONL into a tree store. Returns row counts
    keyed by node_type so callers can assert / log.

    Delegates grouping + driver work to
    `harness.store.document_tree_ingest` (harness-2zf4). The corpus-
    specific config — JO 7110.65's three-level chapter/parent_section/
    section hierarchy plus the `Chapter N` / `§N-M` heading prefixes —
    is the only thing left in the script.
    """
    if rebuild and output_db.exists():
        output_db.unlink()
        # Also drop the WAL / SHM sidecars if they hang around.
        for ext in (".sqlite-wal", ".sqlite-shm"):
            sidecar = output_db.with_suffix(ext)
            if sidecar.exists():
                sidecar.unlink()

    embedder = SentenceTransformersEmbedder(model_name=embedder_repo)
    store = DocumentTreeStore(db_path=output_db, embedder=embedder)
    counts = ingest_blueprints(
        store,
        document_name="JO_7110.65",
        source_uri=str(source_jsonl),
        blueprints=iter_jsonl_nodes(
            source_jsonl,
            depth_fields=("chapter", "parent_section", "section"),
            heading_prefixes=("Chapter ", "§", ""),
            leaf_heading_field="title",
        ),
    )

    if verbose:
        sys.stderr.write(
            f"done: {counts.get('chapter', 0)} chapters, "
            f"{counts.get('section_group', 0)} section groups, "
            f"{counts.get('section', 0)} sections (embedded). "
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
