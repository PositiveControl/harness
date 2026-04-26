"""Propose synonym entries for fixture cases that miss or land at
weak ranks (harness-rup2, Phase 1 of the synonym management plan).

Reads `atc_retrieval_baseline.json`, identifies cases at rank > N or
hard miss, fetches each expected chunk's body from the episodic
store, and asks a small model (default: settings.router_repo) for
3-5 lay-form phrases that bridge the lay-query → doc-text gap. Emits
a YAML draft to stdout, ready for human review and pasting into
`corpus/query_synonyms.yaml` after `scripts/test_synonyms.py`
verifies LIFT.

The suggester NEVER auto-commits. It produces text; the reviewer
edits, runs the tester, and pastes only entries that earn a LIFT
verdict. Per the policy, all entries default to query_synonyms.yaml.

Usage:
    uv run python scripts/suggest_synonyms.py
    uv run python scripts/suggest_synonyms.py --rank-threshold 5
    uv run python scripts/suggest_synonyms.py --output /tmp/suggested.yaml
    uv run python scripts/suggest_synonyms.py --model-repo \\
        mlx-community/Qwen2.5-3B-Instruct-4bit

Lives under harness-rup2. Pairs with `scripts/test_synonyms.py`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.config import settings  # noqa: E402
from harness.evals.atc import default_fixture_path, load_fixture  # noqa: E402
from harness.retrieval.synonym_workflow import (  # noqa: E402
    CandidateMiss,
    emit_suggestion_doc,
    emit_suggestion_yaml_block,
    select_candidates,
    suggest_for_case,
)


def _fetch_chunk_body(store, section: str) -> str:
    """Pull the longest stored body for a section anchor — that's
    usually the canonical content chunk (chunker dedup keeps the
    enriched row over the changelog row when bodies differ; among
    a section's surviving rows the longest typically holds the
    primary substance)."""
    cur = store._con.execute(
        "SELECT body FROM episodic WHERE principle LIKE ? AND superseded_by IS NULL "
        "ORDER BY length(body) DESC LIMIT 1",
        (f"%§{section}%",),
    )
    row = cur.fetchone()
    return str(row[0]) if row else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline JSON to scan for misses. Defaults to "
        "character/<character>/atc_retrieval_baseline.json.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Fixture YAML. Defaults to character/<character>/atc_eval.yaml.",
    )
    parser.add_argument(
        "--rank-threshold",
        type=int,
        default=3,
        help="Cases at rank > N (or hard miss) become candidates. Default 3.",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=None,
        help="HF repo for the suggester adapter. Defaults to settings.router_repo "
        "so the 3B router model serves both routing and suggesting.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Cap the number of candidates handed to the model. 0 = no cap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the YAML to a file instead of stdout.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-candidate progress to stderr while the model runs.",
    )
    args = parser.parse_args(argv)

    character_path = settings.character_path
    fixture_path = args.fixture or default_fixture_path(character_path)
    baseline_path = args.baseline or character_path / "atc_retrieval_baseline.json"
    if not fixture_path.exists():
        sys.stderr.write(f"fixture not found: {fixture_path}\n")
        return 2
    if not baseline_path.exists():
        sys.stderr.write(
            f"baseline not found: {baseline_path}\n"
            "Run `harness eval atc-retrieval --save-baseline` first.\n"
        )
        return 2

    fixture = load_fixture(fixture_path)
    baseline = json.loads(baseline_path.read_text())
    candidates = select_candidates(
        fixture=fixture,
        baseline=baseline,
        rank_threshold=args.rank_threshold,
    )
    if args.max_cases > 0:
        candidates = candidates[: args.max_cases]
    if not candidates:
        sys.stderr.write(
            f"No candidates at rank > {args.rank_threshold} in {baseline_path.name}. "
            "Either retrieval is healthy or the threshold is too generous.\n"
        )
        return 0

    if args.verbose:
        sys.stderr.write(f"candidates: {len(candidates)}\n")
        for cand in candidates:
            rank_repr = "miss" if cand.current_rank is None else f"r{cand.current_rank}"
            sys.stderr.write(f"  {rank_repr:<5} {cand.case_id} → §{cand.expected_section}\n")

    # Lazy adapter + store load — both require sentence-transformers
    # and MLX, which are slow imports we'd rather skip on early
    # validation failures.
    from harness.model.mlx import MLXAdapter
    from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    from harness.store.episodic import EpisodicStore

    repo = args.model_repo or settings.router_repo
    adapter = MLXAdapter(repo=repo)
    embedder = SentenceTransformersEmbedder()
    store = EpisodicStore(settings.character_db_path, embedder=embedder)

    blocks: list[str] = []
    try:
        for cand in candidates:
            body = _fetch_chunk_body(store, cand.expected_section)
            if not body:
                if args.verbose:
                    sys.stderr.write(
                        f"  skipping {cand.case_id}: no chunk for §{cand.expected_section}\n"
                    )
                continue
            cand_with_body = CandidateMiss(
                case_id=cand.case_id,
                query=cand.query,
                expected_section=cand.expected_section,
                current_rank=cand.current_rank,
                chunk_body=body,
            )
            if args.verbose:
                sys.stderr.write(f"  generating for {cand_with_body.case_id}...\n")
            variants = suggest_for_case(
                adapter=adapter,
                query=cand_with_body.query,
                section=cand_with_body.expected_section,
                body=cand_with_body.chunk_body,
            )
            blocks.append(
                emit_suggestion_yaml_block(
                    section=cand_with_body.expected_section,
                    variants=variants,
                    case_id=cand_with_body.case_id,
                    current_rank=cand_with_body.current_rank,
                )
            )
    finally:
        store.close()

    doc = emit_suggestion_doc(blocks)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(doc)
        sys.stderr.write(f"wrote {args.output}\n")
    else:
        sys.stdout.write(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
