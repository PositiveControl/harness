"""Validate a proposed synonym YAML against airton_c1's atc retrieval
eval (harness-rup2, Phase 1 of the synonym management plan).

Reads a proposed YAML (same shape as `corpus/query_synonyms.yaml`),
layers it on top of the live shared + query-only files, runs
`run_atc_retrieval` against the airton_c1 episodic store, and diffs
per-case ranks vs the saved baseline JSON. Emits a per-section
verdict (LIFT / MIXED / DEAD / HARMFUL) so the human reviewer knows
whether to commit.

Test-only — query-side proposals (the policy default destination).
Shared-side proposals (synonyms.yaml) bake into stored rows and
require re-ingest; bench_embedder.py is the right tool for that
flow until --reingest support lands here.

Usage:
    uv run python scripts/test_synonyms.py /tmp/suggested.yaml
    uv run python scripts/test_synonyms.py path/to/proposed.yaml --json

Verdicts:
    LIFT     — target case(s) for a section improved; no other
               case regressed. Safe to commit per policy.
    MIXED    — target improved BUT some other case regressed.
               Tighten the phrasing or accept the trade.
    DEAD     — no case rank moved. Refine or prune.
    HARMFUL  — target unchanged or worse, OR aggregate worse.
               Abandon the entry.

Lives under harness-rup2. Pairs with `scripts/suggest_synonyms.py`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.config import settings  # noqa: E402
from harness.evals.atc import default_fixture_path, load_fixture  # noqa: E402
from harness.evals.atc_retrieval import (  # noqa: E402
    RetrievalHit,
    run_atc_retrieval,
)
from harness.retrieval.query_expander import (  # noqa: E402
    default_query_only_synonyms_path,
    default_synonyms_path,
)
from harness.retrieval.synonym_workflow import (  # noqa: E402
    Verdict,
    case_ranks,
    cases_for_section,
    classify_section,
    expander_with,
    format_verdict_report,
    load_proposed_yaml,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "proposed",
        type=Path,
        help="Path to a proposed-synonyms YAML (same shape as corpus/query_synonyms.yaml).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline JSON to diff against. Defaults to "
        "character/<character>/atc_retrieval_baseline.json.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Fixture YAML. Defaults to character/<character>/atc_eval.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON envelope instead of the human report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any verdict is HARMFUL or MIXED. By "
        "default the script exits 0 even on bad verdicts so the user "
        "can read the report and decide.",
    )
    args = parser.parse_args(argv)

    if not args.proposed.exists():
        sys.stderr.write(f"proposed YAML not found: {args.proposed}\n")
        return 2

    character_path = settings.character_path
    fixture_path = args.fixture or default_fixture_path(character_path)
    baseline_path = args.baseline or character_path / "atc_retrieval_baseline.json"
    if not fixture_path.exists():
        sys.stderr.write(f"fixture not found: {fixture_path}\n")
        return 2
    if not baseline_path.exists():
        sys.stderr.write(
            f"baseline not found: {baseline_path}\n"
            "Run `harness eval atc-retrieval --save-baseline` first so the "
            "tester has a 'before' snapshot to diff against.\n"
        )
        return 2

    proposed = load_proposed_yaml(args.proposed)
    if not proposed:
        sys.stderr.write(
            f"proposed YAML at {args.proposed} parsed to zero entries — "
            "check the shape (`version: 1` + `sections:` mapping).\n"
        )
        return 2

    fixture = load_fixture(fixture_path)
    baseline = json.loads(baseline_path.read_text())

    # Build the under-test expander: live shared + query-only +
    # proposal layered on top. Run retrieval against the airton_c1
    # episodic store with that expander. The "before" numbers come
    # straight from the saved baseline, so this is one eval run.
    expander = expander_with(
        shared_path=default_synonyms_path(character_path),
        query_only_path=default_query_only_synonyms_path(character_path),
        proposed=proposed,
    )

    from harness.character import load_character
    from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    from harness.store.episodic import EpisodicStore

    character = load_character(character_path)  # noqa: F841 — load to validate path
    embedder = SentenceTransformersEmbedder()
    store = EpisodicStore(settings.character_db_path, embedder=embedder)

    try:

        def _search(query: str, depth: int) -> list[RetrievalHit]:
            raw = store.search(expander.expand(query), k=depth, mode="hybrid")
            return [
                RetrievalHit(principle=rec.principle or "", score=float(score))
                for rec, score in raw
            ]

        result = run_atc_retrieval(fixture, _search, k=10)
    finally:
        store.close()

    after_ranks = case_ranks(result.cases)
    before_ranks = {
        str(c["id"]): (
            int(c["rank_of_first_expected"])
            if isinstance(c.get("rank_of_first_expected"), int)
            else None
        )
        for c in baseline.get("cases", [])
        if isinstance(c, dict) and "id" in c
    }

    verdicts = []
    for section, variants in proposed.items():
        target_ids = cases_for_section(fixture, section)
        verdict = classify_section(
            section=section,
            proposed_variants=variants,
            target_case_ids=target_ids,
            before_ranks=before_ranks,
            after_ranks=after_ranks,
        )
        verdicts.append(verdict)

    if args.json:
        envelope: dict[str, object] = {
            "proposed_path": str(args.proposed),
            "baseline_path": str(baseline_path),
            "fixture_path": str(fixture_path),
            "verdicts": [
                {
                    "section": v.section,
                    "verdict": v.verdict.value,
                    "rationale": v.rationale,
                    "proposed_variants": list(v.proposed_variants),
                    "target_changes": [
                        {"case_id": c.case_id, "before": c.before, "after": c.after}
                        for c in v.target_changes
                    ],
                    "collateral_changes": [
                        {"case_id": c.case_id, "before": c.before, "after": c.after}
                        for c in v.collateral_changes
                    ],
                }
                for v in verdicts
            ],
        }
        json.dump(envelope, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_verdict_report(verdicts) + "\n")

    if args.strict and any(v.verdict in (Verdict.HARMFUL, Verdict.MIXED) for v in verdicts):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
