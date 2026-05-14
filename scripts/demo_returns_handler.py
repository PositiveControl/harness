"""Deterministic demo of the returns_handler contract pipeline
(harness-kgpi).

Companion to `docs/returns_handler/demo_session.md` (which captures a
real MLX chat session). This script drives the same retrieval
primitive without a model in the loop — useful for:
  - Showing exactly what assemble_context would return for a given
    `(customer_id, request_summary)` pair.
  - Smoke-testing the character bootstrap + tabular wiring after
    edits to core.yaml, the contract YAML, or the policy seeds.
  - CI / reviewers — no MLX + Qwen download required.

Usage:
    uv run python scripts/demo_returns_handler.py
    uv run python scripts/demo_returns_handler.py --customer C9148 \\
        --request "leather jacket arrived damaged"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from harness.character import load_character  # noqa: E402
from harness.retrieval.context_package import AccessPolicy  # noqa: E402
from harness.retrieval.contract import (  # noqa: E402
    StoreBundle,
    assemble_package,
    load_contract,
)
from harness.retrieval.st_embedder import SentenceTransformersEmbedder  # noqa: E402
from harness.store.episodic import EpisodicStore  # noqa: E402
from harness.store.tabular import build_tabular_store_for_character  # noqa: E402

CHARACTER_PATH = REPO / "character" / "returns_handler"


def _seed_episodic(store: EpisodicStore) -> None:
    """Ingest the character's policy seed memories into the live
    episodic store. Idempotent on external_id."""
    char = load_character(CHARACTER_PATH)
    for seed in char.seed_memories:
        store.ingest(
            external_id=seed.id,
            title=seed.title,
            body=seed.body,
            principle=seed.principle,
            tags=seed.tags,
            tier="seed",
            source="demo_returns_handler",
        )


def run_demo(*, customer_id: str, request_summary: str, embedder_repo: str) -> None:
    char = load_character(CHARACTER_PATH)
    embedder = SentenceTransformersEmbedder(model_name=embedder_repo)

    # Reuse the character's actual on-disk stores so the demo
    # reflects whatever state `harness memory ingest` has produced.
    # Falls back to an in-memory episodic if the on-disk store is
    # empty — keeps the demo runnable on a fresh clone.
    episodic_db = CHARACTER_PATH / "data" / "harness.sqlite"
    episodic = EpisodicStore(db_path=episodic_db, embedder=embedder)
    # Seed if the policies aren't already ingested. Idempotent on
    # external_id so re-runs are safe.
    _seed_episodic(episodic)

    tabular = build_tabular_store_for_character(
        character_path=CHARACTER_PATH,
        embedder=embedder,
        tabular_tables=char.tabular_tables,
    )
    if tabular is None:
        sys.stderr.write(
            "ERROR: returns_handler.core.yaml is missing tabular_tables — "
            "should not happen on a clean checkout.\n"
        )
        sys.exit(2)

    contract = load_contract(CHARACTER_PATH / "contracts" / "returns_handler.yaml")
    package = assemble_package(
        contract,
        variables={
            "customer_id": customer_id,
            "request_summary": request_summary,
        },
        access=AccessPolicy(user_id=customer_id, role="returns_handler"),
        stores=StoreBundle(episodic=episodic, tabular=tabular),
    )

    print()
    print("=" * 72)
    print(f"Contract: {contract.role}")
    print(f"Intent:   {contract.intent}")
    print(f"Customer: {customer_id}")
    print(f"Request:  {request_summary}")
    print(
        f"Tokens:   {package.tokens_used} / {package.budget.max_tokens}  "
        f"(overflow: {package.tokens_overflow})"
    )
    print(f"Complete: {package.is_complete}")
    if package.missing_required_slots:
        print(f"Missing:  {', '.join(package.missing_required_slots)}")
    print("=" * 72)

    for slot in contract.slots:
        hits = package.hits_for_slot(slot.name)
        suffix = "required" if slot.required else "optional"
        print()
        print(f"--- {slot.name}  ({len(hits)} hit{'' if len(hits) == 1 else 's'}, {suffix}) ---")
        if not hits:
            print("  (no hits)")
            continue
        for h in hits:
            print(
                f"  [{h.provenance.store}/{h.provenance.method} "
                f"score={h.provenance.score:.3f}] {h.provenance.record_id}"
            )
            for line in h.body.splitlines():
                print(f"    {line}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--customer",
        default="C2856",
        help="Customer id to demo (default: C2856 — high-value backpack escalation).",
    )
    parser.add_argument(
        "--request",
        default="damaged backpack on arrival",
        help="One-line request summary used in the refund_policy query template.",
    )
    parser.add_argument(
        "--embedder",
        default="BAAI/bge-small-en-v1.5",
        help="HF embedder repo. Default: BAAI/bge-small-en-v1.5",
    )
    args = parser.parse_args(argv)

    run_demo(
        customer_id=args.customer,
        request_summary=args.request,
        embedder_repo=args.embedder,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
