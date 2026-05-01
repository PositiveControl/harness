"""Cosine-floor calibration for the scope gate (harness-8dop).

Samples N random episodic-store chunks and probes their retrieval
score under the same hybrid path the chat / eval pipeline uses. The
top-1 score of each in-scope query characterizes the distribution we
need a scope floor to LET PASS. Compared to the corpus noise floor
(empirically ~0.016 for airton_c1; harness-11ha), the gap tells us
whether a 0.10 cut is structurally separable.

Usage (from repo root):
    HARNESS_CHARACTER_NAME=airton_c1 uv run python scripts/scope_calibration.py
    HARNESS_CHARACTER_NAME=airton_c1 uv run python scripts/scope_calibration.py \\
        --n 200 --seed 7 --query-field title

Query strategies (`--query-field`):
    title       — chunk title alone (e.g. "WAKE TURBULENCE SEPARATION")
    title_body  — title + first 80 chars of body (richer signal,
                  closer to a paraphrased question)

Both should land above 0.10 in 99%+ of samples for airton_c1's
JO 7110.65 corpus; if they don't, the floor is too high.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import sqlite3  # noqa: E402

from harness.config import settings  # noqa: E402
from harness.retrieval.st_embedder import SentenceTransformersEmbedder  # noqa: E402
from harness.store.episodic import EpisodicStore  # noqa: E402

DEFAULT_CHARACTER = "airton_c1"
DEFAULT_N = 200
DEFAULT_SEED = 17


def _query_for(record: object, *, field: str) -> str:
    title = getattr(record, "title", "") or ""
    body = getattr(record, "body", "") or ""
    if field == "title":
        return title.strip()
    if field == "title_body":
        head = body.strip().splitlines()[0] if body.strip() else ""
        return f"{title.strip()} — {head[:80]}".strip(" —")
    raise ValueError(f"unknown --query-field: {field!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--query-field",
        choices=("title", "title_body"),
        default="title_body",
        help="What text to embed as the query (default title_body).",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.10,
        help="Proposed cosine floor; reports fraction of samples below as the false-block rate.",
    )
    args = parser.parse_args(argv)

    character = os.environ.get("HARNESS_CHARACTER_NAME", DEFAULT_CHARACTER)
    db_path = settings.character_db_path
    print(f"character: {character}")
    print(f"store:     {db_path}")

    embedder = SentenceTransformersEmbedder()
    store = EpisodicStore(db_path, embedder=embedder)

    # Pull every seed row's (id, title, body); sample without replacement.
    rows: list[tuple[int, str, str]] = []
    with sqlite3.connect(db_path) as con:
        for rid, title, body in con.execute(
            "SELECT id, title, body FROM episodic WHERE tier='seed'"
        ):
            rows.append((int(rid), title or "", body or ""))
    if not rows:
        print("no seed rows in store — abort", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)  # noqa: S311 — non-cryptographic sampling for stats only
    sampled = rng.sample(rows, k=min(args.n, len(rows)))

    # The gate uses raw cosine (mode='dense') not the RRF-fused hybrid
    # score. RRF flattens to 1/(60+rank) ~= 0.016-0.033 and would be
    # useless as an absolute floor — the bead's "noise 0.016 vs in-scope
    # 0.4-0.7" gap is the cosine-space gap, not the RRF-space gap.
    scores: list[float] = []
    skipped = 0
    for _rid, title, body in sampled:
        rec = type("R", (), {"title": title, "body": body})()
        query = _query_for(rec, field=args.query_field)
        if not query:
            skipped += 1
            continue
        hits = store.search(query, k=3, mode="dense")
        if not hits:
            skipped += 1
            continue
        top_score = hits[0][1]
        scores.append(float(top_score))

    if not scores:
        print("no scoreable samples — abort", file=sys.stderr)
        return 1

    scores.sort()
    n = len(scores)

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, round((p / 100.0) * (n - 1))))
        return scores[idx]

    print(f"\nSampled {n} chunks (skipped: {skipped})")
    print(f"Query field: {args.query_field}")
    print("\nin-scope top-1 cosine distribution:")
    print(f"  min:    {scores[0]:.4f}")
    print(f"  p01:    {pct(1):.4f}")
    print(f"  p05:    {pct(5):.4f}")
    print(f"  p25:    {pct(25):.4f}")
    print(f"  median: {pct(50):.4f}")
    print(f"  p75:    {pct(75):.4f}")
    print(f"  p95:    {pct(95):.4f}")
    print(f"  max:    {scores[-1]:.4f}")
    print(f"  mean:   {sum(scores) / n:.4f}")

    below_floor = sum(1 for s in scores if s < args.floor)
    print(f"\nfloor = {args.floor:.4f}")
    print(f"  in-scope queries below floor: {below_floor}/{n} ({100 * below_floor / n:.1f}%)")
    print("  → these would be FALSE-BLOCKED by the scope gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
