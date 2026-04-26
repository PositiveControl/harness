"""Benchmark a candidate embedder against airton_c1's atc retrieval eval.

Drives the full swap-cycle for one HF embedder repo:

  1. Provision a per-candidate scratch SQLite under
     `character/airton_c1/data/bench/<sanitized-repo>.sqlite` (so the
     real store stays untouched).
  2. Re-ingest the chunks JSONL with the candidate embedder. Times
     ingest; samples peak RSS during the pass.
  3. Run `eval atc-retrieval --json` against the scratch DB.
  4. Emit a single JSON envelope with recall@1/@3/@5/@K, per-case
     ranks, ingest seconds, embed dimension, peak RSS delta, and the
     candidate's repo + sanitized id.

Usage:
    uv run python scripts/bench_embedder.py --repo BAAI/bge-small-en-v1.5
    uv run python scripts/bench_embedder.py --repo BAAI/bge-large-en-v1.5
    uv run python scripts/bench_embedder.py \\
        --repo nomic-ai/nomic-embed-text-v1.5 \\
        --output character/airton_c1/embedder_bench/nomic.json

Multi-candidate sweeps are a thin shell loop on top of this — each run
emits a self-contained JSON, then a small comparator (TBD bead) folds
them into a candidate scoreboard.

The benchmark is idempotent on a stable corpus: same chunks JSONL +
same embedder repo → same numbers. Re-runs after a chunker change
require deleting the scratch SQLite (the script does that automatically
unless --reuse is passed).

Lives under harness-pw9z (Fix D embedder swap). Pairs with
`scripts/bench_router.py` and `scripts/bench_models.py` as the third
benchmark surface — measurement before opinion."""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sanitize_repo(repo: str) -> str:
    """Filesystem-safe slug for a HF repo. `BAAI/bge-small-en-v1.5` →
    `BAAI_bge-small-en-v1.5`. Slash is the only character we transform —
    HF repos otherwise stick to `[A-Za-z0-9._-]`."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", repo)


def _peak_rss_mb() -> float:
    """RSS in MB. macOS reports `ru_maxrss` in bytes, Linux in kB —
    detect on platform. Resolution is per-process peak since startup,
    so deltas across phases give a candidate-load cost."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


@dataclass
class IngestStats:
    seconds: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    rows_ingested: int
    embed_dim: int


@dataclass
class BenchResult:
    repo: str
    sanitized: str
    db_path: str
    fixture: str
    ingest: IngestStats
    eval_seconds: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_k: float
    median_rank: float | None
    cases: list[dict[str, object]]


def _provision_scratch_db(character_path: Path, sanitized: str, *, reuse: bool) -> Path:
    """Place the scratch DB under `<character>/data/bench/<slug>.sqlite`.
    When --reuse isn't passed, any existing file at that path is
    removed so the run starts from a known-empty state."""
    bench_dir = character_path / "data" / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    db_path = bench_dir / f"{sanitized}.sqlite"
    if not reuse and db_path.exists():
        db_path.unlink()
    return db_path


def _ingest_with_embedder(
    *,
    character_name: str,
    chunks_dir: Path,
    db_path: Path,
    embedder_repo: str,
) -> IngestStats:
    """Re-ingest chunks JSONL into a fresh scratch DB under the
    candidate embedder. Routes through the existing
    `scripts/atc_ingest.py` to inherit the dedup + filter rules so
    benchmark numbers reflect what real ingest would produce."""
    rss_before = _peak_rss_mb()
    env = os.environ.copy()
    env["HARNESS_CHARACTER_NAME"] = character_name
    env["HARNESS_EMBEDDER_REPO"] = embedder_repo
    # Point the store at the scratch DB so the live store stays clean.
    # config.db_path_for derives from character + root; the simpler
    # override is to symlink data/harness.sqlite → scratch DB for the
    # duration of the run. We instead invoke a tiny subprocess wrapper
    # that uses the real ingest path against an explicit DB path.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "atc_ingest.py"),
    ]
    # atc_ingest.py respects HARNESS_CHARACTER_NAME for chunk source +
    # DB destination. To redirect ONLY the DB to our scratch path
    # without polluting the live one we use a per-character symlink:
    # set HARNESS_CHARACTER_NAME=airton_c1_bench__<slug>, mkdir + link.
    # The bench dir lives under character/airton_c1_bench__<slug>/.
    bench_char_name = f"{character_name}_bench__{db_path.stem}"
    bench_char_root = REPO_ROOT / "character" / bench_char_name
    bench_char_root.mkdir(parents=True, exist_ok=True)
    # Mirror chunks dir into the bench character so atc_ingest finds it.
    chunks_link = bench_char_root / "corpus" / "chunks"
    chunks_link.parent.mkdir(parents=True, exist_ok=True)
    if chunks_link.exists() or chunks_link.is_symlink():
        chunks_link.unlink()
    chunks_link.symlink_to(chunks_dir.resolve())
    # Ditto for synonyms.yaml so ingest enrichment matches the real run.
    for synonyms_name in ("synonyms.yaml",):
        src = (REPO_ROOT / "character" / character_name / "corpus" / synonyms_name).resolve()
        if src.exists():
            link = bench_char_root / "corpus" / synonyms_name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src)
    # Mirror the scratch DB path to the bench character's data dir.
    bench_data_dir = bench_char_root / "data"
    bench_data_dir.mkdir(parents=True, exist_ok=True)
    bench_db = bench_data_dir / "harness.sqlite"
    if bench_db.exists() or bench_db.is_symlink():
        bench_db.unlink()
    bench_db.symlink_to(db_path.resolve())

    env["HARNESS_CHARACTER_NAME"] = bench_char_name

    t0 = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 — local benchmark
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"atc_ingest.py failed (exit {completed.returncode})")

    # Parse "+ jo_7110_65: 2,092 new, 0 already present" out of the
    # ingest output for a row count. Fall back to 0 on parse failure
    # — the eval downstream will catch an empty store.
    rows = 0
    for line in completed.stdout.splitlines():
        match = re.search(r"^\s*\+ \S+: ([\d,]+) new", line)
        if match:
            rows += int(match.group(1).replace(",", ""))
            break  # one source for now (jo_7110_65)

    # Embed dim — ask the embedder directly, post-load.
    from harness.retrieval.st_embedder import SentenceTransformersEmbedder

    embedder = SentenceTransformersEmbedder(model_name=embedder_repo)
    embedder.embed(["dimension probe"])
    embed_dim = embedder.dimension

    rss_after = _peak_rss_mb()
    return IngestStats(
        seconds=elapsed,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        rss_delta_mb=rss_after - rss_before,
        rows_ingested=rows,
        embed_dim=embed_dim,
    )


def _run_eval(
    *,
    character_name: str,
    db_path: Path,
    fixture: Path,
    embedder_repo: str,
) -> tuple[float, dict[str, object]]:
    """Invoke `harness eval atc-retrieval --json` against the scratch
    DB. Returns (eval_seconds, parsed_envelope). The CLI emits leading
    non-JSON noise (model load progress) before the envelope, so we
    locate the JSON start by the first `{\\n`."""
    bench_char_name = f"{character_name}_bench__{db_path.stem}"
    env = os.environ.copy()
    env["HARNESS_CHARACTER_NAME"] = bench_char_name
    env["HARNESS_EMBEDDER_REPO"] = embedder_repo
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    cmd = [
        "uv",
        "run",
        "harness",
        "eval",
        "atc-retrieval",
        "--fixture",
        str(fixture.resolve()),
        "--json",
    ]
    t0 = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 — local benchmark
        cmd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"eval atc-retrieval failed (exit {completed.returncode})")

    out = completed.stdout
    try:
        idx = out.index("{\n")
    except ValueError as exc:
        raise RuntimeError("eval atc-retrieval emitted no JSON envelope") from exc
    envelope = json.loads(out[idx:])
    return elapsed, envelope


def run_bench(
    *,
    repo: str,
    character_name: str,
    chunks_dir: Path,
    fixture: Path,
    reuse: bool,
) -> BenchResult:
    sanitized = _sanitize_repo(repo)
    character_path = REPO_ROOT / "character" / character_name
    db_path = _provision_scratch_db(character_path, sanitized, reuse=reuse)

    ingest = _ingest_with_embedder(
        character_name=character_name,
        chunks_dir=chunks_dir,
        db_path=db_path,
        embedder_repo=repo,
    )

    eval_seconds, env = _run_eval(
        character_name=character_name,
        db_path=db_path,
        fixture=fixture,
        embedder_repo=repo,
    )

    return BenchResult(
        repo=repo,
        sanitized=sanitized,
        db_path=str(db_path),
        fixture=str(fixture),
        ingest=ingest,
        eval_seconds=eval_seconds,
        recall_at_1=float(env["recall_at_1"]),
        recall_at_3=float(env["recall_at_3"]),
        recall_at_5=float(env["recall_at_5"]),
        recall_at_k=float(env["recall_at_k"]),
        median_rank=env.get("median_rank"),
        cases=list(env.get("cases", [])),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help="HF embedder repo to benchmark (e.g. BAAI/bge-large-en-v1.5).",
    )
    parser.add_argument(
        "--character",
        default="airton_c1",
        help="Character whose corpus + fixture drive the benchmark.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=None,
        help="Override chunks dir. Defaults to character/<char>/corpus/chunks/.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Override eval fixture. Defaults to character/<char>/atc_eval.yaml.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse the scratch DB from a previous run instead of rebuilding.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON envelope here. Defaults to "
        "character/<char>/embedder_bench/<sanitized-repo>.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip the human-readable summary; emit only the JSON envelope on stdout.",
    )
    args = parser.parse_args(argv)

    chunks_dir = args.chunks_dir or REPO_ROOT / "character" / args.character / "corpus" / "chunks"
    fixture = args.fixture or REPO_ROOT / "character" / args.character / "atc_eval.yaml"
    if not chunks_dir.exists():
        sys.stderr.write(f"chunks dir not found: {chunks_dir}\n")
        return 2
    if not fixture.exists():
        sys.stderr.write(f"fixture not found: {fixture}\n")
        return 2

    result = run_bench(
        repo=args.repo,
        character_name=args.character,
        chunks_dir=chunks_dir,
        fixture=fixture,
        reuse=args.reuse,
    )

    output = args.output or (
        REPO_ROOT / "character" / args.character / "embedder_bench" / f"{result.sanitized}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2))

    if args.quiet:
        sys.stdout.write(json.dumps(asdict(result), indent=2))
        return 0

    print(f"=== embedder bench: {result.repo} ===")
    print(f"  embed dim:      {result.ingest.embed_dim}")
    print(f"  rows ingested:  {result.ingest.rows_ingested:,}")
    print(f"  ingest seconds: {result.ingest.seconds:.1f}")
    print(f"  RSS Δ (MB):     {result.ingest.rss_delta_mb:+.0f}")
    print(f"  eval seconds:   {result.eval_seconds:.1f}")
    print(f"  recall@1:       {result.recall_at_1 * 100:.1f}%")
    print(f"  recall@3:       {result.recall_at_3 * 100:.1f}%")
    print(f"  recall@5:       {result.recall_at_5 * 100:.1f}%")
    print(f"  recall@k:       {result.recall_at_k * 100:.1f}%")
    print(f"  median rank:    {result.median_rank}")
    print(f"  → {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
