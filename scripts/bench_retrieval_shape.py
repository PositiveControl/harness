"""bench_retrieval_shape (harness-kj2a / Phase 0 of the data-retrieval-
primitives spike).

Runs the current hybrid retriever (EpisodicStore.search, mode="hybrid")
against three shape-distinct corpora and emits an apples-to-apples
JSON envelope so later phases (tree / table / contract retrievers)
have a fixed baseline to regress against.

Corpora live under `retrieval_eval/corpora/`:

  - prose_journal.yaml      — 12 self-contained prose records + 12 queries.
                              Each record carries text inline; embed-on-load.
  - structured_atc.yaml     — pointer to airton_c1's JO 7110.65 ingest +
                              atc_eval.yaml fixture. Reuses the existing
                              anchor-in-principle matcher.
  - tabular_returns.yaml    — 200-row synthetic returns CSV; rows ingested
                              into an in-memory episodic store with a
                              flattened body template.

Output: `retrieval_eval/baselines/phase0.json` — a single envelope
holding one section per corpus, each with recall@1/3/5/k, MRR, median
rank, wall_ms, and per-case hit_record_ids for debugging.

Phase 0 deliberately does NOT introduce new retrievers — the point is
the comparator. Phase 1+ register additional `RetrieverFactory`s
alongside `_hybrid_retriever`.

Usage:
    uv run python scripts/bench_retrieval_shape.py
    uv run python scripts/bench_retrieval_shape.py --corpus prose_journal
    uv run python scripts/bench_retrieval_shape.py \\
        --output retrieval_eval/baselines/probe.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from harness.evals.atc import load_fixture as _load_atc_fixture_rows  # noqa: E402
from harness.evals.atc_retrieval import (  # noqa: E402
    _expected_anchor_set,
    _extract_anchors,
)
from harness.evals.retrieval_shape import (  # noqa: E402
    ShapeCase,
    ShapeHit,
    ShapeResult,
    run_retrieval_shape,
)
from harness.retrieval.query_expander import (  # noqa: E402
    NullQueryExpander,
    QueryExpander,
    load_query_expander,
)
from harness.retrieval.st_embedder import SentenceTransformersEmbedder  # noqa: E402
from harness.store.episodic import EpisodicStore  # noqa: E402

_DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
_DEFAULT_K = 10


# ----------------------------------------------------------------------
# Result envelope


@dataclass
class CorpusBench:
    """One (corpus, retriever) cell in the output envelope."""

    corpus: str
    shape: str
    retriever: str
    k: int
    case_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_k: float
    mrr: float
    median_rank: float | None
    wall_ms: float
    cases: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BaselineEnvelope:
    embedder: str
    embedder_dim: int
    k: int
    cells: list[CorpusBench] = field(default_factory=list)


def _result_to_cell(
    *,
    result: ShapeResult,
    corpus_name: str,
    shape: str,
    retriever: str,
    wall_ms: float,
) -> CorpusBench:
    return CorpusBench(
        corpus=corpus_name,
        shape=shape,
        retriever=retriever,
        k=result.k,
        case_count=len(result.cases),
        recall_at_1=round(result.recall_at_1, 4),
        recall_at_3=round(result.recall_at_3, 4),
        recall_at_5=round(result.recall_at_5, 4),
        recall_at_k=round(result.recall_at_k, 4),
        mrr=round(result.mrr, 4),
        median_rank=result.median_rank,
        wall_ms=round(wall_ms, 1),
        cases=[
            {
                "id": c.id,
                "query": c.query,
                "expected_record_ids": list(c.expected_record_ids),
                "hit_record_ids": list(c.hit_record_ids),
                "rank_of_first_expected": c.rank_of_first_expected,
                "score_of_first_expected": (
                    round(c.score_of_first_expected, 4)
                    if c.score_of_first_expected is not None
                    else None
                ),
            }
            for c in result.cases
        ],
    )


# ----------------------------------------------------------------------
# Corpus loaders


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected mapping at top level")
    return raw


def _ingest_inline_records(
    store: EpisodicStore,
    records: Sequence[dict[str, Any]],
    *,
    source_label: str,
) -> int:
    """Ingest YAML-inline records (each with `id` + `text`) into a fresh
    episodic store. Returns the row count. Tier='seed' so the rows
    behave like canonical, non-superseded memory under hybrid search."""
    count = 0
    for rec in records:
        if "id" not in rec or "text" not in rec:
            raise ValueError(f"{source_label}: record missing 'id' or 'text': {rec!r}")
        store.ingest(
            external_id=str(rec["id"]),
            title=str(rec["id"]).replace("-", " ").replace("_", " "),
            body=str(rec["text"]).strip(),
            tier="seed",
            source="bench_retrieval_shape",
        )
        count += 1
    return count


def _ingest_csv_rows(
    store: EpisodicStore,
    csv_path: Path,
    ingest_spec: dict[str, Any],
) -> int:
    """Ingest CSV rows under the ingest spec from a tabular corpus YAML.

    Spec fields:
      id_column            — column whose value supplies the row id.
      id_prefix            — prepended to id_column to form external_id.
      record_title_template — Python-style format string over the row.
      record_body_template  — same, multi-line OK.
    """
    id_col = ingest_spec["id_column"]
    id_prefix = ingest_spec.get("id_prefix", "")
    title_tmpl = ingest_spec["record_title_template"]
    body_tmpl = ingest_spec["record_body_template"]

    count = 0
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            ext_id = f"{id_prefix}{row[id_col]}"
            store.ingest(
                external_id=ext_id,
                title=title_tmpl.format(**row),
                body=body_tmpl.format(**row),
                tier="seed",
                source="bench_retrieval_shape",
            )
            count += 1
    return count


# ----------------------------------------------------------------------
# Per-shape eval runners


def _build_episodic_search_fn(
    store: EpisodicStore,
    *,
    expander: QueryExpander | None = None,
) -> Any:
    """Wrap EpisodicStore.search into a ShapeSearchFn that returns
    ShapeHit objects keyed by external_id. The eval doesn't see the
    underlying record.

    `expander`, when provided, rewrites the user query before search —
    the harness-m78r path that lifts airton_c1's QueryExpander into
    the bench. `None` = stock hybrid baseline (Phase 0 default).

    Hits are re-sorted by (-score, record_id) to pin tie-breaks at the
    bench boundary. The underlying hybrid path ranks by score only and
    Python's stable sort then preserves whatever SQLite row-order the
    candidate query returned — and SQLite without ORDER BY isn't
    deterministic. Re-sorting here doesn't change the SET of top-K
    results (the score is identical) — only the within-tie order, so
    r@1 / MRR baselines stop wobbling. The deeper fix belongs in
    episodic._search_dense / _search_text but is out of scope for the
    Phase 0 baseline.
    """

    def _search(query: str, k: int) -> list[ShapeHit]:
        if expander is not None:
            query = expander.expand(query)
        hits = store.search(query, k=k, mode="hybrid", user_id=None)
        out: list[ShapeHit] = []
        for record, score in hits:
            if record.external_id is None:
                continue
            out.append(ShapeHit(record_id=record.external_id, score=float(score)))
        out.sort(key=lambda h: (-h.score, h.record_id))
        return out

    return _search


def _load_corpus_glossary(corpus_yaml: Path, spec: dict[str, Any]) -> QueryExpander | None:
    """Read optional `glossary:` / `glossary_query_only:` fields from a
    corpus spec and return a `QueryExpander`, or None when neither is
    set. Paths in the YAML are relative to the corpus YAML file."""
    primary_raw = spec.get("glossary")
    query_only_raw = spec.get("glossary_query_only")
    if not primary_raw and not query_only_raw:
        return None
    primary = (corpus_yaml.parent / str(primary_raw)).resolve() if primary_raw else None
    query_only = (corpus_yaml.parent / str(query_only_raw)).resolve() if query_only_raw else None
    expander = load_query_expander(primary, query_only_path=query_only)
    if isinstance(expander, NullQueryExpander):
        return None
    return expander


def _run_with_retrievers(
    *,
    cases: list[ShapeCase],
    store: EpisodicStore,
    expander: QueryExpander | None,
    spec: dict[str, Any],
    k: int,
) -> list[CorpusBench]:
    """Run the case list under every registered retriever for this
    corpus (today: `hybrid_episodic` always; `hybrid_plus_expander` when
    a glossary is wired). Returns one CorpusBench per retriever."""
    cells: list[CorpusBench] = []

    # Cell 1: stock hybrid.
    stock_fn = _build_episodic_search_fn(store, expander=None)
    t0 = time.perf_counter()
    stock_result = run_retrieval_shape(cases, stock_fn, k=k)
    stock_ms = (time.perf_counter() - t0) * 1000
    cells.append(
        _result_to_cell(
            result=stock_result,
            corpus_name=str(spec["name"]),
            shape=str(spec["shape"]),
            retriever="hybrid_episodic",
            wall_ms=stock_ms,
        )
    )

    # Cell 2: hybrid + expander. Only emitted when the corpus pointed at
    # a non-empty glossary — harness-m78r's "lifted primitive" cell.
    if expander is not None:
        expander_fn = _build_episodic_search_fn(store, expander=expander)
        t0 = time.perf_counter()
        expander_result = run_retrieval_shape(cases, expander_fn, k=k)
        expander_ms = (time.perf_counter() - t0) * 1000
        cells.append(
            _result_to_cell(
                result=expander_result,
                corpus_name=str(spec["name"]),
                shape=str(spec["shape"]),
                retriever="hybrid_episodic+expander",
                wall_ms=expander_ms,
            )
        )

    return cells


def _run_prose(corpus_yaml: Path, *, k: int, embedder_repo: str) -> list[CorpusBench]:
    spec = _load_yaml(corpus_yaml)
    records = spec.get("records") or []
    raw_cases = spec.get("cases") or []

    embedder = SentenceTransformersEmbedder(model_name=embedder_repo)
    store = EpisodicStore(db_path=Path(":memory:"), embedder=embedder)
    _ingest_inline_records(store, records, source_label=str(corpus_yaml))

    cases = [
        ShapeCase(
            id=str(c["id"]),
            query=str(c["query"]),
            expected_record_ids=tuple(str(x) for x in c["expected_record_ids"]),
        )
        for c in raw_cases
    ]

    expander = _load_corpus_glossary(corpus_yaml, spec)
    return _run_with_retrievers(cases=cases, store=store, expander=expander, spec=spec, k=k)


def _run_tabular(corpus_yaml: Path, *, k: int, embedder_repo: str) -> list[CorpusBench]:
    spec = _load_yaml(corpus_yaml)
    data_path = (corpus_yaml.parent / str(spec["data_path"])).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"tabular data not found: {data_path}")
    raw_cases = spec.get("cases") or []

    embedder = SentenceTransformersEmbedder(model_name=embedder_repo)
    store = EpisodicStore(db_path=Path(":memory:"), embedder=embedder)
    _ingest_csv_rows(store, data_path, spec["ingest"])

    cases = [
        ShapeCase(
            id=str(c["id"]),
            query=str(c["query"]),
            expected_record_ids=tuple(str(x) for x in c["expected_record_ids"]),
        )
        for c in raw_cases
    ]

    expander = _load_corpus_glossary(corpus_yaml, spec)
    return _run_with_retrievers(cases=cases, store=store, expander=expander, spec=spec, k=k)


def _build_structured_atc_search_fn(
    store: EpisodicStore,
    *,
    expander: QueryExpander | None = None,
) -> Any:
    """Per-hit anchor-in-principle search adapter for structured-doc
    corpora. Re-keys store hits from external_id → parsed anchor so
    the generic scorer can match without knowing ATC conventions.

    `expander` mirrors `_build_episodic_search_fn` — applied to the
    query before `store.search`."""

    def _search(query: str, depth: int) -> list[ShapeHit]:
        if expander is not None:
            query = expander.expand(query)
        # Mirror the airton_c1 retrieval path: hybrid, no user scoping
        # (the corpus is shared / NULL user).
        hits = store.search(query, k=depth, mode="hybrid", user_id=None)
        out: list[ShapeHit] = []
        for record, score in hits:
            # If a record's principle holds multiple anchors, emit one
            # synthetic hit per anchor at the same rank score — mirrors
            # how the existing atc_retrieval scorer treats multi-anchor
            # principles. Caller's scorer takes the first matching
            # anchor per case.
            for anchor in _extract_anchors(record.principle or ""):
                out.append(ShapeHit(record_id=anchor, score=float(score)))
        # Same tie-pinning logic as _build_episodic_search_fn — see
        # docstring there. Without this the structured baseline r@1 /
        # MRR wobble across invocations because tied dense-cosine
        # scores are extremely common in this corpus.
        out.sort(key=lambda h: (-h.score, h.record_id))
        return out

    return _search


def _run_structured_atc(corpus_yaml: Path, *, k: int, embedder_repo: str) -> list[CorpusBench]:
    """Re-uses the populated airton_c1 episodic store. We don't ingest
    here — the store is the live one on disk, opened in the same
    embedder so dense vectors line up."""
    spec = _load_yaml(corpus_yaml)
    ext = spec["external_eval"]
    fixture_path = (corpus_yaml.parent / str(ext["fixture"])).resolve()
    store_path = (corpus_yaml.parent / str(ext["store"])).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"atc fixture not found: {fixture_path}")
    if not store_path.exists():
        raise FileNotFoundError(f"airton_c1 store not found: {store_path}")

    # Snapshot the live store to a tempfile before search. Without
    # this, a concurrent writer on the airton_c1 SQLite (dolt server,
    # background harness chat, etc.) shifts BM25 statistics and the
    # rank-0 ↔ rank-1 ordering for tied-score hits drifts between
    # bench invocations. r@5 / r@k stays stable across the variance —
    # only r@1 / MRR are affected — but a baseline that wobbles isn't
    # a baseline. Snapshot pins the read.
    snapshot_dir = Path(tempfile.mkdtemp(prefix="bench_retrieval_shape_"))
    snapshot_path = snapshot_dir / "structured_atc.sqlite"
    shutil.copy2(store_path, snapshot_path)

    embedder = SentenceTransformersEmbedder(model_name=embedder_repo)
    store = EpisodicStore(db_path=snapshot_path, embedder=embedder)
    atc_rows = _load_atc_fixture_rows(fixture_path)

    # Each ATC fixture case becomes a ShapeCase whose expected_record_ids
    # are the flat set of acceptable anchors (e.g. {"2-4-3"}).
    cases: list[ShapeCase] = []
    for row in atc_rows:
        anchors = _expected_anchor_set(row)
        if not anchors:
            continue  # malformed fixture row; skip
        cases.append(
            ShapeCase(
                id=row.id,
                query=row.question,
                expected_record_ids=tuple(sorted(anchors)),
            )
        )

    expander = _load_corpus_glossary(corpus_yaml, spec)
    cells: list[CorpusBench] = []

    stock_fn = _build_structured_atc_search_fn(store, expander=None)
    t0 = time.perf_counter()
    stock_result = run_retrieval_shape(cases, stock_fn, k=k)
    stock_ms = (time.perf_counter() - t0) * 1000
    cells.append(
        _result_to_cell(
            result=stock_result,
            corpus_name=str(spec["name"]),
            shape=str(spec["shape"]),
            retriever="hybrid_episodic",
            wall_ms=stock_ms,
        )
    )

    if expander is not None:
        expander_fn = _build_structured_atc_search_fn(store, expander=expander)
        t0 = time.perf_counter()
        expander_result = run_retrieval_shape(cases, expander_fn, k=k)
        expander_ms = (time.perf_counter() - t0) * 1000
        cells.append(
            _result_to_cell(
                result=expander_result,
                corpus_name=str(spec["name"]),
                shape=str(spec["shape"]),
                retriever="hybrid_episodic+expander",
                wall_ms=expander_ms,
            )
        )

    return cells


# ----------------------------------------------------------------------
# Dispatch + main


_RUNNERS = {
    "prose": _run_prose,
    "tabular": _run_tabular,
    "structured_doc": _run_structured_atc,
}


def _discover_corpora(corpora_dir: Path) -> list[Path]:
    return sorted(p for p in corpora_dir.glob("*.yaml") if p.is_file())


def run_all(
    *,
    corpora_dir: Path,
    output: Path,
    k: int,
    embedder_repo: str,
    corpus_filter: str | None,
) -> BaselineEnvelope:
    embedder_probe = SentenceTransformersEmbedder(model_name=embedder_repo)
    embedder_probe.embed(["dim probe"])
    envelope = BaselineEnvelope(
        embedder=embedder_repo,
        embedder_dim=embedder_probe.dimension,
        k=k,
    )

    for corpus_yaml in _discover_corpora(corpora_dir):
        spec = _load_yaml(corpus_yaml)
        name = str(spec.get("name") or corpus_yaml.stem)
        if corpus_filter and name != corpus_filter:
            continue
        shape = str(spec.get("shape") or "")
        runner = _RUNNERS.get(shape)
        if runner is None:
            sys.stderr.write(f"[skip] {name}: unknown shape {shape!r}\n")
            continue
        sys.stderr.write(f"[run]  {name} ({shape}) ...\n")
        envelope.cells.extend(runner(corpus_yaml, k=k, embedder_repo=embedder_repo))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(envelope), indent=2))
    return envelope


def _print_summary(envelope: BaselineEnvelope) -> None:
    print()
    print(f"=== retrieval-shape baseline (embedder={envelope.embedder}, k={envelope.k}) ===")
    print(
        f"{'corpus':<20} {'retriever':<28} {'cases':>5} "
        f"{'r@1':>6} {'r@3':>6} {'r@5':>6} {'r@k':>6} {'mrr':>6} {'ms':>7}"
    )
    print("-" * 96)
    for cell in envelope.cells:
        print(
            f"{cell.corpus:<20} {cell.retriever:<28} {cell.case_count:>5} "
            f"{cell.recall_at_1 * 100:>5.1f}% {cell.recall_at_3 * 100:>5.1f}% "
            f"{cell.recall_at_5 * 100:>5.1f}% {cell.recall_at_k * 100:>5.1f}% "
            f"{cell.mrr:>6.3f} {cell.wall_ms:>7.0f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora-dir",
        type=Path,
        default=REPO_ROOT / "retrieval_eval" / "corpora",
        help="Directory of corpus YAMLs. Default: retrieval_eval/corpora/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "retrieval_eval" / "baselines" / "phase0.json",
        help="Output JSON path. Default: retrieval_eval/baselines/phase0.json",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=_DEFAULT_K,
        help=f"Top-K to retrieve per case. Default: {_DEFAULT_K}",
    )
    parser.add_argument(
        "--embedder",
        default=_DEFAULT_EMBEDDER,
        help=f"HF embedder repo. Default: {_DEFAULT_EMBEDDER}",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Run only this corpus (by `name:` field). Default: all.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip the human-readable summary table; JSON envelope only.",
    )
    args = parser.parse_args(argv)

    if not args.corpora_dir.exists():
        sys.stderr.write(f"corpora dir not found: {args.corpora_dir}\n")
        return 2

    envelope = run_all(
        corpora_dir=args.corpora_dir,
        output=args.output,
        k=args.k,
        embedder_repo=args.embedder,
        corpus_filter=args.corpus,
    )

    if not args.quiet:
        _print_summary(envelope)
        print()
        try:
            shown = args.output.relative_to(REPO_ROOT)
        except ValueError:
            shown = args.output
        print(f"-> wrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
