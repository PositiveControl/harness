# retrieval_eval/

Character-agnostic fixtures and baselines for the data-retrieval-primitives
spike (branch `me/data-retrieval-primitives`). Lives at the repo root —
not under `character/<name>/` — because the primitives these fixtures
benchmark are meant to be reusable across personas.

Filed against `harness-kj2a` (Phase 0). Phases 1–3 add new retrievers
that get measured against the baselines here.

## Layout

```
retrieval_eval/
├── README.md                       # this file
├── corpora/                        # one YAML per benchmarked corpus
│   ├── prose_journal.yaml          # 12 self-contained prose records (inline)
│   ├── structured_atc.yaml         # pointer to airton_c1 JO 7110.65 ingest
│   └── tabular_returns.yaml        # 200-row synthetic CSV
├── data/
│   └── returns.csv                 # tabular_returns backing data (deterministic)
└── baselines/
    └── phase0.json                 # baseline numbers, regression target
```

## Running the bench

```bash
uv run python scripts/bench_retrieval_shape.py                 # all corpora
uv run python scripts/bench_retrieval_shape.py --corpus prose_journal
uv run python scripts/bench_retrieval_shape.py --k 5
uv run python scripts/bench_retrieval_shape.py \
    --embedder nomic-ai/nomic-embed-text-v1.5
```

Writes `retrieval_eval/baselines/phase0.json` by default. Each new
retriever in later phases lands its own bench cell alongside the
current `hybrid_episodic` cell so deltas stay attributable.

## Corpus YAML schema

Each corpus YAML carries:

```yaml
version: 1
name: <unique slug, also bench dispatch key>
shape: prose | structured_doc | tabular | graph
description: |
  Prose explaining what this corpus measures and what we expect.
```

Then, depending on `shape`:

### `shape: prose` — inline records

```yaml
records:
  - id: <external_id>
    text: |
      Prose body, embedded as-is into a fresh in-memory episodic store.
cases:
  - id: <case_id>
    query: "natural-language question targeting one record"
    expected_record_ids: ["<external_id>"]   # any-match is a hit
```

### `shape: tabular` — CSV-backed records

```yaml
data_path: ../data/<file>.csv     # relative to the YAML

ingest:
  id_column: <csv column>                   # supplies the row id
  id_prefix: "<prefix>"                     # prepended → external_id
  record_title_template: "..."              # Python {field} format
  record_body_template: |
    ...                                     # multi-line OK

cases:
  - id: <case_id>
    query: "..."
    expected_record_ids: ["<id_prefix><value>"]
```

### `shape: structured_doc` — external store + fixture pointer

```yaml
external_eval:
  fixture: ../../path/to/atc_eval.yaml
  store: ../../path/to/harness.sqlite
  match_strategy: anchor_in_principle
```

The bench imports the existing `harness.evals.atc_retrieval` anchor
matcher (`§N-N-N` extracted from `EpisodicRecord.principle`) — caller
doesn't need to teach the generic eval module about anchors.

## Adding a new corpus

1. Create `corpora/<name>.yaml` with one of the shapes above.
2. If shape is `tabular`, drop the CSV under `data/`.
3. Add a sentinel test under `tests/` if your shape needs new wiring
   in `scripts/bench_retrieval_shape.py`.
4. Run the bench and commit the updated `baselines/phase0.json`.

## Adding a new retriever (Phase 1+)

1. Implement a `SearchFn` adapter in `scripts/bench_retrieval_shape.py`
   that returns `ShapeHit(record_id, score)` per the
   `harness.evals.retrieval_shape.ShapeSearchFn` protocol.
2. Wire it into the per-shape runner alongside `hybrid_episodic`. The
   envelope's `retriever` field is what distinguishes the two cells.
3. Re-run, commit `baselines/phase<N>.json`.

## What the baseline says (Phase 0)

| corpus | shape | r@1 | r@3 | r@5 | r@k | MRR | notes |
|---|---|---|---|---|---|---|---|
| prose_journal | prose | 100% | 100% | 100% | 100% | 1.000 | upper bound; flat hybrid is excellent on prose |
| structured_atc | structured_doc | 65.5% | 75.9% | 75.9% | 89.7% | 0.705 | STOCK hybrid only; deployed pipeline runs through a character-specific query expander that lifts this to 79%/100% |
| tabular_returns | tabular | 50.0% | 50.0% | 58.3% | 75.0% | 0.544 | predicted gap — aggregation-shaped queries hard-miss because flat vector retrieval can't sort |

The structured-doc number is intentionally lower than the existing
`character/airton_c1/atc_retrieval_baseline.json` (79%/89%/100%) — that
baseline includes airton_c1's `QueryExpander`, this one does not.
Comparing later phases against an un-augmented stock retriever keeps
them apples-to-apples; layering the expander on top is an additive,
separate measurement.

## Determinism

- `data/returns.csv` is generated with a fixed seed (`/tmp/gen_returns.py`,
  seed `20260514`).
- `prose_journal.yaml` and `tabular_returns.yaml` are static.
- `structured_atc.yaml` resolves against the live `airton_c1` SQLite,
  which evolves with ingests — baseline numbers there are tied to the
  store state at the time of the baseline commit. The bench snapshots
  the live DB to a tempfile at run start to avoid concurrent-writer
  drift mid-run; recommit `phase0.json` after any `atc_ingest` change.
- `EpisodicStore._search_dense` / `_search_text` rank by score only;
  Python's stable sort preserves whatever SQLite row order the
  candidate query returned, which itself isn't deterministic without
  an `ORDER BY` tiebreaker. The bench-side search adapters re-sort by
  `(-score, record_id)` to pin ties — this is hygiene at the
  measurement boundary, not a fix to the underlying retrieval path
  (that's a separate refactor).
