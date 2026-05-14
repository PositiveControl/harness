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
├── contracts/                      # worked role contracts (harness-s3f7)
│   └── returns_handler.yaml        # one slot per shaped store, end-to-end
├── data/
│   └── returns.csv                 # tabular_returns backing data (deterministic)
└── baselines/
    └── phase0.json                 # baseline numbers, regression target
```

## Running the bench

```bash
# First-run setup: build the tree-retriever DB (gitignored, 4 MB)
uv run python scripts/atc_ingest_tree.py --rebuild

# Then run the bench
uv run python scripts/bench_retrieval_shape.py                 # all corpora
uv run python scripts/bench_retrieval_shape.py --corpus prose_journal
uv run python scripts/bench_retrieval_shape.py --k 5
uv run python scripts/bench_retrieval_shape.py \
    --embedder nomic-ai/nomic-embed-text-v1.5
```

The tree-retriever cells skip with a one-line message if
`retrieval_eval/data/tree_atc.sqlite` is missing; everything else still
runs.

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

### Optional `glossary:` wiring (any shape)

```yaml
glossary: ../../path/to/synonyms.yaml
glossary_query_only: ../../path/to/query_synonyms.yaml   # optional
```

When set, the bench loads the glossary through
`harness.retrieval.query_expander.load_query_expander` and emits a
second cell (`hybrid_episodic+expander`) alongside the stock one. The
glossary YAML itself uses either the new `topics:` schema (default
empty prefix) or the legacy `sections:` schema (default `§` prefix);
both accept an explicit `prefix:` field that wins.

### Optional `tree_store:` wiring (structured_doc shape only)

```yaml
tree_store: ../data/tree_atc.sqlite
```

When set, the bench opens a `DocumentTreeStore` SQLite (produced by
`scripts/atc_ingest_tree.py`) and emits a `tree_section_hybrid` cell.
If a `glossary:` is also wired, it also emits
`tree_section_hybrid+expander` so all four combinations of
(flat | tree) × (no expander | + expander) are visible in one
envelope. Tree-store paths resolve relative to the corpus YAML.

### Optional `table:` wiring + per-case `sql:` (tabular shape only)

```yaml
table:
  table_name: returns
  description: |
    Customer return requests, one row per refund. ...
  columns:
    - [order_id, INTEGER, "Unique order identifier"]
    - [amount_usd, REAL, "Refund amount in USD"]
    # ...
cases:
  - id: q_hard_highest_value
    query: "show me the highest-value return"
    expected_record_ids: ["order:10377"]
    sql: |
      SELECT order_id FROM returns
      ORDER BY CAST(amount_usd AS REAL) DESC LIMIT 1
```

When set, the bench registers the CSV as one logical table in an
in-memory `TabularStore`, then runs each case's `sql:` through
`store.query_sql`. The returned `id_column` is prefixed with
`ingest.id_prefix` to build `record_id` strings that align with the
fixture's `expected_record_ids`. Emits a `table_sql_oracle` cell —
"oracle" because the SQL is fixture-supplied rather than NL-generated,
isolating the storage layer from NL→SQL quality. The production tool
`harness.tools.query_table.QueryTableTool` handles NL→SQL at chat time
via the agent itself, not a separate model layer.

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

## What the baseline says

| corpus | retriever | r@1 | r@3 | r@5 | r@k | MRR | notes |
|---|---|---|---|---|---|---|---|
| prose_journal | hybrid_episodic | 100% | 100% | 100% | 100% | 1.000 | upper bound; flat hybrid is excellent on prose |
| structured_atc | hybrid_episodic | 65.5% | 75.9% | 75.9% | 89.7% | 0.705 | Phase 0 baseline — flat chunk hybrid, no glossary |
| structured_atc | hybrid_episodic+expander | 79.3% | 89.7% | 89.7% | 100% | 0.839 | + lifted QueryExpander (harness-m78r) |
| structured_atc | tree_section_hybrid | 65.5% | 82.8% | **89.7%** | 89.7% | 0.753 | section-grained tree (harness-h5ly) — **+13.8pp r@5 vs Phase 0** |
| structured_atc | tree_section_hybrid+expander | 75.9% | 93.1% | **96.5%** | 96.5% | 0.853 | tree + expander — best aggregate result |
| tabular_returns | hybrid_episodic | 58.3% | 58.3% | 66.7% | 75.0% | 0.618 | Phase 0 baseline — flat vector hard-misses on aggregation queries |
| tabular_returns | **table_sql_oracle** | **100%** | **100%** | **100%** | **100%** | **1.000** | table-shape via `TabularStore` (harness-edt6) — **+33.3pp r@5 vs Phase 0** |

**Multi-cell layout:** corpora that declare a `glossary:` field emit a
`+expander` cell; corpora that declare a `tree_store:` field emit
`tree_section_hybrid` (and a `+expander` companion when both are
wired). Phase N+ retrievers report against the relevant prior cells so
each primitive's contribution stays attributable:
- vs Phase 0 stock hybrid → did the primitive move the number on its own?
- vs Phase 0 + expander → does it still earn its keep with the
  cheap, content-driven expander in the mix?
- vs the previous phase's best → is this an additive win or a substitute?

**Phase 1 verdict:** tree retriever lands. The bd decision rule
(harness-h5ly) was "beat flat hybrid on r@5 or revert." Tree
section-grained ingest delivers r@5 = 89.7% (+13.8pp), is ~7× faster
than flat hybrid (374ms vs 2470ms — 675 nodes vs 1854), and composes
cleanly with the expander.

**Phase 2 verdict:** table-shape lands. The bench cell measures the
*architectural ceiling* with fixture-supplied SQL (NL→SQL generation
is the agent's job at chat time; the bench keeps that variable out
of the storage-layer measurement). Table-shape via `TabularStore`
delivers r@1 / r@5 / r@k all at 100% vs Phase 0's 58.3% / 66.7% / 75%
(+33-41pp), in ~1ms. Validates the thesis: tabular data should be
retrieved AS tables. Production tool:
`harness.tools.query_table.QueryTableTool` exposes the registered
schemas in its description so the model can write correct SQL inline,
no separate `list_tables` round-trip.

**Phase 3 verdict:** Context Package + Contract Bundle lands. Per the
data-retrieval-primitives notes: Intent / Access (policy) / Proof
(provenance) / Budget (cost bound). The package
(`harness.retrieval.context_package.RetrievalContextPackage`) wraps
shape-aware retrieval in a single envelope; `ContractBundle`
(`harness.retrieval.contract`) declares per-role what slots must be
filled, from which shaped store, with what query template.
`assemble_package` walks the contract, calls each store, packages
the result. Worked contract: `retrieval_eval/contracts/returns_handler.yaml`
pulls customer history (episodic), refund policy (episodic), and
matching orders (tabular) in one call. Required-slot misses surface
in `missing_required_slots` so the agent gets told *what's missing*
instead of working with silently-incomplete context.

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
