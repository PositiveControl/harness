# airton_c1 — Embedder Candidate Roster (Fix D, harness-pw9z)

Tracking the embedder swap experiment. Each candidate is ranked by
**expected register-gap closure** for lay→doc retrieval on JO 7110.65,
within the **32 GB unified-memory ceiling** of the M4 Pro box.

The current production baseline is **`BAAI/bge-small-en-v1.5`** —
33 M params, 384-dim, ~130 MB resident. Recall@1 73.7%, recall@10
94.7% on the post-catchup atc_eval baseline (19 cases). Lay-language
cosine bottoms out at ~0.016 on `§3-10-3` paraphrases (the
harness-rhto epic's seed observation).

Bench harness: `scripts/bench_embedder.py --repo <hf-repo>`. Each
candidate produces a JSON envelope at
`character/airton_c1/embedder_bench/<sanitized-repo>.json` with
recall@1/@3/@5/@K, ingest seconds, RSS delta, and per-case ranks.
The retrieval comparator (harness-sb6r) does the cross-candidate
diff once two or more envelopes exist.

## Off-the-shelf candidates

Ordered by hypothesis strength: how likely the candidate is to beat
the BGE-small baseline by ≥+15 pp recall@5 (the bead's gate).

### Tier 1 — strong priors

| repo | params | dim | RAM (rough) | rationale |
| ---- | ----- | --- | ----------- | --------- |
| `BAAI/bge-large-en-v1.5` | 335 M | 1024 | ~1.3 GB | Same family as the baseline, ~10× the parameters. MTEB-en average +5 pp over bge-small. Same dual-tower BERT shape — drop-in for the swap path. |
| `mixedbread-ai/mxbai-embed-large-v1` | 335 M | 1024 | ~1.3 GB | Was the harness's previous default before BGE-small. Strong on technical-document retrieval; M4 Pro can run two of these without thrashing. |
| `BAAI/bge-m3` | 568 M | 1024 | ~2.2 GB | Multilingual + dense + sparse hybrid. Sparse signal could lift §-anchor matching specifically (BM25-adjacent). Largest of the tier 1 set; worth the bench cost. |
| `nomic-ai/nomic-embed-text-v1.5` | 137 M | 768 | ~550 MB | Matryoshka representation — can truncate to 256/384/512 dim post-hoc and re-bench cheaply. Apache-2.0 license. |

### Tier 2 — speculative

Worth benching if Tier 1 doesn't close the gap.

| repo | params | dim | rationale |
| ---- | ----- | --- | --------- |
| `Snowflake/snowflake-arctic-embed-l` | 335 M | 1024 | Trained heavily on retrieval-style query-document pairs. Some technical-doc bias. |
| `intfloat/e5-large-v2` | 335 M | 1024 | E5 family is the long-time MTEB top entry for English. Older but well-understood. |
| `Alibaba-NLP/gte-large-en-v1.5` | 434 M | 1024 | Long-context (8k tokens) — useful if we ever drop the chunker's ~1k-char body cap. |

### Tier 3 — fine-tune candidates (Phase B)

If no off-the-shelf model lands the +15 pp gate, fine-tune the
strongest tier-1 candidate on JO 7110.65 (lay query, doc paragraph)
pairs. Three viable paths:

1. **`BAAI/bge-large-en-v1.5` + LoRA** — sentence-transformers
   supports PEFT integration; ~5–25 M trainable params on the
   query encoder side. Smallest disruption.
2. **`BAAI/bge-small-en-v1.5` + full fine-tune** — only 33 M
   total params; full FT fits the M4 Pro and yields a deployable
   replacement embedder under 200 MB.
3. **`nomic-embed-text-v1.5` Matryoshka FT** — train at full
   768-dim, ship at whatever dim the recall lift caps out at.

Training data shape (Phase B prereq, separate bead):

- ~3,000 chunks × ~5 lay paraphrases per canonical question
  → ~15,000 (query, positive chunk) pairs.
- Held-out split: every chunk that appears as an `expected_anchor`
  in `atc_eval.yaml` stays out of training. Eval contamination
  is the single biggest risk for fine-tune work.
- Synthetic-pair generator uses the existing chat adapter (Qwen
  2.5 7B) to expand each chunk's title + first paragraph into
  5 lay-form questions; manual review / filtering on a sample.

## Acceptance gates

For an off-the-shelf swap to ship as the new default:

1. **Recall lift** — ≥ +15 pp on at least one of recall@1 / @3 / @5
   relative to the static-only BGE-small baseline. No regression
   on recall@10.
2. **Per-case stability** — no fixture case drops out of top-K.
   Per-case rank slips of 1–2 ranks within top-K are acceptable
   if aggregate recall holds.
3. **Latency** — ingest seconds ≤ 3× the BGE-small baseline.
   The runtime cost is one embed-per-query, which scales linearly
   with the embedder size; capping ingest cost is the surrogate.
4. **RAM** — peak RSS during ingest ≤ +1.5 GB over the BGE-small
   baseline. Larger candidates are still benchable; this is the
   ship gate, not the bench gate.

For a fine-tune to ship: same 1–4 plus a held-out generalization
test (5+ pp recall lift on a query set the fine-tune did NOT see).

## Run order

1. `BAAI/bge-large-en-v1.5` — same family, biggest expected lift
   per dollar of swap cost.
2. `mixedbread-ai/mxbai-embed-large-v1` — known-good fallback.
3. `BAAI/bge-m3` — sparse-signal hypothesis test.
4. `nomic-ai/nomic-embed-text-v1.5` — license-friendly + dim-flex.
5. Tier 2 candidates only if 1–4 all miss the gate.
6. Fine-tune lane only if the off-the-shelf sweep caps below the
   gate.

## Sweep results (2026-04-25, tier 1)

All four tier-1 candidates benched against the post-catchup
BGE-small baseline (recall@1 73.7 / @3 89.5 / @5 89.5 / @10 100.0
on 19 cases).

| candidate                            | @1   | @3   | @5   | @10  | ingest_s | rss_Δ_MB | dim |
| ------------------------------------ | ---- | ---- | ---- | ---- | -------- | -------- | --- |
| `BAAI/bge-small-en-v1.5` (baseline)  | 73.7 | 89.5 | 89.5 | 100.0| —        | —        | 384 |
| `BAAI/bge-large-en-v1.5`             | **78.9** | 89.5 | **94.7** | 94.7 | 182.8 | +516 | 1024 |
| `mixedbread-ai/mxbai-embed-large-v1` | 73.7 | 89.5 | 94.7 | 94.7 | 113.9   | +505     | 1024 |
| `BAAI/bge-m3`                        | **68.4** | 89.5 | 94.7 | 94.7 | 280.8 | +973 | 1024 |
| `nomic-ai/nomic-embed-text-v1.5`     | 73.7 | 89.5 | 94.7 | 94.7 | 139.7   | +510     |  768 |

### Per-case findings

Two **universal wins** across every candidate vs the baseline:

- `controller_aircraft_to_aircraft_alerts`: rank 6 → 3-4 (lifts
  out of weak-match territory).
- `controller_rbn_mh_usable_distance`: rank 1 → 0 (canonical
  rank-zero hit instead of rank-one neighbour).

One **universal regression** across every candidate:

- `controller_wake_turbulence_concern_lay`: rank 8 → hard miss
  (out of top-10) for ALL four candidates. Same case `harness-zxw6`
  was already tracking as a known-flaky lay-paraphrase outlier.
  Not an embedder problem; investigated separately.

Per-candidate noise:

- `bge-large`: zero additional slips beyond the universal regression.
  Cleanest of the four.
- `mxbai-large`: extra slip on `ifr_clearance_items_order` 0 → 1.
- `bge-m3`: extra slips on `ifr_clearance_items_order` 0 → 1 AND
  `same_runway_arrival_lay_time` 0 → 2; recall@1 itself dropped
  by 5.3 pp.
- `nomic`: extra slip on `ifr_clearance_items_order` 0 → 1.

### Bead gate verdict

Initial reading of the sweep numbers suggested `bge-large-en-v1.5`
as the winner: only candidate that lifted recall@1 (+5.2 pp), only
one with zero additional slips beyond a universal wake_turb
regression. Investigation into that universal regression
(`harness-zxw6`, lay-paraphrase outlier) reframed the conclusion.

**Investigation finding (2026-04-25):** the wake_turb case
(`controller_wake_turbulence_concern_lay`) is **fixture
over-specification**, not an embedder problem.

The lay query — "When does a controller need to worry about a small
plane flying behind a big jet?" — has TWO legitimate JO 7110.65
answers, both surfaced by retrieval:

- §2-1-19 (Wake Turbulence — general application rule).
- §2-1-20 (Wake Turbulence Cautionary Advisories — when to issue
  advisories to aircraft behind larger aircraft).

The fixture had `expected_citations: ["2-1-19"]` only. Every
embedder candidate converges on §2-1-20 because it's semantically
closer to the "controller worries" lay phrasing. Broadening the
fixture to accept either anchor is the structural fix:

```yaml
expected_citations:
  - ["2-1-19", "2-1-20"]
```

Promoting §2-1-19's lay synonyms from `query_synonyms.yaml` to
`synonyms.yaml` (the alternative fix considered) was rejected —
`query_synonyms.yaml`'s own header documents that this regresses
the mirror jargon case `wake_turbulence_application` from rank 0
to rank 9. Fixture broadening is the no-trade-off fix.

### Sweep verdict (post-fixture-broaden)

Re-running bge-large against the broadened-fixture baseline shifted
the picture: bge-small lifts @3 (89.5 → 94.7) and @10 (94.7 → 100)
just from finding §2-1-20 at rank 2 on wake_turb. bge-large does
NOT find either §2-1-19 or §2-1-20 in top-10 for that query —
its top hits are §5-5-4 / §6-1-5 / §3-7-3, surface-token matches
on "wake turbulence" / "behind / separation" without the §2-1-*
abstraction-level connection bge-small still makes.

| candidate                            | @1   | @3   | @5   | @10  | net Δ |
| ------------------------------------ | ---- | ---- | ---- | ---- | ----- |
| `BAAI/bge-small-en-v1.5` (baseline)  | 73.7 | 94.7 | 94.7 | 100.0 | —     |
| `BAAI/bge-large-en-v1.5`             | 78.9 | 89.5 | 94.7 | 94.7  | -1    |

bge-large is **not** a ship-as-default. It picks up +5.2 pp @1
(rbn_mh 1 → 0, aircraft_to_aircraft 6 → 3) but loses @3 + @10
(wake_turb 2 → None, omit_holding 1 → 2).

### Lane outcome

Off-the-shelf swap lane closes without a default change. The actual
recall lift came from the fixture-broadening probe, not from any
candidate swap. Phase B fine-tune (`harness-6l9o`) stays open as a
future option if fixture growth (`harness-44l4` etc.) surfaces new
register-gap regressions the current corpus + synonyms don't cover.

For the record on the ranking: across every tier-1 candidate, the
two universal wins (aircraft_to_aircraft, rbn_mh) replicate, but no
candidate replicates bge-small's wake_turb behaviour AND its other
recall numbers. The 384-dim model trained on the right data
out-resolves a 1024-dim model trained on more general data for this
specific corpus + fixture shape.

Tier 2 candidates (`arctic-embed-l`, `e5-large-v2`, `gte-large-en`)
not run — the tier 1 sweep makes it unlikely a tier 2 candidate
flips the verdict. Re-evaluate if the corpus or fixture grows
non-trivially.

## Result envelope

`scripts/bench_embedder.py` writes one JSON per candidate at
`character/airton_c1/embedder_bench/<sanitized>.json`. Schema:

```jsonc
{
  "repo": "BAAI/bge-large-en-v1.5",
  "sanitized": "BAAI_bge-large-en-v1.5",
  "db_path": "character/airton_c1/data/bench/<sanitized>.sqlite",
  "fixture": "character/airton_c1/atc_eval.yaml",
  "ingest": {
    "seconds": ...,
    "rss_before_mb": ...,
    "rss_after_mb": ...,
    "rss_delta_mb": ...,
    "rows_ingested": ...,
    "embed_dim": ...
  },
  "eval_seconds": ...,
  "recall_at_1": ...,
  "recall_at_3": ...,
  "recall_at_5": ...,
  "recall_at_k": ...,
  "median_rank": ...,
  "cases": [ ... per-case rank + top hits ... ]
}
```

A small comparator (TBD bead) folds N envelopes into a candidate
scoreboard table. Until then, eyeball-compare via:

```bash
jq '{repo, recall_at_1, recall_at_3, recall_at_5, recall_at_k, "ingest_s": .ingest.seconds, "rss_delta": .ingest.rss_delta_mb}' \
  character/airton_c1/embedder_bench/*.json
```
