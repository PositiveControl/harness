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
