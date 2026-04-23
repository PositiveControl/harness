# atc Phase-1 Baseline (2026-04-23)

First end-to-end run of `eval atc` against the full Phase-1 stack:
Qwen 2.5 7B 4-bit + PersonaAdapter + BGE-small retrieval over the
6,688-row ingested FAA corpus (`atc-4` → `atc-5` → `atc-6` chain).

Reproduce:

```bash
HARNESS_CHARACTER_NAME=airton_c HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run harness eval atc --model mlx --json > character/airton_c/atc_baseline.json
```

Raw output lives in `character/airton_c/atc_baseline.json` — tracked so
diffs between runs are visible in git. Overwrite it on each baseline
run; history is the commit log.

## Headline numbers

| scope   | pass | rate |
| ------- | ---- | ---- |
| overall | 12/22 | **54.5%** |
| ifr     |  7/11 | 63.6% |
| ppl     |  5/11 | 45.5% |

Phase-1 target was ≥80%. Baseline is **−25pp**.

## Failure patterns

### Pattern 1 — AIM paragraph citations don't land (4 failures)

Model prefers `§91.xxx` CFR format; doesn't surface `N-N-N` AIM
paragraph format naturally. Substantive content is correct in each
case — only the citation form misses.

| case | expected | got |
| ---- | -------- | --- |
| `ppl_readback_basics` | `AIM 4-4-7` | no AIM cite; content correct (3/2 keyword hits) |
| `ppl_wake_turbulence_avoidance` | `AIM 7-3` | no AIM cite; content correct |
| `ifr_ifr_clearance_limit` | `AIM 5-3` | no AIM cite; content correct |
| `ppl_preflight_action` (also pattern 3) | `§91.103` | `§91.4(a)` — wrong CFR |

### Pattern 2 — Keyword phrasing is brittle (3 failures)

Model produced semantically-equivalent phrasings that the exact-
substring matcher missed:

| case | expected keyword | model's phrasing |
| ---- | ---------------- | ---------------- |
| `ifr_lost_comm_rules` | "last assigned" | "last ATC clearance" |
| `ppl_hemispheric_cruising_altitudes` | "odd" / "even" (altitudes) | used the numbers, not the quality |
| `ppl_class_b_entry_requirements` | "Mode C" / "two-way radio" | partial — missed one exact string |

### Pattern 3 — Wrong retrieval / factual error (3 failures)

These are the interesting ones — content went wrong, not just form:

- **`ppl_vfr_cloud_clearance_above_10k`** (cite ✓, keywords 0/2) —
  model claimed "cloud clearance requirement is not applicable above
  10,000 MSL". Factually wrong. Either retrieval didn't surface the
  §91.155(a) table or the model hallucinated.
- **`ppl_preflight_action`** (cite ✗) — cited `§91.4(a)`; reply
  mentions "control links" and "TFR compliance" — shape of UAS /
  drone-ops content, not general PPL preflight. Retrieval appears to
  have pulled from CFR Part 91 subpart E (UAS) when the pilot-facing
  §91.103 was the right target.
- **`ifr_takeoff_minimums_part_91`** — cited `§91.155` + `§91.179`
  instead of `§91.175`. Topic confusion between VFR weather minimums
  and IFR takeoff minimums.

## Next steps

Three lanes, in order of cheapest-effort-to-highest-lift:

### Lane A — rubric loosening (cheap; probably +3 passes)

Relax exact-string keyword matching where the model is correct but
phrases differently. Concrete edits to `atc_eval.yaml`:

- Accept `"1,000 ft"` / `"1000 feet"` alongside `"1,000 feet"`.
- Accept `"last ATC clearance"` alongside `"last assigned"`.
- Replace `"odd"` / `"even"` with the numeric examples the model
  actually gives (`"3,500"`, `"4,500"`, etc.).

Still strict enough to fail a model that wanders off-topic; just
removes form-over-substance losses.

### Lane B — AIM citation voice samples (cheap; probably +2-3 passes)

Add 1-2 canonical samples to `character/airton_c/voice/canonical.yaml`
that use AIM paragraph format explicitly (e.g., `AIM 4-4-7`, `AIM
5-1-15`). The voice retriever will pick them up for retrieval-form
queries; the PersonaAdapter rewrite will nudge toward AIM-shaped
citation when appropriate. Watch for over-correction — don't want
every reply to cite AIM when CFR is the right source.

### Lane C — retrieval diagnosis (real bug; highest lift)

For each of the 3 factual-error cases, pull the top-3 retrieved
episodic rows via the memory-search CLI and compare against what
atc should have seen:

```bash
HARNESS_CHARACTER_NAME=airton_c HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run harness memory search "What's the VFR cloud clearance above 10,000 MSL?" --k 5
```

Likely findings and candidate fixes:

- **VFR cloud clearance** — if retrieval didn't bring the §91.155(a)
  table, atc-6's filter may have dropped it (short body?) or the
  chunker may have split the table apart. Check `cfr_14_vol2.jsonl`.
- **Preflight action** — if retrieval ranked UAS content above the
  pilot-facing §91.103 row, that's a corpus-balance issue. Consider
  a source-aware retrieval weight or tagging Part 107/Part 89 rows
  so PPL queries prefer Part 61/91.
- **Takeoff minimums §91.175** — maybe §91.175 was filtered out in
  chunker dedup. Verify the row exists.

## What the baseline means

54.5% with a strict rubric, reasonable content quality on most
failures, and three clearly-factual-error cases. This is a **useful
Phase-1 artifact** — the eval is now the gate for:

- Phase-2 voice rewriter changes (must stay ≥ 54.5% to land).
- Phase-3 LoRA (must beat 54.5% materially; otherwise LoRA is not
  earning the complexity).
- Phase-1.5 corpus expansion (must not regress the 54.5% baseline).

The `atc_baseline.json` file is the reference snapshot; rerunning and
diffing the per-case pass set is the smallest unit of feedback when
tuning any of the three lanes.

---

## Run 2 — after lane A (2026-04-23)

**Rescore** of the original replies against the loosened rubric
(`harness-099`): **13/22 = 59.1%** (+4.6 pp over the 54.5% baseline).
Rescore is deterministic — same replies, new scorer. No new MLX
generation.

### Finding: Pattern 2 was overestimated

I originally classified 3 failures as "brittle keyword matching":
`ifr_lost_comm_rules`, `ppl_hemispheric_cruising_altitudes`,
`ppl_class_b_entry_requirements`. After loosening their fixture
keywords to accept multiple alternate phrasings, only ONE flipped to
pass:

- **`ifr_lost_comm_rules`** — legit Pattern 2 win. Model said "last
  ATC clearance" where the fixture expected "last assigned". Alternate
  `["last assigned", "last ATC clearance", ...]` caught it. Also
  matched "radio failure" via the expanded "lost communication" set.

The other two were actually content gaps, not phrasing gaps:

- **`ppl_class_b_entry_requirements`** — model focused on ATC
  clearance phraseology and skipped Mode C / two-way radio entirely.
  Loosening to accept "transponder" / "ADS-B" / "two-way" didn't
  matter — none appeared.
- **`ppl_hemispheric_cruising_altitudes`** — model deflected with
  "the exact values are in 14 CFR §91.159". Neither "odd/even" nor
  any concrete altitude (3,500 / 4,500 / …) appeared.
- **`ifr_instrument_currency`** — model cited the wrong CFR section
  (`§91.171` Maintenance instead of `§61.57`) and talked about flight
  reviews rather than instrument currency. Content is about the wrong
  topic entirely.

These three belong in **lane C** (retrieval / content quality) — not
lane A. The baseline report's "Pattern 2: 3 cases" estimate was wrong;
the real count is 1.

### Takeaway for next runs

Rubric loosening is cheap but has a small ceiling unless the model is
actually phrasing-limited. Before adding alternates to a keyword,
check the raw reply: is the *concept* present in any form, or is the
model talking about something else entirely? If the latter, alternates
won't help — it's a lane B (citation form) or lane C (retrieval)
case.

Schema change shipped alongside: `expected_keywords` entries can now
be either a scalar string (single phrasing, backward-compatible) or a
list of strings (any member matches). See
`src/harness/evals/atc.py:_load_keywords` for the loader semantics,
and the tests `test_keyword_alternates_*` for the contract.

Next:
- **Lane B (`harness-e73`)** — AIM-paragraph citation voice samples;
  should move the 3–4 AIM-cite misses.
- **Lane C (`harness-74n`)** — diagnose the 3 content-gap / wrong-
  retrieval cases. Higher lift, more work.
