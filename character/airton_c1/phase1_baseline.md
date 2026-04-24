# airton_c1 Phase-1 Baseline (2026-04-23)

First end-to-end deterministic run of `eval atc` against airton_c1's
stack: Qwen 2.5 7B 4-bit + PersonaAdapter + BGE-small retrieval over
752 seed rows from JO 7110.65BB (Basic w/ Chg 1 & 2, 2026-01-22).

Reproduce:

```bash
HARNESS_CHARACTER_NAME=airton_c1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run harness eval atc --model mlx --json > character/airton_c1/atc_baseline.json
```

Raw output lives in `character/airton_c1/atc_baseline.json` — tracked
so diffs between runs are visible in git. Overwrite on each baseline
run; history is the commit log.

## Headline numbers

| scope      | pass | rate |
| ---------- | ---- | ---- |
| overall    | 8/12 | **66.7%** |
| controller | 8/12 | 66.7%     |

Phase-1 target: ≥ 80% (10/12). Baseline is **−13.3 pp**.

All 12 cases ran through the persona rewriter + post-rewrite citation
fixup (harness-cco). Temperature 0.0 throughout — run-to-run
deterministic.

## Failure patterns

All 4 failures are citation-form misses; 3 of 4 have correct content
with a wrong-or-absent section number.

### Pattern 1 — retrieval/citation bug on §3-9-6 ↔ §3-10-3 (2 failures)

Two same-runway-separation cases reveal a real bug worth filing.

**`controller_same_runway_departure`** — Q about **departing** aircraft.
Retrieval pulled §3-10-3 (arrival) content; reply correctly identifies
the section as `§3-10-3` but explains arrival separation instead of
departure. Completely off-topic for the question. §3-9-6 is the
correct departure section — it's in the corpus (verified) but lost
the retrieval rank. BM25 lexical bias: the term "same runway"
surfaces §3-10-3 ("Same Runway Separation") ahead of §3-9-6 ("Separation
on the Same Runway") on most phrasings.

**`controller_same_runway_arrival`** — Content is correct (pulled from
actual §3-10-3). But the reply cites a **fabricated section number**:
"§3-12-3". §3-12-3 doesn't exist in JO 7110.65 or in the corpus. Model
is confabulating the section while the retrieved content is right.
Post-rewrite citation fixup couldn't catch this because pass-1 also
had the wrong section (not a rewriter drop — a generation error).

This is airton_c1-specific — airton_c never fabricated section
numbers because its corpus mixes CFR (numeric) and AIM (N-N-N),
giving enough structural anchoring. Pure JO 7110.65's N-N-N format
is more fabricable.

### Pattern 2 — no voice sample → no citation (2 failures)

**`controller_class_b_vfr_clearance`** — Model produced a terse
phraseology-only reply:

> "Cleared through/to enter/out of Bravo airspace, and as appropriate,
> via (route). Maintain (altitude) while in Bravo airspace."

Content correct, verbatim from §7-9-2. But no citation. Constitution
says "cite the source or say you don't know" — the model skipped it.
airton_c1 has **zero voice samples** (the template shipped empty);
there's nothing teaching the model to frame phraseology answers with
a section header.

**`controller_emergency_distress_urgency`** — Content is rich and
correct (Mayday/Pan-Pan, distress/urgency all present). But no
§10-1-1 citation. Same diagnosis: no voice sample exemplifying the
"cite JO 7110.65 § first, then answer" shape.

## Next lanes

Estimated lifts assume the harness-cco post-rewrite fixup + voice
retriever behave the way they do on airton_c.

### Lane V — voice samples (cheap; expect +2 to +3 flips)

Four canonical voice samples in `character/airton_c1/voice/canonical.yaml`,
one per failure, each citing the expected section verbatim in the
first sentence. The sample for each case:

- `§3-9-6` same-runway-departure (targets both the retrieval miss AND
  the citation shape — the sample text itself will become a
  high-ranked voice hit on departure queries, which may counteract
  the BM25 bias on §3-10-3)
- `§3-10-3` same-runway-arrival (stops the §3-12-3 fabrication by
  giving the model a correct example to imitate)
- `§7-9-2` Class B VFR clearance (anchors the phraseology in the
  section)
- `§10-1-1` emergency distress/urgency (anchors the Mayday/Pan-Pan
  answer in the section)

Each follows airton_c's e73 pattern: prompt + gold with the section
cite in the opening sentence. Post-rewrite citation fixup will catch
any that the rewriter drops.

### Lane R — retrieval diagnosis for §3-9-6 (medium lift)

Even with a voice sample, the actual retrieved episodic content for a
departure query still surfaces §3-10-3 first. The voice sample
addresses citation form, not retrieval rank. Investigate:

1. Run `harness memory search "same runway departure"` and inspect
   top-5. Confirm §3-9-6 is absent or below §3-10-3.
2. Compare chunked body text for both sections — §3-9-6's body may
   start with text that's less-lexically-matching the query than
   §3-10-3's "Same Runway Separation" header.
3. Candidate fix: boost section-title BM25 weight, or re-chunk with a
   principle-tag header that includes the full paragraph title so
   "Separation on the Same Runway" vs. "Same Runway Separation"
   disambiguates.

### Lane F — fabricated section numbers (needs investigation)

The §3-12-3 fabrication is a **new** failure mode airton_c didn't
exhibit. Before writing a voice sample, understand why the model
invents section numbers for JO 7110.65 but not for CFR/AIM. Hypothesis:
JO 7110.65's `N-N-N` pattern is lexically closer to the model's priors
for arbitrary citation-shaped strings than numerical `§91.155` is.
Track in a new bead; the voice sample for §3-10-3 may hide the
symptom but the underlying risk remains.

## What the baseline means

66.7% with a 12-case fixture, all failures are citation-form (content
is mostly correct). This is a **useful Phase-1 artifact** — the eval
is now the gate for every subsequent lane on airton_c1. A +1-flip
change is real signal at temp=0.

Path to 80%: +2 flips needed. Lane V covers the two no-cite failures
cleanly (§7-9-2, §10-1-1) — that alone puts us at 83% (10/12). The
other two (same-runway pair) are at higher risk until retrieval +
fabrication are tackled. Worst-case-after-lane-V: 10/12 = 83.3%.
Best-case: 12/12 = 100%.

---

## Run 2 — after lane V (2026-04-23)

Four canonical voice samples added to `character/airton_c1/voice/canonical.yaml`,
one per failure, each citing the expected section verbatim in the opening
sentence (e73 pattern). Voice retriever rank-1 verified for each of the
four eval prompts before the re-run.

**12/12 = 100%** — +4 flips vs. the 8/12 (66.7%) baseline. Zero regressions.
Passed best-case.

| scope      | run 1 | run 2 | Δ         |
| ---------- | ----- | ----- | --------- |
| overall    | 66.7% | 100%  | +33.3 pp  |
| controller | 66.7% | 100%  | +33.3 pp  |

### Per-case flips

**Flipped pass (+4)**:

- `controller_same_runway_departure` — voice sample shifted the
  semantic context enough that the persona adapter answered the
  departure question with §3-9-6 content (the sample's gold explicitly
  anchors to §3-9-6 and describes departure-side separation). Fixed
  both the retrieval miss AND the citation miss in one move.
- `controller_same_runway_arrival` — §3-12-3 fabrication gone. Reply
  cites §3-10-3 cleanly. The voice sample gave the model a correct
  section number to imitate.
- `controller_class_b_vfr_clearance` — reply now opens with
  `JO 7110.65 §7-9-2` before the phraseology. Clean voice-sample
  imitation (the reply is very close to the gold).
- `controller_emergency_distress_urgency` — `JO 7110.65 §10-1-1`
  lands in the first sentence. Reply paraphrases well beyond the
  sample (added a descriptive closing sentence about signaling
  nature of the emergency).

### Known residual issue — content-accuracy beneath the rubric

`controller_same_runway_arrival` passed the fixture but the reply
hallucinates Category-distance numbers. Real §3-10-3 lists:

    Category I behind Category I/II    — 2,000 feet
    Category II behind Category I/II   — 2,500 feet
    (+ other pairings)

The reply instead says 3,000 / 4,500 / 6,000. The fixture didn't catch
this because `expected_keywords` only required `landed`, `clear of the
runway`, `landing threshold` — not the specific distances. The voice
sample itself says "e.g. 2,000 feet for Category I behind Category
I/II" but the model paraphrased past the sample's one illustrative
number and invented three new ones.

Candidate follow-up lane: content-accuracy fixture tightening — add
the specific distance values as required keywords so hallucinated
numbers fail. Orthogonal to the 100% score on the current rubric;
note it as a known gap.

### What 100% means — and doesn't

Phase-1 target (≥80%) met, with headroom. What this *does* mean:

- The citation-form lane is cleanly solved for the 12 fixture topics.
- The voice-sample + post-rewrite fixup pattern transfers from airton_c
  to airton_c1 without any additional machinery.
- Deterministic temp=0 means any +1 flip from here is real signal.

What this *does not* mean:

- airton_c1 is correct outside these 12 topics. The fixture is a gate,
  not a coverage guarantee.
- The §3-9-6 retrieval bias (harness-0lmm) is fixed. It's masked — the
  voice sample out-ranks the bad episodic hit. On a query not covered
  by a voice sample, the bias re-emerges. Lane R still valid.
- The JO 7110.65 fabrication risk (harness-aise) is fixed. It's
  suppressed in this one case by the voice sample. A query far from
  any canonical sample can still fabricate a section. Lane F still
  valid.

### Next

Durability lanes, not score lanes. Lane V bought +33.3 pp cheaply
because it targeted the specific failure modes the fixture scored
against. To harden the score:

1. Grow the fixture to ~30 cases (file as Phase-1.5 bead) — each
   failure of actual student use becomes a regression test.
2. Close lane R (retrieval title bias) so §3-9-6 works on arbitrary
   "same runway" queries, not just the one the voice sample matches.
3. Close lane F (section-number fabrication) — post-model hook that
   verifies every `JO 7110.65 §X-Y-Z` cited appears in the ingested
   corpus; drop + nudge on any that don't.
4. Tighten the same-runway-arrival fixture to require specific
   distance values (addresses the content-accuracy gap).
