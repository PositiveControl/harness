# airton_c1 — Synonym Policy

This document is the source of truth for **how synonym entries
get added, where they live, when they get pruned, and how each
entry justifies itself**. Two YAML files do the actual work
(`corpus/synonyms.yaml`, `corpus/query_synonyms.yaml`). This
file is the policy that governs them.

If you can't trace an entry back to a fixture case + a measured
rank delta, it doesn't belong in either file.

## The two files

| file                          | side             | who reads it                                   |
| ----------------------------- | ---------------- | ---------------------------------------------- |
| `corpus/synonyms.yaml`        | ingest + query   | `scripts/atc_ingest.py` (bakes synonyms into stored row's principle) AND `query_expander.py` (triggers on incoming query) |
| `corpus/query_synonyms.yaml`  | query only       | `query_expander.py` only (does NOT touch ingest) |

The split exists because **stored-row enrichment is double-edged**:
appending lay-term tokens to a row's principle helps lay-language
queries match that row, but it also dilutes the row's TF/IDF on
its own jargon identity. The `query_synonyms.yaml` header
documents the canonical regression: adding "small plane behind
big jet" to §2-1-19's ingest-side synonyms moved the jargon query
"when do wake turbulence procedures apply" from rank 0 to rank 9,
even though the lay query "small plane flying behind a big jet"
moved from hard miss to rank 0. Net-neutral on aggregate recall
but bad on precision.

The split lets us add lay-language coverage without paying that
precision tax everywhere.

## When to add an entry

Only on a **proven miss**. Specifically: a fixture case in
`atc_eval.yaml` at rank > 3 OR hard miss in
`atc_retrieval_baseline.json`. "I think this lay phrasing should
match" is not enough — speculation pollutes both files and grows
the dead-entry tail.

The miss must be reproducible:

```bash
HARNESS_CHARACTER_NAME=airton_c1 \
  uv run harness eval atc-retrieval --json | jq '.cases[] | select(.id == "<case>")'
```

If the case isn't in the fixture, add it to the fixture first
(plan #4 grew the fixture 19 → 29 for exactly this reason).

## Where the entry goes — decision tree

1. **Default = `query_synonyms.yaml`.** It's the safer side. The
   only cost is N+1 string-concat work at query time; nothing
   leaks into stored rows.
2. **Run `eval atc-retrieval --json` post-add.** Read the new
   rank for the target case.
3. **If rank ≤ 3 (top-3 — meaningful lift):** done. Commit with
   audit trail.
4. **If rank still > 3 OR hard miss:** the issue is that the
   stored chunks don't contain matching tokens after the query
   expander adds the synonyms. Two options:
   - (a) Refine the synonym phrasing — more closely matches
     tokens that ARE in the chunk body.
   - (b) Promote to `corpus/synonyms.yaml` so the synonyms
     bake into the stored row's principle on next ingest.
5. **Before promoting** to `synonyms.yaml`, run the **jargon
   regression check**: identify the jargon case(s) on the same
   section in the fixture and confirm their rank doesn't move
   past their pre-promotion rank. Example: §3-10-3's lay
   variants live in `synonyms.yaml`; the paired jargon case
   `controller_same_runway_arrival` stays at rank 0 across
   that promotion. If §2-1-19 entries had been promoted (the
   harness-ajn measurement), the jargon case would have
   regressed 0 → 9. That regression is the gate on promotion.

## Audit trail — every entry justifies itself

Every entry MUST carry a comment naming:

- The fixture case ID it serves (e.g. `controller_omit_holding_instructions`).
- The rank-without-entry → rank-with-entry delta at the time
  the entry was added (e.g. `# rank: hard-miss → 1`).
- The date the entry landed (so future ablation work can tell
  fresh entries from stale ones).

Block format in the YAML:

```yaml
"5-10-11":
  # Missed approach procedure — case
  # `controller_missed_approach_lay`. Lay phrasing uses "go
  # around" + temporal "last second" framing the doc-text
  # ("Before an aircraft starts final descent...") doesn't
  # carry. Rank: hard-miss → 1 (added 2026-04-25).
  - go around at last second
  - missed approach already told pilot
  - ...
```

Without this trail, a reviewer six months from now can't tell
whether an entry is load-bearing or dead. With it, the ablation
script (Phase 1) can confirm the recorded delta still holds.

## Promotion log

When an entry moves from `query_synonyms.yaml` to
`synonyms.yaml`, append a one-liner to the
**Promotion log** section at the bottom of `synonyms.yaml`:

```
# Promotion log
# 2026-04-25: §X-Y-Z promoted from query_synonyms.yaml. Reason:
#   query-only didn't lift case `<case_id>` past rank 3
#   (residual rank 5). Jargon case `<jargon_case_id>` rank
#   confirmed unchanged (0 → 0) post-promotion.
```

Promotion is rare and load-bearing. The log makes it auditable.

## When to retire an entry

An entry is **dead** when the case it cited as its reason no
longer relies on it (rank stable when the entry is removed).
This usually happens because:

- The chunker improved (e.g. `harness-1s4`'s NOTE/PHRASEOLOGY
  fold absorbed the substance the synonym was bridging).
- The fixture case got rephrased.
- A bigger embedder swap closed the gap (Phase D Fix D
  candidate, currently no swap warranted).

Detecting dead entries is the **ablation script** in Phase 1
(`scripts/test_synonyms.py --ablate`). For each entry:

```
1. Remove the entry temporarily.
2. Re-ingest scratch DB if shared-side.
3. Run eval atc-retrieval --json.
4. Diff per-case ranks vs current baseline.
5. If no case rank changed: entry is dead. Prune.
```

Until that script lands, manual ablation on a new entry's
target case is the minimum bar before promoting.

## Anti-patterns (don't)

- **Speculative entries.** "Pilots might call X this." If no
  fixture case proves it, don't add it.
- **Generic phrasings.** Entries should be section-specific.
  "Aircraft separation" is too broad — every section is about
  aircraft separation. Tight to the section's actual topic.
- **Lifting at the cost of a sibling.** A §2-1-19 entry that
  lifts the lay query but breaks the jargon query is a
  precision loss. Test both sides before promoting.
- **Editing without re-snapshotting baseline.** Every YAML
  edit invalidates the comparator gate. After a clean add,
  re-run `--save-baseline` so the gate measures against the
  new state going forward.
- **Bulk additions.** Add one entry per commit unless the
  entries are a coordinated set (e.g. a paired jargon/lay
  pair with the same evidence trail). Atomic adds make
  ablation tractable.

## Phase plan

This document is **Phase 0** — the policy itself plus the two
manual fixes for `controller_missed_approach_lay` and
`controller_handoff_lay` that demonstrate the policy in action.

**Phase 1** (next bead): build `scripts/suggest_synonyms.py`
(small-model proposes candidate entries from misses) and
`scripts/test_synonyms.py` (validates a candidate ablatively
before commit). Workflow becomes:

```
1. uv run python scripts/suggest_synonyms.py > /tmp/suggested.yaml
2. # human edits — picks/refines entries
3. uv run python scripts/test_synonyms.py /tmp/suggested.yaml
4. # if LIFT: paste into the right yaml file, commit
```

**Phase 2** (eventually): CI integration. Suggester runs
nightly against the baseline; ablation runs weekly; new
beads auto-open with proposed YAML attached. Synonym
generation becomes a managed background process; humans
review and merge instead of authoring from scratch.

The architectural conviction: **synonyms are configuration,
not code.** YAML is the source of truth (human-readable,
git-versioned, diff-friendly). Whatever generates the YAML
(hand / script / model) is just a different input pipeline;
the runtime stays simple.
