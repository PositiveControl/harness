# airton_c1 retrieval diagnosis — §3-9-6 vs §3-10-3 (harness-0lmm)

## Symptom

Phase-1 baseline run 1 (`controller_same_runway_departure`): the
persona retrieved §3-10-3 (arrival-side same-runway separation) content
when asked about **departing** aircraft. §3-9-6 (departure-side) exists
in the corpus but got out-ranked.

## Reproduction (episodic-only, voice bypassed)

```
$ harness memory search "same runway departure" --k 5
  #1 JO_7110.65 §3-9-6       ✓
  #3 JO_7110.65 §3-10-3

$ harness memory search "departing aircraft behind another departing aircraft same runway" --k 5
  #1 JO_7110.65 §3-9-7       ← wake turb intersection departures, wrong
  (§3-9-6 not in top-4)

$ harness memory search "takeoff roll behind another aircraft same runway" --k 5
  #1 JO_7110.65 §3-9-7
  #4 JO_7110.65 §3-10-3      ← arrival ahead of §3-9-6 for a departure query

$ harness memory search "landing threshold behind another aircraft" --k 5
  #1 JO_7110.65 §3-12-3      ← sea lane arrival
  #2 JO_7110.65 §3-10-3      ← runway arrival
```

Lane V's voice sample for §3-9-6 was retrieved rank-1 for the exact
eval prompt, which is why the re-run flipped the case to pass. The
underlying episodic bias still stands for any phrasing not covered by
a canonical sample.

## Root cause — shared section titles across parent-section groups

JO 7110.65 reuses subsection titles across Chapter-3 groups:

| parent                                          | §      | title                    |
| ----------------------------------------------- | ------ | ------------------------ |
| 3-9 (Departure Procedures and Separation)       | 3-9-6  | SAME RUNWAY SEPARATION   |
| 3-10 (Arrival Procedures and Separation)        | 3-10-3 | SAME RUNWAY SEPARATION   |
| 3-12 (Seaplane / Water Operations)              | 3-12-3 | ARRIVAL SEPARATION       |
| 3-9 (Departure Procedures and Separation)       | 3-9-7  | WAKE TURBULENCE SEPARATION FOR INTERSECTION DEPARTURES |

The chunker (`scripts/atc_chunk.py`) tracks `parent_section="3-9"` /
`"3-10"` as a field but does NOT capture parent-section titles, even
though the markdown has them as `## **Section 9. Departure Procedures
and Separation**` headers.

The ingest script (`scripts/atc_ingest.py`) composes the principle tag
from `section` only:

```python
def principle_for(row):
    return f"{row['source']} §{row['section']}"
```

So `§3-9-6` and `§3-10-3`'s embed-text headers differ by the section
number alone. After `_build_embed_text` (see
`src/harness/store/episodic.py`), their headers look like:

```
[tier: seed; principle: JO_7110.65 §3-9-6]
SAME RUNWAY SEPARATION
JO_7110.65 §3-9-6
<body>
```

vs.

```
[tier: seed; principle: JO_7110.65 §3-10-3]
SAME RUNWAY SEPARATION
JO_7110.65 §3-10-3
<body>
```

The words "Departure" and "Arrival" appear in **neither** header.

## Retrieval impact

**BM25 (FTS5)**: "same runway departure" query matches the shared
title heavily on both sections — title-column matches dominate. Body
does contain "departing aircraft" (§3-9-6) vs "arriving aircraft"
(§3-10-3) so some disambiguation happens, but the title repetition
inflates both sections' scores equally.

**Dense cosine (BGE-small)**: same story at the embedding level. The
repeated title dominates the embedding semantic signal. Without a
parent-section cue ("Departure" vs "Arrival") in the embed text,
similarity ranks are close enough that body-level variance wins.

**§3-9-7 domination**: "Wake Turbulence Separation for Intersection
Departures" has "Departures" in the title — that's why it out-ranks
§3-9-6 on most departure-flavored queries.

## Candidate fixes

Ordered by effort:

### Fix A — enrich `principle` with parent-section title (recommended)

Modify `scripts/atc_chunk.py` to capture parent-section titles from the
markdown's `## **Section N. Title**` headers alongside `parent_section`.
Modify `scripts/atc_ingest.py` `principle_for()` to use both:

```python
def principle_for(row):
    parent = row.get("parent_section_title") or ""
    section_title = row.get("title") or ""
    if parent:
        return f"{row['source']} §{row['section']} ({parent} — {section_title})"
    return f"{row['source']} §{row['section']}"
```

Resulting embed text for §3-9-6:

```
[tier: seed; principle: JO_7110.65 §3-9-6 (Departure Procedures and Separation — SAME RUNWAY SEPARATION)]
SAME RUNWAY SEPARATION
JO_7110.65 §3-9-6 (Departure Procedures and Separation — SAME RUNWAY SEPARATION)
<body>
```

Now "Departure" appears three times in §3-9-6's embed text and zero
times in §3-10-3's. BM25 and dense cosine both disambiguate cleanly.

**Cost**: chunker update (parse `## **Section N. Title**` headers),
ingest update, full re-ingest of all 752 rows. Eval check: re-run
atc eval at temp=0 and confirm 12/12 holds.

### Fix B — FTS5 column weighting

`episodic_fts` is a FTS5 table with columns `(title, body, principle)`.
In `_hybrid.py`, the BM25 MATCH query could use `bm25(episodic_fts,
0.3, 1.0, 0.5)` style column weights to under-weight the shared title.

**Cost**: small `_hybrid.py` change; no re-ingest. But: affects all
characters (airton, airton_b, airton_c, airton_c1), so a cross-persona
verification run is required. And it addresses only BM25 — dense
cosine keeps the same bias.

### Fix C — content-body header injection

At chunk time, prepend `<Parent-section title> — <section title>` to
each row's body. Inferior to Fix A because it pollutes the body text
that gets shown to the model; Fix A only touches the embed-text
header.

## Decision

**Recommend Fix A**, scoped to a new bead. Diagnostic work in this
bead (harness-0lmm) is complete — the root cause is identified and
the recommended fix is concrete. The fix carries a re-ingest step;
keeping it as its own unit of work lets the atc-eval regression check
gate the change cleanly.

## Relation to harness-aise (fabrication catcher)

The diagnosis reframes harness-aise. The §3-12-3 citation in run 1
was NOT a fabrication — §3-12-3 exists, it's the sea-lane version of
§3-10-3. What happened is: retrieval surfaced §3-12-3 for a runway
query (sea lane + runway arrival share nearly identical procedural
prose), and the model used the retrieved section number as the cite
while paraphrasing. The "fabrication catcher" envisioned in
harness-aise would have false-flagged a real section. Better fix for
that case is the same retrieval disambiguation (Fix A) — it stops
§3-12-3 from being retrieved for runway queries in the first place.
harness-aise remains valid for true fabrications (model invents a
section number that doesn't exist) but is lower-priority than
harness-0lmm expected.
