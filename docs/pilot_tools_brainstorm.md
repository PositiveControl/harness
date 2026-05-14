# Pilot Tools Brainstorm — airton_c* as backend

Free web tools for private pilots, built on the airton_c / airton_c1 retrieval
+ anti-fabrication stack. Avoid the ubiquitous categories: no flight tracker,
no NOTAM viewer, no METAR/TAF decoder, no chart browser.

## What the backend uniquely brings

- ~13k pre-chunked, retrieval-indexed FAA chunks: AIM, 14 CFR vol 1+2,
  JO 7110.65, PCG, PHAK.
- `cite_or_silent` value + `lead_with_citation` rewriter — every reply
  opens with a §-anchor.
- `require_search_memory` forces a grounding tool call before the first
  reply; `UngroundedCitationHook` refuses replies whose citations weren't
  in the retrieved chunks.
- `fetch_url` allowlist — aviationweather.gov, 1800wxbrief.com, faa.gov,
  notams.aim.faa.gov.
- Domain catchers: `scope_redirect`, `reserved_squawk_code`,
  `ambiguous_context`.

Net effect: structurally cannot hallucinate a section number. That is the
moat against "ChatGPT + RAG" lookalikes.

## Ideas, ranked by uniqueness × pain

1. **TFR / unfamiliar-NOTAM plain-English risk explainer.** Paste a NOTAM
   or hit the FAA NOTAM endpoint by ICAO/region. Output: what it means
   for you, with citations into 14 CFR §91.137–145 and the source NOTAM
   text. TFR busts are a top FAA enforcement category and the published
   text is legalese.

2. **Citation-anchored checkride oral prep.** Socratic PPL/IFR/Commercial
   ACS Q&A — every answer cites the FAR/AIM/PHAK section it came from.
   Differentiator vs. Sheppard/Gleim: free, teaches via citations instead
   of memorize-the-test. ACS area-as-URL slug (`/ppl/area-iii-task-b`)
   so CFIs can share specific tasks.

3. **Airspace-transit advisor.** "KPAO → KTRK, what airspace do I cross
   and what's required?" Plain-English ladder of B/C/D/E/G transit rules
   with §91.130/135/155 + AIM 3-2 citations. Most planners show airspace
   on a map; none explain the requirements.

4. **IFR / PPL currency status explainer.** Not yes/no — walk the
   regulation (§61.57, §61.56, §61.23 medical / BasicMed) and tell the
   pilot specifically what they're short and the path back (grace period
   vs. IPC vs. safety-pilot rules).

5. **AD / SB lookup for owners and renters.** Search FAA AD database by
   make/model/N-number, return plain-English compliance impact with
   §39 citations.

6. **Chart symbol & approach plate Q&A.** Snap-to-citation Q&A grounded
   in AIM Chapter 5 + Chart User's Guide. Mobile-first.

7. **ATC phraseology decoder.** Paste a transmission you didn't
   understand → PCG/AIM-grounded translation. Student-pilot pain mostly,
   with commercial spillover (oceanic / CPDLC).

## Recommendation

Start with **#1 (TFR/NOTAM explainer)** or **#2 (checkride oral prep)**.
Both ride existing corpora — no new ingest. Both make airton_c's citation
moat legible to a pilot in the first 30 seconds.

- #1: more enforcement-risk gravity, sharper "save me from a violation"
  hook, lower repeat usage.
- #2: higher daily volume + stickiness, every CFI has students prepping
  for a ride.
