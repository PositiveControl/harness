# TFR Explainer — Plan

Free public web tool that decodes published TFR/NOTAM text into plain-English
risk explanation for Part 91 pilots, grounded in the airton_c anti-fabrication
stack. Sibling doc: [pilot_tools_brainstorm.md](pilot_tools_brainstorm.md).

Tracked in bd as epic **`harness-3jz1`** with children `harness-3jz1.{1..8}`.

## The pain

TFR busts sit in the FAA's top enforcement categories every year. Published
NOTAM text reads like:

> PURSUANT TO 14 CFR SECTION 91.137(A)(1) TEMPORARY FLIGHT RESTRICTIONS…
> AIRCRAFT FLIGHT OPERATIONS ARE PROHIBITED WITHIN AN AREA DEFINED AS 6 NMR
> OF 384154N/0863912W…

Coordinate-encoded geometry, mixed MSL/AGL altitudes, UTC times, recurring
stadium-schedule semantics. Existing decoders just expand abbreviations.
Nobody explains *what it means for you* with cited anchors into the rule.

## The moat

airton_c's stack structurally refuses fabricated section numbers — the
failure mode that has burned pilots on ChatGPT.

- `cite_or_silent` value + `lead_with_citation` rewriter — every reply opens
  with a §-anchor.
- `require_search_memory` forces a grounding tool call before the first
  reply; `UngroundedCitationHook` refuses replies whose citations weren't
  in the retrieved chunks.
- `fetch_url` allowlist — aviationweather.gov, 1800wxbrief.com, faa.gov,
  notams.aim.faa.gov, tfr.faa.gov.
- Domain catchers: `scope_redirect`, `reserved_squawk_code`,
  `ambiguous_context`.

## Scope

**In (MVP):**
- Paste-only input. Textarea takes raw NOTAM text.
- Part 91 framing only. §91.137 / .138 / .141 / .143 / .144 / .145 + AIM 3-5.
- Explanation, not authorization. Verdict + geometry + window +
  what-it-means + citations + caveats.

**Out (v2+):**
- Live TFR feed (tfr.faa.gov) — tracked in `harness-3jz1.8`.
- NOTAM Search integration (no clean API).
- Non-TFR NOTAM categories (runway closures, NAVAID outages).
- Part 135/121 framings.
- Accounts, saved history, mobile app.

## Architecture

```
[ static SPA ]  ──POST /explain──▶  [ FastAPI gateway ]
                                        │
                                        ▼
                            ┌─ deterministic NOTAM parser
                            │   (regex/coords/times → JSON)
                            │
                            ▼
                     [ airton_tfr character ]
                            │   forced search_memory
                            │   lead_with_citation
                            │   UngroundedCitationHook
                            ▼
                       structured JSON
                          │
                          ▼
                     SPA renders
```

Key picks:

1. **New character `airton_tfr`** — clone airton_c, scope corpus to
   §91.137-145 + AIM 3-5 + TFR-relevant PCG entries, tighten
   `scope_redirect` so operational questions ("should I fly?", "is this
   safe?") are refused — explanation only.
2. **Two-stage pipeline.** Parser first (regex + pyproj) extracts geometry
   and window deterministically. Explainer second (LLM) reasons over the
   parsed structure + retrieved chunks. The LLM never parses coordinates —
   it would hallucinate digits.
3. **Pluggable `ModelAdapter`.** Dev: MLX on the Mac. Prod: hosted
   (Anthropic). Adapter boundary already exists in the harness; this is a
   config flag, not a rewrite.
4. **Stateless backend.** No DB, no session. Each request is a clean run.
5. **Tiny SPA.** One HTML + one JS file, vanilla, mobile-first. No
   SvelteKit for a single form.

## JSON contract

```json
{
  "verdict": "Stadium TFR — no transit in the cylinder while active.",
  "type": "stadium",
  "geometry": {
    "center": {"lat": 32.7511, "lon": -97.0825, "ref": "Globe Life Field, Arlington TX"},
    "radius_nm": 3.0,
    "floor": {"value": 0, "ref": "SFC"},
    "ceiling": {"value": 3000, "ref": "AGL"}
  },
  "active": {"from": "2026-05-14T23:05Z", "to": "2026-05-15T03:30Z"},
  "what_it_means": "Plain-English summary.",
  "citations": [
    {"anchor": "§91.145", "doc": "14 CFR", "excerpt": "..."},
    {"anchor": "AIM 3-5-3", "doc": "AIM", "excerpt": "..."}
  ],
  "caveats": ["Ceiling not explicit in source — confirm with 1800wxbrief."],
  "source_url": null,
  "retrieved_at": "2026-05-14T18:22Z"
}
```

The JSON contract is the API. Mobile apps, CFI integrations, scripts — all
just consumers.

## Phases → beads

| Bead | Phase | Notes |
|---|---|---|
| `harness-3jz1.1` | airton_tfr character + scoped corpus | foundation |
| `harness-3jz1.2` | tfr_eval.yaml + `harness eval tfr` | depends on .1 |
| `harness-3jz1.3` | NOTAM text parser | independent — fan out |
| `harness-3jz1.4` | FastAPI `POST /explain` gateway | depends on .1, .3 |
| `harness-3jz1.5` | Hosted `ModelAdapter` (Anthropic) | independent — fan out |
| `harness-3jz1.6` | SPA frontend | depends on .4 |
| `harness-3jz1.7` | Public launch + ops | depends on .2, .4, .5, .6 |
| `harness-3jz1.8` | [v2 deferred] Live TFR feed | parking lot |

Three parallel paths into a single launch join. Start `3jz1.1` / `3jz1.3`
/ `3jz1.5` concurrently as soon as you want to fan out.

## Open forks

1. **Hosted model or local MLX for public traffic?** Local-on-Mac via
   Tailscale works for friends; falls over at ~tens of concurrent users
   and dies when the Mac sleeps. Hosted (Claude) scales but costs ~$0.005-
   0.02/request. **Pick:** hosted for public, MLX for dev/eval. Adapter
   boundary makes this a config flag.
2. **v1 scope strictness: TFRs only, or all NOTAMs?** **Pick:** TFRs only.
   Brand is "TFR explainer." Expanding dilutes value prop and balloons
   eval surface.
3. **Brand / domain?** Affects positioning more than engineering. To be
   resolved during launch phase (`3jz1.7`).
4. **Live feed in v1 or v2?** **Pick:** v2. Paste-only ships faster; the
   use case ("decode this thing my briefer mentioned") is intact without
   a feed.

## Estimate

| Phase | Days |
|---|---|
| `3jz1.1` character + corpus | 3-5 |
| `3jz1.2` eval set + subcommand | 2-3 |
| `3jz1.3` NOTAM parser | 5-7 (text is gnarly) |
| `3jz1.4` FastAPI gateway | 2-3 |
| `3jz1.5` hosted adapter | 3-4 |
| `3jz1.6` SPA | 2 |
| `3jz1.7` launch | 2-3 |

**~2.5-3.5 weeks** of focused work to public launch with fan-out on the
independent paths.

## Close criterion

Public free web tool live at a domain. `tfr_eval.yaml` passes at ≥0.85
citation accuracy with zero fabricated section numbers. Rate-limited
under basic load. Mobile-tested on iOS Safari + Android Chrome. JSON
contract stable and documented.
