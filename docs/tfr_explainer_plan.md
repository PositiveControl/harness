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
[ static SPA ] ──POST /explain──▶ [ harness.web factory · airton_c_tfr router ]
                                        │
                                        ├──── BASE (harness.web — any character)
                                        │      GET /healthz
                                        │      GET /character
                                        │      POST /chat   (single-turn, full grounding stack)
                                        │      GET /capabilities
                                        │
                                        ├──── EXTENSION (airton_c_tfr)
                                        │      POST /explain
                                        │      GET /tfr_list
                                        │      GET /debug/parse
                                        │
                                        ▼  POST /explain pipeline
                            ┌─ deterministic NOTAM parser
                            │   (regex/coords/times → struct)
                            │
                            ▼
                     [ airton_c_tfr character via orchestrator ]
                            │   forced assemble_context (contract)
                            │   request-scoped read_parsed_notam tool
                            │   lead_with_citation rewriter
                            │   UngroundedCitationHook
                            ▼
                  grammar-constrained model output
                  (prose fields only: verdict, what_it_means, caveats)
                            │
                            ▼
                  endpoint stitches:
                    parser-owned fields (geometry / active / type / citations)
                    + model-owned prose fields
                            │
                            ▼
                       buffered JSON contract
                            │
                            ▼
                       SPA renders
```

Key picks:

1. **New character `airton_c_tfr`** — clone airton_c, scope corpus to
   §91.137–145 + AIM 3-5-1…3-5-5 + AIM 5-6 (national security &
   interceptions) + TFR-relevant PCG entries, tighten `scope_redirect`
   so operational questions ("should I fly?", "is this safe?") are
   refused — explanation only.
2. **Two-stage pipeline.** Deterministic NOTAM parser (regex + `pyproj`
   for geodetic math) extracts geometry and window. Explainer (LLM)
   reasons over the parsed struct + retrieved chunks. The LLM never
   parses coordinates — it would hallucinate digits.
3. **Local-first inference.** Dev: MLX on Mac (Qwen 2.5 7B/32B 4-bit,
   the model family the anti-fabrication catchers are tuned against).
   Prod: Ollama on the DGX Spark when that hardware is authorized for
   this project — same Qwen 4-bit, no new adapter, no catcher
   revalidation. Hosted-API adapter (Anthropic/OpenAI) is deferred until
   sustained traffic genuinely exceeds local capacity.
4. **Generalizable `harness.web` factory.** `harness web --character <name>`
   serves any character via the same primitive (`harness-3jz1.9`). Base
   endpoints (`/healthz`, `/character`, `/chat`, `/capabilities`) are
   character-agnostic. Character extensions (TFR's `/explain`, future
   characters' analogues) live at `src/harness/web/characters/<name>.py`
   and get auto-discovered + mounted. New characters get a working web
   surface without touching the harness primitive.
5. **Stateless backend, request-scoped tool registry.** No DB, no
   session. Per request: parse NOTAM → bind parsed struct into a
   `read_parsed_notam` tool via the generalizable request-scoped tool
   registry helper → forced `assemble_context` prelude (the character's
   contract) → grammar-constrained reply. Tool surface for `/explain`:
   `search_memory` (assemble_context contract path) + `read_parsed_notam`
   only (no `fetch_url`, no `search_web` — paste-only input contract
   holds).
6. **Grammar-constrained model output (prose fields only).** Model emits
   only the prose fields (verdict, what_it_means, caveats) under a JSON
   schema enforced via `outlines`. The endpoint stitches in the parser-
   owned fields (geometry, active, type, citations) deterministically
   before returning. The model can never hallucinate a coordinate — that
   slot in the contract is filled by the parser, not the LLM.
7. **Buffered JSON reply.** Gateway returns one complete JSON object;
   reply size is ~1–2KB so buffered latency is bounded. Rich
   request-level logging (raw completion + parsed object) captured under
   `HARNESS_WEB_DEBUG_DIR` (generalizable, not TFR-specific) for replay
   and `tfr_eval` fixture growth. Streaming considered for v1.5 only if
   perceived-latency feedback warrants it.
8. **Tailscale-only ingress.** Invite-the-audience model — Mac (now) /
   Spark (later) sits inside the tailnet, no public domain, no
   Cloudflare Tunnel, no port forward. `harness web` defaults to
   `--host 0.0.0.0`; the Mac firewall + Tailscale ACLs constrain who
   can reach the port. Standard Tailscale pattern. Matches the
   10s-of-users target audience and zero home-network exposure. Public-
   domain launch is a v2 promotion path, not a v1 requirement.
9. **Tiny SPA.** One HTML + one JS file, vanilla, mobile-first. No
   SvelteKit for a single form. Reads `GET /capabilities` at load and
   renders the available actions — so a future character served on the
   same factory gets a usable SPA shell without code changes. Loaded
   over the tailnet by invited users.

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
| `harness-3jz1.1` | airton_c_tfr character + scoped corpus (incl. AIM 5-6, PCG) | foundation |
| `harness-3jz1.2` | tfr_eval.yaml + `harness eval tfr` (hybrid det + judge) | depends on .1 |
| `harness-3jz1.3` | NOTAM text parser (regex + pyproj) | independent — fan out |
| `harness-3jz1.9` | `harness.web` factory: base endpoints, request-scoped tools, grammar reply | independent — fan out |
| `harness-3jz1.4` | airton_c_tfr web extension: `/explain` + `/tfr_list` + `/debug/parse` | depends on .1, .3, .9 |
| `harness-3jz1.5` | Validate Ollama runtime for airton_c_tfr + DGX Spark deployment plan | independent — fan out |
| `harness-3jz1.6` | SPA frontend (Tailscale-loaded, capabilities-driven, buffered JSON consumer) | depends on .4 |
| `harness-3jz1.7` | Invite-audience launch: Tailscale onboarding doc, disclaimer, basic ops | depends on .2, .4, .5, .6 |
| `harness-3jz1.8` | [v2 deferred] Live TFR feed | parking lot |

Four parallel paths into a single launch join. Start `3jz1.1` / `3jz1.3`
/ `3jz1.5` / `3jz1.9` concurrently as soon as you want to fan out. The
TFR-specific gateway (`3jz1.4`) joins after `3jz1.9` (the generalizable
web factory) lands.

## Resolved decisions (2026-05-14 refinement pass)

1. **Inference: local-first.** MLX on Mac (dev/now) → Ollama on DGX Spark
   (prod/when authorized) — same Qwen 2.5 32B 4-bit family the catcher
   roster is tuned against. Hosted-API adapter deferred. Rationale: cheap-
   to-own > rent-for-life productization values; avoids re-validating the
   anti-fabrication stack against a different model's failure modes.
2. **v1 scope: TFRs only.** Brand is "TFR explainer." Expanding dilutes
   value prop and balloons eval surface.
3. **Live feed: v2.** Paste-only ships faster; use case intact.
4. **Corpus: §91.137–145 + AIM 3-5-1…3-5-5 + AIM 5-6 + PCG.** AIM 5-6
   added for §91.139 prohibited / NORAD interception cases.
5. **Eval scoring: hybrid.** Deterministic citations (anchors must appear
   in retrieved chunks) + deterministic geometry (lat±0.005°,
   radius±0.1NM, altitude exact) — then LLM-judge gated behind det≥0.9 to
   distinguish "right-shaped + correct" from "right-shaped + wrong
   meaning." Judge model orthogonal to runtime model (same pattern as
   voice eval).
6. **Parser deps: pyproj.** Proper geodetic math for great-circle
   distance, cylinder rendering, point-in-cylinder checks. Worth the C
   dep.
7. **Ingress: Tailscale-only, invite the audience.** No public domain in
   v1. Matches 10s-of-users target. Public ingress (Cloudflare Tunnel /
   VPS reverse-proxy) is a v2 promotion path.
8. **Reply shape: buffered JSON.** Gateway returns complete JSON object;
   reply is small (~1-2KB) so latency is bounded. NDJSON streaming
   considered and rejected for v1 (UX-only win, debug-only harm). Rich
   gateway logging — raw completion + parsed object — captured under a
   debug-dir env flag for replay and `tfr_eval` fixture growth.
9. **Tool surface: `search_memory` + `read_parsed_notam`.** Forced
   `search_memory` prelude for retrieval grounding. Per-request tool
   registry binds the parsed NOTAM struct into a `read_parsed_notam` tool
   the model can introspect (no `fetch_url`, no `search_web` — paste-only
   input contract).

## Still open

1. **Brand / domain.** Parked until v2 promotion path is on the table. No
   v1 decision needed (Tailscale-only ingress doesn't require a public
   domain).

## Estimate

| Phase | Days |
|---|---|
| `3jz1.1` character + corpus | 3-5 |
| `3jz1.2` eval set + subcommand | 2-3 |
| `3jz1.3` NOTAM parser | 5-7 (text is gnarly) |
| `3jz1.9` harness.web factory + base endpoints | 3-5 |
| `3jz1.4` TFR character extension (/explain + /tfr_list + /debug/parse) | 2-3 |
| `3jz1.5` validate Ollama + Spark plan | 2-3 |
| `3jz1.6` SPA | 2 |
| `3jz1.7` launch | 2-3 |

**~3-4.5 weeks** of focused work to invite-launch with fan-out on the
independent paths (`.1` / `.3` / `.5` / `.9`). `.1`, `.2`, `.3` already
landed (2026-05-14 / 2026-05-15); remaining work centers on `.9` (the
generalizable web factory), `.4` (TFR character extension), and the
`.5` / `.6` / `.7` path.

## Close criterion

Tool live on the tailnet, invite-onboarded for the first wave of
pilot-friend users. `tfr_eval.yaml` passes at ≥0.85 citation accuracy
with zero fabricated section numbers (UngroundedCitationHook clean).
Mobile-tested on iOS Safari + Android Chrome over Tailscale. JSON
contract stable and documented. Spark deployment plan written and ready
to execute when hardware is authorized.
