# Phraseology Linter — Product Path Progress

Tracking doc for the cite-grounded ATC transmission verifier built on
airton_c1 (epic [`harness-0pte`](https://github.com/...)). This file is
the progress journal across the product path; per-issue scope and
acceptance criteria stay in beads.

> **Read order**: roadmap → status table → log (newest first).

---

## Goal

Take an ATC transmission utterance (text in Phase 1, audio later) and
return:

1. **verdict** — `ok` / `wrong` / `incomplete` / `out_of_scope`
2. **expected_section** — JO 7110.65 § citation (e.g. `3-9-10`)
3. **expected_phraseology** — the canonical form template
4. **mismatch** — short reason when verdict is `wrong` or `incomplete`
5. **citation_quote** — anchored chunk text from the corpus

The linter is an adjunct to existing voice/sim products (UFA, RICTE):
they capture and deliver speech, the linter verifies it against the
rulebook. The cite-or-silent guarantee from airton_c1's hardened stack
(BGE-small + FTS5 + RRF + UngroundedCitationHook) carries over.

## Scope decisions

- **No new model.** Reuses airton_c1's existing retrieval + persona +
  catcher stack. New surface is one ToolSpec + one CLI subcommand +
  one fixture.
- **Controller-side phraseology only.** Pilot transmissions
  (`MAYDAY`, `REQUEST CLEARANCE`, etc.) are explicitly `out_of_scope`
  — JO 7110.65 governs what controllers say; pilot phrasing lives in
  AIM and the P/CG. The linter must refuse, not cite-guess.
- **Phase 1 = text-mode demo.** No STT, no streaming, no UX investment
  until text-mode proves the product question.

---

## Roadmap

### Phase 1 — text-mode MVP (sequential)

| # | Bead | Status | Owner | Summary |
|---|---|---|---|---|
| 1 | [`harness-h2iz`](#) | **in_progress** | Mark | Build `phraseology_eval.yaml` (~40 cases, ≥10 per scenario class). Locks eval contract. |
| 2 | [`harness-q35t`](#) | open (blocked on #1) | — | `src/harness/tools/phraseology_lint.py` ToolSpec. Cite-or-silent. New `phraseology` tool profile. CLI subcommand `harness phraseology lint <utterance>`. Gate ≥80% combined accuracy on fixture. |
| 3 | [`harness-15dy`](#) | open (blocked on #2) | — | `harness eval phraseology` Typer subcommand + per-scenario breakdown + `--compare-baseline` against `phraseology_baseline.json` + pre-push hook gating regressions on tool / fixture / corpus paths. |

### Phase 2 — voice (deferred until Phase 1 ships)

| # | Bead | Status | Summary |
|---|---|---|---|
| 4 | [`harness-o92o`](#) | deferred | STT front-end (audio → utterance text). |
| 5 | [`harness-ep9o`](#) | deferred | Real-time streaming verdicts. |

### Adjacent risk lanes (not blockers, but bend verdict surface)

These harden the airton_c1 retrieval the linter inherits. If they slip,
the linter inherits the same failure modes. Track separately; ship
text-mode first, fold their wins into the linter's baseline as they
land.

| Bead | Why it matters for the linter |
|---|---|
| `harness-cwx` (P1 bug) | Main model fabricates web-sourced prose after weak router-prelude. Could leak fabricated citations into linter verdicts on weak retrieval. |
| `harness-7wju` (epic) | Phase-1.6 voice-corpus capacity + rubric durability. Children below. |
| `harness-1ebs` | Cite-grounding actuator (auto-replace) — the same hook the linter uses to reject ungrounded citations. |
| `harness-zsbz` | Retry-tier cite-grounding escalation. |
| `harness-8dop` | OutOfScopeHook cosine-floor scope gate — directly relevant to the linter's `out_of_scope` verdict path. |

---

## Status table — quick read

| Item | State | Notes |
|---|---|---|
| Phase-1 fixture | drafted (40/40) | `character/airton_c1/phraseology_eval.yaml`, awaiting tool to score against |
| Phase-1 tool | not started | blocked on fixture sign-off |
| Phase-1 eval subcommand | not started | blocked on tool |
| Phase-1 baseline JSON | not started | snapshot once tool MVP lands |
| Phase-1 pre-push gate | not started | mirrors `harness-zxqs` pattern |
| Phase-2 STT | deferred | gated on Phase-1 ship |
| Phase-2 streaming | deferred | gated on Phase-1 ship |

---

## Log (newest first)

### 2026-04-26 — fixture drafted (harness-h2iz, in_progress)

**What landed.** `character/airton_c1/phraseology_eval.yaml` — 40 cases,
10 per scenario class (`departure`, `arrival`, `handoff`, `emergency`),
covering 8 JO 7110.65 sections:

- §3-9-10 takeoff clearance (departure ok / wrong / incomplete)
- §3-9-4 line up and wait (departure ok / wrong / incomplete)
- §3-10-5 landing clearance (arrival ok / LAHSO / change-runway / wrong / incomplete / continue)
- §2-1-17 radio communications / contact frequency (handoff ok / wrong / incomplete)
- §5-7-2 speed adjustment methods (handoff ok / wrong / incomplete)
- §2-1-6 safety alert — terrain + traffic-conflict (emergency ok / incomplete)
- §2-1-21 traffic advisories (emergency ok)
- §10-2-6 hijacked aircraft squawk + verify (emergency ok / wrong)

Verdict distribution across the 40: `ok` 21, `incomplete` 8, `wrong` 7,
`out_of_scope` 4. Out-of-scope cases include pilot-side `MAYDAY` per
§10-1-1b (pilot phrasing, not controller), pilot finals report,
"going down to..." pilot frequency-departure, and a non-rule chitchat
sample.

**Scoring spec** (documented inline in `purpose:` block):

- `verdict_accuracy = mean(predicted_verdict == expected_verdict)`
- `citation_accuracy = mean(expected_section in predicted_citations)`
  (skipped on `out_of_scope`)
- `combined = mean(verdict_pass AND citation_pass)`
- `per_scenario` — same metrics broken out by `scenario` field
- Phase-1 ship gate: `combined ≥ 0.80`.

**Wrong/incomplete construction discipline.** Every `wrong` case is a
single-edit mutation of a verbatim canonical form (one swapped word,
one wrong code-class, one wrong unit, one wrong noun-class). Every
`incomplete` case drops one rule-required element (runway number,
call sign, frequency facility, position-of-aircraft, speed value,
hold-short target). This forces the linter to localize the mismatch
rather than just bucket on overall similarity.

**Open questions for Phase-1 tool design** (sibling bead harness-q35t):

1. Does the tool emit a single verdict, or rank top-K candidate sections
   and let the rubric collapse? Cite-or-silent on the corpus suggests
   single-verdict with a `null` fallback when no chunk clears
   `min_score`.
2. How does the tool surface `expected_phraseology` template form
   (with `(slot)` placeholders) vs. the actual canonical wording? The
   fixture stores the template form so it round-trips cleanly; tool
   probably needs to fetch the chunk and quote the PHRASEOLOGY block.
3. `out_of_scope` decision — pure cosine-floor, or does the
   `OutOfScopeHook` from harness-8dop need to land first?

**Sanity check (passed).** Direct sqlite query against the airton_c1
seed store for each cited §. Every fixture section has multiple seed
rows present:

| Section | Seed rows |
|---|---|
| §3-9-10 | 4 |
| §3-9-4 | 9 |
| §3-10-5 | 6 |
| §2-1-17 | 5 |
| §5-7-2 | 7 |
| §2-1-6 | 5 |
| §2-1-21 | 5 |
| §10-2-6 | 2 |

All ≥2 chunks; retrieval has the material to ground every case. (§10-2-6
is the smallest at 2 — short section in the corpus.) Matches the
discipline used for `atc_eval.yaml`.

**Next.** `harness-h2iz` ready to close. Unblock `harness-q35t`
(tool build) — see open questions above for design decisions to lock
before that work starts.
