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
| 1 | [`harness-h2iz`](#) | **closed** | Mark | Build `phraseology_eval.yaml` (40 cases, ≥10 per scenario class). Locks eval contract. |
| 2 | [`harness-q35t`](#) | **in_progress** | Mark | `src/harness/tools/phraseology_lint.py` ToolSpec. Cite-or-silent. New `phraseology` tool profile. CLI subcommand `harness phraseology lint <utterance>`. Gate ≥80% combined accuracy on fixture (gate-validation requires harness-15dy). |
| 3 | [`harness-15dy`](#) | **in_progress** | Mark | `harness eval phraseology` Typer subcommand + per-scenario breakdown + `--compare-baseline` against `phraseology_baseline.json` + pre-push hook gating regressions on tool / fixture / corpus paths. |

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
| Phase-1 fixture | shipped (40/40) | `character/airton_c1/phraseology_eval.yaml`, sanity-checked against seed store |
| Phase-1 tool | shipped | `src/harness/tools/phraseology_lint.py`, `PhraseologyLintTool` + `lint_utterance()`, 14 unit tests |
| Phase-1 CLI subcommand (lint) | shipped | `harness phraseology lint <utterance>` (and `--json`); echo-adapter smoke confirms cite-or-silent gate fires |
| Phase-1 tool profile | shipped | `phraseology` profile in `tools/profiles.py`; inherits atc's rulebook-first `search_memory` framing |
| Phase-1 eval subcommand | shipped | `harness eval phraseology` with `--scenario` / `--save-baseline` / `--compare-baseline` / `--regression-budget`; 22 unit tests on `evals/phraseology.py` |
| Phase-1 baseline JSON | shipped | `character/airton_c1/phraseology_baseline.json` snapshotted at 12/40 = 30% combined; relative-path-portable |
| Phase-1 pre-push gate | shipped | `scripts/phraseology_gate.sh` + `.pre-commit-config.yaml` entry; fires on changes to lint tool, eval scorer, or fixture |
| Phase-1 product gate (≥80%) | **not cleared** | current run 30% — rubric / retrieval gaps documented under "Failure analysis" below; q35t stays in_progress until rubric/retrieval lift the bar |
| Phase-2 STT | deferred | gated on Phase-1 ship |
| Phase-2 streaming | deferred | gated on Phase-1 ship |

---

## Failure analysis (Phase-1 baseline run, 2026-04-27)

Combined accuracy 30% on 40 cases (verdict 40%, citation 65%). Per-
scenario: departure 50% · arrival 20% · emergency 30% · handoff 20%.
Three failure clusters that explain ~all of the gap:

1. **Phonetic-spelled slot values read as missing.** When the utterance
   spells the slot value ("RUNWAY TWO SEVEN, LINE UP AND WAIT") the
   model frequently calls it `incomplete` ("missing runway number")
   even though the slot is filled. Hits canonical-form cases like
   `dep_luaw_canonical`, `dep_luaw_intersection_canonical`,
   `arr_landing_change_runway_canonical`. Fix surface: tighten the
   lint prompt (explicit examples that "TWO SEVEN" is a runway number,
   "JULIETT" is a taxiway designator), or post-rewrite the utterance
   to the digit form before retrieval.

2. **Single-word swaps too lenient.** Wrong-predicate variants like
   "CLEARED TO TAKEOFF" pass as `ok` instead of flagging `wrong`.
   Same for "REDUCE SPEED TO 250 MPH" (should be `wrong` for unit
   error, comes back `ok`). Fix surface: lint prompt rules need to
   explicitly forbid preposition / unit / verb substitutions.

3. **Retrieval misroutes some short transmissions.** "SQUAWK SEVEN
   FIVE ZERO ZERO" retrieves §2-4-17 (Numbers Usage — keyword overlap
   on the spelled digits) instead of §10-2-6 (Hijacked Aircraft).
   "REDUCE SPEED TO TWO FIVE ZERO" retrieves §2-4-17 instead of
   §5-7-2. Fix surface: use the `scenario_hint` more aggressively in
   the retrieval query (right now it appends as bracket text — could
   be a re-rank signal), or query expansion that maps phrase → §
   class. Adjacent to harness-7wju Phase-1.6 hardening lanes.

These belong to harness-q35t (rubric / retrieval) and harness-7wju
(Phase-1.6 hardening) — not 15dy. The 15dy infrastructure ships at
30%; future improvements ratchet the baseline up.

## Log (newest first)

### 2026-04-27 — eval subcommand + baseline + pre-push gate (harness-15dy, in_progress)

**What landed.**

- `src/harness/evals/phraseology.py` — fixture loader, scorer, baseline
  comparator. Mirrors `evals/atc_retrieval.py` shape: `LintFn`
  protocol, `PhraseologyEvalResult` with `verdict_accuracy` /
  `citation_accuracy` / `combined_accuracy` aggregates, per-scenario
  breakdown, `BaselineComparison` with per-case pass-flip detection +
  aggregate-accuracy regression.
- `src/harness/cli.py` — new `eval phraseology` subcommand. Flags:
  `--scenario` filter, `--save-baseline` / `--compare-baseline`,
  `--baseline-path`, `--regression-budget`, `--json`. Renders per-case
  human table + per-scenario summary; `--json` envelope is what
  `--save-baseline` snapshots. Fixture path serialized relative to
  repo root for portability across machines.
- `tests/test_eval_phraseology.py` — 22 unit tests across fixture
  loader (oos null-citation invariant, scalar field validation,
  expected_verdict enum check), pure scoring, per-scenario breakdown,
  baseline comparator (per-case pass flips, aggregate regressions,
  regression budget tolerance, new/dropped cases, missing-aggregate
  field tolerance).
- `tests/test_cli_introspect.py` — pin updated for `eval phraseology`.
- `scripts/phraseology_gate.sh` + `.pre-commit-config.yaml` —
  pre-push gate fires on changes to lint tool, eval scorer, or
  fixture YAML. Exit 0 when no baseline (brand-new clones); exit 1
  when comparator detects regression.
- `character/airton_c1/phraseology_baseline.json` — first snapshot.
  Combined 30% (12/40), verdict 40%, citation 65%. Below the 80%
  q35t product gate; future work ratchets it up.

**Gate verification.** Pre-push hook config validated. Real run
against the same fixture deferred (3-4 min on MLX) — comparator unit
tests cover the no-regression / per-case-flip / aggregate-drop paths
with deterministic fixtures.

**Quality gates.** ruff, ruff format, mypy (197 source files), pytest
(1,676 + 22 = 1,698 passing) all green.

**Bead status.** `harness-15dy` substantial work landed:
✓ subcommand exists, ✓ baseline JSON committed, ✓ comparator + tests,
✓ pre-push gate wired. Live `--compare-baseline` real-MLX verification
is the one remaining check — leave bead `in_progress` until that run
goes green.

**Next.**
1. (15dy close) Run `harness eval phraseology --compare-baseline` against
   the just-snapshotted baseline to confirm self-comparison clears.
2. (q35t close) Lift combined accuracy past 80% via the three failure
   clusters above (rubric + retrieval). Hand off to harness-7wju
   children where they overlap (cite-grounding actuator, OutOfScopeHook,
   query expansion).

### 2026-04-27 — tool + CLI subcommand (harness-q35t, in_progress)

**What landed.**

- `src/harness/tools/phraseology_lint.py` — Phase-1 tool surface:
  - `PhraseologyVerdict` dataclass — five-field structured output
    matching the fixture row shape.
  - `lint_utterance(utterance, *, adapter, episodic_store, ...)` —
    pure function the CLI subcommand and (future) eval harness call
    directly. Two cite-or-silent gates (pre-model on empty retrieval,
    post-model on ungrounded citation).
  - `PhraseologyLintTool` — chat-tier wrapper. JSON-encoded verdict in
    `output`, the section in `citations_grounded`, one `ToolHit` so
    the audit log can read the verdict without reparsing.
- `src/harness/tools/profiles.py` — new `phraseology` profile pairing
  `phraseology_lint` with `search_memory` + `introspect`. Inherits the
  atc rulebook-first `search_memory` description override.
- `src/harness/cli.py` — new `phraseology` Typer sub-app + `lint`
  command. Gates non-atc-family characters at the entry point so the
  prompt and grounding rules don't drift onto unrelated corpora.
- `tests/test_phraseology_lint.py` — 14 unit tests covering: pre-model
  cite-or-silent gate (empty store ⇒ no model call), happy-path ok /
  wrong / incomplete verdicts, post-model cite-grounding gate
  (ungrounded section ⇒ downgrade to oos), unparseable model output ⇒
  diagnostic oos, OOS verdict normalization (null citation fields),
  scenario hint plumbing, tool wrapper contract.

**Design decisions** (the open questions from yesterday's log):

1. *Single-verdict output vs. top-K candidates.* Single verdict +
   structured five-field payload. Top-K candidates are an implementation
   detail of the prompt; the model sees the top-3 chunks (cap 800
   chars each) as labeled candidates and picks one section.
2. *Template form vs. quoted PHRASEOLOGY block.* Both. Tool emits
   `expected_phraseology` (template with `(slot)` placeholders, mirrors
   fixture) AND `citation_quote` (verbatim chunk excerpt). Fixture
   scoring uses the template form; the quote is for human-readable
   output.
3. *OOS decision.* Pre-model gate handles `len(hits) == 0` (no
   candidate sections). Post-model gate handles model-picks-an-
   ungrounded-section. Cosine-floor / `OutOfScopeHook` (harness-8dop)
   is a Phase-1.6 hardening lane, not blocking Phase-1; the tool's
   two gates already enforce cite-or-silent on the substrate it has.

**Cite-or-silent verified.** Smoke run with the echo adapter (which
returns the prompt verbatim, never JSON) on a real airton_c1 store:

```
$ HARNESS_CHARACTER_NAME=airton_c1 uv run harness phraseology lint \\
    --model echo --json "RUNWAY TWO SEVEN, CLEARED FOR TAKEOFF."
{"verdict": "out_of_scope", "expected_section": null, ...,
 "mismatch": "model output unparseable"}
```

Refused without claiming a section — exactly the cite-or-silent
discipline.

**Quality gates.** `ruff check`, `ruff format`, `mypy src tests`,
`pytest` (1,655 passed) all green. Updated CLI command pin
(`tests/test_cli_introspect.py`) to include `phraseology lint`.

**Gate-validation status.** The bead's "≥80% combined accuracy on the
fixture" gate is not yet run — that requires harness-15dy (the eval
subcommand) so the loop is built once, not twice. Bead stays
`in_progress`. Tool + CLI + tests are the substantial deliverable;
gate-clear is one `eval phraseology` run away once 15dy lands.

**Next.** Pick up harness-15dy: `harness eval phraseology` subcommand
modeled on `harness eval atc`, loops fixture cases through
`lint_utterance()`, scores `verdict_accuracy` + `citation_accuracy` +
`combined`, emits `--json` envelope, supports `--compare-baseline`.
Snapshot baseline + wire pre-push hook once gate clears.

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
