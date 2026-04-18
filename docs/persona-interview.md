# Persona interview template

Guided interview for creating a new persona in this harness. Start here when someone says "let's make another one."

**How to use.** Walk the interviewee section by section. Skip what doesn't apply — every section says when. Record answers inline. Revisit earlier sections freely; the template is fluid, not linear. When every **Required** question has an answer and the **Sign-off check** passes, the filled interview becomes the direct input for the character files in §13.

**Worked example.** The `airton_b` exercise (see git history around 2026-04-18) is the reference fill-in. When a question is abstract, compare with how ab answered it.

---

## 0. Meta (30 seconds)

- **Working name** (directory-safe, lowercase, underscores):
- **Alias** (optional, short):
- **Author**:
- **Date**:
- **Relationship to existing personas** (net new / variant of X / replaces Y):

---

## 1. Premise — required

One sentence. Archetype + what it holds so the user doesn't.

- **Draft premise**:

**Gate**: if the premise is longer than two sentences or has an "and also", cut scope before continuing. A persona that does two things does neither.

*Airton:* gruff ex-defense embedded engineer dragged into web work.
*ab:* personal operations lead that holds the shape of your week.

---

## 2. Identity — required

- **Pronoun** (it / they / she / he / …):
- **First-person** (I / we / none):
- **Self-awareness stance** (transparent software / in-character only / hybrid):
- **Era or archetype flavor** (optional — "1990s bare-metal", "dispatcher", "librarian"):
- **Relationship to user** (mentor / steward / peer / assistant / …):

---

## 3. Owns / doesn't own — required

The scope fence. Both halves matter; the *doesn't* list is what keeps the persona focused.

- **Owns** (3-6 bullets):
- **Doesn't own** (3-6 bullets):

**Gate**: anything in "owns" that overlaps with an existing persona → name how they divide or merge the personas. Do not ship overlap.

---

## 4. Core deliverable — required

The thing this persona produces, repeatedly, that the user keeps coming back for. One paragraph describing it, plus a realistic sample.

- **Deliverable name**:
- **When does the user ask for it?**
- **What does a good one look like?** (paste an example)
- **What would a bad one look like?** (failure mode to avoid)

*ab's deliverable:* the daily path — tiered list of today's work with reasons attached.

---

## 5. Domain model — required

What distinctions does this persona track? Usually a small set of scopes + a shared item schema.

- **Scopes** (list, 2-4):
- **First-class entities** (task / project / goal / event / …):
- **Shared item schema** (required fields across all entities):
- **Inheritance rules** (what inherits from what):
- **What's deferred for v1?**

---

## 6. Interaction surface — required

How the user drives the persona. Two halves:

### 6a. Explicit commands
| command | purpose |
|---------|---------|
|         |         |

### 6b. Implicit triggers
Phrases or patterns the persona should act on without being told.

### 6c. Capture flow (skip if persona has no capture)
- **Required fields to mark an item complete**:
- **Clarifying-q budget** (0 / 1 / until resolved):
- **Can the user `skip` optional fields?**:
- **Recursion rule** (can a new parent be created mid-capture?):

---

## 7. Voice + register — required

- **Register source**: reuse existing persona's rewriter / borrow-and-shift / net new / external (e.g., caveman plugin)
- **If borrow-and-shift**: what shifts? (less gruff, more dry, softer landings, …)
- **If external**: which package/skill, what intensity, what config knobs?
- **Auto-clarity surfaces** — when does the register relax to normal prose? (list situations)

**Per-surface intensity map** (fill when the register has intensity levels):

| surface | intensity | reason |
|---------|-----------|--------|
|         |           |        |

---

## 8. Taboos — required

4-8 never-do rules. Each must be behavioral and observable. "Never silent-capture — echo new items back" ✓. "Never be unhelpful" ✗.

1. Never …
2. Never …
3. Never …

**Gate**: every taboo maps to at least one eval fixture in §12 (positive or negative case).

---

## 9. Memory — required

- **Scope**: fully separate silo / shared facts only / shared episodic+facts / shared across the board
- **Per-user scoping**: inherits harness rule (user_id required on retrieval) — confirm yes/no
- **Implementation note** (per-character DB / character_id column / reuse default DB):
- **Seed memories** — 3-5 formative memories the persona starts with. Each one phrased as a lived experience, tagged by principle.

| id | principle | one-line summary |
|----|-----------|------------------|
|    |           |                  |

---

## 10. Learning — required

Pick from the menu; mark each as **ship with v1**, **phase 2**, or **not planned**.

- **(a) Outcome feedback** — store wins/losses against predictions, adjust priors:
- **(b) Style mirroring** — passive log user phrasing, condition rewriter:
- **(c) Policy memory** — semantic facts about user operating style, consulted before ranking/advice:
- **(d) Other** — describe:

For each marked **ship with v1**: what triggers the learning pass? (end-of-day / on capture / on close / scheduled / …)

---

## 11. Rewriter architecture — skip if §7 says "reuse existing"

- **New class name**:
- **Lives at**:
- **Wraps**: (base adapter / existing persona adapter / …)
- **On tool turns**: rewriter on or off? (Airton default: off; reason: rewriter compresses, wrong for investigate replies)
- **Config knobs exposed to user** (intensity, per-surface overrides, …):

---

## 12. Eval fixtures — required

Minimum 20 gold prompts + expected responses, shipped before the persona is considered ready. Categories map to §6 surfaces + §8 taboos + §7 auto-clarity cases.

| count | category | notes |
|-------|----------|-------|
|       | core deliverable outputs (varying load) |  |
|       | capture / acquisition dialogues |  |
|       | re-plan or update notices |  |
|       | drift / nudge |  |
|       | auto-clarity kicks in (destructive confirm, multi-step) |  |
|       | status / rollup |  |
|       | ambiguous-ask clarifying responses |  |
|       | register stress tests (does register survive edge cases?) |  |
| ≥20   | **total**                             |  |

Fixtures go in `character/<name>/voice/canonical.yaml` in the same shape as `character/airton/voice/canonical.yaml`.

---

## 13. Files to create — checklist

Copy into the PR / ticket when the interview is signed off.

```
character/<name>/
  core.yaml              # from §1, §2, §3, §5, §8
  constitution.md        # from §7, §8, §9-authorization, §10
  voice/canonical.yaml   # from §12 (≥20 samples)
  voice/captured.yaml    # empty, grows via /capture
  seed_memories/*.md     # from §9
  router_eval.yaml       # from §6a (commands → tool names)
```

Code changes (only if §11 says net-new rewriter):
```
src/harness/persona/<name>_rewriter.py
  - class, tests, wiring into CLI flags
```

Memory changes (only if §9 says separate DB):
```
- per-character DB path resolution in store/*.py
- migration note for existing data
```

---

## 14. Sign-off check

Before leaving the interview:

- [ ] Premise fits in one sentence.
- [ ] Owns / doesn't own divides cleanly from existing personas.
- [ ] Core deliverable has a realistic sample.
- [ ] Every taboo maps to ≥1 eval fixture.
- [ ] ≥20 eval fixtures sketched (names + categories; gold text can come later).
- [ ] Memory scope decided, implementation path named.
- [ ] Learning choices marked v1 / phase 2 / not planned.
- [ ] Register integration path is concrete (reuse / shift / external / new).
- [ ] File checklist in §13 copied into issue tracker.

If any box is unchecked → go back and fix, not forward and commit.

---

## Appendix — conversational flow

When running the interview live (vs. filling async), this order reads naturally:

1. §1 premise — 90 seconds, no more.
2. §3 owns/doesn't own — before anything else. If this isn't tight, nothing downstream will be.
3. §4 core deliverable — what does this persona *produce*?
4. §5 domain model — only the distinctions the deliverable needs.
5. §6 interaction surface — how the user drives it.
6. §7 voice — often the most fun, deliberately late.
7. §8 taboos — usually easier after voice is decided.
8. §9, §10, §11 — plumbing. Can be async.
9. §12 eval fixtures — drafted last but *before* any code.
10. §14 sign-off.

Skip forward and back. The template is a map, not a script.
