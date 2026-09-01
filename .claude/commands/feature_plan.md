# Feature Plan

Plan a feature: explore → design doc → human approval → sized child beads. Pass an epic bead id or a problem statement: `/feature_plan harness-v9hs` or `/feature_plan "weather tool for aviation characters"`.

## Instructions

### Step 1: Gather context

`$ARGUMENTS` looks like a bead id → fetch it with structure and discussion:

```bash
bd show $ARGUMENTS
bd comments $ARGUMENTS
bd children $ARGUMENTS
bd dep list $ARGUMENTS
```

Otherwise treat `$ARGUMENTS` as the problem statement.

### Step 2: Explore the codebase (high altitude)

1. `CLAUDE.md` — the load-bearing invariants (adapter boundary, user scoping, character-data-not-code, append-first stores). A design that breaks one of those is dead on arrival.
2. `docs/` — existing design docs (`docs/plans/`), architecture notes (`docs/architecture/`), the roadmap (`docs/roadmap.md`).
3. `bd search "<keywords>"` — prior beads on the same ground, open or closed. Closed ones carry decisions.
4. Identify the modules the feature touches and which subsystem owns them (`model/`, `store/`, `tools/`, `orchestrator/`, `driver/`, `web/`, `cli*.py`).
5. Note the patterns to reuse and the constraints (schema migrations on SQLite stores, tool schema budget per profile, strict mypy).

Shape-level understanding, not line-level planning — that happens per-bead in `/task_plan`. Blocked on a genuine design question? Suggest `/segue <question>`.

### Step 3: Write the design doc

Create `docs/plans/YYYY-MM-DD-<slug>-design.md`:

- **Problem** — what and for whom
- **Constraints** — invariants from `CLAUDE.md`, prior decisions, hardware limits (local MLX on M4 Pro 48 GB)
- **Approach** — chosen shape at module level
- **Rejected alternatives** — and why (stops re-litigation later)
- **Open questions** — settle before filing beads, or mark explicitly deferred
- **Decomposition** — proposed vertical slices (store → service → tool/CLI surface → eval → docs, as applicable)
- **Docs impact** — which `docs/` files and which `CLAUDE.md` sections need updating

### Step 4: GATE — design approval

Present a short summary: problem, approach, slice list with size estimates, open questions. **Wait for explicit approval.** The doc is a proposal; beads are commitment. Iterate until approved.

### Step 5: File beads

After approval:

1. Single-slice feature → one bead, no epic
2. Multi-slice → epic + one child bead per slice

Each child bead needs:
- Goal (1–2 sentences) + path to the design doc
- Acceptance criteria — **≤5 testable bullets** (more → split the slice)
- Size forecast targeting **100–600 added lines** per landing
- Sibling ordering noted in the body

```bash
bd create "<title>" --description="<body>" -t task -p 2 --deps parent-child:<EPIC_ID> --json
```

**Children must be DIRECT epic children** — `parent-child:<epic>`, never a dep chain off a sibling. A chained sub-bead never surfaces in `bd ready`, and `harness drive` never sees it.

Epic itself:

```bash
bd create "[epic] <title>" --description="<body + design doc path>" -t epic -p 2 --json
```

Verify the shape before moving on:

```bash
bd children <EPIC_ID>
bd ready --json | python3 -c "import sys,json;print([i['id'] for i in json.load(sys.stdin)])"
```

Every leaf you expect to work next must appear in `bd ready`. If one doesn't, its deps are wrong — fix them now.

### Step 6: Docs impact

Per the design doc's **Docs impact** section, note the files to write at landing time. Don't create empty placeholder docs — record the list in the epic body so `/pr_submit` can check it off. `CLAUDE.md` updates for new commands, tools, or subsystems are part of the last slice, not an afterthought.

### Next step

```
Ready to start? Run: /pick   (or /task_plan <first-child-bead>)
```

Epic of mechanical, gate-checkable leaves? It may be drive-shaped instead: prepare `harness drive plan --epic <EPIC_ID>` and hand the command over. Drives are launched by hand, never by you.

## Reference
- Design docs: `docs/plans/YYYY-MM-DD-<slug>-design.md`
- Architecture docs: `docs/architecture/`
- Sizing: 100–600 added lines per landing; acceptance criteria ≤5 bullets per bead
- Priorities: P0 critical → P4 backlog. Types: `bug`, `feature`, `task`, `epic`, `chore`
- Full bd protocol: `AGENTS.md`
