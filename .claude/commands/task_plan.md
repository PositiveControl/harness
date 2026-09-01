# Task Plan

Plan one bead's implementation: explore → task file → human approval → branch. Planning only — execution is `/implement`. Pass the bead id: `/task_plan harness-5z6y`.

## Instructions

### Step 1: Fetch bead details

```bash
bd show $ARGUMENTS
bd comments $ARGUMENTS
bd dep list $ARGUMENTS
bd children $ARGUMENTS
```

**Shape check:** no acceptance criteria, more than 5 acceptance bullets, `issue_type: epic`, or open children → stop, route to `/feature_plan $ARGUMENTS`. Unshaped work never enters task planning.

### Step 2: Read the design doc

Bead has a parent (or names a design doc) → read the matching `docs/plans/*-design.md` first. Constraints, rejected alternatives, and slice boundaries are already decided there. Do not re-litigate them.

### Step 3: Explore the codebase

1. `CLAUDE.md` — invariants and the commands/gates section
2. `docs/` for the affected area; `bd search` for prior beads on the same ground
3. Read the real files — current behavior beats assumed behavior
4. Find the existing tests for the area (`tests/test_<subsystem>*.py`) — new tests join them, they don't start a new pattern
5. Check for an existing task file: `ls .llm/tasks/$ARGUMENTS*` — exists → consider `/implement $ARGUMENTS` instead

A subsystem-expert subagent (`memory-retrieval-expert`, `orchestrator-expert`, `model-adapter-expert`, `cli-typer-expert`, `persona-voice-expert`, `evals-expert`, `tui-ux-expert`, `bd-workflow-expert`) is the fast path when the bead lands inside one subsystem. Blocked on a decision the bead doesn't answer? Suggest `/segue <question>`.

### Step 4: Create the task file

Copy `.llm/tasks/task_template.md` to `.llm/tasks/<BEAD_ID>_<snake_case_slug>.md` and fill:

- **Goal** — from the bead description
- **Background / Context** — bead + comments + design doc + what exploration found
- **Requirements & Acceptance Criteria** — from the bead; in scope AND out of scope explicit
- **Next Actions** — concrete steps: files to touch, approach, test strategy
- **References** — bead id, parent, design doc, key source files

Conventions live in `CLAUDE.md`. The task file points at it; it never copies rules.

### Step 5: GATE — present the plan for approval

1. **Bead** — id, title, one-line description
2. **Approach** — the shape of the change
3. **Files to change** — each with a brief note
4. **New tests** — what coverage gets added, in which file
5. **Size forecast** — added lines vs the 100–600 target; over → propose the split now
6. **Risks / questions**
7. **Flagged decisions** — new patterns, new deps, anything touching an invariant in `CLAUDE.md` (ALWAYS needs approval)

**Wait for approval** — approve, adjust, or answer before any code.

### Step 6: Set up for implementation

After approval:

**Clean working tree:** `git status` — uncommitted changes → ask: stash or commit first.

**Branch:**
```bash
git branch --list "$ARGUMENTS-*"
```
- Exists → `git checkout <that branch>`
- Not → `git checkout main && git pull && git checkout -b $ARGUMENTS-<short-slug>`

**Claim the bead:**
```bash
bd update $ARGUMENTS --claim --json
```

`--claim` is atomic: it assigns you and sets `in_progress`. That, not the task file, is the state of record.

### Next step

```
Plan approved, bead claimed, branch ready. Run: /implement <BEAD_ID>
```

## Reference
- Task files: `.llm/tasks/<bead-id>_<slug>.md` (local scratch, gitignored)
- Template: `.llm/tasks/task_template.md`
- Branch convention: `<bead-id>-<slug>` (e.g. `harness-k18er-gate-blind-guards`)
- Sizing: 100–600 added lines per landing
- Gates: `uv run ruff check . && uv run ruff format . && uv run mypy src tests && uv run pytest`
