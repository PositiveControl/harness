# Implement

Execute an approved task plan. Idempotent — the first run after `/task_plan` approval and the tenth resume after a lost session are the same command. Pass the bead id, or omit to infer from the branch: `/implement harness-5z6y` or `/implement`.

## Instructions

### Step 1: Resolve the bead

1. `$ARGUMENTS` given → use it
2. Otherwise parse the branch name (`<bead-id>-<slug>`): `git rev-parse --abbrev-ref HEAD`
3. Neither works → ask, or suggest `/pick`

### Step 2: Load state

1. Read `.llm/tasks/<bead-id>_*.md` — goal, acceptance criteria, Next Actions, progress log. **No task file → stop, run `/task_plan <bead-id>` first.** Never implement without an approved plan.
2. `bd show <bead-id>` — status, comments, notes added since planning.
3. Verify the branch. Not on `<bead-id>-*` → check it out. Then `git status` and `git log main..HEAD --oneline` — what already landed.
4. Cross-check the progress log against actual commits. The log lags; commits are truth.

### Step 3: Orient

Three lines before touching code: what's done, what's in flight, what's next. Plan and code disagree → say so and resolve before continuing.

### Step 4: Work loop

Work the Next Actions in order. Per logical unit:

1. Implement per the plan
2. Write tests alongside — real SQLite in `tmp_path`, no mocking the stores (`CLAUDE.md`)
3. Run the affected tests — green before moving on:
   ```bash
   uv run pytest tests/test_<file>.py -q
   ```
4. Lint + type the touched files before committing:
   ```bash
   uv run ruff check . && uv run ruff format . && uv run mypy src tests
   ```
5. Commit — one logical unit, message explains *why* ("scribe: serialize concurrent runs per session — two chats double-wrote candidates", not "update scribe")
6. Append one dated bullet to the task file's progress log

Rules in force:

- **Conventions**: `CLAUDE.md` is the single source. Don't restate its rules in the task file.
- **Invariants**: no model SDK imported outside `src/harness/model/`; character rules stay in `character/<name>/`, not `src/`; `user_id=speaker` on every user-facing `.search()`; stores stay append-first and attributed. Breaking one is a stop-and-ask, not a judgment call.
- **No blanket suppressions**: a failing check gets fixed, or gets a targeted per-file ignore with a comment saying why. Never a bare `# noqa` / `# type: ignore`.
- **Scope escape**: forecast passes ~600 added lines, or new acceptance criteria surface → STOP. File a child bead (`bd create ... --deps parent-child:<epic>` or `discovered-from:<this bead>`), note it in the task file, land the current slice clean.
- **Segue valve**: rabbit hole, plan contradiction, theory-war debugging → `/segue <question>` instead of burning the session.
- Remove debugging scaffolding before you finish.

Test suite going sideways in ways the change doesn't explain → `/test_fix`. Ten daemon/plan tests failing on their own → `bd dolt start` isn't running; that's environment, not a regression.

### Step 5: Done check

All acceptance criteria met, full gates green, tree committed:

```
All acceptance criteria met. Run: /pr_submit <BEAD_ID>
```

Criteria left → keep looping or name the blocker. Blocked on something external → `bd update <bead-id> --status blocked` and record the blocker in a bead comment, so the state survives the session.

## Reference
- Task files: `.llm/tasks/<bead-id>_<slug>.md`
- Branch convention: `<bead-id>-<slug>`
- Single test: `uv run pytest tests/test_character.py::test_load_airton_shape`
- Full gates: `uv run ruff check . && uv run ruff format . && uv run mypy src tests && uv run pytest`
- Sizing: 100–600 added lines per landing
