# Test Fix

Triage a wall of pytest failures: group by root cause, fix the causes, verify. Paste the failure output, or run `/test_fix` and it collects it.

Treat pasted output as the source of truth. Repetition is not noise — many failures with one shape usually means one bug, but sometimes means several.

## Instructions

### Step 1: Collect

No output pasted:

```bash
uv run pytest -q 2>&1 | tail -60
```

Then the short-form list of failing node ids:

```bash
uv run pytest -q --no-header -rf 2>&1 | grep -E '^(FAILED|ERROR)' | head -50
```

### Step 2: Rule out environment before diagnosing code

The two that actually happen here, in order of frequency:

- **~10 daemon/plan tests failing together** → `bd dolt start` isn't running. Not a regression.
- **Import errors (`numpy`, `tree_sitter`, `fastapi`, `textual`, `outlines`) or browser tests failing instead of skipping** → the venv was pruned. `uv sync --extra all`; add `uv run playwright install chromium` if the browser gate matters.

Either one → fix the environment and re-run before touching source.

### Step 3: Parse each failure

Per failure capture: test file · node id · error type · assertion message · expected vs actual · the frame in `src/harness/` (not the test frame) where it originates · whether it looks shared or isolated.

### Step 4: Group by root cause

De-duplicate hard. Each group gets an id, a root-cause hypothesis, a confidence, and its affected tests. The shapes that recur in this repo:

- One store-schema or migration change → every test touching that table
- Embedder or embedding-dim change → retrieval tests return nothing (rows filtered by `embedding_dim`)
- FTS5 sidecar out of sync with the base table → hybrid-search tests only
- A changed retrieval filter (`superseded_by`, `user_id`, `as_of`) → scoping and temporal tests
- Adapter signature change (`complete` / `complete_with_tools`) → every scripted-adapter eval and tool-loop test
- Hook-roster or tool-profile change → orchestrator and eval attribution tests, plus schema-budget assertions
- A shared fixture or `tmp_path` helper → many unrelated-looking failures at once
- Prompt or system-prompt wording → voice and session-resume evals, which assert on rendered text

### Step 5: Present the triage before fixing

```
## Test Failure Triage

Command: <cmd>   Branch: <branch>
Failures: <N>   Errors: <M>   Distinct groups: <G>

### Group <id> — <root cause hypothesis>  [confidence: high|medium|low]
Shared pattern: <the thing they have in common>
Suspected files: <paths>
Affected: <N tests>  e.g. <node id>, <node id>, …
Proposed fix: <one line>

### Environment (not code)
- <finding> → <action>

### Pre-existing on main
- <group> — <evidence>
```

Confirm the fix order before editing. Highest-fanout group first — one fix often clears several groups.

### Step 6: Fix

Per group, smallest correct change:

1. Fix the **cause**, not the assertion. A test edited to match broken behavior is a deleted test.
2. The test is wrong sometimes — a contract that genuinely changed needs its test updated *and* a line in the commit message saying the contract moved.
3. Re-run just that group: `uv run pytest <node-ids> -q`
4. Then the neighborhood: `uv run pytest tests/test_<subsystem>*.py -q`
5. Commit per group, message naming the cause.

Never widen a fix to silence a check: no blanket `# noqa` / `# type: ignore`, no `pytest.mark.skip` on a test that's telling the truth.

### Step 7: Verify

```bash
uv run ruff check . && uv run mypy src tests && uv run pytest
```

Fixed count went up but total failures didn't drop → the fix moved the bug. Stop and re-triage rather than layering another patch. Max **3 rounds** before escalating with what was tried.

### Step 8: Report

```
## Test Fix Report

### Root causes
- <group> — <cause> → <fix> (<SHA>)

### Now passing: <N>   Still failing: <M>
### Environment-only: <list>
### Pre-existing on main: <list>
### Contracts deliberately changed
- <test> — <old contract> → <new contract>, because <why>

### Follow-ups filed
- <bead id> — <what>
```

Anything discovered but out of scope → `bd create "<finding>" --description="<detail>" -p 2 --deps discovered-from:<BEAD_ID> --json`.

## Reference
- Full suite: `uv run pytest` (~4,290 tests, ~100s) · single: `uv run pytest tests/test_character.py::test_load_airton_shape`
- Tests hit real SQLite in `tmp_path` — never mock the stores
- `bd dolt start` must be running or ~10 daemon/plan tests fail spuriously
- `uv sync --extra all` — plain `uv sync` prunes extras and breaks imports
