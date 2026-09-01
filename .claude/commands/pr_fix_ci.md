# Fix CI

Diagnose and fix pipeline failures for a branch or PR. Pass a PR number or nothing (uses the current branch): `/pr_fix_ci 2` or `/pr_fix_ci`.

**This repo has no hosted CI.** The four local gates are the pipeline, so that path comes first. The hosted path below applies only if a GitHub Actions workflow exists (`ls .github/workflows`) — check before assuming there are checks to poll.

## Instructions

### Step 1: Establish which pipeline

```bash
ls .github/workflows 2>/dev/null   # empty/missing → local gates only
git rev-parse --abbrev-ref HEAD
gh pr checks $ARGUMENTS --json name,state,link,workflow 2>/dev/null   # hosted path only
```

PR given and you're not on its branch → `git checkout <headRefName> && git pull`.

### Step 2: Run the gates and collect failures

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -q
```

Run all four before diagnosing anything — one root cause often trips several.

### Step 3: Classify

1. **Branch-introduced** — caused by code on this branch. Fix these.
2. **Environment** — not the code. The two that actually happen here:
   - ~10 daemon/plan tests failing together → `bd dolt start` isn't running
   - Browser-backed tests failing rather than skipping, or import errors from `numpy` / `tree_sitter` / `fastapi` → the venv was pruned. `uv sync --extra all`, then `uv run playwright install chromium` if the browser gate is in play.
3. **Pre-existing** — fails the same way on `main`. Confirm before believing it:
   ```bash
   git stash && git checkout main && uv run pytest <path> -q; git checkout - && git stash pop
   ```

### Step 4: Present the diagnosis before changing anything

```
## <branch | PR #N> Gate Diagnosis

### Failing: <count>

#### Branch-introduced (fixable)
- <gate/test> — <root cause>
  Fix: <proposal>

#### Environment
- <gate/test> — <cause>
  Action: <bd dolt start | uv sync --extra all | playwright install chromium>

#### Pre-existing (note, don't block)
- <gate/test> — <evidence it predates the branch>

### Passing
- <gate>: ✓
```

Ask: proceed with the proposed fixes? anything to handle differently?

### Step 5: Fix by category

**ruff check** — `uv run ruff check --fix .` for the mechanical ones; fix the rest by hand. Never a blanket `# noqa`; a targeted per-file ignore in `pyproject.toml` with a comment, or a real fix.

**ruff format** — `uv run ruff format .`. Layout belongs to the formatter. Formatter and rule disagree → change the rule.

**mypy** — fix the type. Strict mode also flags unused ignores and unreachable code, so a mypy error can be a *stale* suppression, not a new type problem. New third-party import with no stubs → add the targeted `[[tool.mypy.overrides]]` entry, not a global loosening.

**pytest** — read the assertion, not just the summary. Then:
- Is the failing test in a file this branch touched? `git diff --name-only main..HEAD`
- Does it fail on `main` too (Step 3)?
- Store-shaped failure (embedding dim mismatch, superseded rows leaking into retrieval, FTS5 out of sync) → check the retrieval filters, not the test.
- Many failures with one shape → `/test_fix` groups them by root cause.

**Hosted checks (only if workflows exist):**
```bash
gh run view <RUN_ID> --job <JOB_ID> --log-failed
gh run rerun <RUN_ID> --failed        # infrastructure-only failures
```
Check URLs are `https://github.com/<org>/<repo>/actions/runs/<RUN_ID>/job/<JOB_ID>`. `state` values: PENDING, IN_PROGRESS, SUCCESS, FAILURE, SKIPPED — there is no `conclusion` field.

### Step 6: Commit and push

```bash
git add <files>
git commit -m "$(cat <<'MSG'
<what was fixed and why>

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
git push
```

### Step 7: Re-verify

Re-run the full gate set. A fix that turns one failure green and another red is not done. Maximum **3 fix iterations** before escalating to the user with what was tried.

### Step 8: Report

```
## <branch | PR #N> Gate Resolution

### Status: <all green / partially resolved / needs attention>

### Resolved
- <gate>: <cause> → <fix> (<SHA>)

### Environment
- <gate>: <action taken>

### Pre-existing / ignored
- <gate>: <why>

### Still failing
- <gate>: <what was tried> — needs manual investigation

### Commits
- <SHA> — <message>
```

## Reference
- Gates: `uv run ruff check .` · `uv run ruff format .` · `uv run mypy src tests` · `uv run pytest`
- Sync: `uv sync --extra all` — plain `uv sync` PRUNES omitted extras and breaks mypy/pytest/browser runs
- ~10 daemon/plan failures at once → `bd dolt start`
- Browser tests skip without chromium: `uv run playwright install chromium`
- Max 3 fix iterations before escalating
