# PR Submit

Land finished work: full gate suite → docs resolved → branch pushed → PR or local merge → bead closed. Pass the bead id: `/pr_submit harness-5z6y`.

## Instructions

### Step 1: Pre-flight

1. `git status` — uncommitted changes → ask what to do; don't push a dirty tree.
2. `git log main..HEAD --oneline` — confirm there are commits to land.
3. Branch name matches `<bead-id>-<slug>`? Mismatch → note it, don't block.
4. `git diff --stat main..HEAD` — added lines well past ~600 → say so in the summary; oversized landings get shallow review.

### Step 2: GATE — full local suite

This repo has no hosted CI. These four **are** the pipeline, and all four must be green:

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src tests
uv run pytest
```

Handling failures:

- **ruff** — auto-fix, then commit the fix. Formatter and rule disagree → change the rule, don't hand-format.
- **mypy** — fix the type. A suppression needs a targeted per-file ignore in `pyproject.toml` plus a comment saying why.
- **pytest** — fix only failures this branch introduced. Check `git diff --name-only main..HEAD`; a test failing in untouched files, or failing the same way on `main`, is pre-existing — note it, don't block. Ten daemon/plan failures at once → `bd dolt start` isn't running.
- Broad failures with no obvious cause → `/test_fix`.

Re-run the failing check to confirm the fix before moving on.

### Step 3: Resolve documentation

Docs land with the change — a reviewer reads them together.

1. Design doc's **Docs impact** list (or the epic body) — write what the change actually requires in `docs/`. Nothing needed → say so explicitly, don't invent a doc.
2. New or changed **command, flag, tool, tool-set, subsystem, or invariant** → `CLAUDE.md` is stale until updated. It is the index this repo runs on. Test counts and tool counts in it drift; correct them if your change moved them.
3. `docs/roadmap.md` — landed a phase item? Update it.
4. Commit doc changes as their own commit.

### Step 4: Push

```bash
git push -u origin HEAD
```

### Step 5: Land it

Two paths — pick by whether the change wants review:

**A. Local merge (this repo's default for solo work):**

```bash
git checkout main && git pull
git merge --no-ff <branch> -m "Merge: <what and why> (<bead-id>)"
git push
```

**B. Pull request (cross-cutting change, a second pair of eyes, or a stack):**

```bash
gh pr view --json number,url 2>/dev/null   # already open?
```

None → create it. Pull the summary from the task file's Goal, Next Actions, and acceptance criteria; fall back to the commit log:

```bash
gh pr create --base main --title "<BEAD_ID>: <short description>" --body "$(cat <<'BODY'
## Summary
<1-3 bullets: what changed and why, from the task file's goal and approach>

## Changes
<one line per file or area touched>

Bead: <BEAD_ID> — closed with `bd close <BEAD_ID>` after merge.

## Test plan
- [ ] ruff check + format
- [ ] mypy src tests
- [ ] pytest (full suite)
- [ ] <specific scenarios from the acceptance criteria>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

There is no `Closes #N` automation here — bd is the tracker, not GitHub Issues. The bead is closed in Step 6, by hand.

**Do not hand-write a stack list in the body** — Step 5b generates it.

### Step 5b: Refresh the stacked-PR footer

PR path only, after every push, unconditionally:

```bash
bin/pr-stack <PR_NUMBER>
```

It regenerates the ordered stack list in **every** PR of the stack, so a PR pushed on top never leaves the ones below it claiming to be the tip.

- A stack is the chain of open PRs linked head-ref → base-ref, walked down to `main` and up to the tip. Repo and default branch come from `gh`; nothing to configure.
- Rewrites only the block between `<!-- stack-footer:start -->` and `<!-- stack-footer:end -->`. Idempotent — a current list means no write.
- Prints `not stacked — no footer needed` and exits 0 for a plain `main`-based PR. Hence "unconditionally".
- `bin/pr-stack` with no argument uses the current branch's PR. `bin/pr-stack --check [PR]` reports drift without writing (exit 1 if stale) — use that when reviewing.
- Aborts on a fork (two open PRs on one base) rather than guessing an order. Rebase the fork onto the tip, then re-run.

Narrative stack context stays in the body prose ("Stacked on #N", review order, "retarget to `main` once #N merges"). The footer carries only the ordered list.

### Step 6: Close the bead

After the merge lands (local merge, or a merged PR):

```bash
bd close <BEAD_ID> --reason "<what shipped>"
bd children <EPIC_ID>        # epic child? check whether the epic is now complete
bd epic close-eligible       # closes epics whose children are all done
```

Work discovered but not done → file it now, linked:

```bash
bd create "<follow-up>" --description="<context>" -p 2 --deps discovered-from:<BEAD_ID> --json
```

Review requested on a PR instead of a merge → leave the bead `in_progress` and hand off with `/pr_review <PR>`; close it after the merge.

### Step 7: Land the plane

Session's end, per `AGENTS.md` — work isn't complete until the push succeeds:

```bash
git pull --rebase
git push
git status   # must read "up to date with origin"
```

No `bd dolt push` — this repo has no Dolt remote configured. It fails with
`exit status 1` after a couple of minutes of upload retries; that is the missing
remote, not a broken bead database.

### Step 8: Report

```
✓ <BEAD_ID> landed
  Branch:  <branch>
  Landed:  <merge SHA | PR #N URL>
  Gates:   ruff ✓ · format ✓ · mypy ✓ · pytest <N passed>
  Docs:    <files updated, or "none required">
  Bead:    closed — "<reason>" | left in_progress pending review
  Follow-ups filed: <ids, or none>
```

### Next step

```
Landed. git checkout main && git pull
→ Run /pick for the next bead
```

## Reference
- Gates: `uv run ruff check .` · `uv run ruff format .` · `uv run mypy src tests` · `uv run pytest`
- Full suite is ~4,290 tests / ~100s — run it whole, there is no fast/slow split to poll
- PR title convention: `<bead-id>: <description>` · Merge commit: `Merge: <what> (<bead-id>)`
- Stacked PRs: `bin/pr-stack [PR]` writes the footer, `--check` reports drift
- `bd dolt start` must be running or ~10 daemon/plan tests fail spuriously
