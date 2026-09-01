# PR Review

Review a pull request or a branch's diff, then post or present the feedback. Pass a PR number or a branch: `/pr_review 2` or `/pr_review harness-k18er-gate-blind-guards`.

## Instructions

### Step 1: Fetch context

PR number:

```bash
gh pr view $ARGUMENTS --json number,title,body,author,baseRefName,headRefName,additions,deletions,changedFiles,reviews,comments
```

Branch (this repo lands most work by local merge, so this is the common case):

```bash
git log main..$ARGUMENTS --oneline
git diff --stat main...$ARGUMENTS
```

Summarize: title/branch, base ← head, lines changed, file count.

Base ref isn't `main` → the PR sits in a stack:

```bash
bin/pr-stack --check $ARGUMENTS
```

Exit 1 means a member's body shows a stale order. Don't rewrite the author's bodies mid-review — note it under Nitpicks ("stack footer stale — run `bin/pr-stack`") and trust the chain the script prints. Review the diff against the PR's **own base**, not `main`, or you'll flag code a lower PR already introduced.

### Step 2: Fetch the diff

```bash
gh pr diff $ARGUMENTS            # PR
git diff main...$ARGUMENTS       # branch
```

Very large (50+ files or 2000+ lines) → say so and review the load-bearing files first: stores, adapters, the orchestrator loop, hook pipeline, tool definitions.

### Step 3: Read full files, not just hunks

```bash
gh pr diff $ARGUMENTS --name-only
```

Read each non-trivial changed file whole. A diff without surrounding context yields shallow review. Priority order for this repo:

1. `src/harness/store/` — schema, migrations, retrieval filters
2. `src/harness/model/` — the adapter boundary
3. `src/harness/orchestrator/`, `src/harness/router/` — the tool loop and hook pipeline
4. `src/harness/tools/` — new tools, profile membership, schema budget
5. `src/harness/driver/`, `src/harness/turn/`, `src/harness/cli*.py`
6. `tests/`
7. `character/` data and config

Read from the base, not your working tree: `git show main:<path>`. Your branch may hold changes the PR doesn't, and reviewing against them produces wrong findings.

### Step 4: Read existing feedback first

```bash
gh api repos/{owner}/{repo}/pulls/<PR>/comments     # inline
gh api repos/{owner}/{repo}/issues/<PR>/comments    # discussion
```

Separate bots from humans, note reply threads (`in_reply_to_id`), and track which concerns are still open. Then: don't re-raise a point someone already made unless you're adding something — reference it instead ("+1 on the retrieval scoping concern — also affects the fact path"). Do flag an existing concern that looks unaddressed.

### Step 5: Load the standards

`CLAUDE.md` is the primary review criteria: the invariants, the conventions section, and the quality-gate rules. `AGENTS.md` covers bd protocol compliance. Bead context: `bd show <bead-id>` from the branch name or PR title, plus `.llm/tasks/<bead-id>_*.md` if it exists — the acceptance criteria are the spec you're reviewing against.

### Step 6: Categorize findings

**Blocking:**
- Invariant violations — model SDK imported outside `src/harness/model/`, character rules hardcoded into `src/`, a user-facing `.search()` missing `user_id=speaker`, a store mutation with no attribution
- Bugs, logic errors, race conditions on shared SQLite state
- Destructive or non-reversible data migrations without a supersede path
- Missing or broken tests for new behavior; mocked SQLite where a real `tmp_path` store belongs
- Blanket `# noqa` / `# type: ignore` standing in for a fix
- Tool-schema budget blown for a profile

**Suggestions:** clarity and naming · edge cases · reuse of an existing helper instead of a new one · test-coverage gaps · consistency with neighboring modules.

**Nitpicks** (prefix `nit:`): style the linter doesn't catch, naming preferences, doc wording.

**Positive notes:** call out good design and good tests. A review that's only negative is a worse signal.

### Step 7: Present before posting

```
## <PR #N | branch> Review

**Overall:** <approve / request changes / comment only>
**Risk:** <low / medium / high> — <one line>

### Existing feedback
<what others already flagged; what's unresolved>

### Blocking
- [ ] <file:line> — <what breaks and why>

### Suggestions
- <file:line> — <description>

### Nitpicks
- <file:line> — nit: <description>

### Positive notes
- <description>

### Questions for the author
- <ambiguities, design questions>
```

Ask: post as-is? adjust any severity? add or drop anything? which action — APPROVE, REQUEST_CHANGES, or COMMENT?

### Step 8: Post (PR path only)

**Use `--input -` with a heredoc.** The `gh` `-f 'comments[0][path]=...'` form does not work for arrays — it builds a hash. Position comments with `line` + `side` ("RIGHT" for additions, "LEFT" for deletions).

```bash
gh api repos/{owner}/{repo}/pulls/<PR>/reviews -X POST --input - <<'JSON'
{
  "event": "<APPROVE|REQUEST_CHANGES|COMMENT>",
  "body": "<overall summary>",
  "comments": [
    { "path": "src/harness/store/episodic.py", "line": 42, "side": "RIGHT", "body": "…" }
  ]
}
JSON
```

Drop the `comments` array for a review without inline notes. Branch review with no PR → the Step 7 report is the deliverable; nothing to post.

### Step 9: Report

```
✓ Review posted on PR #<N>
  Action: <APPROVE / REQUEST_CHANGES / COMMENT>
  Inline comments: <count>
  URL: <review URL>
```

## Review principles

- Constructive: "Consider…" for non-blocking items, not demands.
- Explain the why — what goes wrong, and the fix.
- Respect a working, readable solution. Don't rewrite it into your own style.
- Be specific: exact lines, concrete examples, links to the pattern already in the repo.
- Assume good intent — odd-looking code may be deliberate. Ask.
- Proportional: a 5-line fix doesn't need 20 comments.

## Reference
- Use `gh api` for review submission — `gh pr review` has thin inline-comment support
- Prefer `line` + `side` over `position`
- Events: APPROVE, REQUEST_CHANGES, COMMENT
- Stacked PRs: `bin/pr-stack --check <PR>` prints the chain and flags a stale footer (exit 1)
- Most work here lands by local merge — branch review with no PR is normal
