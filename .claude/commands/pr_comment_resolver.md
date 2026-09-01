# PR Comment Resolver

Fetch review comments on a PR, address them, resolve the threads. Pass the PR number: `/pr_comment_resolver 2`.

## Instructions

### Step 1: Fetch PR context

```bash
gh pr view $ARGUMENTS --json number,title,url,headRefName,baseRefName,author,additions,deletions,changedFiles
```

Summarize: title, branch, lines changed. Not on the PR's branch → check it out:

```bash
git checkout <headRefName> && git pull
```

### Step 2: Fetch every comment

```bash
gh api repos/{owner}/{repo}/pulls/$ARGUMENTS/comments --jq '.[] | {id, author: .user.login, path, line, body, in_reply_to_id, created_at}'
gh pr view $ARGUMENTS --json reviews --jq '.reviews[] | "[\(.state)] \(.author.login): \(.body[:200])"'
gh api repos/{owner}/{repo}/issues/$ARGUMENTS/comments --jq '.[] | {id, author: .user.login, body: .body[:300], created_at}'
```

### Step 3: Categorize

1. **Drop already-resolved threads** — the author replied with a fix and it landed.
2. **Separate bots from humans.** Bot reviewers pattern-match; humans carry intent.
3. **Rank by severity:**
   - **Blocking** — bugs, invariant violations (see `CLAUDE.md`), missing tests, unsafe store mutations
   - **Suggestions** — clarity, naming, edge cases, reuse
   - **Noise** — bot false positives (a "query in a loop" flag on a test fixture, warnings about deliberate patterns this repo already documents)

### Step 4: Present before changing anything

```
## PR #<N> Review Comments

### Unresolved: <count>

#### Automated (<bot>)
- [ ] <path>:<line> — <summary> (id: <comment_id>)

#### Human (<reviewer>)
- [ ] <path>:<line> — <summary> (id: <comment_id>)

#### Likely noise (recommend skipping)
- <path>:<line> — <summary> — why it's noise
```

Ask: which to address (default: all non-noise)? any "noise" worth doing anyway? any human comment that needs discussion before a fix?

### Step 5: Address them

Per comment: read the full file, understand the actual concern, make the fix, check nothing else broke. A comment that disagrees with a documented convention gets a reply explaining the convention — not a code change.

Group related fixes into logical commits. Then the gates:

```bash
uv run ruff check . && uv run ruff format . && uv run mypy src tests && uv run pytest
```

```bash
git commit -m "$(cat <<'MSG'
Address PR review comments

- <fix per comment>

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

### Step 6: Push and resolve

```bash
git push
gh api repos/{owner}/{repo}/pulls/<PR>/comments/<COMMENT_ID>/replies -X POST \
  -f body="Addressed — <what changed>. See <COMMIT_SHA>."
```

### Step 7: Report

```
PR #<N> comments addressed
  URL: <PR_URL>
  Resolved: <count>   Skipped as noise: <count>
  Gates: ruff ✓ · mypy ✓ · pytest <N passed>
  Commits: <SHAs>
```

Human comments deferred for discussion → remind the user which, and why.

## Reference
- Reply endpoint: `repos/{owner}/{repo}/pulls/<PR>/comments/<ID>/replies`
- `{owner}/{repo}` are resolved by `gh` from the checkout — no need to spell them out
- Gates are local, not hosted: ruff · format · mypy · pytest
