# PR QA

Guide a manual QA pass on a change the test suite can't cover: the chat REPL and TUI, streaming and interrupts, tool-loop behavior with a real model, driver runs, the web gateway. Pass a bead id, PR number, or branch: `/pr_qa harness-5z6y`.

The unit suite pins contracts. This command pins **behavior in the actual runtime**, which is where a local model harness fails in ways tests don't see.

## Instructions

### 1. Understand the intent

- Read the bead (`bd show <id>`), its comments, and `.llm/tasks/<bead-id>_*.md` if there is one — the acceptance criteria are the QA spec.
- Read the PR body or the commit log for what the author claims changed.
- Note explicit limitations, deferred work, and anything the author flagged as risky.

### 2. Inspect what actually changed

```bash
git diff --stat main...<branch>
git diff main...<branch>
```

Group the changes by behavior and classify each:

- **CLI surface** — new command, new flag, changed default
- **Chat-path** — prompt assembly, retrieval, persona rewrite, compaction
- **Tool-loop** — tool definitions, hook pipeline, router, write-tier confirms
- **Model adapter** — MLX/Ollama, streaming, speculative decoding, LoRA
- **Store / memory** — schema, retrieval filters, scribe, consolidation
- **TUI / REPL** — rendering, streaming, slash commands, cancellation
- **Driver / daemon / web** — long-running paths
- **Internal only** — refactor, types, tests

Call out mismatches between what the description claims and what the diff does.

### 3. Build the QA plan

Per behavior change:

**What changed** · **Why it matters** · **How to exercise it** — the exact command, e.g.

```bash
uv run harness chat --model mlx --persona --tools --tool-set coding --workspace /tmp/qa
uv run harness chat --model mlx --persona --tui
uv run harness eval voice --model mlx --top-k 6 --persona --json
uv run harness memory search "<query>"
uv run harness drive plan --epic <id>     # prepare only — Mark launches drives by hand
```

**Preconditions** — model pulled (`hf download mlx-community/…`), `bd dolt start` running, memory seeded (`harness memory ingest`), the right character selected, a scratch `--workspace` so filesystem tools can't touch the repo.

**Steps** — numbered, concrete, with the literal prompt to type. "Ask it something tool-shaped" is not a step; `read_file on src/harness/cli.py, then summarize` is.

**Expected result** — exactly what should appear, including which tools should fire and in what order.

**Extra checks** — the failure modes this runtime actually has:
- Fabrication: does it claim a tool result it never got? Does the catcher fire?
- Persona drift: does the rewrite preserve substance, or compress a multi-step answer into a slogan?
- Retrieval: do the memory and fact blocks contain what they should — and nothing from another user?
- Latency and RAM: does the change make a turn noticeably slower or heavier? (`scripts/bench_*.py`)
- Cancellation: Ctrl+X mid-stream in the TUI — clean stop, no orphaned worker?
- Write-tier: does the confirm still gate on first use per session?
- Sandbox: do filesystem and shell tools still refuse paths outside `--workspace`?

### 4. Run it interactively

Work one area at a time. Mark each case pass / fail / partial / blocked, ask targeted follow-ups, and help separate a real bug from expected behavior from an unclear requirement. Keep a running log — a local model is non-deterministic, so note the seed conditions and whether a failure reproduces on a second run.

### 5. Final report

```
## QA — <bead-id | PR #N>

**Scope tested:** <areas>
**Environment:** model <repo>, tool-set <name>, workspace <path>

### Passed
- <case> — <observation>

### Failed
- <case> — <what happened> — <reproducible? n/N runs>

### Blocked / could not verify
- <case> — <why>

### Risks / concerns
### Suspected regressions
### Gaps in coverage
### Recommended follow-ups
- <bead to file>

**Status:** Pass | Pass with concerns | Needs fixes | Blocked
```

File the follow-ups as beads before you close out:

```bash
bd create "<finding>" --description="<repro steps + observed vs expected>" -p 2 --deps discovered-from:<BEAD_ID> --json
```

## Rules

- Practical, not theoretical. Convert diff lines into commands someone can actually run.
- One run proves nothing on a non-deterministic model. Repeat a suspicious case before calling it a bug.
- Flag anything you cannot verify by hand — say so instead of guessing.
- Don't stop at the happy path. This runtime's bugs live in tool loops, retrieval scoping, and cancellation.
- Never launch a drive. Prepare the command and hand it over.
