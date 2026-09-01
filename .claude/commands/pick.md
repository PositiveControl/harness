# Pick

Entry door to the workflow. Show my prioritized ready work from bd, then route the selected bead to the right next step based on its shape and state.

## Instructions

### Step 1: Fetch ready work

```bash
bd ready --json
```

`bd ready` already excludes beads blocked by an open dependency. Also pull anything already claimed, which `ready` may hide:

```bash
bd list --status in_progress --json
bd list --status blocked --json
```

### Step 2: Filter to mine

Keep beads where `assignee` is me (`git config user.name`) or `null` (unclaimed, fair game). **Drop `assignee == "airton_b"`** — that is ab's internal scratchpad, not human work. Include it only if I ask for internal beads.

### Step 3: Enrich with structure

Per candidate, one call each — cheap, and only for beads that have relations (`dependency_count`/`dependent_count` > 0):

```bash
bd dep list <id>      # parents (via parent-child) and blockers
bd children <id>      # renders the epic tree itself — don't hand-draw it
```

An epic with open children is a container, not a task.

### Step 4: Present prioritized summary

Sort: priority (P0→P4) → status (blocked → in_progress → open) → type (bug → task → feature → epic → chore).

Markdown table: Priority, Status, ID, Title, Type, Blockers. Under it, paste the `bd children` tree for any epic in the list, and for a leaf with a parent show the one-line parent from `bd dep list`.

### Step 5: I select → route

`bd show <id>` for full detail, then route:

| Selected bead | Route |
|---|---|
| `epic` with open children | Pick a child instead — show the `bd children` tree and re-run Step 5 on the chosen child |
| `epic` with no children | `/feature_plan <id>` — design doc, then children filed as **direct** epic children (`--deps parent-child:<epic>`), never a dep chain, or the child never reaches `bd ready` |
| Vague — no acceptance criteria, or a body spanning several unrelated changes | `/feature_plan <id>` — shape or split before any implementation |
| `blocked` | Show blockers from `bd dep list <id>`; no route until cleared |
| `in_progress` — task file exists in `.llm/tasks/` | `/implement <id>` |
| `in_progress` — no task file | `/task_plan <id>` — the plan was never made, or was lost |
| Ready, shaped, single-concern | `/task_plan <id>` |
| Ready and drive-shaped (mechanical, gate-checkable, an epic of leaf beads) | Prepare the `harness drive loop` command for the epic — print it, do **not** launch it. Drives are run by hand. |
| Landed, awaiting review | `/pr_review <pr>` · `/pr_qa <id>` · `/pr_comment_resolver <pr>` |

State the route explicitly: "Shaped P2 task, no blockers → run `/task_plan harness-abcd`." Shaping is enforced here, at the door — an epic or an unshaped bead never goes straight to task planning.

### Step 6: Hand off to the chain

Each command names the next one. `/pick` only opens the door — planning, implementation, gates, and `bd close` belong to `/task_plan`, `/implement`, and `/pr_submit`. Full map: `docs/dev_workflow.md`.

## Reference
- Me: `git config user.name`
- Statuses: `open`, `in_progress`, `blocked`, `deferred`, `pinned`, `hooked`, `closed`
- Priorities: P0 critical → P1 high → P2 medium (default) → P3 low → P4 backlog
- Types: `bug`, `feature`, `task`, `epic`, `chore`
- `bd` fails on a daemon/plan-shaped query → `bd dolt start` may not be running
- Full bd protocol: `AGENTS.md`. Repo commands + gates: `CLAUDE.md`.
