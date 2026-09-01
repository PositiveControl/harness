# Dev Workflow — Spec & Diagrams

Command-driven lifecycle for developing **this repo** with Claude Code. It is dev tooling for the harness, not a harness feature — nothing here ships in `src/harness/`. State lives in bd; the commands are markdown in `.claude/commands/`.

Diagrams are mermaid — they render on GitHub and in VS Code.

## The chain

```
/pick (entry, routes by state) ⇢ /feature_plan → /task_plan → /implement → /pr_submit → land → bd close
```

Every command ends by naming the next one, so the workflow self-navigates. `/pick` is the entry door, not a chain stage: it surfaces prioritized ready work and routes by bead shape and state.

## 1. Lifecycle master map

```mermaid
flowchart TB
  E((" "))
  PK["/pick — ENTRY — routes by shape + bead state"]
  E --> PK
  subgraph R1 [" "]
    direction LR
    FP["/feature_plan — design doc → docs/plans/ — epic + child beads"]
    G1{"G1 — design approved"}
    TP["/task_plan — task file + impl plan — branch id-slug — bd claim"]
    G2{"G2 — plan approved"}
    FP --> G1
    TP --> G2
  end
  PK -. "epic / unshaped: decompose" .-> FP
  G1 -. "children in bd ready" .-> PK
  PK -- "shaped ready bead" --> TP
  subgraph R2 [" "]
    direction LR
    IM["/implement — code + tests — commit per logical unit"]
    G3{"G3 — gates green"}
    PS["/pr_submit — gates + docs → merge or PR"]
    RV["review — /pr_review · /pr_qa · /pr_comment_resolver"]
    G4{"G4 — comments resolved"}
    MG["land: local merge or human merge"]
    IM --> G3 --> PS --> RV --> G4 --> MG
  end
  subgraph R3 [" "]
    direction LR
    CL["bd close — reason recorded"]
    LP["land the plane — git pull --rebase · git push"]
    CL --> LP
  end
  SG1["/segue thread (planning valve)"]
  SG2["/segue thread (implementation valve)"]
  DR["harness drive — prepared, launched by hand"]

  G2 --> IM
  MG --> CL
  PK -. "in_progress: resume" .-> IM
  PK -. "drive-shaped epic" .-> DR
  TP -. blocked on a decision .-> SG1 -. findings merge .-> TP
  IM -. rabbit hole .-> SG2 -. findings merge .-> IM
  IM -. "scope escape: forecast &gt;600 lines" .-> FP

  style E fill:#0F7480,stroke:#0F7480
  style PK fill:#0F7480,stroke:#0B3A40,color:#FFFFFF
  style G1 fill:#F6E9D8,stroke:#A6651A,color:#5C3A0F
  style G2 fill:#F6E9D8,stroke:#A6651A,color:#5C3A0F
  style G3 fill:#E3EFE6,stroke:#3E7C4F,color:#1F4229
  style G4 fill:#F6E9D8,stroke:#A6651A,color:#5C3A0F
  style MG fill:#F6E9D8,stroke:#A6651A,color:#5C3A0F
  style CL fill:#E3EFE6,stroke:#3E7C4F,color:#1F4229
  style LP fill:#E3EFE6,stroke:#3E7C4F,color:#1F4229
  style FP fill:#E3F0F1,stroke:#0F7480,color:#0B3A40
  style TP fill:#E3F0F1,stroke:#0F7480,color:#0B3A40
  style IM fill:#E3F0F1,stroke:#0F7480,color:#0B3A40
  style PS fill:#E3F0F1,stroke:#0F7480,color:#0B3A40
  style RV fill:#E3F0F1,stroke:#0F7480,color:#0B3A40
  style SG1 fill:#FDFDFB,stroke:#5F6E7C,stroke-dasharray:4 3,color:#3D4954
  style SG2 fill:#FDFDFB,stroke:#5F6E7C,stroke-dasharray:4 3,color:#3D4954
  style DR fill:#FDFDFB,stroke:#5F6E7C,stroke-dasharray:4 3,color:#3D4954
  style R1 fill:none,stroke:none
  style R2 fill:none,stroke:none
  style R3 fill:none,stroke:none
```

Legend: teal = command · filled teal = entry · amber = human gate · green = automated gate / completion · dashed grey = optional side path.

## 2. Bead state machine

bd is the single state home. Commands fire the transitions; there is no board automation — `bd close` is a deliberate step in `/pr_submit`. State lives in bd, never in a session's memory.

```mermaid
stateDiagram-v2
  direction LR
  [*] --> open : /feature_plan files beads
  open --> in_progress : /task_plan — bd update --claim
  in_progress --> blocked : dependency or open question
  blocked --> in_progress : unblocked
  in_progress --> closed : /pr_submit — merged, bd close --reason
  closed --> [*]
  open --> deferred : parked deliberately
  deferred --> open : revived
```

`bd ready` = open, unblocked, and not someone else's. `--claim` is atomic (assign + `in_progress`) and is what makes concurrent sessions safe.

## 3. Gates

| # | Gate | Where | Kind | Blocks |
|---|---|---|---|---|
| G1 | Design doc approved | `/feature_plan` | Human | Filing beads — beads are commitment, the doc is a proposal |
| G2 | Implementation plan approved | `/task_plan` | Human | Any code |
| G3 | Local gate suite green | `/pr_submit` | Automated | Push / merge |
| G4 | Review comments resolved | `/pr_submit` + review | Human | Merge |

There is no hosted CI here. G3 is `ruff check` · `ruff format` · `mypy src tests` · `pytest` — and pre-commit runs them on every commit, with pytest gated at push time. That makes G3 the hardest gate in the chain; the human gates are the soft ones.

## 4. The thread ID

The linchpin. One bead id connects every artifact, which is what lets each command rebuild state from scratch — any session, any machine, no handoff notes.

```mermaid
flowchart LR
  ID(("bead harness-xxxx"))
  TF["task file — .llm/tasks/harness-xxxx_slug.md (local)"]
  BR["branch — harness-xxxx-slug"]
  CM["commits — subsystem: what and why"]
  PR["PR title — harness-xxxx: description"]
  MG["merge commit — Merge: what (harness-xxxx)"]
  BD["bd status — the lifecycle state"]
  DD["design doc — docs/plans/ (via the epic)"]

  ID --> TF
  ID --> BR
  ID --> CM
  ID --> PR
  ID --> MG
  ID --> BD
  ID --> DD

  style ID fill:#0F7480,stroke:#0B3A40,color:#FFFFFF
```

## 5. Command inventory

| Command | Job |
|---|---|
| `/pick` | Entry door: prioritized ready work + epic trees; routes epic/unshaped → `/feature_plan`, shaped ready → `/task_plan`, in_progress → `/implement`, blocked → show blocker, drive-shaped → prepare `harness drive` |
| `/feature_plan` | Explore → design doc (G1) → epic + sized child beads as **direct** children |
| `/task_plan` | Read design doc → task file + plan (G2) → branch → `bd update --claim` |
| `/implement` | Idempotent resume: load the task file, orient, execute, commit per logical unit |
| `/pr_submit` | Gate suite (G3) → docs resolved → push → local merge or PR (+ `bin/pr-stack`) → comments (G4) → `bd close` → land the plane |
| `/pr_review` | Reviewer side: full-context review of a PR or a branch diff |
| `/pr_qa` | Manual QA of what tests can't cover: chat, TUI, tool loop, streaming, driver |
| `/pr_fix_ci` | Gate-failure triage: branch-introduced vs environment vs pre-existing |
| `/pr_comment_resolver` | Fetch, address, and resolve PR review comments |
| `/test_fix` | Group pytest failures by root cause, fix causes not assertions |
| `/run_lint` | ruff + mypy pass with the no-blanket-suppression rule |
| `/update_docs` | Deep doc pass; keeps `CLAUDE.md` honest against the code |
| `/segue` ×5 | Isolated discussion thread with findings-only merge-back (global commands in `~/.claude/commands/`, not vendored here) |

Inclusion principle: a command earns its place only if a lifecycle gate or transition breaks without it.

## 6. Contract slots

Exactly one owner per slot. Ambient tooling (memory systems, indexers, background agents) must not claim authority over any of them.

| Slot | Owner |
|---|---|
| State machine | bd — `open` → `in_progress` → `closed`, plus `blocked` / `deferred` |
| Resumability artifact | Task files in `.llm/tasks/` + design docs in `docs/plans/` — `/implement` reloads them in any fresh session |
| Memory home | `CLAUDE.md` (invariants, commands, conventions) + `AGENTS.md` (bd protocol) + `docs/` |

## 7. Sizing

A bead fits when its acceptance criteria fit in ≤5 testable bullets. More is an epic; decompose.

| Size | Added lines | Guidance |
|---|---|---|
| XS | <100 | Fine as-is; consider batching with related work |
| S | 100–300 | Ideal |
| M | 300–600 | Healthy upper bound |
| L | 600–1,200 | Split if possible |
| XL | 1,200+ | Must split |

**Scope-escape rule:** a mid-implementation forecast past ~600 added lines, or new acceptance criteria appearing → stop, file a child bead, land the current slice clean.

## 8. Repo-specific rules that override generic advice

- **`bd dolt start` must be running** or ~10 daemon/plan tests fail. Looks like a regression; isn't.
- **`uv sync --extra all`**, always. Plain `uv sync` prunes omitted extras and silently breaks mypy, pytest, symbol reads, and the browser smoke gate.
- **Drives are prepared, never launched.** `harness drive` commands get handed to Mark; the loop artifacts get analyzed afterward.
- **Decomposed beads must be direct epic children** (`parent-child:<epic>`), never a dep chain — a chained sub-bead never surfaces in `bd ready` and the driver never sees it.
- **Drive-behavior fixes are FSM-path only** — the default `--no-fsm` bypasses the gate-quality and revive hardening.
- **ab (`airton_b`) beads are internal scratchpad.** `/pick` filters them out; don't route human work to them.
