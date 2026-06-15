# Agent-Framework Benchmark Plan

Compare open-source agent frameworks **similar to this harness** by giving each the
same autonomous build task — **build a GTA-style game** — against a single shared
remote model (**gx10**). The harness enters as the home-team baseline.

The benchmark is designed to be **repeatable** (pinned everything, N runs, full env
manifest), **recordable** (every transcript, diff, and artifact captured for replay),
and **adjustable** (metrics are pluggable scorers listed in a config — add/remove a
metric without touching the runner).

> Scope decisions locked with the requester: build target = *build a GTA-style game*;
> remote model = *gx10 over an OpenAI-compatible `/v1` endpoint (vLLM)*; candidate
> language = *any, best-in-class*.

---

## 1. Candidate libraries

Two buckets. **Turnkey coding agents** drive their own multi-step build loop, exactly
like this harness does — apples-to-apples. **Build-your-own frameworks** need a thin
GTA-builder scaffold written per library, so they measure *your scaffold quality* as
much as the library; include them as a second tier and flag the caveat in results.

Every candidate below is confirmed to support a custom OpenAI-compatible `base_url`
(directly or via LiteLLM), so all can be repointed at gx10 with config only.

### Tier 1 — turnkey coding agents (primary comparands)

| Framework | Repo | gx10 wiring | Notes |
|---|---|---|---|
| **This harness** | (local) | native vLLM / OpenAI adapter | baseline / reference entry |
| **OpenHands** (ex-OpenDevin) | https://github.com/All-Hands-AI/OpenHands | LiteLLM → `base_url` | best open SWE-bench scaffold; Docker sandbox; fully autonomous |
| **Aider** | https://github.com/Aider-AI/aider | `--openai-api-base` / LiteLLM | veteran, diff-based edits, terminal-native, scriptable |
| **OpenCode** | https://github.com/sst/opencode | OpenAI-compatible provider in `opencode.json` | terminal agent, 75+ providers, easy headless |
| **Goose** (Block) | https://github.com/block/goose | built-in OpenAI provider → vLLM `/v1` | Apache-2.0, MCP extensions, vLLM guide exists |
| **mini-SWE-agent** | https://github.com/SWE-agent/mini-swe-agent | LiteLLM | minimal canonical reference loop; great control |
| **SWE-agent** | https://github.com/SWE-agent/SWE-agent | LiteLLM | research scaffold; heavier than mini |

Optional / harder to headless: **Cline** (https://github.com/cline/cline, VS-Code-bound),
**Codex CLI** (https://github.com/openai/codex, supports custom provider base_url).

### Tier 2 — build-your-own frameworks (need a per-lib GTA-builder scaffold)

| Framework | Repo | gx10 wiring |
|---|---|---|
| **smolagents** (HF) — `CodeAgent` | https://github.com/huggingface/smolagents | `OpenAIServerModel(api_base=...)` |
| **Pydantic-AI** | https://github.com/pydantic/pydantic-ai | `OpenAIProvider(base_url=...)` |
| **LangGraph** | https://github.com/langchain-ai/langgraph | `ChatOpenAI(base_url=...)` |
| **AutoGen / AG2** | https://github.com/microsoft/autogen · https://github.com/ag2ai/ag2 | OpenAI-compatible `config_list` |
| **CrewAI** | https://github.com/crewAIInc/crewAI | LiteLLM / `base_url` |
| **Agno** | https://github.com/agno-agi/agno | OpenAI-compatible model class |

**Recommended starting set:** harness + OpenHands + Aider + OpenCode + Goose +
mini-SWE-agent (Tier 1), plus smolagents `CodeAgent` as the one Tier-2 control. Six to
seven entries keeps a full N-run matrix tractable; add more once the rig is proven.

---

## 2. The benchmark product — GTA-style game

A fixed spec file (`spec/gta-spec.md`) is the **only** task input, identical for every
framework. It defines a top-down 2D GTA-style game with graded milestones so partial
success is measurable, not pass/fail:

1. **M1 — window + loop**: game window opens, runs headless, exits cleanly.
2. **M2 — player car**: a controllable car renders; arrow/WASD move + rotate.
3. **M3 — world**: top-down tiled map / roads larger than viewport; camera follows car.
4. **M4 — physics**: acceleration, steering, friction (not instant teleport).
5. **M5 — NPC traffic**: ≥1 AI-driven vehicle moving on the map.
6. **M6 — collision**: car vs world/NPC collision detected and resolved.
7. **M7 — objective**: a mission/score loop (reach waypoint, pickup, or wanted level) + HUD.

Pin the stack in the spec to remove a free variable (recommend **Python + pygame** —
trivially headless-testable via `SDL_VIDEODRIVER=dummy`, no GPU/browser needed). The
spec is git-tracked; its commit SHA goes in every result row so a spec change is never
silently mixed into a results series.

---

## 3. gx10 as the shared remote model

One model endpoint for all frameworks isolates *framework* as the only variable.

- gx10 serves the model via **vLLM** exposing OpenAI `/v1/chat/completions` over Tailscale.
- Every adapter gets the same `base_url`, `model` id, and a dummy API key.
- **Pin and record** in the run manifest: model id + quant, `max_model_len`,
  `temperature` (0 for max determinism), `top_p`, `seed` (if the vLLM build honors it),
  and a hash of the server launch args.
- **Serialize runs** (or hard rate-limit) so concurrent agents don't contend on gx10
  and skew wall-clock / latency numbers.
- Health-gate before each run: probe `/v1/models`; abort the batch if the served model
  id ≠ the pinned id (catches a silently swapped endpoint).

---

## 4. Harness architecture

Mirror this repo's load-bearing patterns: an **adapter boundary** per framework, an
**append-first attributed store** for results, and **config-as-data** for metrics.

```
agent-bench/
  spec/gta-spec.md          # the one task input (git SHA recorded per run)
  bench.yaml                # frameworks, runs-per-framework, gx10 endpoint, active metrics
  metrics.yaml              # ordered list of scorer plugins to run (the adjustable surface)
  adapters/                 # one thin wrapper per framework — the only framework-specific code
    base.py                 #   Adapter protocol: prepare(workspace) -> invoke(spec) -> Result
    openhands.py  aider.py  opencode.py  goose.py  mini_swe.py  smolagents_codeagent.py ...
  scorers/                  # pluggable metric collectors (drop-in, listed in metrics.yaml)
    builds.py  runs_headless.py  feature_checklist.py  cost.py  process.py  quality_judge.py
  runner.py                 # orchestrates the matrix; framework-agnostic
  store.py                  # SQLite + JSONL, append-first, attributed, timestamped
  results/                  # per-run artifacts: transcript, diff, repo tarball, scores.json
  report.py                 # aggregate -> markdown/CSV tables + per-metric distributions
```

**Adapter protocol** (the framework boundary — keep framework SDKs *only* in here):

```python
class Adapter(Protocol):
    name: str
    version: str                       # pinned, recorded in manifest
    def prepare(self, workspace: Path) -> None: ...   # fresh, isolated build dir
    def invoke(self, spec: str, gx10: Endpoint) -> RunArtifacts: ...
        # returns transcript, final diff, token/turn counters, exit status
```

**Run loop** (per `(framework, run_idx)`):

1. Fresh isolated workspace — a git worktree or clean temp dir, so runs never cross-contaminate.
2. Health-gate gx10; stamp the manifest (framework version, model id, spec SHA, params).
3. `adapter.invoke(spec, gx10)` with a wall-clock timeout and a turn cap.
4. Capture artifacts: full stdout/stderr transcript, final `git diff`, repo tarball,
   raw token/turn counters from the framework or proxied off gx10.
5. Run every scorer listed in `metrics.yaml` against the artifacts → `scores.json`.
6. Append one row to the store; write artifacts under `results/<framework>/<run_idx>/`.

---

## 5. Repeatable · recordable · adjustable

**Repeatable**
- Pin framework versions in a lockfile; record each in the manifest. Containerize Tier-1
  agents (most ship a Docker image) so host drift can't leak in.
- Pin gx10 model/params/seed; `temperature=0`. Accept that agent loops stay
  non-deterministic even at temp 0 → run **N ≥ 5** per framework and report
  **distributions** (success rate, median, IQR), never a single number.
- One command: `bench run --runs 5` replays the whole matrix from `bench.yaml`.
- Fresh workspace per run; spec SHA pinned per row → a spec edit starts a new series, never blends.

**Recordable**
- Every run persists: transcript, diff, repo tarball, token/turn counters, scores, manifest.
- Append-first store (SQLite + JSONL) keyed by
  `(framework, version, run_id, started_at, spec_sha, model_id)`. Nothing is overwritten —
  re-runs append, so trends over time are queryable.
- `report.py` regenerates tables/plots from the store at any time; raw artifacts allow
  full offline replay and dispute resolution.

**Adjustable**
- `metrics.yaml` is the dial: an ordered list of scorer plugin names (+ optional weights).
  The runner loads and runs *only* what's listed.
- Add a metric = drop a `scorers/<name>.py` implementing the `Scorer` protocol
  (`score(artifacts) -> dict[str, float|bool]`) and add its name to `metrics.yaml`.
  No runner edit, no schema migration — new keys just appear in `scores.json`.
- Remove/disable a metric = delete its line. Old rows keep their historical keys; the
  store is schemaless JSON per row for scores.

```yaml
# metrics.yaml — illustrative; swap freely (the requester sets the real set)
metrics:
  - builds              # does the generated project import/compile?
  - runs_headless       # SDL dummy driver: boots and exits without crash?
  - feature_checklist   # M1..M7 milestones probed statically + at runtime
  - cost                # prompt/completion tokens, wall-clock, gx10 GPU-seconds
  - process             # agent turns, tool calls, files touched, diff LOC, retries, interventions
  - quality_judge       # optional LLM-judge (0-5) on code quality + playability
```

---

## 6. Build order

1. **Skeleton + store + one adapter** (Aider — simplest to script) + 2 cheap scorers
   (`builds`, `runs_headless`). Prove one row end-to-end against gx10.
2. **Write `spec/gta-spec.md`** (M1–M7) and the `feature_checklist` scorer that probes them.
3. **Add Tier-1 adapters**: OpenHands, OpenCode, Goose, mini-SWE-agent.
4. **Add `cost` + `process` scorers** (parse counters / proxy gx10 token usage).
5. **N-run matrix + `report.py`** distributions; lock the manifest/repeatability story.
6. **Tier-2 scaffold** (smolagents `CodeAgent`) + `quality_judge`; expand candidates as needed.

## 7. Known risks / caveats

- **Tier-2 measures the scaffold, not just the library** — report it in its own section; don't rank it head-to-head with turnkey agents.
- **Non-determinism** survives `temperature=0`; only distributions over N runs are trustworthy.
- **Sandboxing**: agents run/install arbitrary generated code — isolate every run (container or throwaway worktree), never the host.
- **Cline / IDE-bound agents** resist headless automation — keep optional until a scripted entrypoint exists.
- **gx10 contention** skews latency — serialize runs or the wall-clock metric is noise.

---

### Sources

- [Best Open Source Coding Agents 2026 — Open Source AI Review](https://www.opensourceaireview.com/blog/best-open-source-coding-agents-in-2026-reviewed-ranked)
- [OpenHands vs SWE-agent (2026) — CodeSOTA](https://www.codesota.com/agentic/openhands-vs-swe-agent)
- [OpenHands evaluation harness](https://github.com/OpenHands/benchmarks)
- [OpenCode docs — config / providers](https://opencode.ai/docs/config/)
- [Goose — Configure LLM Provider](https://goose-docs.ai/docs/getting-started/providers/) · [block/goose vLLM provider](https://github.com/block/goose)
- [Pydantic-AI — OpenAI-compatible models](https://ai.pydantic.dev/models/openai/)
- [Open-source agent framework comparison — Langfuse](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)
- [Python AI agent library comparison 2026](https://jangwook.net/en/blog/en/python-ai-agent-library-comparison-2026/)
