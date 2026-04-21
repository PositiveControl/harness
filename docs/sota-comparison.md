# Harness vs. SOTA — Architecture Comparison & Upgrade Plan

> Peer-review briefing comparing the current harness implementation against
> leading-edge agent-harness tools, frameworks, and research as of Q1 2026.
>
> Scope: architecture, memory, retrieval, tool use, persona/voice,
> self-improvement, local inference, observability.
>
> Last updated: 2026-04-21.

## TL;DR

Harness is already uncommonly close to SOTA on three dimensions:

- **Architecture** — the `orchestrator/tool_loop.py` is shaped like Claude
  Agent SDK's loop, with several hallucination catchers that aren't even
  named in the public literature yet.
- **Memory** — tiered (seed / working / consolidated) + per-user scoping
  (`user_id IS NULL` for shared) is ahead of Mem0's flat-scope model and
  parallel to Letta's core/recall/archival design, under different names.
- **Eval discipline** — three fixture-based evals (voice / router /
  session-resume) with both heuristic + LLM-judge scoring is more than most
  shipped agent harnesses have.

The gaps are at the *edges*: retrieval sophistication (pure dense cosine, no
hybrid / rerank), observability (no structured tracing), and closing the
self-improvement loop between the `bd` thought-graph and future-turn
behavior. 2–3 focused weeks of work would push harness **ahead** of SOTA
for the single-user long-running personal-agent use case.

## How to read this document

Each of the 8 sections has the same shape:

1. **Current** — what harness implements today, grounded in actual files.
2. **SOTA** — what the leading tools / papers are doing as of Q1 2026.
3. **Upgrades** — concrete, named changes with benefit and tradeoff.

A priority-ordered punch list lives at the end.

---

## 1. Agent architecture

### Current

`src/harness/orchestrator/tool_loop.py` (844 LOC) is a hand-rolled
Claude-Agent-SDK-shaped loop:

```
adapter.complete_with_tools
  → execute each tool call (with write-tier confirmation)
  → append result
  → loop until no tool calls
```

Features already present:

- `Tool` protocol + `ToolResult` — uniform abstraction.
- Tool-set profiles (`minimal` / `core` / `coding` / `memory` /
  `diagnostic` / `research`) with `--tools-add` / `--tools-drop`.
- Hallucination catchers: fabricated tool-call success, fabricated search
  results, bare tool-intent with no call, quoted-snippet numbered lists,
  paired meta-confirm.
- Wrap-up cap (default 384 tokens).
- Duplicate-call short-circuit.
- Retrieval degradation when on-rails (skip VoiceRetriever).
- Small-model intent router fronts the loop on read-tier calls.

### SOTA (Q1 2026)

The leading frameworks have converged on a common shape:

- **Claude Agent SDK** (GA late 2025). Streaming async-generator query
  loop. `Tool` as the single abstraction. `hooks` for intercept / modify /
  block at each loop boundary. `subagents` for context-isolated parallel
  workers. MCP was donated to the Linux Foundation's Agentic AI Foundation
  in Dec 2025 (97M monthly SDK downloads by March 2026).
- **LangGraph** — production-grade stateful. Every step compiles to a
  stateful graph with checkpoints, time-travel replay, per-node token
  streaming, LangSmith tracing. The winner for "we need to debug this in
  six months."
- **OpenAI Agents SDK** (née Swarm). Lightweight, tool-centric, handoffs
  between agents, built-in tracing + guardrails. Low ceiling, fast stand-up.
- **CrewAI** — role-based "crews" DSL; ~40% faster to deploy than
  LangGraph (per their numbers), weaker checkpointing.
- **AutoGen / AG2** — moved to maintenance; Microsoft Agent Framework is
  the active successor. Don't start new work on AutoGen.
- **Pydantic-AI** — strict typed I/O via Pydantic schemas, dependency
  injection. Best for "structured task agent," thin for multi-agent.
- **smolagents** (HF) — `CodeAgent`: model emits Python, sandbox
  executes, result feeds back. Much denser than JSON tool calls for
  multi-step data plumbing.
- **DSPy** — not an agent framework per se; a *program* abstraction where
  prompts are compiled from signatures and optimized by MIPROv2 / SIMBA /
  GEPA. Pairs with any runtime.

### Upgrades

#### 1.A — Subagent primitive

A `spawn_subagent(system_prompt, tools, budget)` meta-tool that runs in an
isolated context window and returns a summary.

- **Benefit:** "search-and-summarize" or "multi-file grep-and-reason"
  stops polluting main context with 40 tool results. Biggest
  architectural lever remaining. Maps onto the existing tool loop cleanly
  — a subagent is just a recursive tool-loop invocation.
- **Tradeoff:** adds a second concurrent inference; on M4 Pro you'll want
  a rate-limit. Subagent summaries are a new failure surface (info loss
  in the summary step).

#### 1.B — Hooks refactor of hallucination catchers

Your catchers are currently inline `if` branches in `tool_loop.py`.
Extract into a typed pipeline:

```python
class HookPipeline:
    pre_tool: list[Hook]
    post_tool: list[Hook]
    pre_model: list[Hook]
    post_stream: list[Hook]
```

- **Benefit:** testable in isolation, configurable per profile, easier to
  add the 12th catcher without growing the file further. Matches Claude
  Agent SDK's `hooks` vocabulary.
- **Tradeoff:** moving-boxes refactor with zero user-visible features.
  Only worth doing immediately before adding the next catcher.

#### 1.C — Do NOT migrate to LangGraph / Claude Agent SDK

Harness is already specific to your invariants (MLX-first, local-first,
per-user scoping, ab-owned beads). A migration would cost more than it
gains. Borrow *ideas* — subagents, hooks — rather than adopting the
framework wholesale.

---

## 2. Memory systems

### Current

`src/harness/store/episodic.py` (419 LOC), `semantic.py` (357 LOC),
`consolidate/consolidator.py`, `scribe/`.

- Three tiers: `seed` / `working` / `consolidated`.
- Per-user scoping: `user_id IS NULL` for shared, `user_id=X` for private.
  Retrieval filters `(user_id IS NULL OR user_id = <speaker>)`.
- Async scribe extracts candidates from transcript; consolidator runs
  single-link clustering at cosine ≥ 0.80 inside each `user_id` partition.
- Superseded rows retained for audit (`superseded_by IS NOT NULL`
  filtered out of retrieval).
- `embedder_id` + `embedding_dim` per row — new embedder doesn't crash;
  `rebuild_embeddings()` migrates in place.
- Semantic triples: `(subject, predicate, object)` + confidence +
  provenance.

### SOTA (Q1 2026)

- **Letta** (MemGPT successor). Tiered memory — *core* (in-context,
  self-editable), *recall* (searchable conversation), *archival* (cold
  tool-call storage). Runtime, not a layer — the agent lives inside
  Letta.
- **Mem0** — memory *layer* you bolt on. User / session / agent scopes;
  hybrid vector + graph + KV. Highest adoption on "drop-in" axis.
- **Zep / Graphiti** — temporal knowledge graphs. Facts as edges with
  validity windows (`valid_from`, `valid_to`). Hybrid semantic + BM25 +
  graph-traversal retrieval. The most interesting SOTA for
  "relationship memory that changes over time."
- **LangMem** — background memory manager; async extract / consolidate
  off the hot path. (Their consolidation is a batch job, same shape as
  your scribe → consolidator.)
- **GraphRAG** (Microsoft) — community-detection over extracted
  entity-relation graph + LLM-precomputed community summaries.

Field has converged on: hybrid vector + graph, temporal facts,
episode-to-semantic distillation.

### Mapping: harness ↔ SOTA

| Harness | Letta | Mem0 | Zep/Graphiti |
|---|---|---|---|
| `tier=seed` (shared) | core | agent-scope | seed facts |
| `tier=working` | recall | session-scope | recent episodes |
| `tier=consolidated` | archival | user-scope | consolidated KG |
| `user_id IS NULL` | — | global | — |
| `superseded_by` | — | — | temporal edge end |
| Scribe + consolidator | MemoryManager | add_memory | extractor |

Harness's per-user scoping with shared rows is *ahead* of Mem0. Harness's
`superseded_by` is the bones of temporal facts but not exposed as such.

### Upgrades

#### 2.A — Temporal fields on `SemanticStore`

Add `valid_from`, `valid_to`, `asserted_at` columns. Retrieval filters by
`now BETWEEN valid_from AND valid_to` by default, with `--as-of` override.

- **Benefit:** "X was true last March but isn't now" stops being lossy.
  Consolidator stops hiding history. Essential for a long-running
  personal agent that outlives job titles, relationships, project scopes.
- **Tradeoff:** two more columns, a migration, slightly more retrieval
  logic. Small.

#### 2.B — Episode→fact distillation in the consolidator

When an episodic cluster stabilizes (n ≥ 3, cosine-adjacent, same topic),
emit a semantic fact with `derived_from=[ep_ids]`.

- **Benefit:** today episodes and facts are parallel pipes; you're
  leaving the "what do I actually believe about X?" derivation on the
  table. Fixes the "40 episodes about Mark's preferences but no fact
  asserting any of them" gap.
- **Tradeoff:** consolidator latency grows; topic-extraction strategy
  needs a decision (LLM pass vs embedding-cluster centroid vs frontmatter
  principle).

#### 2.C — Procedural tier

`tier="procedural"` for learned behaviors ("when Mark says X, he wants
format Y"). Source: `bd` `thought:decision` / `thought:observation` beads
(see §6).

- **Benefit:** closes the self-improvement loop from `bd` retros to
  future-turn behavior.
- **Tradeoff:** promotion policy is subtle — too eager and you calcify
  mistakes; too conservative and you never learn. Gate on outcome
  confirmation or explicit user approval.

#### 2.D — Do NOT migrate to Kuzu yet

An open bd issue suggests a graph layer. Premature — your `bd`
thought-graph + semantic triples won't cross 50k edges for a long time.
SQLite + judicious JOINs wins until then.

---

## 3. Retrieval

### Current

`src/harness/retrieval/voice_retriever.py` (54 LOC) + store-side search
in `episodic.py::search`, `semantic.py::search`.

- Pure dense cosine over BLOBs in SQLite.
- Default embedder: `BAAI/bge-small-en-v1.5` (384 dim, ~130 MB, MPS on
  Mac). Config-overridable.
- L2-normalized vectors so cosine = dot product.
- `embedder_id` + `embedding_dim` per row guards against mismatched dims.
- FTS5 exists on the `transcript` store but **NOT** on `episodic` or
  `semantic`.
- No reranker. No hybrid fusion. No contextual chunking.

### SOTA (Q1 2026)

- **Hybrid BM25 + dense fused with RRF** is table stakes. Benchmark-
  winning stack: BM25 ∪ SPLADE ∪ dense → ColBERT reranker (highest nDCG
  on multi-way evals).
- **bge-reranker-v2-m3** — CPU-fine, ~560 MB, multilingual, matches
  Cohere Rerank at $0. Self-hostable.
- **Jina-ColBERT-v2** — +6.5% over original ColBERT v2, 89 languages,
  Matryoshka output dims. Token-level MaxSim is the SOTA reranker
  primitive.
- **Contextual chunking** (Anthropic) — prepend doc-scope context to
  each chunk; cuts retrieval failure rate 35–50%.
- **HyDE** — for short/ambiguous queries, generate hypothetical answer
  and embed *that*. Usually 5–15% recall lift.
- **Small-to-big** — retrieve at chunk grain, return parent document.

### Upgrades

#### 3.A — FTS5 + cosine RRF on episodic + semantic (highest ROI)

Mirror the transcript FTS5 setup onto episodic and semantic. Fuse
results with Reciprocal Rank Fusion (k=60):

```
score = 1 / (k + rank_fts) + 1 / (k + rank_cos)
```

- **Benefit:** catches proper nouns, code identifiers, exact phrases
  (`BeadsAdapter`, `user_id IS NULL`) that dense cosine loses. Personal
  memory has lots of these.
- **Tradeoff:** keep the FTS index in sync via SQLite triggers (cheap).
  RRF implementation is ~50 lines.

#### 3.B — Optional reranker pass

Retrieve top-20 by hybrid RRF, rerank to top-5 with bge-reranker-v2-m3.

- **Benefit:** measurable lift on "should've gotten this memory but
  didn't."
- **Tradeoff:** +500 ms latency per turn (CPU), +560 MB resident. Gate
  behind `--rerank`; do not enable by default.

#### 3.C — Contextual chunking for episode bodies

Prepend `[principle: X; session: Y; date: Z]` to the embedded text.
Frontmatter already exists on seed memories — propagate into the
embedding, not just the metadata.

- **Benefit:** retrieval by lesson or timeframe starts working
  ("what did we decide about identity?" finds principle-tagged episodes).
  Near-free.
- **Tradeoff:** requires one `rebuild_embeddings` run (you already have
  that command).

#### 3.D — Skip HyDE and ColBERT for now

Marginal gains over 3.A–C at much higher complexity. Revisit when basic
hybrid is in and still insufficient.

---

## 4. Tool use + intent routing

### Current

- 18 tools across filesystem, shell, git, memory, web, self-introspection,
  ab-ops.
- Tool profiles (`minimal`, `core`, `coding`, `memory`, `diagnostic`,
  `research`) grouped by use case; ≤ ~1,500 tokens of schema overhead
  per profile.
- Workspace sandbox (`--workspace`).
- `--router` fronts the tool loop with a small model (default
  `mlx-community/Hermes-3-Llama-3.2-3B-4bit`) in free-JSON or outlines-
  grammar-constrained mode.
- Write-tier confirmation per session, per tool.
- Hallucination catchers in the loop.
- No MCP. No tool-result summarization. Tools return raw content to the
  model.

### SOTA (Q1 2026)

- **MCP** — de facto standard. Linux Foundation governance since Dec
  2025. 97M monthly downloads. Supported by ChatGPT, Cursor, Gemini,
  Copilot, VS Code. Perplexity publicly moved *off* MCP for first-party
  tools citing context-window burn + clunky auth — MCP is great for
  third-party, mixed for first-party.
- **Constrained decoding**:
  - **Outlines** — FSM pre-compute, some startup cost (~1 GB RAM).
  - **LM Format Enforcer (LMFE)** — permissive (whitespace / ordering
    agnostic); best false-positive rates across zero/1/2-turn benchmarks;
    near-zero RAM.
  - **Guidance / llguidance** — 50 μs/token, fastest.
  - vLLM supports all three.
- **Programmatic Tool Calling** (Anthropic, April 2026) — model
  orchestrates tools *via code* rather than round-tripping each call.
  Huge for parallel execution + fewer `tool_result` messages chewing
  context.
- **Tool-result summarization at ingestion** — named pattern: old tool
  results > 200 chars outside the protected tail get compressed. Context
  drift (not context *limit*) is the #1 cited enterprise agent failure
  mode.
- **Planner/executor split** — plan emitted once as strict JSON;
  executor dynamically provisioned with *only* the tools it needs per
  step. Devin: Planner / Coder / Critic.

### Upgrades

#### 4.A — Tool-result summarizer hook

In `tool_loop.py` between execute and append-to-messages: if
`len(result) > 1 KB`, summarize via small model (reuse the Hermes-3
router) while preserving identifiers / paths verbatim.

- **Benefit:** directly attacks the 65%-of-agent-failures-are-context-
  drift number. Matters most on `grep`, `list_dir`, `search_web` results.
- **Tradeoff:** info-loss risk; preservation rules need care. Start
  conservative — only tools flagged `high_noise=True`.

#### 4.B — MCP adapter (out + in)

Wrap existing 18 tools as an MCP server (outgoing). Wrap incoming MCP
tools as `Tool`-protocol shims (incoming).

- **Benefit:** Airton can be hosted into Claude Desktop / Cursor; can
  consume the MCP ecosystem (GitHub, Filesystem, Slack, etc.) without
  rewriting. Strategic — the whole ecosystem has moved.
- **Tradeoff:** MCP's context cost is real (Perplexity's concern). Keep
  internal `Tool` protocol as primary; MCP as interop adapter, not
  replacement.

#### 4.C — Benchmark LM Format Enforcer vs outlines

Add `--router-mode lmfe` alongside existing `grammar` mode.

- **Benefit:** ~1 GB RAM back (outlines FSM); potentially better
  zero-turn accuracy.
- **Tradeoff:** another optional dep. If outlines isn't causing
  problems, this is a quiet-week project, not a priority.

#### 4.D — Programmatic / planner split for multi-file edits

Today model decides edit-by-edit. Two-pass: plan in `bd` (already
structured!) → execute each step with only that tool provisioned.

- **Benefit:** fewer tokens, higher success on write-heavy tasks. `bd`
  integration means you have the plan substrate already.
- **Tradeoff:** more state to manage; write-tier confirmations get more
  complex. Only worth it once multi-file edits are a common workload.

---

## 5. Voice / persona

### Current

- Two-pass persona rewriter (`PersonaAdapter`) + optional chain-of-
  rewrite concrete-substitution.
- Retrieval-picked voice few-shots from canonical + captured corpus.
- `voice/canonical.yaml` (32 samples) + `voice/captured.yaml` (grows via
  `/edit` or `harness voice capture`).
- `harness eval voice` — heuristic + optional LLM judge.
- Rewriter off when tools ran (`rewrite_on_tools=False`) — correct call,
  rewriter compresses which breaks investigate/summarize replies.

### SOTA (Q1 2026)

- **LoRA / QLoRA on persona corpus** — pragmatic gold standard once ≥
  100 samples. Axolotl + QLoRA on Qwen 2.5 7B is textbook.
- **Persona vectors** (Anthropic, July 2025, arXiv 2507.21509).
  Activation-space direction extracted from contrast pairs, added at a
  specific layer at inference. MLX has the layer-hook primitives.
- **BILLY** (Oct 2025, arXiv 2510.10157) — blends multiple persona
  vectors for compositional persona.
- **Style-modulation heads** (arXiv 2603.13249) — a sparse subset of
  attention heads governs persona; targeting only those heads gives
  robust persona without broad-steering coherency degradation.
- **Persona DPO / ORPO** — preference-tuning with (good, bad) pairs;
  dominant when soft prompting isn't enough but full LoRA is overkill.
- **Voice drift detection** (Hamming's framework, 4M+ calls) —
  baseline at launch, LLM-judge continuously, alert on >10% deviation.
  Named as distinct from model drift.

### Upgrades

#### 5.A — LoRA on voice corpus (once captured > ~100)

Your Tier 3 roadmap item. Reserve 20% as held-out val.

- **Benefit:** voice lock without per-turn rewriter pass (latency win);
  removes the "rewriter compresses tool outputs" bug-shape entirely.
- **Tradeoff:** one LoRA per model-repo; every time you bump the base
  model you retrain. Worth it.

#### 5.B — Persona vector as inference-time knob (research spike)

MLX supports layer hooks. Extract a direction from (airton-tone, neutral)
contrast pairs drawn from canonical samples.

- **Benefit:** zero-training; stacks with rewriter; adjustable strength;
  better than prompting for subtle voice cues.
- **Tradeoff:** research-territory stability. Keep behind a flag, never
  default. Spike first, commit only if it wins.

#### 5.C — Weekly voice-drift cron

`uv run harness eval voice --json` → persist aggregate score in a
`drift/` table → alert (emit `bd thought:observation`) on >10% drop.

- **Benefit:** catches embedder migrations, base-model bumps, rewriter-
  prompt edits that silently break voice.
- **Tradeoff:** ~5 min of compute per week. Cron config, not code.

#### 5.D — Decouple judge from generator

You already flagged this as an open thread. Use Llama-3.3-70B via
Ollama locally, or Claude Haiku API, as second judge.

- **Benefit:** removes 5–7% self-enhancement bias.
- **Tradeoff:** second model to manage; Ollama adds RAM pressure
  alongside MLX.

---

## 6. Self-improvement / evals

### Current

- `harness eval voice` — heuristic + judge.
- `harness eval router` — tool-name + arg-shape accuracy vs
  `character/<name>/router_eval.yaml`.
- `harness eval session-resume` — contains / not_contains per scenario
  vs `character/<name>/session_resume_eval.yaml`.
- `bd` thought-graph with `thought:hypothesis` / `question` / `decision`
  / `observation` labels.
- `/retro` REPL slash command + auto-retro on session-end (`/exit`,
  `:q`, Ctrl-C).
- LLM judge is same-model-as-generator (circular).

### SOTA (Q1 2026)

- **DSPy optimizers**:
  - **MIPROv2** — Bayesian opt over instructions + few-shots jointly.
  - **SIMBA** — introspective failure analysis on stochastic batches.
  - **GEPA** — reflect-and-propose over trajectories.
- **Voyager-style skill libraries** — skills as executable programs
  indexed by description embedding; compositional and transferable.
  2026 wave extends to "AI Skills as institutional knowledge primitive"
  (arXiv 2603.14805).
- **Agent-as-Judge** — agent evaluators with reasoning traces; better
  than one-shot scalar judges for multi-step tasks.
- **Judge panels** — multiple judges' mean + disagreement mitigates
  single-judge bias; CoT on judge +10–15% reliability; position-bias
  debiasing (swap order, average).
- **Calibration**: monthly re-run against locked calibration set.

### Key observation

**Harness has the Voyager substrate and isn't harvesting it.** `bd` with
`thought:*` labels + `/retro` is literally the skill-library primitive;
you built it for a different reason but it's the same mechanic.

### Upgrades

#### 6.A — Harvest `bd` into a skill library

Embed `thought:decision` + `thought:observation` beads from closed foci;
retrieve on relevant user intents; inject into system prompt as "last
time X came up, you did Y."

- **Benefit:** this is Voyager's core mechanic — the key missing piece
  for self-improvement. Substrate already exists.
- **Tradeoff:** noise-in-noise-out; wrong "decisions" get re-injected.
  Gate on `thought:decision` confirmed by outcome or explicit user
  approval.

#### 6.B — DSPy-compile rewriter + router system prompts

Your voice eval + router eval fixtures are the natural trainsets.
MIPROv2 optimize over (instruction, few-shots).

- **Benefit:** expected 5–15% voice aggregate lift; router latency may
  drop from fewer few-shots needed; auto-improvement flywheel.
- **Tradeoff:** DSPy brings its own weight (optimizer loops are long);
  you trade hand-craft for machine-search — some current prompts may
  already be near-optimal.

#### 6.C — Judge panel, not single judge

Heuristic scorer + Qwen judge + Llama-3.3 judge (or Haiku). Report mean
+ disagreement. Flag samples with disagreement > threshold for human
review.

- **Benefit:** calibration + signal; removes self-enhancement bias.
- **Tradeoff:** more moving parts in the eval CLI.

---

## 7. Local inference stack (Apple Silicon)

### Current

- `src/harness/model/mlx.py` (347 LOC) — MLX 4-bit Qwen-2.5-7B default,
  32B via `--model-repo`. LoRA loading via `--lora-path`.
- Ollama adapter as fallback with tool-call + token-streaming support.
- Single-stream mlx-lm.
- **No speculative decoding**.
- No serving framework (vLLM, vllm-mlx).

### SOTA (Q1 2026)

- **vllm-mlx** — continuous batching. 21–87% higher throughput than
  llama.cpp. Up to 525 tok/s on M4 Max. Best Apple Silicon *serving* as
  of early 2026.
- **mlx-lm** — native KV-cache management + speculative decoding
  (DFlash). ~230 tok/s baseline, higher with draft model. Zero quality
  loss.
- **llama.cpp** — still king for fine-grained control + universal GGUF;
  ~150 tok/s short-context.
- **Distributed MLX** — JACCL tensor parallelism over Thunderbolt-
  connected Macs (experimental but real). MoE expert parallelism.
- **Quantization**:
  - GGUF K-quants (Q4_K_M default).
  - IQ-quants — importance-weighted, better quality at same size.
  - AWQ — activation-aware, Apple-Silicon-friendly.
  - EXL2 — variable bits-per-weight, best quality/size (CUDA-centric).
  - MLX: native 4-bit group quant; growing set of mixed-precision MLX
    variants.
- **M5 GPU neural accelerators** — Apple ML Research Q1 2026 shows
  non-trivial gains for LLM workloads. Relevant for next hardware cycle.

### Upgrades

#### 7.A — Speculative decoding in mlx-lm (highest single-change ROI)

Draft model: Qwen-2.5-0.5B or 1.5B alongside 7B/32B target.

- **Benefit:** 1.5–2× single-stream throughput on 32B. Mathematically
  identical output distribution — **zero quality loss**. Single highest-
  ROI change on this entire list.
- **Tradeoff:** +1–2 GB RAM for draft model; mlx-lm API surface change
  — small refactor inside the MLX adapter.

#### 7.B — vllm-mlx as a second serving adapter (when gateways arrive)

Only when you stand up Slack / web gateways or concurrent TUI sessions.

- **Benefit:** continuous batching unlocks real multi-user throughput.
- **Tradeoff:** premature today — you're single-user CLI. Park behind
  a `vllm-mlx` branch of `model/factory.py` for Phase 4.

#### 7.C — IQ-quants on 32B

Rebench against voice eval.

- **Benefit:** potentially noticeable quality lift at same memory
  footprint.
- **Tradeoff:** one afternoon. If it loses, no-op; if it wins, update
  default `HARNESS_MODEL_REPO`.

#### 7.D — Don't bother with multi-Mac / tensor parallel

Not resource-bound on M4 Pro 48 GB for the workload.

---

## 8. Observability / tracing

### Current

- Transcript store (SQLite + FTS5) as the poor-man's trace log.
- TUI metrics strip (elapsed, context) as ad-hoc observability.
- **No structured tracing, no spans, no propagation.**

### SOTA (Q1 2026)

- **Langfuse** — OSS, self-host in 5 min via Docker Compose. Feature
  parity between cloud + self-hosted. Native OTLP endpoint at
  `/api/public/otel`. Default pick for self-hosted LLM obs in 2026.
- **Arize Phoenix** — OSS, OTEL-native, Postgres backend. Strong for
  local dev. Enterprise (Arize AX) is separate.
- **OpenLLMetry** — instrumentation library; OpenTelemetry semantic
  conventions for LLM spans. Ships traces into any OTEL backend. Not a
  backend itself.
- **OpenObserve** — OSS; covers LLM + infra in one deployment.
- **OTEL semantic conventions for GenAI** — now stable
  (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, etc.). Makes
  backends swappable.

### Upgrades

#### 8.A — Minimum-viable trace stack

OpenLLMetry SDK → OTLP → self-hosted Langfuse in Docker Compose.

- **Benefit:** per-turn spans for model call / tool call / retrieval /
  rewriter / router; P50/P95 latency; token cost tracking; session
  replay; eval runs as traced datasets with per-span regression diffs.
  Unblocks data-driven iteration on everything else.
- **Tradeoff:** introduces Docker + a second persistent service. Real
  weight gain for a local-first system. Alternative: Phoenix in-process
  is lighter but less featureful.

#### 8.B — OTEL decorator on the adapter boundary

Wrap `complete`, `complete_with_tools`, each `Tool.execute`,
`VoiceRetriever.top_k`, rewriter passes, scribe, consolidator.

- **Benefit:** one decorator; every existing feature becomes traced.
- **Tradeoff:** ~50 lines + config plumbing. No user-visible change
  until a backend is up.

---

## Priority-ordered punch list

Organized by ROI ÷ effort. Anything above the fold is "do this quarter."

| # | Change | Effort | Lever |
|---|---|---|---|
| 1 | Speculative decoding in mlx-lm | 1d | ~2× throughput on 32B, zero quality loss |
| 2 | FTS5 + cosine RRF on episodic + semantic | 1d | Recall win on proper nouns / identifiers |
| 3 | Tool-result summarizer hook | 1d | Attacks #1 agent failure mode (context drift) |
| 4 | Temporal fields on `SemanticStore` | 2d | Unlocks "as-of" queries; essential for longevity |
| 5 | OpenLLMetry → self-hosted Langfuse | 2d | Unblocks data-driven iteration |
| 6 | Contextual chunking (frontmatter → embedding) | 0.5d | Near-free retrieval lift |
| 7 | Harvest `bd` thought-graph → skill library | 3d | Self-improvement flywheel; substrate exists |
| 8 | MCP adapter (out + in) | 2d | Strategic ecosystem interop |
| 9 | Subagent primitive | 3d | Biggest architectural lever for complex tasks |
| 10 | Episode→fact distillation in consolidator | 2d | Closes episodic/semantic gap |
| 11 | DSPy-compile rewriter + router prompts | 3d | Auto-improvement; 5–15% voice lift plausible |
| 12 | LoRA on voice corpus (when captured > 100) | 1w | Voice lock without rewriter latency |
| 13 | Weekly voice-drift cron | 0.5d | Cheap insurance against silent regressions |
| 14 | Judge panel (2nd model) + panel scoring | 1d | Calibration; removes self-enhancement bias |
| 15 | bge-reranker-v2-m3 optional pass | 1d | Gate behind `--rerank`; measurable recall lift |
| 16 | IQ-quants on 32B bench | 0.5d | Possible free quality bump |

### Deliberately NOT in the list

- **Kuzu migration** — too early; wait until graph crosses ~50k edges.
- **vllm-mlx adapter** — too early for single-user; relevant only when
  gateways arrive.
- **Persona vectors as default** — research stability; keep as a spike
  behind a flag.
- **ColBERT v2 / Jina-ColBERT-v2** — over-complex for the lift; basic
  hybrid + reranker wins first.
- **Full framework migration to LangGraph / Claude Agent SDK** —
  harness is already more specific to your invariants than these. You'd
  lose more than you'd gain. Borrow *ideas* (subagents, hooks) rather
  than frameworks.

---

## Sources

### Agent frameworks

- [Claude Agent SDK — agent loop](https://platform.claude.com/docs/en/agent-sdk/agent-loop)
- [Inside Claude Code architecture deep dive](https://zainhas.github.io/blog/2026/inside-claude-code-architecture/)
- [Claude Code subagents](https://platform.claude.com/docs/en/agent-sdk/subagents)
- [Claude Managed Agents deep dive](https://dev.to/bean_bean/claude-managed-agents-deep-dive-anthropics-new-ai-agent-infrastructure-2026-3286)
- [CrewAI vs LangGraph vs AutoGen vs OpenAI Agents SDK comparison](https://openagents.org/blog/posts/2026-02-23-open-source-ai-agent-frameworks-compared)
- [OpenAI Agents SDK vs LangGraph vs CrewAI 2026 matrix](https://www.digitalapplied.com/blog/openai-agents-sdk-vs-langgraph-vs-crewai-matrix-2026)
- [AI agent frameworks 2026 — SDKs, ACP, tradeoffs](https://www.morphllm.com/ai-agent-framework)
- [smolagents multi-agent code execution](https://earezki.com/ai-news/2026-04-16-a-coding-implementation-to-build-multi-agent-ai-systems-with-smolagents-using-code-execution-tool-calling-and-dynamic-orchestration/)
- [DSPy MIPROv2 API](https://dspy.ai/api/optimizers/MIPROv2/)
- [DSPy optimizers — SIMBA, GEPA](https://dspy.ai/learn/optimization/optimizers/)

### Memory

- [Mem0 vs Zep vs Letta benchmark](https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3)
- [Mem0 vs Letta (MemGPT)](https://vectorize.io/articles/mem0-vs-letta)
- [Graphiti — temporal knowledge graphs for agent memory (Neo4j)](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [Graph RAG in 2026: what actually works](https://medium.com/graph-praxis/graph-rag-in-2026-a-practitioners-guide-to-what-actually-works-dca4962e7517)
- [LangMem async background manager](https://langchain-ai.github.io/langmem/)
- [LangMem — long-term memory for AI agents (Digital Ocean)](https://www.digitalocean.com/community/tutorials/langmem-sdk-agent-long-term-memory)
- [Mem0 vs Zep vs LangMem 2026](https://dev.to/anajuliabit/mem0-vs-zep-vs-langmem-vs-memoclaw-ai-agent-memory-comparison-2026-1l1k)

### Retrieval

- [Jina-ColBERT v2 multilingual late interaction](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/)
- [Advanced RAG — hybrid search + reranking](https://dev.to/kuldeep_paul/advanced-rag-from-naive-retrieval-to-hybrid-search-and-re-ranking-4km3)
- [Infinity multi-way retrieval evaluation](https://infiniflow.org/blog/multi-way-retrieval-evaluations-on-infinity-database)
- [bge-reranker-v2-m3 vs Jina Reranker v2](https://agentset.ai/rerankers/compare/baaibge-reranker-v2-m3-vs-jina-reranker-v2-base-multilingual)
- [Best rerankers for RAG leaderboard](https://agentset.ai/rerankers)

### Tool use / MCP / constrained decoding

- [MCP donation to Linux Foundation (Anthropic)](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)
- [A year of MCP: 2025 review](https://www.pento.ai/blog/a-year-of-mcp-2025-review)
- [Why Model Context Protocol won (The New Stack)](https://thenewstack.io/why-the-model-context-protocol-won/)
- [vLLM structured outputs](https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html)
- [llguidance super-fast structured outputs](https://github.com/guidance-ai/llguidance)
- [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)
- [Guided decoding — generating structured outputs benchmark (arXiv 2501.10868)](https://arxiv.org/html/2501.10868v1)
- [Anthropic advanced tool use — programmatic tool calling](https://www.anthropic.com/engineering/advanced-tool-use)
- [Planner-executor architecture (Ema)](https://www.ema.ai/additional-blogs/addition-blogs/build-plan-execute-agents)

### Persona / voice

- [Anthropic persona vectors research](https://www.anthropic.com/research/persona-vectors)
- [Persona vectors — monitoring and controlling character traits (arXiv 2507.21509)](https://arxiv.org/abs/2507.21509)
- [Style modulation heads — robust persona control (arXiv 2603.13249)](https://arxiv.org/abs/2603.13249)
- [BILLY — merging persona vectors (arXiv 2510.10157)](https://arxiv.org/html/2510.10157)
- [Voice agent drift detection (Hamming)](https://hamming.ai/blog/voice-agent-drift-detection-guide)
- [LLM-as-judge guide 2026](https://labelyourdata.com/articles/llm-as-a-judge)
- [Best LLM drift monitoring platforms 2026 (Galileo)](https://galileo.ai/blog/best-llm-output-drift-monitoring-platforms)

### Self-improvement

- [Voyager — open-ended embodied agent](https://voyager.minedojo.org/)
- [AI skills as institutional knowledge primitive (arXiv 2603.14805)](https://arxiv.org/html/2603.14805v1)

### Local inference

- [llama.cpp vs MLX vs Ollama vs vLLM — Apple Silicon 2026](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026)
- [2026 Mac inference framework: vllm-mlx vs Ollama vs llama.cpp](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html)
- [mlx-lm — MLX LLM runtime](https://github.com/ml-explore/mlx-lm)
- [MLX distributed inference](https://localai.io/features/mlx-distributed/)
- [Apple ML Research — LLMs on MLX with M5 neural accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Quantization methods compared — GGUF, AWQ, GPTQ, EXL2](https://ai.rs/ai-developer/quantization-methods-compared)

### Observability

- [Langfuse self-hosted LLM observability](https://github.com/langfuse/langfuse)
- [Langfuse OTEL integration](https://langfuse.com/integrations/native/opentelemetry)
- [Arize Phoenix AI observability](https://github.com/Arize-ai/phoenix)
- [Top LLM observability tools 2026 (SigNoz)](https://signoz.io/comparisons/llm-observability-tools/)

### Context management

- [Automatic context compression in LLM agents](https://medium.com/the-ai-forum/automatic-context-compression-in-llm-agents-why-agents-need-to-forget-and-how-to-help-them-do-it-43bff14c341d)
- [Context compression strategies for long-running agent sessions](https://zylos.ai/research/2026-02-28-ai-agent-context-compression-strategies)

### Shops / practitioners

- [Devin vs Cursor — autonomous coding verdict 2026](https://fordelstudios.com/research/devin-vs-cursor-agent-mode-autonomous-coding-verdict-2026)
- [Cursor — best practices for agent coding](https://cursor.com/blog/agent-best-practices)
