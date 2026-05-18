# AGI Extrapolation — Hypothetical Plan

Status: speculative. This is a substrate-shape exercise, not a sprint backlog. Tracked in bd as `harness-N..N+7` (Phase 1–8 epics).

## TL;DR

The harness is a competent **single-agent substrate**: persistent memory, tool use, voice, bd-backed task graph, hook-mediated self-checks. Roughly 10–15% of what "AGI" implies as a software stack, ~0% of what it implies as a cognition stack — the base model is doing essentially all the thinking; the harness is scaffolding around it. The architectural choices are mostly fine for AGI substrate, with a few load-bearing assumptions that will need to break.

## What's actually built (the substrate today)

| Layer | Status | AGI relevance |
|---|---|---|
| Model adapter (MLX/Ollama/echo) | Solid abstraction | Required; the boundary holds |
| Tool loop + 4-phase hooks | ~20 tools, write-tier confirms, fabrication catchers | Necessary, not sufficient |
| Memory: 4 tiers + RRF hybrid + temporal validity | Mature for ~10k rows | Correct shape, wrong scale |
| Scribe → consolidate batch loop | Watermark-tracked, locked | Discrete-event; AGI wants streaming |
| Intent router (model + grammar variants) | Advisory, fall-through | Right idea, single-step only |
| Voice retrieval + 2-pass rewriter | Per-turn comp + LOO eval | Persona, not cognition |
| Bd thought-graph + harvest into procedural tier | `airton_b` data-plane, idempotent `external_id` | Cute, externalized; see below |
| Sandboxed workspace (`--workspace`) for fs/git/shell | Real | Foundation for self-modification |
| Compaction at context threshold | Auto-scribes before folding | Memory survives compression — good |
| Subagent spawning | Depth-1, read-only | Toy; not multi-agent |

## The honest gap

AGI-relevant capabilities the codebase doesn't have:

1. **No closed learning loop.** Voice corpus → LoRA is candidate-roadmap, not built. Nothing in the system updates *weights* from experience. Retrieval surfaces past examples; it doesn't compound them.
2. **No continuous operation.** Everything is turn-driven. There is no clock-driven heartbeat, no background introspection, no agent-scheduled future action. `/loop` and `/schedule` are user-invocable, not agent-invocable.
3. **No world model.** Memory is RAG over facts and episodes. Facts have temporal validity but no *relations* between facts, no causal links, no event/action/outcome graph. Kuzu is mentioned as a future graph layer — not started.
4. **No self-modification protocol.** Character files are human-edited. Adding a new tool requires a Python file + restart. Constitution updates aren't agent-initiated.
5. **No goal persistence inside the runtime.** Bd holds goals, but bd is an external subprocess; the orchestrator doesn't have a first-class plan/goal/subgoal type — it has tool calls and final replies.
6. **No reward signal.** Voice-judge produces scalars; they don't propagate anywhere. Nothing prefers the answers that worked over the ones that didn't.
7. **Single modality.** Text in, text out. Tools can fetch URLs and read files, but no vision, no audio, no sensors.
8. **Single agent.** "One identity, one memory, one orchestrator" is invariant #1 in CLAUDE.md. AGI-grade behavior probably emerges from competing/cooperating sub-policies, not one monolithic policy.

## Architectural decisions that will fight you

These are choices that are *correct for the current product* but will constrain the AGI extrapolation:

- **Invariant #1 (one identity, one orchestrator).** The hard one. Self-organization across multiple sub-agents requires fluid identity boundaries; the current invariant treats Airton as a singleton. The bd `airton_b` split is the seed of a precedent (data-plane isolation for an internal voice), but it's a two-character convention, not a protocol for N.
- **`ModelAdapter.complete()` as the cognition primitive.** Per-turn single model. Mixture-of-experts, parallel deliberation, self-play, draft/verify chains — all possible but bolted on top of an orchestrator that assumes "model produced reply, now what."
- **Tool registry is static Python.** `profiles.py` enumerates; ~1.5k-token schema budget partitions are pre-curated. Tool *synthesis at runtime* (write tool → register → use → persist) needs a registry protocol the codebase doesn't have.
- **Hooks pipeline is deterministic regex/string-matching Python.** Fabrication catchers work by pattern. AGI-grade self-monitoring wants a meta-model evaluating the primary policy, not `if "I'll search" in reply: bail`.
- **SQLite + FTS5 + BLOB embeddings.** Documented graduation path to LanceDB. Beyond that, AGI scale wants a graph store (Kuzu, candidate) plus distributed vector. The store interfaces are clean enough to swap, but `_hybrid.py` RRF lives at the wrong layer for that.
- **Bd as authoritative task substrate.** Subprocess-based, JSONL-on-disk. Externalizing the goal graph to a separate CLI was the right call for collaboration with Claude Code; it's the wrong call for an agent reading its own plan 100x/turn.
- **Compaction is lossy at the edges.** Auto-scribe before folding mitigates this, but the model only sees a summary of older turns. AGI-grade continuity wants the summary *and* on-demand re-expansion of arbitrary past windows — currently you only get retrieval hits, not full replay.
- **Single user_id partition.** Shared vs per-user. Fine for "Mark and a few friends," wrong for "agent with hierarchical context stacks."
- **`character/<name>/` is read-only at runtime.** Identity, voice, constitution all loaded fresh each session. Agent updating its own constitution is not a code path.

## Phased extrapolation

Order matters. Each phase unlocks the next.

### Phase 1 — Streaming consolidation + relational memory

Replace batch scribe/consolidate with streaming; add a graph store (Kuzu candidate) alongside the vector store for entity/event/relation graph. This is the substrate everything else needs. Memory stops being RAG and starts being a *world model under construction*.

Touches: `src/harness/scribe/`, `src/harness/consolidate/`, `src/harness/store/`, new `src/harness/store/graph.py`.

### Phase 2 — Goal graph as first-class runtime type

Move from "bd is the source of truth" to "runtime has typed plans/subgoals/active-actions; bd is one of N persistence backends." Lets the orchestrator reason about its own plan structure without a subprocess hop. Bd remains useful for human collaboration; the runtime stops depending on it for self-knowledge.

Touches: new `src/harness/plan/`, `src/harness/orchestrator/tool_loop.py`, `src/harness/store/bd_adapter.py` (becomes one backend among N).

### Phase 3 — Agent heartbeat loop

Clock-driven loop separate from user turns. Runs compaction, consolidation, plan revision, drift checks, scheduled tool calls. The diff between reactive tool-user and goal-pursuing agent. `/loop` and `/schedule` exist for the user; this builds the agent's version.

Touches: new `src/harness/runtime/heartbeat.py`, lifecycle hooks in `cli.py`, daemon-mode entry point.

### Phase 4 — Dynamic tool registry + tool synthesis

Protocol for the agent to author, validate (in sandbox), register, and persist new tools at runtime. Requires hot-reload of `tools/` and a meta-tool that compiles + tests candidate tools before promoting them to the active registry. The ~1.5k-token schema budget becomes a *working set* over a larger persistent tool library.

Touches: `src/harness/tools/profiles.py`, new `src/harness/tools/registry.py`, new `src/harness/tools/synth.py`.

### Phase 5 — Online learning loop

LoRA delta updates from voice corpus + judge-scored conversations + tool-outcome rewards. Adapter boundary already supports `--lora-path`; missing is the training side, the reward propagation, and the merge cadence. Probably nightly LoRA fits on captured + judge-approved turns, gated by a regression eval.

Touches: new `src/harness/learn/`, `src/harness/evals/voice.py` (reward channel), `src/harness/model/mlx.py` (hot-swap LoRA between sessions).

### Phase 6 — Multi-agent self-organization

Break invariant #1. N orchestrators sharing memory but with distinct priors / tool-sets / policies, coordinating via the goal graph. The `airton_b` precedent extends from a 2-character convention to an N-character registry. Identity becomes a stack frame, not a singleton.

Touches: invariant #1 in CLAUDE.md, `src/harness/orchestrator/`, `src/harness/store/_user_scope.py`, character loader, bd `assignee` semantics.

### Phase 7 — Multi-modal adapter

Vision and audio at the `ModelAdapter` boundary. Likely the cleanest single addition — the abstraction already supports it; `ChatMessage` grows attachments, adapters implement what their underlying model supports. Tool surface for screenshot capture, microphone capture, image generation.

Touches: `src/harness/model/adapter.py`, `src/harness/model/mlx.py`, `src/harness/model/ollama.py`, new perception tools under `src/harness/tools/`.

### Phase 8 — Self-modifying identity

Agent edits own `core.yaml` / `constitution.md` under human-in-the-loop approval; later autonomously. This is where alignment work compounds — the gate is review tooling and provenance, not the file edit. Probably implemented as a write-tier tool that lands constitution patches in a draft branch for review.

Touches: `character/<name>/`, write-tier confirmation pipeline, new `propose_constitution_patch` tool, review UI.

## The frank caveat

None of this *makes* AGI. The capability ceiling is the base model. The harness can give a Qwen-2.5-7B more memory, more tools, more time — it cannot give it more reasoning. A useful framing: this codebase is becoming a good **chassis for an agent**; whether the engine you bolt in is AGI-capable is a question about frontier models, not about whether the chassis has a graph store.

The most leveraged single addition for "AGI-like behavior on the current substrate" is probably **Phase 3 (heartbeat) + Phase 2 (typed goal graph)** — that's the difference between a reactive tool-user and something that pursues goals across days.
