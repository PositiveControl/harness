# Roadmap

Living document. The authoritative architecture summary is `CLAUDE.md`; this file tracks sequencing and open questions across phases.

## Phase structure at a glance

- **Phase 0 — Skeleton.** Complete. Character package, model adapter boundary, SQLite transcript, CLI chat with echo adapter, quality gates (ruff + mypy-strict + pre-commit).
- **Phase 1 — Memory + persona.** In progress. Broken into sub-phases as the voice work expanded:
  - **1a** — MLX adapter for Qwen 2.5 32B + voice-drift eval (done).
  - **1a.1** — Few-shot voice examples + leave-one-out eval flag (done).
  - **1a.2** — Voice suite expanded from 6 to 32 samples (done). Regression on hard prompts revealed the "more examples" strategy has a ceiling — motivated 1b.0.
  - **1b.0** — Retrieval-based few-shot (`sentence-transformers` + `VoiceRetriever`). Same substrate that Phase 1b memory uses (done).
  - **1d** — Persona-voice post-pass (`PersonaAdapter`, two-pass generation; done).
  - **1d.1** — Voice scorer (length / openers / bullets) + tightened rewriter prompt (done).
  - **1d.2** — LLM-judge scoring + filler-pattern detector + chain-of-rewrite option (done).
  - **1b.0** — Episodic memory store (SQLite + BLOB embeddings, cosine scan). Seed memories ingested from character on chat startup. Retrieval wired into the chat system prompt. Validated live against Qwen 2.5 32B — memory-driven responses reproduce seed-memory specifics (done).
  - **1b.1** — Semantic store (atomic facts, same pattern) + batch scribe that extracts episodic + semantic candidates from transcript turns. Watermark-tracked for incremental reruns. Validated end-to-end (done).
  - **1b.2** — Consolidator: cluster episodic near-duplicates, group semantic facts by (subject, predicate), promote to consolidated, mark originals superseded. Search filters out superseded rows (done).
  - **1b.3** — Dimension tracking in stores (`embedder_id` / `embedding_dim` columns) + `memory rebuild-embeddings` command, so future embedder switches are non-destructive. Rows with mismatched dims sit quietly until rebuilt (done).
- **Phase 2 — Multi-user.** Web gateway, ACLs, affective + procedural memory.
- **Phase 3 — Gateways + roles.** Slack + Matrix; Kuzu graph layer; multi-agent orchestrator.
- **Phase 4 — Concurrent + always-on.** Concurrent sessions, launchd, backup target landed.
- **Phase 5 — Life.** Scheduler, initiative, dreams, autonomy.

## Voice durability — the permanent path

Voice quality has a ceiling that prompt engineering cannot reach on its own. The path below is ordered from cheapest and most immediate to deepest and most durable.

### Tier 1 — Smarter scoring (landed in 1d.2)

- **Heuristic scorer** (`harness.evals.voice_score`): length / openers / bullets / filler. Fast, deterministic, interpretable. Catches regressions reliably.
- **LLM-as-judge**: small rubric-based rating (1-10) against gold, via the model adapter. Orthogonal to the heuristic scorer; catches mid-sentence register drift that regex cannot. Uses the loaded generation model for simplicity; swap in a different/stronger judge later.
- **Held-out eval**: once the voice suite passes ~50 samples, reserve a 20% slice that never appears in few-shot or training. The only honest generalization test.

### Tier 2 — Rewriter refinements (tail of 1d)

- **Chain-of-rewrite**: two rewrite passes — first for length + openers, second for concrete-action substitution (*"ensure X"* → *"do X"*; quoted rules → the action that follows). Opt-in because it doubles post-pass latency.
- **Anti-pattern injection**: feed the rewriter its draft's concrete violations (*"you opened with 'That sounds like a solid'"*) rather than abstract rules. Corrections land harder than constraints.
- **Length-cap enforcement**: pass a hard character-count target derived from the gold distribution instead of a fuzzy *"tighten it."*

### Tier 3 — LoRA fine-tune (Phase 1e candidate)

The first move that changes the model's weights, not its prompt. Train a LoRA adapter on Qwen 2.5 32B using the voice suite as (prompt, gold) pairs.

- **Feasibility**: `mlx-lm` has LoRA support. Fine-tuning 32 to 100 pairs takes a few hours on the M4 Pro.
- **Outcome**: the base model's register shifts toward Airton's without few-shot. No rewrite pass required for easy cases.
- **Risk**: catastrophic forgetting on general ability. Mitigation — curate a mixed training set that includes generic prompts with high-quality generic responses to preserve general competence.
- **Gate**: wait until the heuristic + LLM-judge scorer is trustworthy enough to tell "did this fine-tune help or hurt" before attempting. Otherwise it's flying blind.

### Tier 4 — Corpus growth loop (Phase 2-adjacent, ongoing)

The compounding play. Every time Mark edits Airton's response before sending (or flags a response as "off"), that edit becomes a new (prompt, gold) pair. The corpus grows with use.

- **Infrastructure**: `harness chat` records edits; the voice suite's canonical set stays curated but new captures pool into a growth set.
- **Payoff**: in six months, hundreds of real-world (prompt, gold) pairs. In a year, enough for a non-trivial SFT pass.
- **Dependency**: web UI or richer CLI where editing is a first-class action (part of Phase 2).

### Tier 5 — Preference learning (DPO)

Once the corpus passes ~200 samples, Direct Preference Optimization beats SFT for register capture. Train the model to prefer gold-like responses over generic drafts. The draft/gold pairs already produced by `eval voice --persona` are exactly the right shape.

### Tier 6 — Full SFT on an expanded corpus (stretch)

With 1000+ curated Airton-voiced responses, do a full supervised fine-tune (not LoRA). Produces a model whose default register IS Airton's. No rewrite pass, no few-shot. This is the ceiling — voice becomes a property of the weights.

At this stage it's worth reconsidering the base model. Qwen 2.5 Instruct's "helpful assistant" tuning is stubborn. A base (non-Instruct) Qwen or a Mistral base may take persona more cleanly.

## Near-term ordering

1. **Finish Tier 1 scoring** (1d.2) — LLM-judge + filler detector land first so subsequent changes are measurable.
2. **Land Tier 2 rewriter improvements** where heuristic + judge show material gains.
3. **Return to Phase 1b (memory)**. Voice is good enough to ship; memory is the headline feature that makes this a harness rather than a chatbot. Voice loops back after memory, at which point:
4. **LoRA fine-tune (Phase 1e)** — first permanent move, once scoring is trustworthy and memory foundations exist.
5. **Corpus growth infrastructure** goes in whenever chat UX work lands (Phase 2 web gateway is the natural place).

## Open decisions

- **Backup destination** (still TBD). Blocks Phase 2. Candidates: Backblaze B2, iCloud Drive, S3, NAS, another Mac.
- **LLM-judge model**: same Qwen for now (circular, low-cost), separate small model later for orthogonal signal.
- **Voice eval cadence**: run automatically on every commit that touches `src/harness/character.py`, `src/harness/persona/`, or `character/airton/`? Would need GitHub-Actions-equivalent locally; defer until Phase 4 hardening.

## Principle

Voice is three layers deep: **prompt → pipeline → weights**. Each layer is more permanent than the one above it and more expensive to change. The ladder above walks down it in order. Don't skip rungs.
