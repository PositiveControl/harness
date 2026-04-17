# Roadmap

Living document. Authoritative architecture + commands reference is `CLAUDE.md`; daily-use workflow is `docs/usage.md`. This file tracks phase sequencing and open questions.

## Where we are (2026-04-16)

**Working end-to-end**:

- Chat with Airton locally via MLX-hosted Qwen 2.5 32B Instruct.
- Voice-rewrite post-pass keeps Airton's register on responses the base model would otherwise drift on.
- Episodic and semantic memory: seed + scribe-written + consolidator-promoted tiers, all retrieval-indexed with mxbai-embed-large-v1 (1024 dim).
- Per-user relationship scoping — Alice can't see Bob's private memories; shared seeds reach everyone.
- Corpus growth: `harness voice capture` turns a corrected reply into a new voice sample; the loader picks it up on next start; retrieval surfaces it on similar future prompts.
- Validated end-to-end twice: the "junior asks for the fix" and "transceiver bug" validation turns both reproduced seed-memory specifics; the onboarding prompt demonstrated corpus growth fixing a previously-generic response.
- ~110 tests green under ruff + mypy strict + pre-commit.

**What you can't do yet**: talk to Airton from anywhere but a terminal on the M4. No always-on daemon, no gateway, no multi-agent roles.

## Phases landed

- **Phase 0 — Skeleton.** Character package, model adapter boundary, SQLite transcript, CLI + quality gates.
- **Phase 1a — MLX adapter + voice eval** (incl. 1a.1 few-shot + leave-one-out, 1a.2 voice suite expansion to 32 samples).
- **Phase 1b.0 — Retrieval substrate** (`sentence-transformers` + `VoiceRetriever`).
- **Phase 1d — Persona-voice post-pass** (incl. 1d.1 heuristic scorer + tightened rewriter, 1d.2 LLM-judge + filler detector + chain-of-rewrite, 1d.3 nuanced bullet rule + bullet-density scorer).
- **Phase 1b — Memory proper**:
  - **1b.0** — Episodic store (SQLite + BLOB embeddings + cosine scan).
  - **1b.1** — Semantic store + batch scribe + watermark-tracked incremental reruns.
  - **1b.2** — Consolidator: cluster-merge episodes, group-merge fact triples, mark superseded.
  - **1b.3** — Dimension tracking on stores + `rebuild-embeddings` for non-destructive embedder switches.
- **Phase 2.0 — Relationship memory.** Per-user scoping on episodic + semantic search; scribe tags candidates by user; CLI `--user` / `--shared` flags.
- **Phase 2.1 — Corpus growth.** `harness voice capture` writes captured YAML; character loader merges canonical + captured.

## Voice durability — the permanent path

Voice quality has a ceiling that prompt engineering cannot reach on its own. Ordered from cheapest to most durable.

### Tier 1 — Smarter scoring (landed)

- Heuristic scorer: length / openers / bullet-discipline / bullet-density / filler.
- LLM-as-judge (rubric-based 1–10, same model for now).
- Held-out eval slice — not yet; worthwhile once the voice suite passes ~50 samples.

### Tier 2 — Rewriter refinements (mostly landed)

- Chain-of-rewrite (opt-in second pass focused on concrete-action substitution).
- Nuanced bullet rule: dashes with one-clause items are fine; numbered lists wrong; multi-sentence bullets are the tutorial tell.
- Remaining: anti-pattern injection ("your draft opened with 'X'"), hard character-count targeting for length.

### Tier 3 — LoRA fine-tune (Phase 1e candidate)

First permanent move. Train a LoRA on Qwen 2.5 32B using the voice suite (canonical + captured) as training data.

- **Feasibility**: `mlx-lm` has LoRA support; 32 + captured samples fine-tune in a few hours on the M4.
- **Outcome**: base model's register shifts toward Airton's without few-shot. No rewrite pass needed for easy cases.
- **Risk**: catastrophic forgetting. Mitigate with a mixed training set.
- **Gate**: wait until heuristic + judge scorer are trustworthy enough to tell "did this fine-tune help or hurt."

### Tier 4 — Corpus growth loop (landed in 2.1)

Infrastructure in place. Compounds every time Mark captures an edit.

Remaining niceties:
- In-chat `/edit` command invoking `$EDITOR` with Airton's reply pre-loaded — currently the capture flow requires leaving chat to run the CLI command.
- Automatic capture prompt after each Airton turn (opt-in; keystroke to skip).

### Tier 5 — Preference learning (DPO)

Once captured hits ~200 samples, DPO beats SFT for register capture. Train the model to prefer gold-like responses over the draft shape. The draft/gold pairs produced by `eval voice --persona` are exactly the right shape.

### Tier 6 — Full SFT on an expanded corpus (stretch)

With 1000+ Airton-voiced responses, do a full supervised fine-tune (not LoRA). Voice becomes a property of the weights. At this stage it's worth reconsidering the base model — Qwen 2.5 Instruct's "helpful assistant" tuning is stubborn.

## Next-up candidates (no priority implied)

All independent; pick any.

- **In-chat `/edit` for voice capture.** Lower the friction of corpus growth. Small CLI change.
- **Consolidator user-awareness.** Currently clusters across users; harmless with one user, needs fixing before adding a second.
- **LoRA fine-tune (Phase 1e).** First permanent voice move. Eval suite is dialed in enough to judge the result.
- **Web gateway.** FastAPI + SvelteKit ops console. Unblocks Slack/Matrix, turns Airton into something other people can reach.
- **Launchd daemon + scheduled consolidation.** Make Airton actually "always-on"; run scribe + consolidate nightly.
- **Multi-agent roles** (planner / researcher / executor / critic / persona). The architecture promised these; currently everything runs as a single single-model ReAct-ish loop.
- **Kuzu graph layer.** When we want relationship graphs over entities (who-works-with-whom, project-depends-on-project).
- **Scheduled initiative.** Phase 5 — Airton opens threads unprompted, reacts to external events.

## Robustness backlog

Deferred items from a 2026-04-16 critique pass. Not load-bearing for a single-user local CLI; relevant as the system opens to multiple gateways, multiple users, or automated scheduling. Filed here so we don't lose them.

- **Auth / authz on gateways.** Tailscale covers the network layer today. When web / Slack / Matrix gateways land, each needs its own identity check before writes hit memory. Blocks any gateway that isn't terminal-local.
- **Rate limiting.** Per-user throughput cap on turns + tool calls. Matters once something other than Mark can trigger generation.
- **Backup / restore workflow.** Beyond the existing open decision on destination: need a tested restore path and a nightly job. Likely pairs with launchd daemon.
- **Consolidator user-awareness.** Currently clusters across users; harmless with one user, incorrect before a second arrives. (Already on the next-up list — mirrored here because it's a robustness blocker, not a feature.)
- **Error logging + structured telemetry.** Nothing emits structured events today. Minimum: a rotating JSONL log for tool-call outcomes and scribe runs. Scales up to OpenTelemetry if we grow out of that.
- **Health check / readiness endpoint.** Needed for the launchd daemon and any gateway. Can start as `harness health` returning store + embedder + model status.
- **Database corruption recovery.** SQLite WAL survives power loss, but we have no `integrity_check` + restore-from-backup runbook. Write one once backup destination is chosen.
- **Resource caps on long-running sessions.** Transcript tail already limits history to 50 turns; memory store is naturally bounded by consolidation. Revisit when a session runs for days without restart.
- **Scheduled-consolidation hardening.** When consolidation runs nightly via launchd, it must not overlap with interactive scribe / chat writes. Depends on the scribe lock landing first (Item 5 below).

## Open decisions

- **Backup destination.** Blocks Phase 2 multi-user. Candidates: Backblaze B2, iCloud Drive, S3, NAS, another Mac. Default until chosen: SQLite WAL + local Time Machine + nightly tarball of `data/`.
- **LLM-judge model.** Same Qwen for now (circular, low cost). Swap in a different/stronger judge — smaller model for speed, or cloud model for orthogonal signal — when the heuristic + circular judge plateau.
- **Voice eval cadence.** Automatic on commits that touch `src/harness/character.py`, `src/harness/persona/`, or `character/airton/`? Would need a pre-commit or CI hook. Defer until we have a "real" deployment.

## Principle

Voice is three layers deep: **prompt → pipeline → weights**. Each layer is more permanent than the one above it and more expensive to change. Walk down it in order. Don't skip rungs.

Memory is three layers deep: **seed → scribe → consolidate**. Seeds are curated, scribes are noisy, consolidation is how noise becomes signal. Every memory retrieval in live chat is supposed to be a consolidated one; working-tier is a staging area.

Retrieval is three layers deep for Airton now: **voice few-shot → episodic memories → semantic facts**. Each layer has a similarity floor — irrelevant content should not reach the prompt. More retrieval is not better; *better* retrieval is better.
