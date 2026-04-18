# Roadmap

Living document. Authoritative architecture + commands reference is `CLAUDE.md`; daily-use workflow is `docs/usage.md`. This file tracks phase sequencing and open questions.

## Where we are (2026-04-17)

**Working end-to-end**:

- Chat with Airton locally via MLX-hosted Qwen 2.5 (7B default for dev speed; 32B available via `--model-repo`), swappable to any MLX HF repo, or to Ollama via `--model ollama`.
- Optional LoRA adapter on top of the base MLX model (`--lora-path`) — plumbed through chat / eval / scribe.
- Voice-rewrite post-pass keeps Airton's register on responses the base model would otherwise drift on.
- Episodic and semantic memory: seed + scribe-written + consolidator-promoted tiers, all retrieval-indexed with mxbai-embed-large-v1 (1024 dim).
- Per-user relationship scoping — Alice can't see Bob's private memories; shared seeds reach everyone.
- Corpus growth: `harness voice capture` turns a corrected reply into a new voice sample; the loader picks it up on next start; retrieval surfaces it on similar future prompts.
- **Tool use (Phase 3)**: `--tools` hands the model 17 built-in tools across filesystem read (`read_file`, `list_dir`, `grep`, `glob`), filesystem write (`edit_file`, `write_file`, `shell`), git read (`git_status`, `git_diff`, `git_log`), memory (`search_memory`, `search_facts`, `remember_fact`, `remember_event`, `scribe_session`, `consolidate_memory`), and web (`search_web`). Named profiles (`minimal` / `core` / `coding` / `memory` / `diagnostic` / `research`) group them by use case; escape hatches `--tools-add` / `--tools-drop`. The orchestrator runs the tool-call loop, confirms write-tier tools once per session, streams tokens, renders a context meter, auto-compacts older turns, short-circuits duplicate calls, catches fabricated tool-call successes / bare tool-intent / meta-confirm, and caps wrap-up rounds. Filesystem + git tools are sandboxed to `--workspace`.
- **Intent router**: `--router` fronts the tool loop with a small model (default `mlx-community/Hermes-3-Llama-3.2-3B-4bit`). `--router-mode free` does tolerant JSON parsing; `--router-mode grammar` does JSON-schema-constrained decoding via `outlines`. Advisory: `null` / write-tier / unparseable router results fall through. `harness eval router` scores tool-selection accuracy against a YAML fixture.
- **Textual TUI**: `--tui` launches a full chat app — persistent input, scrolling RichLog, live ctx + elapsed footer, inline tool-loop event rendering, token-delta streaming, write-tier confirmation modal, history replay on mount, `/exit` + `:q` slash commands. Behind the optional `tui` extra.
- **Voice-capture ergonomics**: in-chat `/edit` / `/capture` opens `$EDITOR` on the last reply; saving captures a new voice sample without leaving the session.
- Robustness hardening: scribe `fcntl` session lock (no overlapping scribe runs corrupting the watermark); `PRAGMA busy_timeout = 5000` across all SQLite stores; HF/transformers/ST startup noise suppressed; Ollama adapter supports tool calls + token streaming; stream-level meta-confirm filter gated behind `--dev`.
- Validated end-to-end: the "junior asks for the fix" and "transceiver bug" validation turns both reproduced seed-memory specifics; the onboarding prompt demonstrated corpus growth fixing a previously-generic response.
- ~470 tests green under ruff + mypy strict + pre-commit.

**What you can't do yet**: talk to Airton from anywhere but a terminal on the M4. No always-on daemon, no gateway, no multi-agent roles.

**Captured voice corpus** (as of this refresh): 32 canonical + 1 captured. Most LoRA voice lift needs more capture — aim for ~50 captured before pulling the Tier-3 LoRA trigger.

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
- **Phase 3.0 + 3.1 — Tool use.** Initial 5 built-in tools (read_file, write_file, shell, search_memory, search_facts), `--tools` flag, orchestrator loop, Ollama adapter with tool-call + token-streaming support, streaming tokens, spinner, context meter, automatic compaction (`--compact-at`, `--compact-keep-recent`), retrieval degradation when a turn is clearly on-rails, `--workspace` sandbox, write-tier confirmation, `--rewrite-on-tools` toggle for the persona rewriter. Alongside: `--model-repo` and `--lora-path` flags across chat / eval / scribe, scribe fcntl session lock, SQLite `busy_timeout` pragma, HF/transformers noise suppression.
- **Phase 3.2 — Tool expansion + orchestrator hardening.** Grew from 5 to 17 tools: filesystem read trio (`list_dir`, `grep`, `glob`), partial-file `edit_file` (empty `old_string` = append), `write_file` now refuses overwrite by default (nudges toward `edit_file`), git read (`git_status`, `git_diff`, `git_log`), memory write (`remember_fact`, `remember_event`, `scribe_session`, `consolidate_memory`), web (`search_web` — DuckDuckGo HTML, stdlib only). Tool-set profiles (`minimal` / `core` / `coding` / `memory` / `diagnostic` / `research`) with `--tool-set`, `--tools-add`, `--tools-drop`. Orchestrator catches fabricated tool-call success, fabricated search results, bare tool-intent with no call, numbered-list quoted snippets, paired meta-confirm; caps wrap-up rounds (default 384 tokens) with widened cap on truncated recovery; short-circuits duplicate calls within a turn. `consolidate` partitions by user_id before clustering. Stream-level meta-confirm filter + pre-validated approve UX (dev markers gated behind `--dev`). Structured loading header shows every active flag. In-chat `/edit` slash command for voice capture.
- **Phase 3.3 — Intent router.** `Router` protocol + `ModelRouter` (free-form JSON + tolerant parse, tool-name + arg validation, system prompt with null rubric + few-shots) + `GrammarRouter` (JSON-schema-constrained decoding via `outlines` and MLX, warn-once fallback on failure). `--router`, `--router-repo`, `--router-mode` flags; default repo `mlx-community/Hermes-3-Llama-3.2-3B-4bit`. `harness eval router` fixture-based accuracy scoring; `scripts/` router-on-vs-off benchmark with RAM tracking. `outlines` pinned `<1.0` with the `datasets` transitive pin. Grammar extra in `pyproject.toml`.
- **Phase 3.4 — Textual TUI.** Seven-phase build: scaffold → adapter + persona + retrieval wiring → live metrics footer → tool loop + inline event rendering → token-delta streaming into RichLog → write-tier confirmation modal → history replay on mount + `/exit` / `:q` slash commands. Behind the optional `tui` extra and `--tui` flag; classic REPL untouched.

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
- Automatic capture prompt after each Airton turn (opt-in; keystroke to skip).

### Tier 5 — Preference learning (DPO)

Once captured hits ~200 samples, DPO beats SFT for register capture. Train the model to prefer gold-like responses over the draft shape. The draft/gold pairs produced by `eval voice --persona` are exactly the right shape.

### Tier 6 — Full SFT on an expanded corpus (stretch)

With 1000+ Airton-voiced responses, do a full supervised fine-tune (not LoRA). Voice becomes a property of the weights. At this stage it's worth reconsidering the base model — Qwen 2.5 Instruct's "helpful assistant" tuning is stubborn.

## Next-up candidates (no priority implied)

Tracked as `bd` issues now — run `bd ready` for the live list. Summary of current open candidates:

- **LoRA fine-tune (Phase 1e)** (`harness-kr4`) — first permanent voice move. Eval suite is dialed in enough to judge the result. Gate on captured corpus size (currently 1 — too low; capture more first).
- **Launchd daemon + scheduled consolidation + nightly backup** (`harness-bi3`, blocked on backup destination `harness-cxd`).
- **Web gateway (epic)** (`harness-g7y`) — FastAPI + SvelteKit ops console. Blocked on auth (`harness-55p`), rate limit, and backup destination.

Not yet filed in `bd` (forward look):

- Slack + Matrix gateways (after web gateway).
- Multi-agent roles (planner / researcher / executor / critic / persona). The architecture promised these; currently everything runs as a single-model tool-use loop.
- Kuzu graph layer. When we want relationship graphs over entities (who-works-with-whom, project-depends-on-project).
- Scheduled initiative. Phase 5 — Airton opens threads unprompted, reacts to external events.

## Robustness backlog

Deferred items from a 2026-04-16 critique pass. Not load-bearing for a single-user local CLI; relevant as the system opens to multiple gateways, multiple users, or automated scheduling. Filed here so we don't lose them.

- **Auth / authz on gateways.** Tailscale covers the network layer today. When web / Slack / Matrix gateways land, each needs its own identity check before writes hit memory. Blocks any gateway that isn't terminal-local.
- **Rate limiting.** Per-user throughput cap on turns + tool calls. Matters once something other than Mark can trigger generation.
- **Backup / restore workflow.** Beyond the existing open decision on destination: need a tested restore path and a nightly job. Likely pairs with launchd daemon.
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
