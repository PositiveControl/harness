# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local-first model-agent harness for a persistent character ("Airton"), built to be always-reachable by a small trusted circle over Tailscale. Starts as a single-user CLI co-worker; grows into a multi-gateway (web / Slack / Matrix), multi-agent, concurrent, self-maintaining system with memory, tools, and eventually initiative.

The model is plug-n-play: all model-specific code lives behind an adapter. Primary runtime is MLX on Apple Silicon (M4 Pro 48 GB). A Linux/larger Mac target may follow.

For daily-use workflow (how Mark actually talks to Airton), see `docs/usage.md`. For sequencing across phases, see `docs/roadmap.md`. This file is the architectural + commands reference.

## Commands

Environment setup (one time):

- `uv sync --extra all` — install the full core set in one shot: dev + MLX + retrieval + grammar-router + Textual TUI + tree-sitter symbol layer (`outline` / `read_file symbol=`) + browser (Playwright smoke gate) + web gateway. `all` is an aggregate extra = `harness[dev,mlx,retrieval,grammar,tui,code,browser,web]`. ALWAYS sync `all` — `uv sync` PRUNES any extra you omit, so a partial sync (e.g. `--extra browser` alone) silently rips out numpy / tree-sitter / fastapi and the next mypy / pytest / drive run breaks. Add a niche extra additively: `uv sync --extra all --extra asr`. Then `uv run playwright install chromium` once for the browser smoke gate.
- `uv run hf download mlx-community/Qwen2.5-7B-Instruct-4bit` — pull the default MLX model (~4 GB). For the fuller 32B model: `uv run hf download mlx-community/Qwen2.5-32B-Instruct-4bit` (~18 GB).
- `uv run pre-commit install --install-hooks && uv run pre-commit install --hook-type pre-push` — install git hooks.

Daily chat (see `docs/usage.md` for the intended workflow):

- `uv run harness chat --model mlx --persona --memories 3 --facts 5` — MLX + persona + 3 episodic + 5 semantic facts + retrieval-picked voice few-shot.
- `uv run harness chat --model mlx --persona --tools` — adds tool-use orchestrator (default `core` tool-set; `--tool-set {minimal,core_minimal,core,coding,memory,diagnostic,research,ops,reckon,full,atc,phraseology,notes,scholar,contract}`; escape hatches `--tools-add`, `--tools-drop`; `--workspace DIR` to sandbox elsewhere).
- `uv run harness chat --model mlx --persona --tools --router` — small-model intent router fronts tool loop. `--router-mode grammar` for schema-constrained decoding via `outlines`; `--router-repo` to override.
- `uv run harness chat --model mlx --persona --tui` — Textual chat app instead of REPL (requires `--extra tui`).
- `uv run harness chat --model ollama --model-repo qwen2.5-coder:32b-instruct --persona` — Ollama backend.
- `uv run harness chat` — echo-adapter dry run.
- `uv run harness describe` — dump Airton's resolved character sheet.

Edge-case chat flags (discover via `uv run harness chat --help`): `--model-repo`, `--lora-path`, `--draft-repo` (MLX speculative decoding), `--summarize-tool-results`, `--no-harvest-skills`, `--no-harvest-memories`, `--rewrite-on-tools`, `--compact-at`, `--compact-keep-recent`, `--auto-scribe`, `--dev`.

Session replay (paste-ready transcript dump for Claude Code or any other reader):

- `uv run harness session list` — list every recorded chat session, newest-active first, with turn counts and time range.
- `uv run harness session show [SESSION_ID]` — dump a session as markdown (no id = most recent). `--json` for JSONL, `--follow` to keep streaming as new turns land.

Voice corpus:

- `uv run harness voice capture --session X --gold "…"` — capture a corrected reply as a new voice sample for session X's last user prompt. Goes to `character/<name>/voice/captured.yaml`.
- Inside chat, type `/edit` at the `you ›` prompt — `$EDITOR` opens with Airton's last reply pre-loaded; save edits to capture a new voice sample without leaving the session.
- `uv run harness voice list-captured` — inspect the captured set.

Memory operations:

- `uv run harness memory ingest` — seed episodic from `character/<name>/seed_memories/` (idempotent).
- `uv run harness memory {list,search}` — browse / semantic-search episodic. `list` takes `--tier seed|consolidated|working|procedural`.
- `uv run harness memory {fact-list,fact-search,fact-add}` — browse / search / insert semantic facts. `fact-add SUBJECT PREDICATE OBJECT [--user NAME]` (shared if `--user` unset).
- `uv run harness memory scribe --session X --user NAME --model mlx` — extract candidate memories from unprocessed turns.
- `uv run harness memory consolidate` — cluster dupes (episodic) + merge fact groups, supersede losers.
- `uv run harness memory {harvest-skills,harvest-memories}` — on-demand bd harvest; chat does these at session start by default.
- `uv run harness memory rebuild-embeddings` — re-embed after embedder switch or chunking-format change. Non-destructive.
- `uv run harness memory wipe --yes` — clear episodic + semantic + scribe watermarks (transcripts preserved).

Evals (append `--json` for machine-readable output):

- `uv run harness eval voice --model mlx --top-k 6 --persona` — current best config. Flags: `--chain-rewrites` (2nd concrete-sub pass), `--judge` (LLM-judge), `--no-leave-one-out` (ceiling), `--top-k 0` (no retrieval), `--sample SAMPLE_ID`.
- `uv run harness eval router` — replay `character/<name>/router_eval.yaml`, score tool-selection accuracy. `--router-mode grammar --tool-set coding` for the constrained variant.
- `uv run harness eval session-resume` — replay `session_resume_eval.yaml`, pin the `build_resume_summary` contract (focus / in-progress / memories / drift).
- `uv run harness eval tool-loop` — replay `tool_loop_eval.yaml` against scripted adapters; includes fabrication-catcher attribution.

Other subcommand groups (`uv run harness <group> --help` for each):

- `drive` — multi-turn autonomous driver: `plan`, `lint-epic`, `loop`, `auto-iterate`, `logs`.
- `plan` — inspect / bootstrap / manage runtime-typed plans.
- `daemon`, `daemon-status` — heartbeat loop for scheduled maintenance, plus its state sidecar.
- `web` — serve the current character over HTTP.
- `tool` — inspect + manage the tool catalog. `denylist` — manage the `fetch_url` denylist.
- `phraseology` — cite-grounded ATC transmission verifier (airton_c1, JO 7110.65).

Quality gates (all must stay green; pre-commit runs them on every commit):

- `uv run ruff check .` — lint.
- `uv run ruff format .` — format in place.
- `uv run mypy src tests` — strict type-check (src + tests).
- `uv run pytest` — full test suite (currently ~4,290 tests across 197 files, ~100s).
- `uv run pytest tests/test_character.py::test_load_airton_shape` — single test.
- `uv run pre-commit run --all-files` — run all hooks against the working tree.

## Architecture

Load-bearing invariants — they shape almost every decision:

1. **One identity, one memory, one orchestrator — many gateways.** The character is a single coherent being. Gateways (CLI now; web/Slack/Matrix later) normalize to a common envelope and hand off to the orchestrator.
2. **Concurrency-ready from day one, concurrent from day N.** No globals. Every turn is keyed by `(user, channel, turn_id)`. Even the current CLI is written as if it were one of many sessions.
3. **Models stay behind an adapter.** Only `harness.model.*` imports MLX / llama.cpp / Ollama / OpenAI SDKs. The rest of the system speaks `ChatMessage` + `ModelAdapter.complete()`.
4. **Auditable memory.** Every mutation is attributed to a speaker, turn, and reason. Memory writes are two-phase: scribe emits candidates, consolidator promotes on a periodic pass. Superseded rows stay for audit; they drop out of retrieval.
5. **Hybrid memory scoping.** Shared character memory (seeds, project facts) lives at `user_id IS NULL` and is visible to everyone. Per-user relationship memory is siloed — `search(..., user_id=<this user>)` only returns that user's rows plus shared ones.

### Repo layout

- `character/<name>/` — character as data (config, not code). `core.yaml` (identity), `constitution.md`, `voice/{canonical,captured}.yaml`, `seed_memories/*.md` (frontmatter-tagged by principle).
- `src/harness/` — runtime (package `harness`).
  - `model/` — **adapter boundary**. Only place that imports MLX / Ollama / model SDKs. `adapter.py` defines `ChatMessage` + `ModelAdapter` (incl. `complete_with_tools`); `mlx.py` (Qwen 2.5 7B 4-bit default, supports `--model-repo`, `--lora-path`, `--draft-repo` speculative decoding); `ollama.py`; `echo.py`; `factory.py`.
  - `persona/rewriter.py` — voice-rewrite post-pass. Off by default when tools ran (`rewrite_on_tools=False`) — rewriter compresses.
  - `retrieval/` — `Embedder` protocol + `SentenceTransformersEmbedder` (default `BAAI/bge-small-en-v1.5`, 384-dim; override via `HARNESS_EMBEDDER_REPO`). `voice_retriever.py`.
  - `store/` — SQLite + BLOB embeddings + FTS5 sidecar. `transcript.py`, `episodic.py`, `semantic.py`, `_hybrid.py` (RRF fusion). Rows carry `embedder_id` + `embedding_dim`; mismatched dims filtered. Episodic embed text is contextually chunked (`[tier: X; principle: Y; date: Z]` header). Facts have temporal validity (`valid_from` / `valid_to` / `asserted_at`; `search(as_of=...)`). All stores set `PRAGMA busy_timeout = 5000`. Graduate to LanceDB past ~10k rows.
  - `scribe/` — batch transcript → memory extraction. Watermark per session; `fcntl` per-session lock serializes concurrent runs.
  - `consolidate/consolidator.py` — partition by `user_id`, single-link episodic cluster at cosine ≥ 0.80, merge facts by `(subject, predicate)`. Shared (`user_id IS NULL`) is its own partition.
  - `skills/` — idempotent bd harvesters into episodic tier=`procedural`. `harvester.py` ingests closed `thought:decision` / `thought:observation` beads (`external_id=bead_id`); `memory_harvester.py` mirrors `bd remember` / `retro record` (`external_id=bd-mem:<key>`).
  - `compaction/` — folds older turns at `compact_at × window`. Auto-scribes unprocessed turns *before* folding so `search_memory` survives compression.
  - `tools/` — 60 built-in tools in `catalog.py`'s `BUILTIN_TOOL_METADATA` (ops 21, filesystem 10, memory 8, reckon 7, research 6, meta 4, git 3, atc 1); 58 of them are reachable from a profile, the other two (`citation_lookup`, `query_table`) only via `--tools-add` / `load_tool`. `profiles.py` defines 15 named profiles — general-purpose (`minimal`, `core_minimal`, `core`, `coding`, `memory`, `diagnostic`, `research`, `ops`, `reckon`, `full`) plus character-scoped (`atc`, `phraseology`, `notes`, `scholar`, `contract`) — each with its own schema budget (see the module docstring; `minimal`/`diagnostic`/`phraseology` ≤1k tokens, `core` ≤2k, `coding`/`atc`/`scholar`/`ops` ≤3.5k, `full` unbounded). Write-tier marked + requires per-session confirmation. Filesystem + git + shell tools sandboxed to `--workspace`. Includes `introspect` (scope enum: tools/model/memory/character/commands/session/all), `spawn_subagent` (read-only, depth-1, shared hooks), `fetch_url` (HTTPS-only, 2MB cap, research/coding only), `search_web`, `remember_{fact,event}`, `scribe_session`, `consolidate_memory`.
  - `router/` — small-model intent router. `Router` protocol + `RouterResult`; `ModelRouter` (tolerant-JSON); `GrammarRouter` (`outlines` schema-constrained). Advisory: null / write-tier / unparseable falls through.
  - `orchestrator/` — `tool_loop.py` runs model↔tool cycle. `hooks.py` typed four-phase pipeline: `post_model` / `bail` (first-match: Truncated, Unparseable, Teaser, FalseSuccess, MetaConfirm, FabricatedSearch, FabricatedItemization, AbFabrication, ToolIntent) / `pre_tool` (DuplicateCall, ArgumentGrounding) / `post_tool` (opt-in ToolResultSummarizer) / `finalize` (FabricationFallback). Per-hook toggles via `disabled: frozenset[str]` for attribution evals.
  - `tui/` — Textual chat app (optional, `--tui`).
  - `turn/service.py` — one grounded chat turn (retrieval → prompt assembly → history → tool loop → rewrite → persistence → audit), lifted out of CLI-only assembly so CLI, TUI, web, and daemon callers all run the same path.
  - `driver/` — multi-turn autonomous driver behind `harness drive` (`plan` / `lint-epic` / `loop` / `auto-iterate` / `logs`). 23 modules, the largest subsystem here: phase FSM (`turn_fsm.py`, `fsm_turn.py`, `state.py`), gate-blindness rejection (`gate_blind.py`), gate synthesis (`gate_synth.py`, `runtime_gate_synth.py`), park/revive guards (`premise_guard.py`, `workspace_guard.py`), critic convergence (`auto_iterate.py`, `critic.py`), browser smoke (`smoke_runner.py`, needs chromium). Reference doc pending — see `harness-mgmz`.
  - `plan/` — runtime-typed plan structure (`Plan` values the orchestrator reasons over directly, no subprocess hop per turn). bd is one backend among N: `bd_source.py`, `writeback.py`, `store.py`. CLI surface: `harness plan`.
  - `runtime/` — clock-driven heartbeat loop separate from chat turns; runs maintenance tasks (compaction, consolidation, scheduled tool calls) at configured intervals. CLI surface: `harness daemon` / `harness daemon-status`.
  - `web/` — FastAPI factory that serves any character over HTTP (`harness web`); per-character extensions under `web/characters/`.
  - `notam/parser.py` — standalone NOTAM/TFR parser for `airton_c_tfr`. Import-clean (stdlib + optional `pyproj`, lazily imported), no `harness.*` imports, so it can be vendored or split out.
  - `character_templates/` — archetype skeletons (currently `atc/`) that `scripts/character_from_template.py` copies into `character/<new_name>/`, substituting `{{CHARACTER_NAME}}`.
  - `evals/` — voice / router / session-resume / tool-loop.
  - `cli.py` — Typer app. See Commands above.
- `tests/` — pytest, hits real SQLite in `tmp_path`. ~4,290 tests. Browser-backed tests gate on `tests/browser_probe.py` (needs `uv run playwright install chromium`, else they skip).
- `scripts/` — benchmarks (model-speed, tool-use with `--measure-tokens`, router-on-vs-off with RAM tracking).

### Voice stack

Per-turn composition:

1. **Retrieval** — `VoiceRetriever.top_k(user_message, k=6)` picks the 6 most similar voice samples by cosine.
2. **System prompt** — `Character.system_prompt(include_samples=retrieved)` renders identity + values + taboos + style rules + retrieved examples.
3. **Episodic memory** — `EpisodicStore.search(user_message, k=3, min_score=0.5, user_id=speaker, mode="hybrid")` appends a memory block when hits clear the floor. Hybrid mode fuses BM25 (FTS5) and dense cosine via reciprocal-rank fusion so identifier-heavy and paraphrase-shaped queries both land. Harvested bd thought-graph rows (tier=`procedural`) ride the same path.
4. **Semantic facts** — `SemanticStore.search(user_message, k=5, min_score=0.45, user_id=speaker, mode="hybrid", as_of=now)` appends a facts block when hits clear the floor. `as_of` filters out facts whose validity window doesn't cover the query time — expired or future-only facts stay in the store for audit but don't reach the prompt.
5. **Pass 1 (substance)** — `base_adapter.complete([system, ...history])`.
6. **Pass 2 (voice rewrite)** — `base_adapter.complete(build_rewriter_messages(character, draft))`. Preserves substance, fixes register.
7. **Optional pass 3** — concrete-substitution rewrite (`chain_rewrites=True`). Opt-in; doubles persona latency.

### Memory stack

Four layers of writes:

- **Seeds** (`tier="seed"`, `user_id IS NULL`) — loaded from `character/<name>/seed_memories/` and `character/<name>/voice/canonical.yaml`. Shared across everyone.
- **Working** (`tier="working"`, `user_id=<speaker>` or `NULL`) — scribe-written candidates from recent transcript. Not yet curated.
- **Consolidated** (`tier="consolidated"`) — consolidator-promoted records that merge near-duplicate working-tier entries.
- **Procedural** (`tier="procedural"`, `user_id IS NULL`) — two substrates, same destination: (a) bd thought-graph harvest (`harness-vu3`) of closed `thought:decision` / `thought:observation` beads via `skills/harvester.py`, idempotent on `external_id=bead_id`; (b) bd-memory mirror (`harness-9yd`) of every `bd remember` / `retro record` entry via `skills/memory_harvester.py`, idempotent on `external_id=bd-mem:<key>`. Shared for both.

Retrieval filters: `superseded_by IS NULL AND embedding_dim = <current> AND (user_id IS NULL OR user_id = <speaker>)`. Semantic search also applies the temporal-validity filter against `as_of` (default = now).

### Tool-use stack

When `--tools` is set, the chat loop hands the adapter a tool schema and enters `orchestrator/tool_loop.py`:

1. Adapter calls `complete_with_tools(messages, tools)` — MLX and Ollama both implement this.
2. If the model emits tool calls, the loop executes each tool (write-tier tools prompt for confirmation on first use per session), appends the result to the working message list, and loops.
3. When the model returns a final reply (no tool calls), the loop exits and that reply is the turn's output.
4. Persona rewrite is **off by default** when tools ran — the rewriter compresses, which is wrong for multi-step investigations. `--rewrite-on-tools` opts back in for casual tool use.
5. If the prompt + tool results approach `--compact-at × context_window` tokens, `compaction/` folds older turns into a session summary. The transcript is unchanged — only the model-visible history shrinks.

Filesystem tools (`read_file` / `list_dir` / `grep` / `glob` / `edit_file` / `write_file` / `shell`) and git-read tools (`git_status` / `git_diff` / `git_log`) are sandboxed to `--workspace` (default: the harness repo root). Memory + transcripts stay under the harness data dir regardless.

When `--router` is on, the orchestrator hands the turn to a small-model intent router *before* round 0: on a confident read-tier classification it executes the tool itself and the main model only does a wrap-up round — skipping the fabricate-and-nudge rounds that small models are prone to. `null` / write-tier / unparseable router results fall through to the normal loop.

### Roadmap

Landed work lives in `git log` + bd closed issues. Open work tracked in `bd ready`; see `AGENTS.md` for workflow. Not-yet-started candidates (no priority): LoRA fine-tune on Qwen 2.5 32B, web gateway (FastAPI + SvelteKit; blocked on auth/rate-limit/backup), Slack + Matrix gateways, scheduled nightly consolidation, off-box backup destination, multi-agent roles, Kuzu graph layer.

## Conventions

- **Don't import MLX, llama.cpp, or any model SDK outside `src/harness/model/`.** The adapter boundary is load-bearing.
- **Every store is append-first, attributed, and timestamped.** New memory layers start from `store/episodic.py` as the template.
- **Tests hit real stores.** No mocking of SQLite. Use `tmp_path`.
- **Character data is configuration, not code.** Never hardcode Airton's rules into `src/`. The runtime reads `character/<name>/`.
- **Identity first, then voice, then content.** System prompt starts with premise + self-awareness + values + taboos, then retrieved examples, then memory/facts. If you find yourself re-stating identity inline in code, it belongs in `core.yaml`.
- **User scoping is mandatory on retrieval.** Any `.search()` call that serves a user-facing turn must pass `user_id=speaker`. Omitting it is an owner-tier view — fine for dev tools, wrong for chat.
- **Ab thought-graph beads carry `assignee=airton_b`.** User-captured work is assigned to the user. The adapter's `default_exclude_assignee` hides ab-owned beads from default plan/list/drift/search views unless `--include-internal` (or `--dev`) is set. Any new ab-scoped code path must respect the convention so the user doesn't get buried under ab's scratchpad.
- **Thought labels are free-form, `thought:` prefix.** Common examples: `thought:hypothesis`, `thought:question`, `thought:decision`, `thought:observation`. Not an enum — pick the term that names the cognitive step.

## Quality tooling

Pre-commit hooks are installed (`pre-commit` + `pre-push`) and enforce:

- **ruff** (`check` + `format`) with rules: `E W F I UP B C4 SIM RET TID PT S N RUF`. Global ignore: `S101`. Per-file ignores in `pyproject.toml`.
- **mypy** in `strict` mode — `warn_unused_ignores`, `warn_redundant_casts`, `warn_return_any`, `warn_unreachable`. The `harness` package ships a `py.typed` marker.
- **pytest** at push time — gate pushes, not individual commits.

Rules of engagement:

- Don't silence a failing check with blanket `# noqa` / `# type: ignore`. Fix it, or add a targeted per-file ignore with a comment explaining why.
- `ruff format` owns layout; don't hand-format. If formatter and rule disagree, change the rule.
- New deps go in `pyproject.toml`. Run `uv sync --extra all` after (`uv sync` prunes any extra you omit; `all` is the aggregate that keeps the full core set installed). A new *core* extra must also be added to the `all` list in `pyproject.toml`.
