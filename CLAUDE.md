# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local-first model-agent harness for a persistent character ("Airton"), built to be always-reachable by a small trusted circle over Tailscale. Starts as a single-user CLI co-worker; grows into a multi-gateway (web / Slack / Matrix), multi-agent, concurrent, self-maintaining system with memory, tools, and eventually initiative.

The model is plug-n-play: all model-specific code lives behind an adapter. Primary runtime is MLX on Apple Silicon (M4 Pro 48 GB). A Linux/larger Mac target may follow.

For daily-use workflow (how Mark actually talks to Airton), see `docs/usage.md`. For sequencing across phases, see `docs/roadmap.md`. This file is the architectural + commands reference.

## Commands

Environment setup (one time):

- `uv sync --extra dev --extra mlx --extra retrieval --extra grammar --extra tui` — install runtime + dev + MLX + retrieval + grammar-router + Textual TUI.
- `uv run hf download mlx-community/Qwen2.5-7B-Instruct-4bit` — pull the default MLX model (~4 GB). For the fuller 32B model: `uv run hf download mlx-community/Qwen2.5-32B-Instruct-4bit` (~18 GB).
- `uv run pre-commit install --install-hooks && uv run pre-commit install --hook-type pre-push` — install git hooks.

Daily chat (see `docs/usage.md` for the intended workflow):

- `uv run harness chat --model mlx --persona --memories 3 --facts 5` — full-stack chat: MLX + persona rewriter + 3 episodic memories + 5 semantic facts + retrieval-picked voice few-shot.
- `uv run harness chat --model mlx --persona --tools` — same, plus the tool-use orchestrator (defaults to the `core` tool-set; pass `--tool-set coding` for full read/write/shell/git, `memory` for recall-plus-write, `research` for read + search_web, etc.). Add `--workspace DIR` to point the filesystem tools at another repo. Escape hatches: `--tools-add X,Y` and `--tools-drop Z`.
- `uv run harness chat --model mlx --persona --tools --router` — same, with a small-model intent router (default `mlx-community/Hermes-3-Llama-3.2-3B-4bit`) fronting the tool loop: when it confidently classifies the turn into a read-tier tool call, the orchestrator executes the tool itself and the main model only does a wrap-up round. `--router-mode grammar` switches to JSON-schema-constrained decoding via `outlines` (requires the `grammar` extra). `--router-repo` overrides the router model.
- `uv run harness chat --model mlx --persona --tui` — launch the Textual chat app instead of the classic REPL: persistent input at the bottom, scrolling RichLog above, live context + elapsed metrics footer, write-tier confirmation modal, history replay on mount, `/exit` + `:q` slash commands. Requires `--extra tui`.
- `uv run harness chat --model mlx --model-repo mlx-community/Qwen2.5-Coder-32B-Instruct-4bit --persona` — override the default model repo. Works for `chat`, `eval voice`, and `memory scribe`.
- `uv run harness chat --model mlx --lora-path ./adapters/airton --persona` — apply a LoRA adapter (directory from `mlx_lm.lora`) on top of the base MLX model.
- `uv run harness chat --model ollama --model-repo qwen2.5-coder:32b-instruct --persona` — Ollama backend.
- `uv run harness chat` — echo-adapter dry run (no model, just wiring).
- `uv run harness describe` — dump Airton's resolved character sheet.

Voice corpus:

- `uv run harness voice capture --session X --gold "…"` — capture a corrected reply as a new voice sample for session X's last user prompt. Goes to `character/<name>/voice/captured.yaml`.
- Inside chat, type `/edit` at the `you ›` prompt — `$EDITOR` opens with Airton's last reply pre-loaded; save edits to capture a new voice sample without leaving the session.
- `uv run harness voice list-captured` — inspect the captured set.

Memory operations:

- `uv run harness memory ingest` — seed the episodic store from `character/<name>/seed_memories/` (idempotent).
- `uv run harness memory list [--tier seed|consolidated|working]` — table of episodic records.
- `uv run harness memory search "query"` — semantic top-K over episodic.
- `uv run harness memory fact-list [--tier X] [--subject Y]` — semantic facts.
- `uv run harness memory fact-search "query"` — semantic top-K over facts.
- `uv run harness memory fact-add SUBJECT PREDICATE OBJECT [--user NAME]` — manual fact insert (shared if `--user` unset).
- `uv run harness memory scribe --session X --user NAME --model mlx` — batch-extract candidate memories from the session's unprocessed turns.
- `uv run harness memory consolidate` — cluster near-duplicate episodes, merge fact groups by `(subject, predicate)`, supersede the losers.
- `uv run harness memory rebuild-embeddings` — re-embed all active rows with the current embedder. Non-destructive; use after an embedder switch.
- `uv run harness memory wipe --yes` — clear episodic + semantic + scribe-watermark data (transcripts preserved).

Voice + persona evals:

- `uv run harness eval voice --model mlx --top-k 6 --persona` — current best config.
- `uv run harness eval voice --model mlx --persona --chain-rewrites --judge` — add the concrete-substitution second rewrite pass and LLM-judge scoring.
- `uv run harness eval voice --model mlx --no-leave-one-out` — ceiling diagnostic: every sample sees the full example set.
- `uv run harness eval voice --model mlx --top-k 0` — disable retrieval, show every sample.
- `uv run harness eval voice --model mlx --sample SAMPLE_ID` — run one sample.
- Append `--json` for machine-readable output.

Router evals:

- `uv run harness eval router` — replay `character/<name>/router_eval.yaml` through the configured router and score tool-selection accuracy. Locks in quality before swapping router models or editing the router prompt.
- `uv run harness eval router --router-mode grammar --tool-set coding` — grammar-constrained router scored against a different profile's tool set.
- Append `--json` for machine-readable output.

Quality gates (all must stay green; pre-commit runs them on every commit):

- `uv run ruff check .` — lint.
- `uv run ruff format .` — format in place.
- `uv run mypy src tests` — strict type-check (src + tests).
- `uv run pytest` — full test suite (currently ~470 tests).
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

- `character/airton/` — the character as data. Treat as configuration, not code.
  - `core.yaml` — immutable identity: premise, pronouns, values, taboos, directives, deep/shallow domains, on-being-wrong register.
  - `constitution.md` — principles the critic-style pass enforces at generation time.
  - `voice/canonical.yaml` — 32 curated voice samples; doubles as drift-eval suite.
  - `voice/captured.yaml` — accrues live captures from `harness voice capture`. Loaded alongside canonical at character load.
  - `seed_memories/*.md` — formative "lived experiences" written into episodic memory on day 0. Frontmatter-tagged by principle so retrieval can surface them by lesson as well as by content.

- `src/harness/` — runtime (src layout, package `harness`).
  - `config.py` — pydantic-settings; env-prefixed `HARNESS_*`.
  - `character.py` — loads `character/<name>/` into a frozen `Character` dataclass; renders a fallback `system_prompt()`. Merges `voice/canonical.yaml` + `voice/captured.yaml`.
  - `_quiet.py` — suppresses HF / transformers / sentence-transformers startup noise so the CLI stays readable.
  - `model/` — adapter boundary. Anything model-specific lives here and nowhere else.
    - `adapter.py` — `ChatMessage` + `ModelAdapter` protocol (including `complete_with_tools` for tool-use).
    - `echo.py` — deterministic adapter for wiring tests.
    - `mlx.py` — MLX-backed adapter (Qwen 2.5 7B Instruct 4-bit by default; 32B / Qwen2.5-Coder available via `--model-repo`). Streams tokens. Accepts `model_repo` and `lora_path` overrides. Lazy load.
    - `ollama.py` — Ollama-backed adapter with tool-call + token-streaming support. Model tag via `model_repo`.
    - `factory.py` — `make_adapter("echo" | "mlx" | "ollama", model_repo=, lora_path=)`. Backends imported lazily.
  - `persona/rewriter.py` — `PersonaAdapter` wraps a base adapter with a voice-rewrite post-pass. Optional chain-of-rewrite (`chain_rewrites=True`) adds a second concrete-substitution pass. Off by default when a turn used tools (`rewrite_on_tools=False`) — rewriter compresses, which is wrong for investigate/summarize replies.
  - `retrieval/` — embedding-backed similarity search.
    - `embed.py` — `Embedder` Protocol. Implementations return L2-normalized vectors so cosine similarity is a dot product.
    - `st_embedder.py` — `SentenceTransformersEmbedder`; default `mixedbread-ai/mxbai-embed-large-v1` (1024 dim, MPS on Mac).
    - `voice_retriever.py` — embeds all voice samples once on construction; `top_k(query, k=, exclude_ids=)`.
  - `store/` — persistent stores (SQLite + BLOB embeddings + cosine scan). Graduate to LanceDB when past ~10k rows.
    - `transcript.py` — append-only transcript (WAL + FTS5). `fetch_after(session, after_id=)` for scribe.
    - `episodic.py` — `EpisodicStore`: title / body / principle / tags / tier / source / user_id / `superseded_by` + embedding BLOB. Tiers: `seed` · `consolidated` · `working`.
    - `semantic.py` — `SemanticStore`: `(subject, predicate, object)` triples with confidence, provenance, tier, `supersedes`, `superseded_by`.
    - Both stores carry `embedder_id` + `embedding_dim` per row; search filters mismatched dims so a new embedder doesn't crash. `rebuild_embeddings()` migrates in place.
    - All stores set `PRAGMA busy_timeout = 5000` so concurrent writes wait instead of erroring.
  - `scribe/` — batch extraction from transcript to memory stores.
    - `extractor.py` — `extract_candidates(adapter, character, turns)`; strict-JSON rubric, tolerates markdown fences + garbage.
    - `runner.py` — `run_scribe(..., user_id=)`; watermark-tracked per session; non-overlapping windows.
    - `locks.py` — `fcntl`-based per-session lock so concurrent scribe runs on the same session serialize instead of racing the watermark.
  - `consolidate/consolidator.py` — `run_consolidation(episodic, semantic)`. Episodic partitions by `user_id` then single-link clusters at cosine ≥ 0.80 within each partition, picking most-recent as representative. Semantic groups by `(user_id, subject, predicate)` with the subject/predicate case-folded, picking highest-confidence-then-recent. Shared rows (`user_id IS NULL`) form their own partition — never merged with private memory. Originals marked `superseded_by → new_id`; retrieval filters them out.
  - `compaction/` — context-window management.
    - `store.py` — persists per-session compaction summaries so reruns don't re-summarize unchanged history.
    - `summarizer.py` — folds older turns into a single session summary using the same adapter.
    - `runner.py` — `maybe_compact(messages, tokens_used, window, compact_at, keep_recent)`; fires when the context meter crosses `--compact-at` (default 0.8 of the window) and leaves `--compact-keep-recent` turns verbatim.
  - `tools/` — built-in tools for the agent loop. Currently 17 tools across filesystem, shell, git, memory, and web domains.
    - `base.py` — `Tool` protocol + `ToolResult`; tools declare schema, execute given a workspace-scoped context, return content + optional metadata. Write-tier tools are marked and trigger per-session user confirmation.
    - `profiles.py` — named tool-set profiles (`minimal`, `core`, `coding`, `memory`, `diagnostic`, `research`) that group tools by use case. `resolve_tool_names(profile, add=, drop=)` returns the final set. Profiles may list forward-compatible names that don't exist yet; the CLI warns + skips. Each profile targets ≤ ~1,500 tokens of schema overhead.
    - Filesystem read: `read_file.py`, `list_dir.py`, `grep.py`, `glob.py`.
    - Filesystem write: `edit_file.py` (partial edits; empty `old_string` = append), `write_file.py` (refuses overwrite by default; nudges toward `edit_file`), `shell.py`. All sandboxed to `--workspace`.
    - Git read: `git.py` — `git_status`, `git_diff`, `git_log`.
    - Memory read: `search_memory.py`, `search_facts.py` — scoped to the speaker.
    - Memory write: `remember.py` — `remember_fact`, `remember_event`. Ops: `ops.py` — `scribe_session`, `consolidate_memory`.
    - Web read: `search_web.py` — stdlib DuckDuckGo HTML scrape.
  - `router/` — small-model intent router that fronts the tool loop.
    - `intent.py` — `Router` protocol + `RouterResult`. Advisory: `null` / write-tier / unparseable intents fall through to the main loop.
    - `model_router.py` — free-form JSON router (tolerant parse, tool-name + arg validation against the active tool set). System prompt has few-shots + a null rubric.
    - `grammar_router.py` — JSON-schema-constrained decoding via `outlines` and MLX. Guarantees valid output + valid tool name by construction; costs ~1 GB RAM for the FSM. Warns once and falls back to free mode if outlines is missing or the schema fails.
  - `orchestrator/tool_loop.py` — runs the adapter's `complete_with_tools` loop: model → tool calls → execute → feed results back → repeat until no more tool calls. Handles display labels, token counting, streaming, retrieval degradation (skips retrieval when a tool run is clearly on rails), duplicate-call short-circuit, wrap-up cap (default 384 tokens) with a widened cap on truncated recovery, and hallucination catchers (fabricated tool-call success, fabricated search results, bare tool-intent with no call, quoted-snippet numbered lists, paired meta-confirm).
  - `tui/` — Textual chat app (optional, behind the `tui` extra and `--tui` flag).
    - `chat_app.py` — scaffold → live adapter wiring → streaming RichLog → tool-loop inline event rendering → live ctx + elapsed footer → write-tier confirmation modal → history replay on mount + `/exit` / `:q` slash commands.
    - `confirm_screen.py` — modal screen that asks for write-tier confirmation.
  - `evals/` — offline evals.
    - `voice.py` — `run_voice_eval(..., retriever=, persona=, use_judge=, chain_rewrites=)`.
    - `voice_score.py` — heuristic scorer: length / openers / bullet-discipline / bullet-density / filler. Aggregate is the mean.
    - `voice_judge.py` — LLM-as-judge; parses 1-10 from the adapter.
    - `router.py` — fixture-based router eval: loads `character/<name>/router_eval.yaml`, runs each prompt through the router, scores tool-name accuracy + arg-shape match.
  - `cli.py` — Typer app: `chat` (with `--tools`, `--tool-set`, `--tools-add/drop`, `--workspace`, `--model-repo`, `--lora-path`, `--rewrite-on-tools`, `--compact-at`, `--compact-keep-recent`, `--router`, `--router-repo`, `--router-mode`, `--tui`, `--dev`, in-chat `/edit` + `/capture` for voice capture), `describe`, `eval {voice,router}`, `memory {list,search,scribe,consolidate,wipe,rebuild-embeddings,fact-*,ingest}`, `voice {capture,list-captured}`.

- `tests/` — pytest. Tests hit real stores (SQLite in `tmp_path`) rather than mocks. ~470 tests across character, stores, retrieval, scribe, consolidator, persona, voice eval, dimension tracking, relationship memory, voice capture, tool loop, all 17 tools, compaction, Ollama adapter, CLI helpers, router (model + grammar), router eval, TUI chat app.

- `scripts/` — benchmarks and one-offs (model-speed benchmark, tool-use benchmark with `--measure-tokens` for per-tool schema + result cost, router-on-vs-off benchmark with RAM tracking).

### Voice stack

Per-turn composition:

1. **Retrieval** — `VoiceRetriever.top_k(user_message, k=6)` picks the 6 most similar voice samples by cosine.
2. **System prompt** — `Character.system_prompt(include_samples=retrieved)` renders identity + values + taboos + style rules + retrieved examples.
3. **Episodic memory** — `EpisodicStore.search(user_message, k=3, min_score=0.5, user_id=speaker)` appends a memory block when hits clear the floor.
4. **Semantic facts** — `SemanticStore.search(user_message, k=5, min_score=0.45, user_id=speaker)` appends a facts block when hits clear the floor.
5. **Pass 1 (substance)** — `base_adapter.complete([system, ...history])`.
6. **Pass 2 (voice rewrite)** — `base_adapter.complete(build_rewriter_messages(character, draft))`. Preserves substance, fixes register.
7. **Optional pass 3** — concrete-substitution rewrite (`chain_rewrites=True`). Opt-in; doubles persona latency.

### Memory stack

Three layers of writes:

- **Seeds** (`tier="seed"`, `user_id IS NULL`) — loaded from `character/<name>/seed_memories/` and `character/<name>/voice/canonical.yaml`. Shared across everyone.
- **Working** (`tier="working"`, `user_id=<speaker>` or `NULL`) — scribe-written candidates from recent transcript. Not yet curated.
- **Consolidated** (`tier="consolidated"`) — consolidator-promoted records that merge near-duplicate working-tier entries.

Retrieval filters: `superseded_by IS NULL AND embedding_dim = <current> AND (user_id IS NULL OR user_id = <speaker>)`.

### Tool-use stack

When `--tools` is set, the chat loop hands the adapter a tool schema and enters `orchestrator/tool_loop.py`:

1. Adapter calls `complete_with_tools(messages, tools)` — MLX and Ollama both implement this.
2. If the model emits tool calls, the loop executes each tool (write-tier tools prompt for confirmation on first use per session), appends the result to the working message list, and loops.
3. When the model returns a final reply (no tool calls), the loop exits and that reply is the turn's output.
4. Persona rewrite is **off by default** when tools ran — the rewriter compresses, which is wrong for multi-step investigations. `--rewrite-on-tools` opts back in for casual tool use.
5. If the prompt + tool results approach `--compact-at × context_window` tokens, `compaction/` folds older turns into a session summary. The transcript is unchanged — only the model-visible history shrinks.

Filesystem tools (`read_file` / `list_dir` / `grep` / `glob` / `edit_file` / `write_file` / `shell`) and git-read tools (`git_status` / `git_diff` / `git_log`) are sandboxed to `--workspace` (default: the harness repo root). Memory + transcripts stay under the harness data dir regardless.

When `--router` is on, the orchestrator hands the turn to a small-model intent router *before* round 0: on a confident read-tier classification it executes the tool itself and the main model only does a wrap-up round — skipping the fabricate-and-nudge rounds that small models are prone to. `null` / write-tier / unparseable router results fall through to the normal loop.

### Phase progress

Done:

- **Phase 0** — Skeleton.
- **Phase 1a/b/d** — MLX adapter, voice eval suite, few-shot + retrieval, persona rewriter, heuristic + judge scoring, chain-of-rewrite, episodic store, semantic store, scribe, consolidator, dimension tracking.
- **Phase 2.0** — Relationship memory (per-user scoping).
- **Phase 2.1** — Corpus growth via `voice capture`.
- **Phase 3.0+3.1** — Tool use: initial 5 built-in tools (read_file, write_file, shell, search_memory, search_facts), orchestrator loop, `--tools` flag, `--workspace` sandbox, write-tier confirmation, streaming tokens, context meter, automatic compaction, retrieval degradation when on-rails, Ollama adapter with tool-call support. Robustness hardening: scribe fcntl session lock, SQLite `busy_timeout` pragma, HF/transformers startup noise suppression, `--model-repo` and `--lora-path` flags across chat/eval/scribe.
- **Phase 3.2 — Tool expansion + hardening.** 17 tools across filesystem read/write (`list_dir`, `grep`, `glob`, `edit_file` with append semantics, `write_file` overwrite refusal), shell, git read (`git_status`, `git_diff`, `git_log`), memory write (`remember_fact`, `remember_event`, `scribe_session`, `consolidate_memory`), and web (`search_web`). Tool-set profiles (`minimal` / `core` / `coding` / `memory` / `diagnostic` / `research`) with `--tool-set`, `--tools-add`, `--tools-drop`. Orchestrator hallucination catchers (fabricated tool-call success, fabricated search results, bare tool-intent, numbered-list quoted snippets, paired meta-confirm), wrap-up round cap, duplicate-call short-circuit, stream-level meta-confirm filter gated behind `--dev`, structured loading header.
- **Phase 3.3 — Intent router.** `Router` protocol + `ModelRouter` (free-form JSON + tolerant parse) + `GrammarRouter` (JSON-schema-constrained decoding via `outlines`). `--router`, `--router-repo`, `--router-mode` flags. Default router model: `mlx-community/Hermes-3-Llama-3.2-3B-4bit`. `harness eval router` with fixture-based scoring.
- **Phase 3.4 — Textual TUI.** `--tui` flag launches a full Textual chat app: persistent input, scrolling RichLog, live ctx + elapsed footer, tool-loop inline event rendering, token-delta streaming, write-tier confirmation modal, history replay on mount, `/exit` + `:q` slash commands. Behind the optional `tui` extra.
- **Voice-capture ergonomics.** In-chat `/edit` (and alias `/capture`) opens `$EDITOR` with Airton's last reply pre-loaded; saving captures a new voice sample without leaving the session.

Available but not started (no priority implied — tracked as `bd` issues):

- LoRA fine-tune on Qwen 2.5 32B using the voice suite as training data (roadmap Tier 3).
- Web gateway (FastAPI + SvelteKit ops console). Blocked on auth + rate limit + backup decisions.
- Slack + Matrix gateways.
- Scheduled consolidation (launchd → nightly).
- Off-box backup destination — decision still open.
- Multi-agent roles (planner / researcher / executor / critic / persona).
- Kuzu graph layer.

Issue tracker: `bd ready` — see `AGENTS.md` for workflow.

## Conventions

- **Don't import MLX, llama.cpp, or any model SDK outside `src/harness/model/`.** The adapter boundary is load-bearing.
- **Every store is append-first, attributed, and timestamped.** New memory layers start from `store/episodic.py` as the template.
- **Tests hit real stores.** No mocking of SQLite. Use `tmp_path`.
- **Character data is configuration, not code.** Never hardcode Airton's rules into `src/`. The runtime reads `character/<name>/`.
- **Identity first, then voice, then content.** System prompt starts with premise + self-awareness + values + taboos, then retrieved examples, then memory/facts. If you find yourself re-stating identity inline in code, it belongs in `core.yaml`.
- **User scoping is mandatory on retrieval.** Any `.search()` call that serves a user-facing turn must pass `user_id=speaker`. Omitting it is an owner-tier view — fine for dev tools, wrong for chat.

## Quality tooling

Pre-commit hooks are installed (`pre-commit` + `pre-push`) and enforce:

- **ruff** (`check` + `format`) with rules: `E W F I UP B C4 SIM RET TID PT S N RUF`. Global ignore: `S101`. Per-file ignores in `pyproject.toml`.
- **mypy** in `strict` mode — `warn_unused_ignores`, `warn_redundant_casts`, `warn_return_any`, `warn_unreachable`. The `harness` package ships a `py.typed` marker.
- **pytest** at push time — gate pushes, not individual commits.

Rules of engagement:

- Don't silence a failing check with blanket `# noqa` / `# type: ignore`. Fix it, or add a targeted per-file ignore with a comment explaining why.
- `ruff format` owns layout; don't hand-format. If formatter and rule disagree, change the rule.
- New deps go in `pyproject.toml`. Run `uv sync --extra dev --extra mlx --extra retrieval --extra grammar` after.

## Open threads

- **Backup destination**: still TBD. Default until chosen: SQLite WAL + local Time Machine + nightly tarball of the `data/` dir. Candidates: Backblaze B2, iCloud Drive, S3, NAS, another Mac.
- **LLM-judge model**: currently uses the same Qwen that generated the reply (circular). Swap in a different/stronger judge when we want orthogonal signal.
- **Voice corpus held-out split**: once captured samples cross ~50, reserve ~20% for a true generalization eval that never appears in few-shot.
