# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local-first model-agent harness for a persistent character ("Airton"), designed to be always-reachable by a small trusted circle over Tailscale. Starts as a single-user CLI co-worker; grows into a multi-gateway (web / Slack / Matrix), multi-agent, concurrent, self-maintaining system with memory, tools, and eventually initiative.

The model is plug-n-play: all model-specific code lives behind an adapter. Primary runtime is MLX on Apple Silicon (M4 Pro 48 GB). A Linux/larger Mac target may follow.

## Commands

Environment management with `uv` (Python 3.12):

- `uv sync --extra dev` — install runtime + dev deps.
- `uv sync --extra dev --extra mlx --extra retrieval` — add MLX and the retrieval stack (required for `--model mlx` and for retrieval-based few-shot respectively).
- `uv run hf download mlx-community/Qwen2.5-32B-Instruct-4bit` — pull Qwen 2.5 32B MLX build (one-time, ~18 GB).
- `uv run harness describe` — dump Airton's resolved character sheet.
- `uv run harness chat` — CLI chat loop with the echo adapter.
- `uv run harness chat --model mlx` — CLI chat with Qwen 2.5 32B.
- `uv run harness chat --model mlx --persona` — chat with the voice-rewrite post-pass on (Airton's register enforced).
- `uv run harness eval voice --model mlx` — run the voice-drift eval; Rich table of `prompt | gold | actual` per sample (leave-one-out by default).
- `uv run harness eval voice --model mlx --persona` — eval with the persona post-pass active; JSON output includes a `draft` field (pass-1 substance) alongside `actual` (pass-2 voiced).
- `uv run harness eval voice --model mlx --no-leave-one-out` — ceiling diagnostic: every sample sees the full example set.
- `uv run harness eval voice --model mlx --top-k 0` — disable retrieval, show all samples (Phase 1a.2 baseline).
- `uv run harness eval voice --model mlx --top-k 6 --persona` — retrieval-picked few-shot + voice-rewrite post-pass.
- `uv run harness eval voice --model mlx --persona --chain-rewrites --judge` — adds the concrete-substitution second rewrite pass and LLM-judge scoring.
- `uv run harness memory ingest` — seed Airton's episodic memory from the character's seed memories (idempotent).
- `uv run harness memory list` — tabular view of every episodic record.
- `uv run harness memory search "query"` — semantic top-K search over episodic memory.
- `uv run harness chat --model mlx --memories 3 --facts 5` — chat with top-3 episodic memories and top-5 semantic facts retrieved each turn.
- `uv run harness memory wipe --yes` — clear episodic + semantic + scribe-watermark data (transcripts preserved). Required when switching embedder dimensions.
- `uv run harness memory fact-list` / `fact-search "query"` / `fact-add subject predicate object` — semantic store CRUD/search.
- `uv run harness memory scribe --session ID --model mlx` — batch-extract memories from unprocessed transcript turns, watermark-tracked.
- `uv run harness memory consolidate` — cluster near-duplicate episodes, merge fact groups by (subject, predicate), supersede retired rows.
- `uv run harness memory rebuild-embeddings` — re-embed every active record with the current embedder. Non-destructive; use after an embedder switch.

Quality gates (all four must stay green; pre-commit runs them on every commit):

- `uv run ruff check .` — lint.
- `uv run ruff format .` — format in place.
- `uv run mypy src tests` — strict type-check (src + tests).
- `uv run pytest` — full test suite.
- `uv run pytest tests/test_character.py::test_load_airton_shape` — single test.
- `uv run pytest --cov` — tests with coverage.
- `uv run pre-commit run --all-files` — run all hooks against the working tree.
- `uv run pre-commit install --install-hooks && uv run pre-commit install --hook-type pre-push` — (re-)install git hooks after a fresh clone.

## Architecture

The system is designed against these invariants — they shape almost every decision:

1. **One identity, one memory, one orchestrator — many gateways.** The character is a single coherent being. Gateways (CLI now; web/Slack/Matrix later) normalize to a common envelope and hand off to the orchestrator.
2. **Concurrency-ready from day one, concurrent from day N.** No globals. Every turn is keyed by `(user, channel, turn_id)`. Even the Phase 0 CLI is written as if it were one of many sessions.
3. **Models stay behind an adapter.** Only `harness.model.*` imports MLX / llama.cpp / Ollama / OpenAI SDKs. The rest of the system speaks `ChatMessage` + `ModelAdapter.complete()`.
4. **Auditable memory.** Every mutation is attributed to a speaker, turn, and reason. Memory writes are two-phase: a scribe emits candidates, a consolidator promotes on a periodic pass.

### Repo layout

- `character/airton/` — the character as data. Treat as configuration, not code.
  - `core.yaml` — immutable identity: premise, pronouns, values, taboos, directives, deep/shallow domains, on-being-wrong register.
  - `constitution.md` — principles enforced by the critic model at generation time.
  - `voice/canonical.yaml` — voice calibration + drift-eval suite (6 locked samples; 14 more to extrapolate in Phase 1).
  - `seed_memories/*.md` — formative "lived experiences" written into episodic memory on day 0. Frontmatter-tagged by principle so retrieval can surface them by lesson as well as by content.
- `src/harness/` — runtime (src layout, package name `harness`).
  - `config.py` — pydantic-settings; env-prefixed `HARNESS_*`.
  - `character.py` — loads `character/<name>/` into a frozen `Character` dataclass; renders a fallback `system_prompt()` for single-model loops.
  - `model/` — adapter boundary. Anything model-specific lives here and nowhere else.
    - `adapter.py` — `ChatMessage` + `ModelAdapter` protocol.
    - `echo.py` — deterministic adapter used for wiring tests and running the loop without a model.
    - `mlx.py` — MLX-backed adapter (Qwen 2.5 32B by default). Lazy load; only this file imports `mlx_lm`.
    - `factory.py` — `make_adapter(name)` resolves `"echo" | "mlx"` to an instance. MLX is imported lazily inside, so environments without MLX still work.
  - `evals/` — offline evaluations. Pure functions that take a `Character` + `ModelAdapter` and return structured results.
    - `voice.py` — `run_voice_eval(...)` compares model output against the gold responses in `character/airton/voice/canonical.yaml`. Supports `leave_one_out` (default True) and `persona` (default False).
  - `persona/` — voice enforcement on top of the base adapter.
    - `rewriter.py` — `PersonaAdapter` wraps a base adapter with a two-pass flow: pass 1 produces substance via the caller's system prompt, pass 2 rewrites in the character's register using a dedicated rewriter prompt. `build_rewriter_messages(character, draft, include_samples=...)` is the pure function the eval uses inline so sample selection stays consistent across both passes.
  - `retrieval/` — embedding-backed few-shot selection. Currently used for voice samples; the same machinery will back episodic and semantic memory lookup in Phase 1b.
    - `embed.py` — `Embedder` Protocol. Implementations must return L2-normalized vectors so cosine similarity is a dot product.
    - `st_embedder.py` — `SentenceTransformersEmbedder` (BAAI/bge-small-en-v1.5 by default; MPS on Mac, CPU elsewhere; lazy-loaded).
    - `voice_retriever.py` — `VoiceRetriever` embeds all sample prompts once at construction; `top_k(query, k=, exclude_ids=)` returns the most similar samples. The CLI (`chat`, `eval voice`) and the voice eval build one per character load.
  - `store/transcript.py` — append-only SQLite transcript with WAL + FTS5. All future stores (episodic, semantic, graph, procedural, affective, identity, world) follow this shape: append-first, indexed for retrieval.
  - `store/episodic.py` — `EpisodicStore` holds narrative records (title, body, principle, tags, tier) in SQLite alongside float32-BLOB embeddings. `search(query, k)` is an in-process cosine scan — fine up to ~10k records, upgrade to LanceDB when we need more. `ensure_seeds_ingested(character, store)` is idempotent and runs on chat startup. Tier vocabulary: `seed` (loaded from character YAML) · `consolidated` (promoted by future consolidator) · `working` (scribe and ad-hoc writes).
  - `store/semantic.py` — `SemanticStore` holds atomic (subject, predicate, object) triples with confidence, provenance, tier, and a `supersedes` self-reference. Same BLOB-embedding-plus-cosine-scan pattern as episodic. `search(query, k, min_confidence)` embeds *"subject predicate object"* for natural-language retrieval.
  - `scribe/` — batch extraction from transcript to the memory stores.
    - `extractor.py` — `extract_candidates(adapter, character, turns)` calls the model with a strict-JSON rubric and returns parsed `EpisodicCandidate` + `SemanticCandidate` tuples. `parse_scribe_output` tolerates markdown fences and garbage, drops malformed items rather than failing the whole window.
    - `runner.py` — `run_scribe(...)` walks unprocessed transcript turns in windows past a per-session watermark, persists candidates to the stores at tier="working", advances the watermark. Non-overlapping windows; malformed output is logged in the summary and processing continues.
  - **Embedder**: `mixedbread-ai/mxbai-embed-large-v1`, 1024 dims, Matryoshka-trained. One embedder shared across voice retrieval, episodic memory, and semantic memory so they live in a comparable geometry. Changing the embedder is currently destructive — existing BLOBs are dim-locked — so plan to add `embedder_id` / `dimension` columns to the stores before any future switch.
  - `cli.py` — Typer app: `chat` and `describe` commands.
- `tests/` — pytest. Tests hit real stores (SQLite in `tmp_path`) rather than mocks.

### Phasing (where we are, where we're going)

- **Phase 0 — Skeleton (current).** Character package, model adapter boundary, SQLite transcript, CLI chat with echo adapter, loadable character sheet.
- **Phase 1 — Memory + persona.** MLX adapter (primary model: Qwen 2.5 32B, MLX 4-bit; A/B vs Llama 3.3 70B once scaffold is stable). Episodic + semantic layers with LanceDB. Retrieval fusion. Single-model ReAct tool loop. Persona-voice post-pass. Extrapolate the voice suite to 20 canonical samples; wire it as a drift-eval.
- **Phase 2 — Multi-user.** Web gateway (FastAPI + SvelteKit ops console). Per-user ACLs (enforced in code, tested). Affective + procedural memory. Scribe/consolidator split.
- **Phase 3 — Gateways + roles.** Slack (Bolt) and Matrix (matrix-nio) gateways. Kuzu graph layer. Multi-agent roles (planner / researcher / executor / critic / persona). Model router (heuristic table over `ModelAdapter.id`).
- **Phase 4 — Concurrent + always-on.** Concurrent sessions, launchd daemon, audit + policy hardening, off-box backups (destination TBD — SQLite WAL + LanceDB/Kuzu snapshots → rclone).
- **Phase 5 — Life.** Scheduler, initiative, dreams (consolidation runs), proactive triggers, autonomy.

### Decisions already locked

- **Runtime**: Python 3.12, asyncio (entering in Phase 1).
- **Stores**: SQLite + FTS5 (transcript, semantic, procedural, affective, audit) · LanceDB (vectors) · Kuzu (graph). All embedded, single-box, no extra daemons.
- **Models**: MLX primary; llama.cpp as fallback; OpenAI-compatible adapter as a cloud escape hatch for the router only.
- **Transport surface (eventual)**: web + Slack + Matrix, reachable via Tailscale. Small-trusted-circle auth; per-user capabilities in Phase 2.
- **Character**: Airton. `it` pronouns. Self-aware as software. 5 values, 8 taboos + 1 positive directive. Voice + 5 seed memories locked.

## Conventions

- **Don't import MLX, llama.cpp, or any model SDK outside `src/harness/model/`.** The adapter boundary is load-bearing. Violations break the plug-n-play guarantee.
- **Every store is append-first, attributed, and timestamped.** If you're writing a new memory layer, start from `store/transcript.py` as the template and diverge only where you must.
- **Tests hit real stores.** No mocking of SQLite or LanceDB. Use `tmp_path`.
- **Character data is configuration, not code.** Never hardcode Airton's rules into `src/`. The runtime reads `character/airton/` and should work just as well with `character/<someone-else>/`.
- **Identity first, then voice, then content.** When composing prompts, the system prompt starts with premise + self-awareness + values + taboos. If you find yourself re-stating identity inline in a prompt, that belongs in `core.yaml`.

## Quality tooling

Pre-commit hooks are installed (`pre-commit` + `pre-push`) and enforce:

- **ruff** (`check` + `format`) with rules: `E W F I UP B C4 SIM RET TID PT S N RUF`. The only global ignore is `S101` (asserts); per-file ignores live in `pyproject.toml`.
- **mypy** in `strict` mode — `warn_unused_ignores`, `warn_redundant_casts`, `warn_return_any`, `warn_unreachable`. The `harness` package ships a `py.typed` marker so downstream type-checking works.
- **pytest** at push time — quick on pre-commit would be too slow once we add the MLX adapter, so tests gate pushes, not individual commits.

Rules of engagement:

- Don't silence a failing check with a blanket `# noqa` / `# type: ignore`. Either fix it or add a targeted, commented per-file ignore in `pyproject.toml`.
- `ruff format` owns layout; don't hand-format. If the formatter and a rule disagree, change the rule.
- New dependencies go in `pyproject.toml` — never into the venv directly. Run `uv sync --extra dev` after editing.

## Open threads

- **Backup destination**: TBD. Default until chosen: SQLite WAL + local Time Machine + nightly tarball of Kuzu/LanceDB in `~/backups/harness/`. Pick an off-box target before Phase 2.
- **Voice suite extrapolation**: 14 pending scenarios listed in `character/airton/voice/canonical.yaml` under `pending_to_extrapolate`. Draft these when the MLX adapter is online so we can calibrate against the real model.
- **5th value already locked** ("If I can't measure it, I can't trust it."); no pending character gaps.
