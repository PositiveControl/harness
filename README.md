# harness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

Run a persistent AI character on your own hardware. Local models (MLX or Ollama) behind one adapter, attributed memory that survives restarts, a sandboxed tool loop, and the character itself defined as configuration rather than code.

A local-first model-agent harness for a persistent character ("Airton"), built to be always-reachable by a small trusted circle over Tailscale.

Starts as a single-user CLI co-worker; grows into a multi-gateway (web / Slack / Matrix), multi-agent, concurrent, self-maintaining system with memory, tools, and eventually initiative.

Primary runtime is **MLX on Apple Silicon** (tested on M4 Pro 48 GB). A local **Ollama** adapter is also supported.

---

## Features

**Model + voice**
- Plug-in model adapter (`echo`, `mlx`, `ollama`) — all model-specific code behind a single boundary.
- Persona rewriter: two-pass generation (substance → voice rewrite) keeps responses in character.
- Optional chain-of-rewrite: second concrete-substitution pass for harder cases.
- Retrieval-picked few-shot voice examples from a curated + captured corpus.
- Live voice capture: turn any corrected reply into a new training sample in one command.

**Memory**
- Append-only, attributed, timestamped episodic + semantic stores (SQLite + BLOB embeddings).
- Three-tier write path: `seed` (curated) → `working` (scribe-extracted) → `consolidated` (merged).
- Per-user relationship scoping — private memory per speaker, shared memory visible to all.
- Embedding-agnostic: dimension-tracked rows, non-destructive `rebuild-embeddings` for embedder swaps.
- Semantic triples (`subject`, `predicate`, `object`) with confidence, provenance, and supersession.

**Agent loop**
- Tool-use orchestrator with 17 built-in tools across filesystem (`read_file`, `list_dir`, `grep`, `glob`, `edit_file`, `write_file`, `shell`), git (`git_status`, `git_diff`, `git_log`), memory (`search_memory`, `search_facts`, `remember_fact`, `remember_event`, `scribe_session`, `consolidate_memory`), and web (`search_web`).
- Named tool-set profiles (`minimal`, `core`, `coding`, `memory`, `diagnostic`, `research`) — each ≤ ~1,500 tokens of schema overhead.
- Optional small-model intent router (`--router`) that fronts the tool loop: confident read-tier classifications skip the fabricate-and-nudge rounds. Free-form JSON or JSON-schema-constrained decoding via `outlines`.
- Streaming token output, spinner, context-window meter, automatic compaction, duplicate-call short-circuit, fabricate-and-nudge catchers.
- Optional Textual TUI (`--tui`) — persistent input, scrolling RichLog, live metrics footer, write-tier confirmation modal, history replay.
- Scribe lock prevents concurrent extraction runs from corrupting the watermark.

**Evals + quality**
- Voice eval suite: leave-one-out retrieval scoring + heuristic scorer + LLM-as-judge.
- Router eval: fixture-based tool-selection accuracy check (`harness eval router`).
- ~470 tests against real SQLite stores — no mocks.
- `ruff` + `mypy --strict` + `pytest` wired into pre-commit and pre-push hooks.

**Character as configuration**
- Identity, voice samples, seed memories, constitution all live in `character/airton/` as YAML/Markdown.
- Swap or add characters without touching `src/`.

---

## Characters

The harness ships with several characters, each shaped by a different tool surface and constitution. The active character is selected via `HARNESS_CHARACTER_NAME` (default: `airton`). Naming convention: `airton_<single-letter>` for top-level archetypes; `airton_<letter><suffix>` for nested specializations under an existing archetype (e.g. `airton_c1`, `airton_c_tfr`).

| Name | Archetype — one-line premise | Tool profile | Launcher |
|------|------------------------------|--------------|----------|
| **airton** | engineer-mentor — gruff ex-defense embedded engineer dragged into web work; TDD-first | `coding` / `full` | `scripts/chat.sh` |
| **airton_b** (ab) | personal operations lead — holds the shape of your week; bd-routed only | `ops` | `HARNESS_CHARACTER_NAME=airton_b uv run harness chat …` |
| **airton_c** | FAA NAS generalist — student-pilot-facing reference over JO + AIM + CFR + PCG + PHAK | `atc` | `HARNESS_CHARACTER_NAME=airton_c uv run harness chat …` |
| **airton_c1** | FAA controller-side — JO 7110.65 only, paired with `phraseology_lint` | `atc` / `phraseology` | `HARNESS_CHARACTER_NAME=airton_c1 uv run harness chat …` |
| **airton_c_tfr** | TFR/NOTAM explainer — narrowed airton_c for plain-English NOTAM interpretation | `atc` | `HARNESS_CHARACTER_NAME=airton_c_tfr uv run harness chat …` |
| **airton_d** | notes curator — capture / query / triage over `character/airton_d/workspace/` | `notes` | `scripts/chat_d.sh` |
| **airton_e** | RESERVED — placeholder slot for a future character (see `character/airton_e/RESERVED.md`) | — | — |
| **airton_f** | scholar — generic markdown doc-tree reader on the contract primitive; bounded external lookup via Google Scholar / arXiv / DOI | `scholar` | `scripts/chat_f.sh` |
| **returns_handler** | tabular-decision demo — refund decisions over policy seeds + order rows | `contract` | `HARNESS_CHARACTER_NAME=returns_handler uv run harness chat …` |

Each character's data lives under `character/<name>/` — `core.yaml`, `constitution.md`, `voice/`, `seed_memories/`, plus optional `contracts/`, `seed_documents/`, `data/`, and `workspace/` subdirs depending on archetype. Character data is configuration, not code: nothing in `src/` hardcodes a character's rules.

---

## Setup

Requirements: macOS (Apple Silicon for MLX), Python 3.11+, [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies. `--extra all` pulls the full core set (dev + mlx +
#    retrieval + grammar + tui + code + browser + web). Always sync `all` —
#    `uv sync` PRUNES any extra you omit, so a partial sync silently breaks
#    later mypy / pytest / drive runs.
uv sync --extra all

# 2. Pull the default MLX model (~4 GB, 4-bit quantized Qwen 2.5 7B Instruct)
uv run hf download mlx-community/Qwen2.5-7B-Instruct-4bit

# 3. Install git hooks (ruff + mypy + pytest gates)
uv run pre-commit install --install-hooks
uv run pre-commit install --hook-type pre-push

# 4. Seed episodic memory from the character's formative narratives
uv run harness memory ingest
```

Optional extras:

- `--extra grammar` — pulls `outlines` + `datasets`. Required for `--router-mode grammar` (JSON-schema-constrained router decoding).
- `--extra tui` — pulls `textual`. Required for `--tui` chat.
- `--extra code` — pulls `tree-sitter` + `tree-sitter-language-pack`. Powers the `outline` tool and symbol-addressed reads (`read_file symbol=Foo.bar`); without it those degrade to line-based reads.

Both can be omitted for a smaller headless install.

### Running a different model

By default the `mlx` adapter loads `mlx-community/Qwen2.5-7B-Instruct-4bit` (smaller + faster for daily use). Override with `--model-repo` for bigger models:

```bash
# Qwen 2.5 32B — full-power base model
uv run harness chat --model mlx \
  --model-repo mlx-community/Qwen2.5-32B-Instruct-4bit \
  --persona
```

More examples:

```bash
# Qwen 2.5 Coder 32B on MLX
uv run harness chat --model mlx \
  --model-repo mlx-community/Qwen2.5-Coder-32B-Instruct-4bit \
  --persona

# A smaller MLX model for speed
uv run harness chat --model mlx \
  --model-repo mlx-community/Qwen2.5-7B-Instruct-4bit

# Ollama with a specific tag
uv run harness chat --model ollama \
  --model-repo qwen2.5-coder:32b-instruct \
  --persona
```

`--model-repo` works for `chat`, `eval voice`, and `memory scribe` — anywhere an adapter is constructed. Pull the weights first (`uv run hf download <repo>` for MLX; `ollama pull <tag>` for Ollama).

### Running with a LoRA adapter

Apply a LoRA trained with `mlx_lm.lora` on top of the base MLX model:

```bash
uv run harness chat --model mlx \
  --model-repo mlx-community/Qwen2.5-32B-Instruct-4bit \
  --lora-path ./adapters/airton-voice-lora \
  --persona
```

`--lora-path` expects a **directory** (the output of `mlx_lm.lora` training, containing `adapter_config.json` plus weight files). Requires `--model mlx`.

### Environment variables

All settings are prefixed `HARNESS_*` (pydantic-settings). Common ones:

- `HARNESS_DATA_DIR` — where `harness.sqlite` lives. Defaults to `./data/`.
- `HARNESS_CHARACTER` — which character to load. Defaults to `airton`.

---

## Quick start

```bash
# Full-stack chat: model + persona rewriter + memory + fact retrieval
uv run harness chat --model mlx --persona --memories 3 --facts 5

# Dump the character's resolved identity
uv run harness describe

# Echo adapter dry-run (no model — just validates wiring)
uv run harness chat
```

For the intended daily-use workflow, see **[`docs/usage.md`](docs/usage.md)**.

---

## Usage reference

### Chat

```bash
uv run harness chat [OPTIONS]
```

Key flags:

| Flag | Purpose |
|---|---|
| `--model {echo,mlx,ollama}` | Pick the model adapter. |
| `--model-repo REPO` | Override the model id. For `mlx`: an HF repo (e.g. `mlx-community/Qwen2.5-Coder-32B-Instruct-4bit`). For `ollama`: a model tag. Ignored for `echo`. |
| `--lora-path DIR` | Apply a LoRA adapter (directory from `mlx_lm.lora` training) on top of the base MLX model. Requires `--model mlx`. |
| `--persona` | Enable two-pass voice rewriter. Without it, the base model drifts to generic-assistant prose. |
| `--chain-rewrites` | Add a third concrete-substitution pass. More character, ~1.5× latency. |
| `--memories N` | Retrieve up to N episodic memories per turn (default 3). |
| `--facts N` | Retrieve up to N semantic facts per turn (default 5). |
| `--memories-threshold F` | Similarity floor for episodic retrieval (default 0.5). |
| `--facts-threshold F` | Similarity floor for semantic retrieval (default 0.45). |
| `--session NAME` | Session id — determines which transcript the scribe later reads. |
| `--speaker NAME` | Who you are — scopes relationship memory. |
| `--tools` | Enable the tool-use orchestrator loop. Which tools register depends on `--tool-set`. Write-tier tools prompt for confirmation on first use per session. |
| `--tool-set NAME` | Named profile: `minimal`, `core` (default, read-only), `coding` (full dev loop), `memory`, `diagnostic`. Each targets ≤ ~1,500 tokens of schema overhead. |
| `--tools-add X,Y` | Comma-separated tool names to add on top of the profile. |
| `--tools-drop X,Y` | Comma-separated tool names to drop from the profile. |
| `--workspace DIR` | Directory the `read_file` / `write_file` / `shell` tools operate inside. Defaults to the harness repo root. Only takes effect with `--tools`. Memory and transcripts still live under the harness data dir regardless. |
| `--rewrite-on-tools` | When tools ran in a turn, also run the persona rewriter on the final reply. Off by default — the rewriter compresses, which is wrong for summarize / investigate tasks. |
| `--compact-at F` | Fraction of context window at which to auto-summarize older turns (default 0.8, set 0 to disable). |
| `--compact-keep-recent N` | Number of most-recent turns to leave verbatim when compaction fires (default 10). |
| `--router` | Front the tool loop with a small-model intent router. Confident read-tier classifications run the tool directly, skipping fabricate-and-nudge rounds. |
| `--router-repo REPO` | HF repo for the router model. Default `mlx-community/Hermes-3-Llama-3.2-3B-4bit` (~2 GB RAM). |
| `--router-mode {free,grammar}` | Routing strategy. `free` = tolerant JSON parse. `grammar` = JSON-schema-constrained decoding via `outlines` (requires `--extra grammar`, ~1 GB extra RAM). |
| `--tui` | Launch the Textual chat app instead of the classic REPL. Requires `--extra tui`. |
| `--dev` | Surface internal signals (stream-filter suppression markers, etc.) — for tuning, not daily use. |

### Working in another repo (`--workspace`)

With `--tools`, the `read_file`, `write_file`, and `shell` tools are sandboxed to a single directory. By default that's the harness repo root. Point it elsewhere to use Airton as a co-worker on a different codebase:

```bash
uv run harness chat --model mlx --persona --tools \
  --workspace ~/dev/some-other-project \
  --session other-project
```

Memory, transcripts, and the character's voice corpus still live under the harness data dir — only tool filesystem access is scoped to `--workspace`. Use a distinct `--session` name so the scribe can later extract memories specific to that project.

### Voice corpus

```bash
# Capture a corrected reply as a new voice training sample (from the shell)
uv run harness voice capture --session <id> --gold "What the character should have said."

# Inspect captured samples
uv run harness voice list-captured
```

Inside chat, type `/edit` (or `/capture`) at the `you ›` prompt — `$EDITOR` opens with the last assistant reply pre-loaded. Save edits to capture a new sample without leaving the session.

### Memory

```bash
# Seed / ingest
uv run harness memory ingest

# Inspect
uv run harness memory list [--tier seed|working|consolidated]
uv run harness memory search "query"
uv run harness memory fact-list [--tier X] [--subject Y]
uv run harness memory fact-search "query"

# Write
uv run harness memory fact-add SUBJECT PREDICATE OBJECT [--user NAME] [--confidence 0.9]

# Extract + consolidate
uv run harness memory scribe --session <id> --user <name> --model mlx
uv run harness memory consolidate

# Maintenance
uv run harness memory rebuild-embeddings
uv run harness memory wipe --yes
```

### Evals

```bash
# Voice — current best config
uv run harness eval voice --model mlx --top-k 6 --persona

# Voice — full eval: chain rewrites + LLM judge
uv run harness eval voice --model mlx --persona --chain-rewrites --judge

# Voice — single sample
uv run harness eval voice --model mlx --sample SAMPLE_ID

# Voice — machine-readable
uv run harness eval voice --model mlx --persona --json

# Router — replay the router-eval fixture and score tool-selection accuracy
uv run harness eval router
uv run harness eval router --router-mode grammar --tool-set coding
```

### Quality gates

All must stay green; pre-commit runs them on every commit.

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src tests
uv run pytest
uv run pre-commit run --all-files
```

---

## Architecture at a glance

```
character/airton/        identity, voice corpus, seeds, constitution (data, not code)
src/harness/
  model/                 adapter boundary — the ONLY place model SDKs are imported
  persona/               voice rewriter post-pass
  retrieval/             embedder + voice sample retriever
  store/                 transcript, episodic, semantic (SQLite + embeddings)
  scribe/                batch extract transcript → memory candidates
  consolidate/           cluster + merge near-duplicates; promote tier
  compaction/            context-window management
  tools/                 17 built-in tools for the agent loop (fs/git/memory/web) + profiles
  orchestrator/          tool-use loop
  router/                small-model intent router (free-form + grammar)
  tui/                   Textual chat app (optional, behind `--extra tui`)
  evals/                 voice + router evals, scorer + judge
  cli.py                 Typer entrypoint
```

Key invariants:

1. **One identity, one memory, one orchestrator — many gateways.**
2. **Concurrency-ready from day one.** No globals; every turn keyed by `(user, channel, turn_id)`.
3. **Models stay behind an adapter.** Only `harness.model.*` imports MLX / Ollama / etc.
4. **Auditable memory.** Every mutation is attributed; scribe emits candidates, consolidator promotes on a pass.
5. **Hybrid memory scoping.** `user_id IS NULL` → shared; `user_id = X` → private to X.

Full architecture reference: **[`CLAUDE.md`](CLAUDE.md)**.
Phase sequencing and what's next: **[`docs/roadmap.md`](docs/roadmap.md)**.
Tool-loop internals: **[`docs/tool_loop_flow.md`](docs/tool_loop_flow.md)**.

---

## Project layout

- `character/<name>/` — character as configuration.
- `src/harness/` — runtime (src layout, package `harness`).
- `tests/` — pytest against real stores.
- `scripts/` — benchmarks and one-offs.
- `data/` — SQLite store (gitignored).
- `docs/` — usage, roadmap, and design notes.
- `CLAUDE.md` / `AGENTS.md` — authoritative architecture + conventions reference.

---

## License

[MIT](LICENSE) © Mark Evans.

Model weights, character corpora, and third-party dependencies carry their own licenses — this covers the harness code only.
