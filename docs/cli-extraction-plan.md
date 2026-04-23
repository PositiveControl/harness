# CLI extraction plan

`src/harness/cli.py` is 3,159 LOC. The goal is to cut it down by moving cohesive helper clusters into siblings, so each file has a single reason to change and the sibling UIs (`cli_classic.py`, `cli_repl.py`, `cli_tui.py`) can use top-level imports instead of deferred `from harness.cli import ...` calls inside function bodies.

## Guiding decisions

- **Flat siblings, not a `cli/` package.** Pattern is already established (`cli_introspect.py`, `cli_classic.py`, `cli_repl.py`, `cli_tui.py`). A package would just relocate the sprawl into `__init__.py`, and the entry point `harness.cli:app` would either need to change or have a re-export shim.
- **Typer sub-apps are the one piece of shared state that forces structure.** `@sub_app.command()` captures the sub-app object at import time, so any file that registers a command needs to import the same `eval_app` / `memory_app` / `voice_app` object that `cli.py` mounts onto the root `app`. Solution: introduce `cli_apps.py` that owns all four Typer app objects; every handler file imports from there.
- **No behavior changes.** Every extraction is a pure move + import rewrite. Line-for-line preservation. Tests must stay green on each step.
- **Pre-commit runs ruff + mypy strict + pytest on push.** Each step is a commit; pushes verify the whole suite.

## Target file layout (end state)

```
src/harness/
  cli.py                  # root Typer app + `chat` + `describe` + top-level wiring (~500 LOC target)
  cli_apps.py             # Typer app objects (app, eval_app, memory_app, voice_app)
  cli_adapter.py          # _resolve_adapter
  cli_bd.py               # bd adapter + harvest glue
  cli_chat_shared.py      # transcript codec, renderers, streaming, retrieval composition
  cli_eval.py             # eval subcommand handlers
  cli_introspect.py       # (exists)
  cli_classic.py          # (exists)
  cli_memory.py           # memory subcommand handlers
  cli_repl.py             # (exists)
  cli_store.py            # embedder cache + store openers
  cli_tools.py            # tool registry construction + grounding block
  cli_tui.py              # (exists)
  cli_voice.py            # voice subcommand handlers
```

## Extraction sequence

Order matters — later steps depend on earlier ones landing cleanly. Each step is one commit; gate on `uv run pre-commit run --all-files` + `uv run pytest`.

### Step 1 — `cli_store.py` (HIGH, ~60 LOC)

**Move:**
- `_load_embedder`
- `_maybe_retriever`
- `_open_episodic_store`
- `_open_semantic_store`
- `_cached_embedder` module-level singleton + `_EMBEDDER_SENTINEL`

**Why first:** Lowest coupling. Every subsequent step needs these (tool registry, memory handlers, eval handlers, sibling UIs). Extracting now means later steps can import from the real home instead of threading through `cli.py`.

**Landmine:** The embedder cache is process-wide. Every caller must go through `cli_store._load_embedder()` — don't create a second cache in `cli.py`. Grep for `_cached_embedder` after the move to confirm single ownership.

**Verify:** `uv run harness memory list` + `uv run harness chat --model echo --persona` (smoke, no model load needed).

---

### Step 2 — `cli_chat_shared.py` (HIGH, ~500 LOC)

**Move:**
- Transcript codec: `_TOOL_CALLS_SENTINEL`, `_encode_assistant_with_tool_calls`, `_decode_transcript_message`, `_persist_tool_exchange`
- Tool-call validation: `_pre_validate_write_call`, `_describe_call`
- Editor integration: `_open_in_editor`
- Prompt-block renderers: `_render_chat_header`, `_render_fact_block`, `_render_memory_block`, `_render_ab_memories_block`, `_format_ctx_meter`
- Streaming: `_ThinkingSpinner`, `_StreamRenderer`, `_render_tool_event`, `_stream_or_complete`
- Retrieval composition: `_RetrievalState`, `_retrieve_turn_context`

**Why:** These are the exact symbols `cli_classic.py`, `cli_repl.py`, `cli_tui.py` reach back into `cli.py` for via deferred imports. Formalizing the dependency direction eliminates the intra-harness circular-import dance.

**Coupling to untangle:**
- `_StreamRenderer` imports orchestrator regexes (`_FABRICATED_SEARCH_RE`, etc.). Keep as top-level imports in the new file.
- Rendering helpers take a `rich.console.Console` instance; no hidden state.
- `_persist_tool_exchange` writes to `TranscriptStore` — pass the store in as a parameter (should already be the pattern).

**Alternative homes to consider (don't act yet, just note):**
- `_encode/_decode_assistant_with_tool_calls` + sentinel logically belong in `harness/store/transcript.py` since they serialize into transcript rows. Move later as a follow-up if a second caller appears.
- `_render_fact_block` / `_render_memory_block` could live in `harness/persona/` since they're part of system-prompt assembly. Keep in `cli_chat_shared.py` for now; only move if persona needs them standalone.

**Verify:** Full `uv run pytest tests/test_cli*.py` + `uv run harness chat --model echo --persona --tools` smoke.

---

### Step 3 — `cli_bd.py` (MEDIUM, ~125 LOC)

**Move:**
- `_maybe_bd_adapter`
- `_maybe_ab_bd_adapter` (back-compat alias)
- `_maybe_harvest_skills`
- `_maybe_harvest_bd_memories`
- `_print_session_end_retro`

**Why:** Self-contained bd adapter lifecycle. `cli_tui.py` already imports `_maybe_harvest_bd_memories` / `_maybe_harvest_skills` from `cli.py` via a deferred block — consolidating here makes that dependency explicit.

**Verify:** `uv run harness chat --model echo --persona` (harvest-at-start path) + existing bd adapter tests.

---

### Step 4 — `cli_tools.py` (HIGH, ~400 LOC)

**Move:**
- `OPS_TOOL_NAMES`
- `_missing_builder_reason`
- `_ab_tool_builders`
- `_build_tool_registry_for_tui`
- `_make_introspect_tool`
- `_router_id_label`
- `_build_tool_grounding_block` (the ~100-LOC prompt string belongs in one place, not in cli.py)
- `_resolve_router_tool_specs`

**Why:** All tool-registry construction in one place. Biggest single-file payoff after chat-shared.

**Dependencies:** Must come after steps 1–3 — tool registry construction calls `_maybe_bd_adapter` (step 3), `_maybe_retriever` (step 1), and needs shared renderers (step 2).

**Coupling to untangle:**
- `_make_introspect_tool` takes `app: Typer` as a parameter already — no circular import.
- `_ATC_FETCH_URL_ALLOWED_HOSTS` (module constant at line ~115) — character-scoped allowlist. Follow-up: move to `character/airton_c/` config or `harness/tools/ab_ops.py`. Not in scope for this step.

**Verify:** `uv run harness chat --persona --tools` + `uv run harness eval router --tool-set coding`.

---

### Step 5 — `cli_apps.py` + subcommand splits (MEDIUM, ~1,300 LOC)

This is the most structurally invasive step. Do it as **three sub-commits** after `cli_apps.py` lands:

**5a — `cli_apps.py`** (~30 LOC)

Owns the four Typer app objects:
```python
app = typer.Typer(...)
eval_app = typer.Typer(...)
memory_app = typer.Typer(...)
voice_app = typer.Typer(...)
app.add_typer(eval_app, name="eval")
app.add_typer(memory_app, name="memory")
app.add_typer(voice_app, name="voice")
```

`cli.py` imports `app` from here. Entry point stays `harness.cli:app` (re-export). Handler files import the sub-apps from here.

**5b — `cli_memory.py`** (~600 LOC)

All `@memory_app.command(...)` handlers: `memory_list`, `memory_search`, `fact_list`, `fact_search`, `fact_add`, `scribe`, `consolidate`, `rebuild_embeddings`, `wipe`, `ingest`, `harvest_skills`, `harvest_memories`. Plus `_parse_as_of` helper.

**5c — `cli_voice.py`** (~130 LOC)

`_write_voice_capture`, `voice_capture`, `voice_list_captured`.

**5d — `cli_eval.py`** (~580 LOC)

All `@eval_app.command(...)` handlers + `_print_router_eval_table`. Imports `_resolve_router_tool_specs` from `cli_tools.py`.

**Import-order landmine:** Handler files must be imported by `cli.py` (or by `cli_apps.py`) so their `@sub_app.command()` decorators actually run. Two options:
- (a) `cli.py` ends with `from . import cli_memory, cli_voice, cli_eval  # register commands`  — explicit registration.
- (b) `cli_apps.py` imports the handler modules at the bottom — centralizes registration but couples `cli_apps.py` to every handler file.

**Recommend (a).** Registration sits next to `app` wiring where a reader expects it.

**Verify:** Every subcommand path. `uv run harness memory list`, `harness voice list-captured`, `harness eval router`, etc. Plus full `uv run pytest`.

---

### Step 6 — `cli_adapter.py` (LOW, ~70 LOC)

**Move:** `_resolve_adapter`.

**Why low:** Already well-isolated. Nice-to-have for testability — lets tests import the composition contract without importing the full `cli.py` Typer tree.

**Defer** unless a test specifically needs it. Can be done any time after step 1.

---

## What stays in `cli.py`

After all six steps, `cli.py` should contain:
- The `chat` and `describe` Typer commands (the root-level handlers — `chat` is the big one).
- Top-level module constants used only by `chat` (defaults, `--help` strings).
- The registration line importing handler modules for side effects.

Target: ~500 LOC. If it's still over 800 after step 6, the `chat` command itself needs decomposition — that's a separate design pass (probably pulling the chat-loop body out into `cli_chat.py`).

## Out of scope for this plan

These surfaced during survey but are follow-ups, not part of the extraction:

- Move `_encode/_decode_assistant_with_tool_calls` into `harness/store/transcript.py`.
- Move `_ATC_FETCH_URL_ALLOWED_HOSTS` into character config.
- Decompose the `chat` command body (only if step 6 doesn't get us to ~500 LOC).
- Consider moving `_render_fact_block` / `_render_memory_block` into `harness/persona/` if persona ever needs to assemble prompts standalone.

## Per-step checklist

For every step:
- [ ] New file created, old symbols removed from `cli.py`.
- [ ] All intra-harness imports updated (grep for old import paths after the move).
- [ ] `uv run ruff check . && uv run ruff format .` clean.
- [ ] `uv run mypy src tests` clean.
- [ ] `uv run pytest` green.
- [ ] Smoke: `uv run harness chat --model echo --persona` (adjust flags per step).
- [ ] Commit with a single `cli: extract X` message.
- [ ] `git push` at the end of each step (not batched).
