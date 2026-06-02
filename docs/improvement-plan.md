# Harness Improvement Plan

Generated from the structural/functionality review on 2026-06-02.

This document is a sequencing guide, not the issue tracker. `bd` remains the
source of truth for ownership, status, dependencies, and completion. The beads
named below should be updated or closed as implementation proceeds.

## Goals

1. Close the authorization gap where a read-tier tool can write files.
2. Move shared chat/turn behavior out of CLI-only assembly so CLI, TUI, web,
   and daemon paths can reuse the same grounded pipeline.
3. Replace ad hoc tool argument handling with typed validation that also
   generates tool schemas.
4. Reduce core-module pressure in the CLI, orchestrator hooks, and store layer.
5. Add operational hardening only after the core service boundaries are stable.

## Current Strengths To Preserve

The project already has strong local-first fundamentals: model adapters,
character-as-configuration, optional extras, strict `ruff`/`mypy` settings, a
large test suite, a mature tool-loop event model, retrieval evals, and explicit
domain catchers. The plan below should preserve those patterns rather than
replace them with a larger framework.

## Phase 0: Safety And Contract Corrections

### Step 0.1: Fix argument-dependent write authorization

Tracking: `harness-qcukc`

Problem: `stream_edit` advertises `tier="read"` while `in_place=True` overwrites
files. The orchestrator confirms only `spec.tier == "write"`, so this is a real
capability leak.

Implementation steps:

1. Add a capability API that can answer whether a specific `ToolCall` requires
   confirmation. Prefer one of these shapes:
   - split `stream_edit` into read-only `stream_edit` and write-tier
     `stream_rewrite`
   - add `Tool.requires_confirmation(arguments) -> bool`
   - add `ToolSpec.effective_tier(arguments) -> Literal["read", "write"]`
2. Update router prelude logic so argument-dependent write calls cannot be
   auto-routed as read-tier.
3. Update CLI/TUI/driver confirmation paths to call the new capability API.
4. Add regression tests for `stream_edit(in_place=False)` and
   `stream_edit(in_place=True)`.

Acceptance:

`in_place=True` must go through the write approval path everywhere the tool loop
is used. Read-only stream editing should keep its low-friction path.

### Step 0.2: Reuse the vLLM HTTP client

Tracking: `harness-5muh`

Problem: `VllmAdapter` constructs new `httpx.Client` instances for model
discovery, non-streaming calls, and streaming calls. This wastes connection
setup and can fail if the certifi path changes mid-run.

Implementation steps:

1. Add `self._client = httpx.Client(timeout=timeout)` in `VllmAdapter.__init__`.
2. Reuse it in `_discover_model`, `_post`, and `_post_stream`.
3. Add `close()`, `__enter__`, and `__exit__` or an equivalent lifecycle path.
4. Update tests to assert multiple calls construct one client.

Acceptance:

All vLLM requests share one adapter-owned client, and the adapter exposes a
clean close path.

### Step 0.3: Keep known operational hardening visible

Existing beads:

- `harness-e0z`: health/readiness command
- `harness-e1k`: resource caps on long-running sessions
- `harness-7wn`: consolidation locking
- `harness-cs9`: chunked output/model splitting when long wrap-ups become common

These are not blockers for the architecture work, but they should be part of
the same operational roadmap once the shared turn service lands.

## Phase 1: Tool Contracts And Turn Service

### Step 1.1: Add typed tool argument models

Tracking: `harness-5cjj9`

Problem: `ToolRegistry.call()` forwards raw dict arguments into `tool.call()` and
then tries to recover from Python exceptions. Pydantic is already a hard
dependency, so schemas and runtime validation should come from one typed source.

Implementation steps:

1. Define an optional `args_model` contract for tools.
2. Generate `ToolSpec.parameters` from `args_model.model_json_schema()` when
   present.
3. Validate and coerce arguments before dispatch in `ToolRegistry.call()`.
4. Return structured validation errors naming the bad field and expected shape.
5. Convert representative tools first:
   - one simple read tool, such as `read_file`
   - one write tool, such as `write_file`
   - one complex tool, such as `stream_edit`
6. After the pilot, migrate remaining tools incrementally by profile.

Acceptance:

The registry can reject malformed calls before invoking tool code, and the
model-visible JSON Schema stays synchronized with runtime validation.

### Step 1.2: Introduce a shared `TurnService`

Tracking: `harness-fl313`

Problem: the most complete grounded chat path is assembled in CLI modules. The
web `/chat` endpoint currently skips tool loop, retrieval, transcript history,
and `assemble_context`.

Service responsibilities:

1. Resolve character, adapter, and persona wrapper.
2. Load transcript history and compaction summary.
3. Retrieve voice samples, episodic memory, semantic facts, and contract context.
4. Build the tool registry and hook pipeline.
5. Run one model turn, with optional streaming events.
6. Persist transcript/tool exchange.
7. Record audit metadata.
8. Return a result object that CLI, TUI, web, and daemon callers can render.

Implementation steps:

1. Create a small service module, likely `src/harness/turn/service.py`.
2. Move CLI helper logic only when it is directly needed by the service.
3. Start with a no-tool, no-stream turn that preserves CLI behavior.
4. Add retrieval and memory blocks.
5. Add tool-loop execution and event streaming.
6. Switch web `/chat` to the service once the service has parity.
7. Switch CLI/TUI call sites after web parity is proven.

Acceptance:

CLI/TUI behavior remains stable, and web `/chat` can use the same grounded
tool-capable path when configured.

## Phase 2: CLI And Hook Modularization

### Step 2.1: Finish CLI extraction

Existing artifact: `docs/cli-extraction-plan.md`

Problem: the extraction plan is directionally right, but it is stale. It says
`cli.py` is 3,159 LOC; it is now 6,611 LOC. The module still owns Typer app
registration, command bodies, store openers, registry construction, rendering,
and multiple eval surfaces.

Implementation steps:

1. Refresh `docs/cli-extraction-plan.md` with current line counts and command
   groups.
2. Extract app registration into `cli_apps.py`.
3. Extract store/embedder openers into `cli_store.py`.
4. Extract transcript codec, renderers, and slash helpers into
   `cli_chat_shared.py` or move transcript serialization into
   `store/transcript.py` where appropriate.
5. Extract tool registry construction into `cli_tools.py`.
6. Extract memory, voice, session, eval, plan, web, denylist, and tool commands
   into command-specific modules.
7. Keep `harness.cli:app` as the entry point and compatibility facade.

Acceptance:

`cli.py` becomes a small root command module and compatibility facade. Sibling UI
modules stop importing private helpers from `harness.cli`.

### Step 2.2: Split hook policies by domain

Tracking: `harness-usbhw`

Problem: `HookPipeline` is a good abstraction, but `hooks.py` now contains
universal, AB, ATC, scholar, web-research, citation, table, and numeric policies
in one large module.

Implementation steps:

1. Keep `HookPipeline`, context dataclasses, and outcome types in a small core
   module.
2. Move universal catchers to `orchestrator/hooks/core.py`.
3. Move domain catchers to domain modules:
   - `ab.py`
   - `atc.py`
   - `scholar.py`
   - `web_research.py`
   - `citation.py`
   - `numeric.py`
4. Keep `default_hook_pipeline()` as the composition point.
5. Preserve catcher ordering exactly during the split.
6. Keep compatibility exports until downstream imports are migrated.
7. Regenerate and verify `docs/hooks.md`.

Acceptance:

Existing hook tests and attribution eval fixtures pass with unchanged behavior,
while domain-specific policies no longer widen the core hook module.

## Phase 3: Store Infrastructure

### Step 3.1: Add shared SQLite connection helpers

Tracking: `harness-rkijl`

Problem: store modules repeat connection setup, PRAGMAs, busy timeout, defensive
schema creation, ad hoc migrations, and FTS rebuild logic.

Implementation steps:

1. Add a common connection factory that applies:
   - parent directory creation
   - `journal_mode = WAL`
   - `synchronous = NORMAL`
   - `busy_timeout = 5000`
   - optional `foreign_keys = ON`
2. Add a tiny migration helper with named idempotent migrations.
3. Pilot the helper in `EpisodicStore` and `SemanticStore`.
4. Move shared FTS tokenizer/rebuild checks into a helper.
5. Expand to transcript, audit, document tree, tabular, compaction, and
   denylist stores after the pilot.

Acceptance:

Existing databases remain compatible, and schema evolution becomes explicit and
testable instead of embedded in store constructors.

### Step 3.2: Decide whether to keep custom migrations or adopt Alembic

Recommendation:

Start with a small in-repo migration helper. Adopt Alembic only if the schema
starts needing coordinated multi-table migrations, downgrade support, or
SQLAlchemy metadata. The current local-first SQLite shape does not require a
large migration framework yet.

Acceptance:

The decision is documented after the shared helper pilot, not before.

## Phase 4: Observability And Operations

### Step 4.1: Add trace boundaries after `TurnService`

Recommendation:

Do not add OpenTelemetry before the service boundary exists. Once `TurnService`
is the single path, add spans around:

1. adapter call
2. router prelude
3. retrieval sources
4. tool execution
5. hook pipeline phases
6. transcript/audit writes
7. compaction/scribe side effects

Acceptance:

One turn can be inspected end-to-end across CLI, web, and daemon surfaces.

### Step 4.2: Promote health and resource checks

Existing beads:

- `harness-e0z`
- `harness-e1k`
- `harness-7wn`

Implementation steps:

1. Add `harness health` with store, embedder, model, and bd checks.
2. Make `/healthz` call the same health service instead of only returning
   uptime.
3. Add log/file growth caps for long sessions.
4. Add consolidation locking before scheduled consolidation is enabled.

Acceptance:

The daemon and web surfaces can distinguish liveness from readiness, and long
sessions have explicit resource limits.

## Suggested Sequence

1. `harness-qcukc`: close the `stream_edit` authorization gap.
2. `harness-5muh`: reuse the vLLM HTTP client.
3. `harness-5cjj9`: add typed tool argument validation, starting with a pilot.
4. `harness-fl313`: introduce `TurnService` and migrate web `/chat`.
5. Refresh and execute the CLI extraction plan.
6. `harness-usbhw`: split hook policies by domain.
7. `harness-rkijl`: centralize store connection and migration helpers.
8. Add trace spans and operational health/readiness once the shared service
   boundary is stable.

## Verification Strategy

Each implementation slice should run the narrow affected tests first, then the
standard project gates:

1. Targeted unit tests for the modified component.
2. `uv run ruff check .`
3. `uv run mypy src tests`
4. `uv run pytest`
5. Relevant smoke commands, such as:
   - `uv run harness chat --model echo --persona`
   - `uv run harness chat --model echo --tools --tool-set coding`
   - `uv run harness web serve --model echo --host 127.0.0.1 --port <free-port>`

## Notes

The plan intentionally favors local abstractions over large framework adoption.
The project already has the essential pieces; the main work is moving policy
and lifecycle ownership to the right layer so future features do not keep
expanding `cli.py`, `hooks.py`, and individual store constructors.
