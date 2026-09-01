# Update Docs

On-demand deep documentation pass. Not part of the per-bead chain — `/pr_submit` handles the docs a single change owes. This is for when the docs have drifted from the code.

## The doc canon

```
CLAUDE.md            The index. Commands, architecture, invariants, conventions, quality tooling.
                     Load-bearing: it is what a fresh session reads first.
AGENTS.md            bd protocol + session-completion workflow.
docs/roadmap.md      Phase sequencing.
docs/usage.md        How the harness is actually used day to day.
docs/plans/          Design docs from /feature_plan (YYYY-MM-DD-<slug>-design.md).
docs/architecture/   Current state of a subsystem: structure, data flow, schema.
docs/<topic>.md      Focused references (tools-and-tool-sets.md, hooks.md, tool_loop_flow.md, …).
docs/bench/          Benchmark results and their conditions.
```

Consolidate hard. No overlap between files. A fact lives in exactly one place and everywhere else links to it.

## When asked to update the docs

1. Read `CLAUDE.md` first — it names what exists. Anything it doesn't mention is either missing or dead.
2. Diff docs against reality for the area in question:
   - Commands: does every `harness <group> <cmd>` in `CLAUDE.md` exist? `uv run harness --help`, then each group's `--help`.
   - Counts: test count (`uv run pytest --collect-only -q | tail -1`), tool count and profile count (`uv run harness tool list`), module counts in prose.
   - Flags: does each documented chat flag still exist? `uv run harness chat --help`.
   - Layout: does the repo-layout section match `ls src/harness/`?
3. Fix what drifted. A stale command or flag in `CLAUDE.md` is worse than an undocumented one — it sends the next session down a dead path.
4. New subsystem with no doc → add `docs/architecture/<subsystem>.md` and one line in `CLAUDE.md` pointing at it. Don't inline the whole thing into `CLAUDE.md`.
5. Landed work → `docs/roadmap.md`. Superseded design doc → say so at its top (`**Superseded by:** …`), don't delete it; the rejected alternatives still carry value.
6. Finish by re-reading `CLAUDE.md` end to end as if you'd never seen the repo. Anything that misleads a fresh session is a bug.

## When asked to initialize docs for an area

- Read the code first, all of it for that subsystem — entry points, data flow, persistence, tests.
- Write one `docs/architecture/<subsystem>.md`: what it does, the modules and their jobs, the data it owns, the invariants it enforces, the seams other subsystems touch it through.
- Add one line to `CLAUDE.md` under the matching repo-layout bullet.
- Stop there. A second file only earns its place when the first one is genuinely too long.

## Rules

- Docs describe the code as it is, not as intended. An aspirational doc is a lie with a timestamp.
- Character behavior is configuration: it belongs in `character/<name>/`, described in `docs/character-authoring.md` — never restated in `src/` docs.
- Never leave a `**Status:** Draft` placeholder behind after the work it describes has landed.
