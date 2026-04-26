# Plan: session-scoped context state machine (item C)

## Problem

Even with daily-rotated session ids (harness-y7ua) in place, retrieval is
still user-scoped: every episodic + semantic row written under
`user_id = mark` can resurface on any future `mark` turn, regardless of
which session it came from. A junior-engineer thread from week 1 can be
pulled into a database-migration thread on week 3 because both queries
share lexical or semantic overlap.

Goal: a per-turn "context state" that knows which session(s) the current
turn belongs to and either prefers, restricts, or excludes memory rows
based on that state — without losing the cross-session learning that
makes the harness valuable.

## Design

### 1. Tag rows with `source_session` at write time

Add a nullable `source_session TEXT` column to:

- `episodic` — written by scribe, harvest-skills, harvest-memories,
  ingest, consolidator (consolidated rows inherit the session of their
  cluster's most-recent member; ties broken by max(id)).
- `semantic` — written by scribe, fact-add, consolidator.
- Existing rows stay `NULL` (= legacy / shared / unsourced). Retrieval
  treats `NULL` as "always eligible" so we don't silently hide pre-
  migration memory.

Write sites that need updating:

- `src/harness/scribe/scribe.py` — already knows the session; thread
  it through the row inserts.
- `src/harness/skills/harvester.py` and `memory_harvester.py` — bd
  beads have no session of origin; write `NULL` (procedural memory is
  shared and session-agnostic by construction).
- `src/harness/consolidate/consolidator.py` — pick the
  most-recent-session label across the cluster.
- `src/harness/cli.py memory_fact_add` — accept `--session NAME`
  override; default to `NULL` (shared).

### 2. Add `--memory-scope` to chat

CLI flag with three values:

- `all` (default, current behavior) — every row matching the user_id /
  shared filter is eligible. `source_session = NULL` rows always pass.
- `current-session` — only `source_session = <this session>` OR
  `source_session IS NULL`. Useful when the user explicitly wants to
  treat this conversation as a clean slate that only learns from
  itself.
- `recent` — `source_session IN (last N session ids by last_at)` OR
  `source_session IS NULL`. N is `--memory-scope-window N` (default
  3). Useful when the user is mid-project and wants context to span a
  few days but not three weeks.

### 3. Per-row recency boost in hybrid RRF

Independent of `--memory-scope`. Today RRF fuses BM25 + dense cosine
with rank-based weights. Add a third term: a recency rank derived from
`source_session`'s last activity (newer session = higher rank). Keep
the recency weight tunable (`HARNESS_RETRIEVAL_RECENCY_WEIGHT`, default
0 = off so this lands behind a flag).

### 4. Topic-boundary signal in the prompt

When `--memory-scope=current-session` is active OR `/clear` has fired
in this process, prepend a one-line system note:

> "This is a fresh conversation. Earlier exchanges from previous
> sessions are not part of this thread."

Cheap to add, costs ~15 tokens, gives the model a frame for ignoring
spurious bleed even when retrieval lets a few rows through.

### 5. Migration

The new column is nullable so the schema change is backwards-compatible.
A one-shot `harness memory rebuild-source-sessions` walks the transcript
table, joins each scribe-written row to the session it came from via
created_at proximity, and backfills `source_session`. Best-effort —
unmatched rows stay NULL.

## Test plan

- Unit: scribe writes rows with the correct `source_session`.
- Unit: consolidator picks max-session correctly across clusters.
- Unit: `EpisodicStore.search(scope="current-session", session=X)`
  filters NULL-or-X.
- Integration: a `current-session` chat with one prior session of
  unrelated content + one prior session on-topic does NOT pull from
  the unrelated session, but DOES pull procedural (NULL) rows.
- Integration: backfill script populates source_session for at least
  80% of scribed rows on the existing local dev db.

## Non-goals (intentional)

- Per-turn session inference (auto-detecting topic shift mid-session).
  Too fuzzy; bring it back if `current-session` + `/clear` aren't
  enough.
- Removing or deprecating user_id scoping. Session is a finer grain
  ON TOP of user_id, not a replacement.
- Encrypting or hashing session ids. Plain text — the transcript is
  already plaintext.

## Sequencing

1. Schema migration (nullable column + indexes).
2. Write-side tagging across scribe / harvest / fact-add / consolidator.
3. Read-side `--memory-scope` flag wiring through cli.py + retrieval.
4. Topic-boundary prompt note.
5. Recency-boost RRF (gated behind env flag).
6. Backfill script + one-time migration on dev db.

Ship 1-3 as the MVP (covers the user's complaint). 4-6 can land
incrementally.

## Open questions

- Does `source_session` need to be an array? A consolidated row may
  legitimately span multiple sessions. Probably yes — store as a
  comma-separated TEXT or a join table; lean toward join table for
  query simplicity.
- Should `harness session show` surface which memory rows are tagged
  to that session id, so the user can audit what survived past it?
  Useful but separate — file as a follow-up bead if (1)–(3) ship.
