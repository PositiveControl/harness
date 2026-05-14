"""Hybrid retrieval primitives shared by episodic + semantic stores.

- `sanitize_fts_query` turns arbitrary user text into a safe FTS5
  MATCH clause that won't trip the parser on punctuation or special
  tokens.
- `reciprocal_rank_fusion` fuses multiple ranked lists of record ids
  into a single ranking via RRF (Cormack, Clarke, Büttcher 2009).
  Used to combine dense-cosine hits with FTS5 BM25 hits so queries
  with proper nouns / code identifiers don't lose to semantic
  matches that share no lexical overlap.

The stores own their schema; this module owns only the pure
computation.
"""

from __future__ import annotations

import re

# Matches FTS5 non-reserved content: alphanumerics + underscore. We
# strip everything else (quotes, parens, colons, `*`, etc.) so the
# user's query can't break the MATCH parser. `_` is kept because
# identifiers like `user_id` / `BeadsAdapter.get_focus` should round-
# trip through the FTS tokenizer as they do in the stored body.
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# Matches hyphen- or slash-joined compounds: 2+ alphanumeric-underscore
# runs glued by `-` or `/`. Drives the phrase-query pass added for
# harness-5uq-followup — without this, FTS5's unicode61 tokenizer
# shreds "aircraft-to-aircraft" into [aircraft, to, aircraft] and the
# distinctive n-gram is lost in an OR-sea of common tokens. Phrase
# queries preserve adjacency, so the ranker sees "aircraft to aircraft"
# as a contiguous match on §13-1-2 rather than three disjunctive token
# hits against every oceanic chapter mentioning 'aircraft' 50+ times.
# Alternation `[-/]` covers both corpus conventions (hyphens in
# aircraft-to-aircraft, L/MF, non-standard; slashes in CA/MCI).
_FTS_COMPOUND_RE = re.compile(r"[A-Za-z0-9_]+(?:[-/][A-Za-z0-9_]+)+")


def sanitize_fts_query(query: str, *, preserve_punctuation: str = "") -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH clause.

    Strategy (two-pass):

    Pass 1: extract alphanumeric+underscore tokens, wrap each in
    double quotes so FTS5 treats them as literal phrase-search
    terms (never reserved operators like AND / OR / NOT / NEAR),
    join with explicit OR.

    Pass 2: extract hyphen/slash compounds (`aircraft-to-aircraft`,
    `L/MF`, `non-standard`). Each becomes a quoted FTS5 phrase
    query with the internal punctuation replaced by a single
    space — `"aircraft to aircraft"` — so the underlying tokenizer
    sees the three-token sequence and BM25 can score adjacency.
    OR'd onto the pass-1 tokens.

    OR rather than the FTS5-default AND matters because the hybrid
    ranker combines with the dense-cosine pass via RRF — precision
    comes from the fusion, not from requiring every user token to
    appear literally.

    Quoting also defangs a user who happens to type "AND" as part
    of their actual query — without the quotes, FTS5 would try to
    parse it as an operator and throw a syntax error mid-search.

    `preserve_punctuation` (harness-q6zl) lets a caller declare
    additional characters that should NOT be split out of pass-1
    tokens. The tree store passes `.-:` so multi-segment section
    paths (`91.131`, `2-4-3`, `CFR_14_Vol2:91.131`) round-trip as
    single quoted FTS5 phrases — aligned with its `tokenchars '.-:'`
    tokenizer config so the indexed token and the query token use
    the same character set. The default (empty string) preserves
    the original episodic / semantic behavior.

    Returns empty string when no valid tokens remain; callers should
    short-circuit the search (FTS5 errors on empty MATCH clauses).
    """
    if preserve_punctuation:
        escaped = re.escape(preserve_punctuation)
        token_re = re.compile(rf"[A-Za-z0-9_{escaped}]+")
        # Strip leading/trailing punctuation so a query that includes
        # `§91.131:` doesn't produce a phrase `"91.131:"` (a different
        # token from the indexed `91.131`). Keep internal punctuation
        # intact so multi-segment paths survive.
        strip_chars = preserve_punctuation
    else:
        token_re = _FTS_TOKEN_RE
        strip_chars = ""
    raw_tokens = token_re.findall(query)
    if strip_chars:
        tokens = [t.strip(strip_chars) for t in raw_tokens]
        tokens = [t for t in tokens if t]
    else:
        tokens = raw_tokens
    if not tokens:
        return ""
    clauses = [f'"{t}"' for t in tokens]
    for compound in _FTS_COMPOUND_RE.findall(query):
        phrase = re.sub(r"[-/]+", " ", compound).strip()
        if phrase and " " in phrase:
            clauses.append(f'"{phrase}"')
    return " OR ".join(clauses)


def session_scope_filter(
    allowed_sessions: tuple[str, ...] | None,
    *,
    column: str = "session_id",
) -> tuple[str, tuple[str, ...]]:
    """Return a (sql_fragment, params) pair for an optional session-
    scope filter, intended to be appended to a WHERE clause.

    `allowed_sessions=None` → no filter (no-op fragment + empty params).
    `allowed_sessions=()` → only `column IS NULL` rows pass (procedural /
        shared seeds; everything session-tagged is hidden).
    `allowed_sessions=(s1, s2, ...)` → `column IS NULL OR
        column IN (s1, s2, ...)`. NULL rows always pass — the C-plan
        contract treats them as 'always eligible'.

    Driving harness-w3mo's `--memory-scope` flag. The fragment leads
    with ` AND ` so callers can splice it into an existing WHERE
    chain without conditional whitespace gymnastics."""
    if allowed_sessions is None:
        return "", ()
    if not allowed_sessions:
        return f" AND {column} IS NULL", ()
    placeholders = ",".join("?" * len(allowed_sessions))
    return (
        f" AND ({column} IS NULL OR {column} IN ({placeholders}))",
        allowed_sessions,
    )


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists of record ids into one ranking by
    weighted RRF: score(id) = Σ_i w_i / (k + rank_i), where rank_i
    is the id's 1-indexed position in the i-th list (absent = no
    contribution from that list).

    k=60 is the Cormack/Clarke/Büttcher default. Low-rank items in
    the top-10 dominate the score; items ranked past ~60 contribute
    diminishing returns — which is exactly what we want for "candidate
    is in the hybrid top-k iff at least one source strongly believes
    in it."

    `weights` (harness-w3mo) — optional per-list scaling factor. None
    defaults to all 1.0 (the original RRF). Use it to dial a
    secondary signal (e.g. recency) up or down without rewriting
    the call site. A weight of 0.0 is equivalent to omitting the
    list entirely.

    Returns id → score pairs, sorted high-to-low. Callers re-hydrate
    the full records from their own side-indexes.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights length {len(weights)} != rankings length {len(rankings)}")
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        if w == 0.0:
            continue
        for rank, rid in enumerate(ranking, start=1):
            scores[rid] = scores.get(rid, 0.0) + w * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda t: t[1], reverse=True)


def build_session_recency_ranks(
    sessions_newest_first: list[str],
) -> dict[str, int]:
    """Map each session id to its 1-indexed recency rank (1 = most
    recently active). Used by `_search_hybrid` to build the third
    RRF ranking for the recency-weighted retrieval gate
    (harness-w3mo). NULL-session rows are intentionally absent —
    callers skip them when building the recency ranking, so seeds /
    procedural memory ride solely on dense + BM25 contributions.

    Pass the output of `Transcript.list_sessions()` already ordered
    by `last_at DESC` (its default) to make this a one-shot
    map-build. Caller is expected to refresh the dict at session
    boundaries — within one chat, a stale dict is fine because the
    only new session is the running one and it'd be rank 1 anyway."""
    return {sid: rank for rank, sid in enumerate(sessions_newest_first, start=1)}
