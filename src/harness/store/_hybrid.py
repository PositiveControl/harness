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


def sanitize_fts_query(query: str) -> str:
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

    Returns empty string when no valid tokens remain; callers should
    short-circuit the search (FTS5 errors on empty MATCH clauses).
    """
    tokens = _FTS_TOKEN_RE.findall(query)
    if not tokens:
        return ""
    clauses = [f'"{t}"' for t in tokens]
    for compound in _FTS_COMPOUND_RE.findall(query):
        phrase = re.sub(r"[-/]+", " ", compound).strip()
        if phrase and " " in phrase:
            clauses.append(f'"{phrase}"')
    return " OR ".join(clauses)


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists of record ids into one ranking by
    RRF: score(id) = Σ_i 1 / (k + rank_i), where rank_i is the id's
    1-indexed position in the i-th list (absent = no contribution).

    k=60 is the Cormack/Clarke/Büttcher default. Low-rank items in
    the top-10 dominate the score; items ranked past ~60 contribute
    diminishing returns — which is exactly what we want for "candidate
    is in the hybrid top-k iff at least one source strongly believes
    in it."

    Returns id → score pairs, sorted high-to-low. Callers re-hydrate
    the full records from their own side-indexes.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, rid in enumerate(ranking, start=1):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda t: t[1], reverse=True)
