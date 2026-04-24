"""Query-side synonym expansion (harness-ajn).

Mirrors the ingest-side enrichment in `scripts/atc_ingest.py`: when the
corpus's `synonyms.yaml` defines lay-term variants for a section, those
variants get appended to every stored row's principle so BM25 can hit
lay-language queries and dense cosine sees the full semantic field. The
ingest side was already solving half the problem — the query side needed
the matching half.

Without query expansion, a user query "shortest distance on approach"
DOES hit §3-10-3 via BM25 (the stored synonym header contains the exact
phrase) but only weakly via dense cosine: the user's query vector is
384-dim over the literal phrase, while the stored vector is 384-dim over
`body + title + principle + all-synonyms` — a much larger semantic
field. Expanding the query with the other synonyms from the same section
shifts the query vector toward that field, lifting dense cosine without
changing BM25's hits.

For section-less queries (no lay term matches), `expand()` returns the
query unchanged — identity-preserving so there's no per-query cost when
expansion doesn't apply. Characters without a `synonyms.yaml` file get
a `NullQueryExpander` from `load_query_expander()` and pay no cost at
all.

The expander is character-agnostic on API: only the synonyms.yaml path
is wired from character state. Ingest-side's `_load_synonyms` is a peer
of this code; the two independently parse the same file because the
ingest path lives in scripts/ (off the import path for harness) and
cross-importing would drag the CLI into the script namespace. YAGNI
beats sharing one parser between two small files.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml

# Minimal English stopword set. Tight by design — over-stripping hurts
# precision more than under-stripping hurts recall, since each retained
# stopword in a lay term is just one more word the query has to also
# contain. The list covers function words and short qualifiers that
# appear in almost any fixture question but aren't topical ("the",
# "is", "on", etc.); topical words like "behind", "before", "after"
# stay in content because they often carry the procedural distinction
# between sections (§3-9-6 departure vs §3-10-3 arrival, etc.).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _content_tokens(text: str) -> frozenset[str]:
    """Lowercase `text`, split on word boundaries, drop stopwords. Used
    by both lay-term indexing and query matching so the two sides agree
    on what counts as a 'content' token."""
    if not text:
        return frozenset()
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS)


class QueryExpander:
    """Load lay-term → section mappings from a `synonyms.yaml` file and
    expand incoming queries with symmetric enrichment. Immutable after
    construction — callers that want a refreshed index must build a new
    instance.

    Matching strategy: a lay term triggers on a query when every one of
    the lay term's content tokens (post-stopword-strip) also appears as
    a content token in the query. Substring matching was the first cut
    and was too strict — natural-language paraphrases rarely contain
    curated lay phrases verbatim. Content-token-subset matches the real
    paraphrase pattern without over-triggering (the lay term still has
    to supply every topical word; extras in the query don't hurt)."""

    def __init__(self, sections: dict[str, tuple[str, ...]]) -> None:
        # Canonical storage: section_id → tuple of lay-term variants.
        # Tuples so the instance is effectively frozen (no mutation via
        # returned references).
        self._sections: dict[str, tuple[str, ...]] = {
            sec: tuple(variants) for sec, variants in sections.items() if variants
        }
        # Pre-compute (content_token_set, section) for every lay term.
        # A term with zero content tokens (e.g. "the on") is dropped — a
        # zero-token subset matches every query, which is pure noise.
        self._term_tokens: tuple[tuple[frozenset[str], str], ...] = tuple(
            (tokens, section)
            for section, variants in self._sections.items()
            for term in variants
            if (tokens := _content_tokens(term))
        )

    @property
    def is_empty(self) -> bool:
        """True when the synonyms source was missing or parsed to zero
        section entries. Callers can short-circuit construction /
        persistence when the expander would be a no-op."""
        return not self._sections

    def triggered_sections(self, query: str) -> tuple[str, ...]:
        """Section IDs whose lay-term list has at least one
        content-token-subset hit in `query`. Order is stable (insertion
        order of `self._sections`) so the output is deterministic. Each
        section appears at most once even if multiple terms matched."""
        query_tokens = _content_tokens(query)
        if not query_tokens:
            return ()
        hit: dict[str, None] = {}  # ordered set
        for term_tokens, section in self._term_tokens:
            if term_tokens.issubset(query_tokens):
                hit.setdefault(section, None)
        return tuple(sec for sec in self._sections if sec in hit)

    def expand(self, query: str) -> str:
        """Return `query` augmented with the aggregated lay-term list
        of every triggered section. When nothing triggers, returns the
        original query unchanged — identity-preserving.

        Format matches the ingest-side `[synonyms: a; b; c]` header so
        BM25 tokenization and dense embedding treat the two symmetrically:
          "<query> [related: §<sec>: term1; term2; ... | §<sec2>: ...]"

        Section anchors are embedded in the expansion so FTS5 can still
        hit the `[section: N-N-N]` tag in the stored text — useful when
        the user's query is a jargon-light paraphrase but maps cleanly
        to a single section."""
        sections = self.triggered_sections(query)
        if not sections:
            return query
        chunks: list[str] = []
        for sec in sections:
            variants = self._sections[sec]
            chunks.append(f"§{sec}: " + "; ".join(variants))
        return f"{query} [related: " + " | ".join(chunks) + "]"


class NullQueryExpander(QueryExpander):
    """Identity expander for characters without a synonyms file. Keeps
    call sites branch-free: `expander.expand(q)` works regardless of
    whether the character has corpus synonyms."""

    def __init__(self) -> None:
        super().__init__({})

    def expand(self, query: str) -> str:
        return query


def _parse_sections_file(path: Path) -> dict[str, tuple[str, ...]]:
    """Parse one synonyms-shaped YAML into a {section: (variants,)} dict.
    Returns {} for missing / malformed files — degrades gracefully."""
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return {}
    if not isinstance(raw, dict):
        return {}
    sections_raw = raw.get("sections")
    if not isinstance(sections_raw, dict):
        return {}
    parsed: dict[str, tuple[str, ...]] = {}
    for section, variants in sections_raw.items():
        if not isinstance(variants, list):
            continue
        cleaned = tuple(str(v).strip() for v in variants if isinstance(v, str) and v.strip())
        if cleaned:
            parsed[str(section)] = cleaned
    return parsed


def load_query_expander(
    synonyms_path: Path | None,
    *,
    query_only_path: Path | None = None,
) -> QueryExpander:
    """Load a `QueryExpander` from one or two YAML files.

    `synonyms_path` is the shared file consumed by both the ingest
    script and the query expander (character/<name>/corpus/synonyms.yaml
    convention). `query_only_path` is an optional additive file whose
    entries ONLY influence query-side expansion, never ingest-side row
    augmentation (character/<name>/corpus/query_synonyms.yaml
    convention). When both files carry entries for the same section,
    the variants are merged (duplicates deduped).

    Returns a `NullQueryExpander` when both files are missing or empty.

    Malformed YAML or unexpected shape silently degrades — this helper
    prefers missing entries over a failed chat bootstrap. Inspect with
    `is_empty` or `isinstance(..., NullQueryExpander)` for a strict
    check.
    """
    shared: dict[str, tuple[str, ...]] = (
        _parse_sections_file(synonyms_path) if synonyms_path is not None else {}
    )
    query_only: dict[str, tuple[str, ...]] = (
        _parse_sections_file(query_only_path) if query_only_path is not None else {}
    )
    if not shared and not query_only:
        return NullQueryExpander()
    # Merge: union the section keys; for sections present in both, merge
    # variants (preserving insertion order from shared first, then
    # query-only, deduped).
    merged: dict[str, tuple[str, ...]] = {}
    for section in list(shared) + [s for s in query_only if s not in shared]:
        seen: dict[str, None] = {}
        for source in (shared.get(section, ()), query_only.get(section, ())):
            for variant in source:
                seen.setdefault(variant, None)
        merged[section] = tuple(seen)
    return QueryExpander(merged) if merged else NullQueryExpander()


def default_synonyms_path(character_path: Path) -> Path:
    """Conventional location under `character/<name>/` — matches the
    ingest script's path. Kept as a function so the CLI, tests, and any
    downstream tool all agree on where to look."""
    return character_path / "corpus" / "synonyms.yaml"


def default_query_only_synonyms_path(character_path: Path) -> Path:
    """Query-only synonyms file — read by the expander, ignored by the
    ingest script. Entries here don't bake into stored row text, so
    they won't dilute BM25 / dense precision for jargon queries. Use
    for lay-paraphrase entries that should trigger expansion but
    shouldn't change stored principle tokens."""
    return character_path / "corpus" / "query_synonyms.yaml"


def expand_many(expander: QueryExpander, queries: Iterable[str]) -> tuple[str, ...]:
    """Batch helper. Exists so eval / benchmark harnesses can expand a
    fixture in one call rather than threading an expander handle through
    their loops. Identity-preserving when `expander` is a
    `NullQueryExpander` (no allocations beyond the tuple materialization)."""
    return tuple(expander.expand(q) for q in queries)
