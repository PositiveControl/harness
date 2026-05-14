"""Query-side synonym expansion (harness-ajn).

Mirrors the ingest-side enrichment in `scripts/atc_ingest.py`: when a
corpus's glossary YAML defines lay-term variants for a topic, those
variants get appended to every stored row's principle so BM25 can hit
lay-language queries and dense cosine sees the full semantic field. The
ingest side was already solving half the problem — the query side needed
the matching half.

Without query expansion, a user query "shortest distance on approach"
DOES hit §3-10-3 via BM25 (the stored synonym header contains the exact
phrase) but only weakly via dense cosine: the user's query vector is
384-dim over the literal phrase, while the stored vector is 384-dim over
`body + title + principle + all-synonyms` — a much larger semantic
field. Expanding the query with the other synonyms from the same topic
shifts the query vector toward that field, lifting dense cosine without
changing BM25's hits.

For unmatched queries (no lay term matches), `expand()` returns the
query unchanged — identity-preserving so there's no per-query cost when
expansion doesn't apply. Characters without a glossary file get a
`NullQueryExpander` from `load_query_expander()` and pay no cost at
all.

Character-agnostic by design. The mechanism (token-subset matching +
augmentation) is reusable for any character with a domain glossary:
legal citation lookups, medical codes, internal product names, etc.
The glossary YAML schema declares its own output prefix so a non-ATC
glossary doesn't get the `§` prefix bolted on (harness-m78r):

  # legacy / ATC-flavored — implicit `§` prefix
  sections:
    "3-10-3":
      - shortest distance on approach

  # character-agnostic — no implicit prefix, explicit one when wanted
  prefix: "ICD-10 "
  topics:
    "E11.9":
      - "high blood sugar"

Both top-level shapes (`sections:` and `topics:`) are accepted; missing
`prefix:` defaults to `§` for `sections:` (back-compat) and `""` for
`topics:` (new convention).

Ingest-side's `_load_synonyms` is a peer of this code; the two
independently parse the same file because the ingest path lives in
scripts/ (off the import path for harness) and cross-importing would
drag the CLI into the script namespace. YAGNI beats sharing one parser
between two small files.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import yaml

if TYPE_CHECKING:
    from harness.model.adapter import ChatMessage

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
    """Load lay-term → topic mappings from a glossary YAML and expand
    incoming queries with symmetric enrichment. Immutable after
    construction — callers that want a refreshed index must build a new
    instance.

    Matching strategy: a lay term triggers on a query when every one of
    the lay term's content tokens (post-stopword-strip) also appears as
    a content token in the query. Substring matching was the first cut
    and was too strict — natural-language paraphrases rarely contain
    curated lay phrases verbatim. Content-token-subset matches the real
    paraphrase pattern without over-triggering (the lay term still has
    to supply every topical word; extras in the query don't hurt).

    `topic_prefix` (harness-m78r) — string spliced before each topic id
    in the expansion output. The legacy default is `§` because the
    airton_c1 glossary's ingest-side enrichment uses `§N-N-N` anchors
    and the query side must match for FTS5 to hit the stored tag. Empty
    string is the right default for any new glossary — only ATC-style
    section-anchor data needs the §.
    """

    def __init__(
        self,
        topics: dict[str, tuple[str, ...]],
        *,
        topic_prefix: str = "§",
    ) -> None:
        # Canonical storage: topic_id → tuple of lay-term variants.
        # Tuples so the instance is effectively frozen (no mutation via
        # returned references).
        self._topics: dict[str, tuple[str, ...]] = {
            topic: tuple(variants) for topic, variants in topics.items() if variants
        }
        self._topic_prefix = topic_prefix
        # Pre-compute (content_token_set, topic) for every lay term.
        # A term with zero content tokens (e.g. "the on") is dropped — a
        # zero-token subset matches every query, which is pure noise.
        self._term_tokens: tuple[tuple[frozenset[str], str], ...] = tuple(
            (tokens, topic)
            for topic, variants in self._topics.items()
            for term in variants
            if (tokens := _content_tokens(term))
        )

    # `_sections` is preserved as a back-compat read-only alias for the
    # `LLMQueryExpander` subclass which sets `self._sections = {}` in
    # its constructor to opt out of the static-table path. Removing the
    # attribute outright would silently mask bugs in subclasses.
    @property
    def _sections(self) -> dict[str, tuple[str, ...]]:
        return self._topics

    @_sections.setter
    def _sections(self, value: dict[str, tuple[str, ...]]) -> None:
        self._topics = value

    @property
    def is_empty(self) -> bool:
        """True when the glossary source was missing or parsed to zero
        topic entries. Callers can short-circuit construction /
        persistence when the expander would be a no-op."""
        return not self._topics

    def triggered_topics(self, query: str) -> tuple[str, ...]:
        """Topic IDs whose lay-term list has at least one
        content-token-subset hit in `query`. Order is stable (insertion
        order of `self._topics`) so the output is deterministic. Each
        topic appears at most once even if multiple terms matched."""
        query_tokens = _content_tokens(query)
        if not query_tokens:
            return ()
        hit: dict[str, None] = {}  # ordered set
        for term_tokens, topic in self._term_tokens:
            if term_tokens.issubset(query_tokens):
                hit.setdefault(topic, None)
        return tuple(topic for topic in self._topics if topic in hit)

    # Back-compat alias — many call sites and tests reference the old
    # name. New code should prefer `triggered_topics()`.
    def triggered_sections(self, query: str) -> tuple[str, ...]:
        return self.triggered_topics(query)

    def expand(self, query: str) -> str:
        """Return `query` augmented with the aggregated lay-term list
        of every triggered topic. When nothing triggers, returns the
        original query unchanged — identity-preserving.

        Format matches the ingest-side `[synonyms: a; b; c]` header so
        BM25 tokenization and dense embedding treat the two symmetrically:
          "<query> [related: <prefix><topic>: term1; term2; ... | ...]"

        The `<prefix>` defaults to `§` for back-compat with ATC-flavored
        glossaries whose stored data carries section-anchor tags; new
        glossaries override via `topic_prefix=""` in the constructor or
        a `prefix:` field in their YAML."""
        topics = self.triggered_topics(query)
        if not topics:
            return query
        chunks: list[str] = []
        for topic in topics:
            variants = self._topics[topic]
            chunks.append(f"{self._topic_prefix}{topic}: " + "; ".join(variants))
        return f"{query} [related: " + " | ".join(chunks) + "]"


class NullQueryExpander(QueryExpander):
    """Identity expander for characters without a glossary file. Keeps
    call sites branch-free: `expander.expand(q)` works regardless of
    whether the character carries corpus synonyms."""

    def __init__(self) -> None:
        super().__init__({})

    def expand(self, query: str) -> str:
        return query


@dataclass(frozen=True)
class _ParsedGlossary:
    """Result of parsing one glossary YAML. `topics` is the variant map;
    `prefix` is the topic-output prefix (None when the YAML didn't
    declare one, so the caller can apply the legacy-vs-new default).
    `had_legacy_key` records whether the parsed file used the old
    `sections:` top-level key — drives the back-compat default for
    `prefix` when the YAML is silent on it."""

    topics: dict[str, tuple[str, ...]]
    prefix: str | None
    had_legacy_key: bool


def _parse_glossary_file(path: Path) -> _ParsedGlossary:
    """Parse one glossary-shaped YAML.

    Accepts both top-level keys: `topics:` (new convention) and
    `sections:` (legacy ATC convention). If both are present, `topics:`
    wins (encourages migration without breaking back-compat).

    Optional `prefix:` at the top level overrides the default
    topic-prefix used in the expander's output. When missing, the
    caller applies the legacy-vs-new default — see `load_query_expander`.

    Returns an empty parsed result for missing / malformed files —
    degrades gracefully."""
    empty = _ParsedGlossary(topics={}, prefix=None, had_legacy_key=False)
    if not path.exists():
        return empty
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return empty
    if not isinstance(raw, dict):
        return empty
    topics_raw = raw.get("topics")
    legacy = False
    if not isinstance(topics_raw, dict):
        topics_raw = raw.get("sections")
        if isinstance(topics_raw, dict):
            legacy = True
        else:
            return empty
    parsed: dict[str, tuple[str, ...]] = {}
    for topic, variants in topics_raw.items():
        if not isinstance(variants, list):
            continue
        cleaned = tuple(str(v).strip() for v in variants if isinstance(v, str) and v.strip())
        if cleaned:
            parsed[str(topic)] = cleaned
    prefix_raw = raw.get("prefix")
    prefix = str(prefix_raw) if isinstance(prefix_raw, str) else None
    return _ParsedGlossary(topics=parsed, prefix=prefix, had_legacy_key=legacy)


# Back-compat alias — exported under the old name for downstream callers
# (`scripts/suggest_synonyms.py`, `scripts/test_synonyms.py` reach into
# the parser for their own validation). New code should prefer
# `_parse_glossary_file`.
def _parse_sections_file(path: Path) -> dict[str, tuple[str, ...]]:
    return _parse_glossary_file(path).topics


def load_query_expander(
    synonyms_path: Path | None,
    *,
    query_only_path: Path | None = None,
) -> QueryExpander:
    """Load a `QueryExpander` from one or two glossary YAML files.

    `synonyms_path` is the shared file consumed by both the ingest
    script and the query expander (character/<name>/corpus/synonyms.yaml
    convention). `query_only_path` is an optional additive file whose
    entries ONLY influence query-side expansion, never ingest-side row
    augmentation (character/<name>/corpus/query_synonyms.yaml
    convention). When both files carry entries for the same topic, the
    variants are merged (duplicates deduped).

    Returns a `NullQueryExpander` when both files are missing or empty.

    Topic-prefix resolution (harness-m78r):
      - If either YAML declares `prefix: "..."`, that wins. When both
        declare a prefix the shared file wins (it owns ingest-side data).
      - Else if either YAML uses the legacy `sections:` key, prefix
        defaults to `§` (back-compat with airton_c1 / airton_c).
      - Else (new `topics:` schema only) prefix defaults to `""`.

    Malformed YAML or unexpected shape silently degrades — this helper
    prefers missing entries over a failed chat bootstrap. Inspect with
    `is_empty` or `isinstance(..., NullQueryExpander)` for a strict
    check.
    """
    shared = (
        _parse_glossary_file(synonyms_path)
        if synonyms_path is not None
        else _ParsedGlossary(topics={}, prefix=None, had_legacy_key=False)
    )
    query_only = (
        _parse_glossary_file(query_only_path)
        if query_only_path is not None
        else _ParsedGlossary(topics={}, prefix=None, had_legacy_key=False)
    )
    if not shared.topics and not query_only.topics:
        return NullQueryExpander()
    # Merge: union the topic keys; for topics present in both, merge
    # variants (preserving insertion order from shared first, then
    # query-only, deduped).
    merged: dict[str, tuple[str, ...]] = {}
    for topic in list(shared.topics) + [t for t in query_only.topics if t not in shared.topics]:
        seen: dict[str, None] = {}
        for source in (shared.topics.get(topic, ()), query_only.topics.get(topic, ())):
            for variant in source:
                seen.setdefault(variant, None)
        merged[topic] = tuple(seen)
    if not merged:
        return NullQueryExpander()

    if shared.prefix is not None:
        prefix = shared.prefix
    elif query_only.prefix is not None:
        prefix = query_only.prefix
    elif shared.had_legacy_key or query_only.had_legacy_key:
        prefix = "§"
    else:
        prefix = ""
    return QueryExpander(merged, topic_prefix=prefix)


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


# ---------- LLM query expansion (harness-hvu1) ----------


_DEFAULT_LLM_EXPAND_PROMPT = """\
Convert the user's question into 3-5 short keyword phrases matching
the wording {context_hint} would use for THE QUESTION'S SPECIFIC TOPIC.

Rules:
- Output ONLY the phrases, one per line. No numbering, no bullets,
  no quotes, no commentary, no examples.
- Stay on the question's topic. Do not emit generic separation /
  spacing / approach phrases unless the question is about those.
- Each phrase 2-6 words.
- Skip the user's literal phrasing — that's already in the query.

User question: {query}

Phrases:"""


def default_llm_expand_prompt_path(character_path: Path) -> Path:
    """Per-character override location. When this file exists its
    contents replace the default prompt — useful for a character with
    a non-FAA corpus (the default is JO 7110.65-flavoured) or for
    iterating on prompt wording without a code change."""
    return character_path / "retrieval_prompts" / "query_expansion.md"


def _load_llm_expand_prompt(prompt_path: Path | None) -> str:
    """Read a prompt template from disk, fall back to default. Caller
    formats with `.format(context_hint=..., query=...)`. Missing /
    empty file returns the default."""
    if prompt_path is None:
        return _DEFAULT_LLM_EXPAND_PROMPT
    if not prompt_path.exists():
        return _DEFAULT_LLM_EXPAND_PROMPT
    text = prompt_path.read_text(encoding="utf-8").strip()
    return text or _DEFAULT_LLM_EXPAND_PROMPT


# Lines we drop from model output: numbering, bullets, surrounding quotes.
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s*")
_QUOTE_RE = re.compile(r'^\s*[\'"]?\s*(.*?)\s*[\'"]?\s*$')


def _parse_rewrites(raw: str, *, max_rewrites: int) -> tuple[str, ...]:
    """Pluck up to `max_rewrites` keyword phrases from the model's reply.
    Tolerates numbered / bulleted / quoted output — the prompt asks for
    plain newline-delimited phrases but small models drift, and we'd
    rather return three usable phrases than zero."""
    out: list[str] = []
    for line in raw.splitlines():
        cleaned = _LIST_PREFIX_RE.sub("", line).strip()
        if not cleaned:
            continue
        match = _QUOTE_RE.match(cleaned)
        if match:
            cleaned = match.group(1).strip()
        if not cleaned:
            continue
        # Drop "Phrases:" / "Output:" header echoes the model sometimes
        # repeats from the prompt.
        if cleaned.endswith(":") and len(cleaned) <= 16:
            continue
        out.append(cleaned)
        if len(out) >= max_rewrites:
            break
    # Dedupe while preserving order — small models occasionally repeat
    # themselves and a duplicate phrase doesn't add retrieval signal.
    deduped: list[str] = []
    seen_lower: set[str] = set()
    for phrase in out:
        key = phrase.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        deduped.append(phrase)
    return tuple(deduped)


@runtime_checkable
class _LLMAdapterProto(Protocol):
    """Structural type matching the slice of `ModelAdapter` we use.
    Kept as a local Protocol so test stubs can implement just
    `complete()` without taking on the full adapter contract
    (`id`, `context_window`, etc.)."""

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str: ...


class LLMQueryExpander(QueryExpander):
    """Pre-retrieval rewrite using a small model (harness-hvu1).

    Lay-language queries have low overlap with corpus phrasing — even
    after the static synonym expander runs (harness-ajn), novel
    paraphrases the synonym table doesn't cover still miss. This
    expander asks a small model (typically the same one driving the
    intent router) to emit 3-5 doc-style keyword phrases for the user's
    question, then appends them to the query so both BM25 and dense
    cosine see the jargon-space rewriting.

    Composition: `chain_to` (typically the static `QueryExpander`)
    runs AFTER the LLM rewrite, so per-section synonym lookups still
    fire on the original lay query AND on any rewrite that happens to
    name a section keyword. Order chosen because the static expander
    is cheap, deterministic, and identity-preserving when nothing
    triggers — running it last is free.

    Failure handling: any exception from the adapter (timeout,
    decode error, OOM) falls through silently to the chained
    expander. Retrieval still runs on the original query plus
    static synonyms — degradation is graceful, not fatal.

    Caching: stateless across instances. A per-process cache is
    deliberately omitted in v1; the bead allots ≤200ms p50 for the
    rewrite call, and small models hit that budget without help.
    Add a TTL cache here if a benchmark shows repeated queries
    dominating cost."""

    def __init__(
        self,
        adapter: _LLMAdapterProto,
        *,
        chain_to: QueryExpander | None = None,
        prompt_template: str = _DEFAULT_LLM_EXPAND_PROMPT,
        context_hint: str = "FAA Order JO 7110.65 (Air Traffic Control)",
        max_rewrites: int = 5,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        # Skip the parent constructor's topic-table init — we don't
        # use the static-table path. `chain_to`, when present, owns
        # the topic-table semantics for the chained pass.
        self._topics: dict[str, tuple[str, ...]] = {}
        self._topic_prefix = ""
        self._term_tokens = ()
        self._adapter = adapter
        self._chain_to = chain_to
        self._prompt_template = prompt_template
        self._context_hint = context_hint
        self._max_rewrites = max_rewrites
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def is_empty(self) -> bool:
        # Never identity-empty: if the chained expander is empty we
        # still emit LLM rewrites. Keeps the load_query_expander API
        # consistent (callers that branch on `is_empty` to skip
        # building still get a working expander when the LLM path is
        # the only signal).
        return False

    def expand(self, query: str) -> str:
        from harness.model.adapter import ChatMessage

        prompt = self._prompt_template.format(
            context_hint=self._context_hint,
            query=query,
        )
        messages = [ChatMessage(role="user", content=prompt)]
        try:
            raw = self._adapter.complete(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception:
            # Adapter failure → no rewrites; fall through to chained
            # expander on the original query.
            rewrites: tuple[str, ...] = ()
        else:
            rewrites = _parse_rewrites(raw, max_rewrites=self._max_rewrites)

        if rewrites:
            joined = "; ".join(rewrites)
            augmented = f"{query} [paraphrases: {joined}]"
        else:
            augmented = query

        if self._chain_to is not None:
            return self._chain_to.expand(augmented)
        return augmented
