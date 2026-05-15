"""Read-tier academic-paper search across Semantic Scholar + OpenAlex.

Replaces the dead-end `site:scholar.google.com` approach. Google
Scholar publishes paper-search URLs dynamically (`/scholar?q=...`),
which DDG doesn't index deeply — a `site:` scope returns author
profile pages instead of paper results. Hitting structured
academic-paper APIs directly avoids that asymmetry.

Two free, no-auth-required APIs:

  * Semantic Scholar (`api.semanticscholar.org/graph/v1/paper/search`).
    ~200M papers. Anonymous tier: 100 req / 5 min, 1 req/s. Returns
    abstracts as plain text, citation counts, externalIds (DOI,
    arXiv), open-access PDF URLs.

  * OpenAlex (`api.openalex.org/works`). ~250M works. No rate limit
    when a `mailto=<email>` query param is included (the "polite
    pool"). Returns abstracts as inverted indexes (word → positions)
    that need reconstruction.

Fan-out strategy: hit both sequentially, parse to a common `_Paper`
struct, dedupe by DOI → arXiv ID → normalized title, rank by
(in-both-APIs, citation_count, year). A failure from one API doesn't
poison the other — returns whatever the surviving call produced.

Privacy note: queries leave the Tailscale boundary. Both APIs log
the query as part of normal operation. Users who care should disable
search_scholar via --tools-drop."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harness.tools.base import ToolSpec

_DEFAULT_TIMEOUT_S = 15
_MAX_TIMEOUT_S = 60
_DEFAULT_MAX_RESULTS = 5
_ABSTRACT_EXCERPT_CHARS = 300

_USER_AGENT = "Mozilla/5.0 (compatible; harness-scholar/1.0)"

_S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,abstract,year,authors,citationCount,externalIds,openAccessPdf,paperId,url"
_OPENALEX_ENDPOINT = "https://api.openalex.org/works"
# Default mailto for OpenAlex's polite pool. Override via
# HARNESS_OPENALEX_MAILTO. Without a mailto OpenAlex still serves
# results but rate-limits more aggressively.
_DEFAULT_OPENALEX_MAILTO = "harness@local"

# Strip punctuation + whitespace from titles before comparing. Same
# normalisation used for both APIs so a dedupe across them works
# even when one has trailing-period quirks.
_TITLE_NORM_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _Paper:
    """Common paper struct across both APIs. `sources` records which
    APIs returned this paper so the merge step can rank papers that
    appear in BOTH above papers that appear in one — appearance in
    both is the strongest relevance signal."""

    title: str
    authors: tuple[str, ...]
    year: int | None
    citation_count: int | None
    abstract: str
    doi: str | None
    arxiv_id: str | None
    open_access_pdf: str | None
    source_url: str
    sources: frozenset[str] = field(default_factory=frozenset)


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used as
    a fallback dedupe key when neither DOI nor arXiv ID is available."""
    stripped = _TITLE_NORM_RE.sub(" ", title.lower())
    return _WS_RE.sub(" ", stripped).strip()


def _dedupe_key(paper: _Paper) -> str:
    """Canonical key for cross-API dedupe. DOI wins (it's globally
    unique by definition); arXiv ID is second-best (preprint
    identifier); normalized title is the last-resort tie-breaker
    when the paper has no persistent identifier (rare for published
    work but happens for early preprints)."""
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id.lower()}"
    return f"title:{_normalize_title(paper.title)}"


def _abstract_excerpt(text: str | None) -> str:
    """Trim an abstract to a one-paragraph excerpt for the tool output.
    The model gets enough context to triage relevance; the full
    abstract is reachable via fetch_url on the source URL if needed."""
    if not text:
        return ""
    cleaned = _WS_RE.sub(" ", text).strip()
    if len(cleaned) <= _ABSTRACT_EXCERPT_CHARS:
        return cleaned
    return cleaned[:_ABSTRACT_EXCERPT_CHARS].rstrip() + "…"


def _format_authors(authors: tuple[str, ...]) -> str:
    """Render an author list as `A, B, C` for 1-3 authors or
    `A et al` for 4+. The "et al" cutoff is intentional: full
    author lists for some-papers run to 50+ names and waste tokens."""
    if not authors:
        return "(unknown authors)"
    if len(authors) <= 3:
        return ", ".join(authors)
    return f"{authors[0]} et al"


def _reconstruct_openalex_abstract(inverted: dict[str, Any] | None) -> str:
    """OpenAlex stores abstracts as `{word: [positions...]}` (an
    inverted index). Reconstruct the running text by ordering each
    (word, position) pair and joining. Returns empty string when the
    field is absent or unusable; the rest of the parser tolerates
    missing abstracts."""
    if not isinstance(inverted, dict) or not inverted:
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                positioned.append((pos, word))
    positioned.sort()
    return " ".join(word for _, word in positioned)


def _parse_s2(payload: Any) -> list[_Paper]:
    """Parse a Semantic Scholar `/paper/search` response. Tolerant
    of missing fields — papers without an ID or title are dropped;
    papers with partial metadata are kept with the missing fields
    set to None / empty."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    out: list[_Paper] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        external = row.get("externalIds") or {}
        doi = external.get("DOI") if isinstance(external, dict) else None
        arxiv_id = external.get("ArXiv") if isinstance(external, dict) else None
        authors_raw = row.get("authors") or []
        authors: list[str] = []
        if isinstance(authors_raw, list):
            for author in authors_raw:
                if isinstance(author, dict):
                    name = author.get("name")
                    if isinstance(name, str) and name.strip():
                        authors.append(name.strip())
        open_access = row.get("openAccessPdf") or {}
        oa_url = open_access.get("url") if isinstance(open_access, dict) else None
        # Source URL preference: DOI > arXiv > Semantic Scholar's own
        # paper page. The model picks the citation form to match.
        if doi:
            source_url = f"https://doi.org/{doi}"
        elif arxiv_id:
            source_url = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            paper_id = row.get("paperId")
            source_url = (
                f"https://www.semanticscholar.org/paper/{paper_id}"
                if isinstance(paper_id, str)
                else row.get("url", "") or ""
            )
        citation_count = row.get("citationCount")
        year = row.get("year")
        raw_abstract = row.get("abstract")
        abstract = raw_abstract if isinstance(raw_abstract, str) else ""
        out.append(
            _Paper(
                title=title.strip(),
                authors=tuple(authors),
                year=int(year) if isinstance(year, int) else None,
                citation_count=int(citation_count) if isinstance(citation_count, int) else None,
                abstract=abstract,
                doi=doi if isinstance(doi, str) else None,
                arxiv_id=arxiv_id if isinstance(arxiv_id, str) else None,
                open_access_pdf=oa_url if isinstance(oa_url, str) else None,
                source_url=source_url,
                sources=frozenset({"s2"}),
            )
        )
    return out


def _parse_openalex(payload: Any) -> list[_Paper]:
    """Parse an OpenAlex `/works` response. Same tolerance contract
    as `_parse_s2`. Abstract is reconstructed from
    `abstract_inverted_index`; DOI is canonicalised by stripping the
    `https://doi.org/` prefix to match S2's bare-DOI format."""
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    out: list[_Paper] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        title = row.get("title") or row.get("display_name")
        if not isinstance(title, str) or not title.strip():
            continue
        doi_full = row.get("doi")
        doi = None
        if isinstance(doi_full, str):
            # OpenAlex returns `https://doi.org/10.x/y`; strip prefix
            # to match S2's bare `10.x/y` form for dedupe.
            doi = doi_full.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        # OpenAlex doesn't expose arXiv ID directly; check ids dict.
        ids = row.get("ids") or {}
        arxiv_id = None
        if isinstance(ids, dict):
            arxiv_url = ids.get("arxiv")
            if isinstance(arxiv_url, str):
                # arxiv ID embedded in URL like https://arxiv.org/abs/2301.08243
                match = re.search(r"arxiv\.org/abs/([^/\s]+)", arxiv_url)
                if match:
                    arxiv_id = match.group(1)
        authorships = row.get("authorships") or []
        authors: list[str] = []
        if isinstance(authorships, list):
            for ship in authorships:
                if not isinstance(ship, dict):
                    continue
                author = ship.get("author")
                if isinstance(author, dict):
                    name = author.get("display_name")
                    if isinstance(name, str) and name.strip():
                        authors.append(name.strip())
        open_access = row.get("open_access") or {}
        oa_url = open_access.get("oa_url") if isinstance(open_access, dict) else None
        # Source URL: prefer DOI, then arXiv abs, then OpenAlex's own
        # work URL (returned in row.id).
        if doi:
            source_url = f"https://doi.org/{doi}"
        elif arxiv_id:
            source_url = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            source_url = row.get("id", "") if isinstance(row.get("id"), str) else ""
        citation_count = row.get("cited_by_count")
        year = row.get("publication_year")
        abstract = _reconstruct_openalex_abstract(row.get("abstract_inverted_index"))
        out.append(
            _Paper(
                title=title.strip(),
                authors=tuple(authors),
                year=int(year) if isinstance(year, int) else None,
                citation_count=int(citation_count) if isinstance(citation_count, int) else None,
                abstract=abstract,
                doi=doi or None,
                arxiv_id=arxiv_id,
                open_access_pdf=oa_url if isinstance(oa_url, str) else None,
                source_url=source_url,
                sources=frozenset({"openalex"}),
            )
        )
    return out


def _merge_papers(
    s2_papers: list[_Paper], openalex_papers: list[_Paper], *, max_results: int
) -> list[_Paper]:
    """Dedupe across both API result sets and rank.

    Dedupe key: DOI > arXiv ID > normalized title. When the same
    paper appears in both APIs, the merged record carries
    `sources={'s2', 'openalex'}` — strongest relevance signal, used
    in the rank step.

    Rank:
      1. Papers in BOTH APIs (sources size 2) come first.
      2. Within each tier, sort by citation_count descending
         (None treated as 0).
      3. Then by year descending (newer first; ties broken by None
         coming last).

    S2 record wins on metadata conflicts (better abstracts, more
    consistent author formatting). OpenAlex fills in fields S2 left
    blank (citation counts sometimes differ, abstracts sometimes
    only one side has, year occasionally absent on one side)."""
    by_key: dict[str, _Paper] = {}
    # S2 first so its fields take precedence on conflicts.
    for paper in s2_papers:
        by_key[_dedupe_key(paper)] = paper
    for paper in openalex_papers:
        key = _dedupe_key(paper)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = paper
        else:
            # Merge: keep S2's primary fields, fill in OpenAlex
            # where S2 left None. Record both sources.
            by_key[key] = _Paper(
                title=existing.title,
                authors=existing.authors or paper.authors,
                year=existing.year if existing.year is not None else paper.year,
                citation_count=(
                    existing.citation_count
                    if existing.citation_count is not None
                    else paper.citation_count
                ),
                abstract=existing.abstract or paper.abstract,
                doi=existing.doi or paper.doi,
                arxiv_id=existing.arxiv_id or paper.arxiv_id,
                open_access_pdf=existing.open_access_pdf or paper.open_access_pdf,
                source_url=existing.source_url,
                sources=existing.sources | paper.sources,
            )
    # Rank.
    merged = list(by_key.values())
    merged.sort(
        key=lambda p: (
            -len(p.sources),
            -(p.citation_count or 0),
            -(p.year or 0),
        )
    )
    return merged[:max_results]


def _format_output(papers: list[_Paper], query: str) -> str:
    """Render the merged paper list as the tool's return string.

    Header gives the model the query echo + result count so it can
    cross-check its own claims about what came back. Each paper line
    carries: position, year, title, authors, citation count, source
    badges, abstract excerpt, source URL. Source badges (`[s2]`,
    `[openalex]`, `[s2,openalex]`) tell the model whether a paper is
    cross-validated."""
    if not papers:
        return f"(no scholar results for {query!r})"
    lines = [f"{len(papers)} result(s) for {query!r}:"]
    for idx, paper in enumerate(papers, start=1):
        year_str = f"{paper.year}" if paper.year is not None else "n.d."
        authors_str = _format_authors(paper.authors)
        citation_str = (
            f"cited {paper.citation_count}" if paper.citation_count is not None else "cited ?"
        )
        sources_str = "+".join(sorted(paper.sources))
        line = f"{idx}. [{year_str}] {paper.title} — {authors_str} ({citation_str}) [{sources_str}]"
        lines.append(line)
        excerpt = _abstract_excerpt(paper.abstract)
        if excerpt:
            lines.append(f"   {excerpt}")
        if paper.source_url:
            lines.append(f"   {paper.source_url}")
    return "\n".join(lines)


_OpenerFn = Callable[..., Any]


def _call_s2(
    query: str, *, max_results: int, timeout_s: int, opener: _OpenerFn | None
) -> list[_Paper]:
    """Hit Semantic Scholar. Empty list on any failure (network,
    HTTP error, JSON decode); the caller falls back to OpenAlex."""
    params = urllib.parse.urlencode(
        {"query": query, "limit": str(max_results), "fields": _S2_FIELDS}
    )
    url = f"{_S2_ENDPOINT}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    request_opener = opener or urllib.request.urlopen
    try:
        with request_opener(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return []
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    return _parse_s2(payload)


def _call_openalex(
    query: str,
    *,
    max_results: int,
    timeout_s: int,
    mailto: str,
    opener: _OpenerFn | None,
) -> list[_Paper]:
    """Hit OpenAlex. Same empty-on-failure contract as `_call_s2`."""
    params = urllib.parse.urlencode(
        {
            "search": query,
            "per-page": str(max_results),
            "mailto": mailto,
        }
    )
    url = f"{_OPENALEX_ENDPOINT}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
    request_opener = opener or urllib.request.urlopen
    try:
        with request_opener(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return []
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    return _parse_openalex(payload)


@dataclass
class SearchScholarTool:
    """Search academic papers across Semantic Scholar + OpenAlex.

    Fans out to both APIs, merges the results by DOI / arXiv ID /
    normalized title, and returns a ranked paper list with citation
    counts, authors, year, and abstract excerpts. Pairs with
    `fetch_url` (whose allowlist includes doi.org, arxiv.org,
    scholar.google.com): the model picks a result, fetches the
    paper page for full content, cites with the corresponding
    `[doi:...]` / `[arxiv:...]` / `[scholar:...]` form.

    Instance knobs are exposed for tests; production wiring uses
    defaults. `request_opener` lets the test suite monkeypatch HTTP
    without patching urllib globally."""

    default_max_results: int = _DEFAULT_MAX_RESULTS
    default_timeout_s: int = _DEFAULT_TIMEOUT_S
    openalex_mailto: str = ""
    request_opener: _OpenerFn | None = None

    def __post_init__(self) -> None:
        # Env-var override for OpenAlex's polite-pool mailto. Resolved
        # lazily so changing the env between sessions takes effect
        # without re-import.
        if not self.openalex_mailto:
            self.openalex_mailto = os.environ.get(
                "HARNESS_OPENALEX_MAILTO", _DEFAULT_OPENALEX_MAILTO
            )

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_scholar",
            description=(
                "ACADEMIC / SCIENTIFIC SEARCH. **Default for any query "
                "about a research topic, scientific concept, technical "
                "acronym, or recent academic work.** Use for phrasings "
                "like 'search and summarize articles about X', 'find "
                "papers on Y', 'recent work on Z', 'research about <topic>', "
                "'who proposed <method>', 'what is <acronym>' when the "
                "acronym names an algorithm / model / scientific concept "
                "(JEPA, BERT, RLHF, CRISPR, mRNA, transformer, etc.). "
                "Searches Semantic Scholar and OpenAlex (free, no-auth). "
                "Returns a numbered list of papers with title, authors, "
                "year, citation count, source badges (`[s2]`, `[openalex]`, "
                "`[s2+openalex]`), abstract excerpt, and DOI / arXiv URL. "
                "Prefer this over search_web for ANY academic / scientific / "
                "research-shaped question; search_web is for general web "
                "orientation (Wikipedia, blogs, news, vendor docs). Follow "
                "up with fetch_url on a paper's DOI or arXiv URL when you "
                "need the full content. Citation form: [doi:<id>] or "
                "[arxiv:<id>]; use the URL the tool returned, not a "
                "fabricated identifier."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Free-form natural language works "
                            "(e.g. 'self-supervised representation learning JEPA'). "
                            "Quoted phrases get exact-match weight; lowercase "
                            "is fine."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Number of results to return (default 5). Cap at "
                            "~10 to keep the output readable."
                        ),
                    },
                },
                "required": ["query"],
            },
            tier="read",
            display_name="Search scholar",
            high_noise=True,
        )

    def call(self, *, query: str, max_results: int | None = None) -> str:
        if not query.strip():
            raise ValueError("query must not be empty")
        n = max_results if max_results is not None else self.default_max_results
        if n <= 0:
            raise ValueError("max_results must be > 0")
        timeout_s = min(self.default_timeout_s, _MAX_TIMEOUT_S)

        # Sequential fan-out. ~1-2s per API on a warm path; total
        # ~2-4s. Concurrent fan-out would shave half but adds
        # threading complexity for marginal gain at v1.
        s2 = _call_s2(query, max_results=n, timeout_s=timeout_s, opener=self.request_opener)
        oa = _call_openalex(
            query,
            max_results=n,
            timeout_s=timeout_s,
            mailto=self.openalex_mailto,
            opener=self.request_opener,
        )
        merged = _merge_papers(s2, oa, max_results=n)
        return _format_output(merged, query)
