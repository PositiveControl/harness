"""Read-tier web search via DuckDuckGo's HTML endpoint. Stdlib only —
no new deps, at the cost of a regex-based HTML parser that can break
if DDG changes their markup. If that happens, swap in the `ddgs`
package (optional extra).

Privacy note: search queries leave the Tailscale boundary. Users
who care about that should disable this tool via --tools-drop
search_web. It is not included in the `core` profile by default; the
CLI enables it only when requested explicitly."""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from harness.tools.base import ToolSpec

_DEFAULT_TIMEOUT_S = 10
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 harness-search"

# DuckDuckGo's non-JS HTML endpoint. Emits a reasonably stable structure:
# each result lives in <a class="result__a" href="...">TITLE</a> followed by
# a <a class="result__snippet">…SNIPPET…</a> in an adjacent div.
_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"

_TITLE_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Detect a `site:` operator anywhere in the query — case-insensitive,
# anchored on word boundary so "website:foo" doesn't false-positive.
# Used by SearchWebTool to skip the default_site_filter prepend when
# the caller already named a site scope (the override path).
_SITE_OPERATOR_RE = re.compile(r"\bsite\s*:", re.IGNORECASE)


def _strip_tags(fragment: str) -> str:
    """Remove HTML tags and collapse whitespace, ASCII-safe."""
    no_tags = _TAG_RE.sub("", fragment)
    # DuckDuckGo HTML-escapes & and " but leaves most unicode intact.
    for entity, replacement in (
        ("&amp;", "&"),
        ("&quot;", '"'),
        ("&#x27;", "'"),
        ("&#39;", "'"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
    ):
        no_tags = no_tags.replace(entity, replacement)
    return _WS_RE.sub(" ", no_tags).strip()


def _unwrap_ddg_redirect(url: str) -> str:
    """DDG wraps links as /l/?uddg=<target>&... — pull the target out."""
    if url.startswith("//duckduckgo.com/l/") or url.startswith("/l/"):
        parsed = urllib.parse.urlparse(url if url.startswith("//") else "https:" + url)
        # Only covers the `//` case; the `/l/` relative case resolves the same way.
        if parsed.query:
            qs = urllib.parse.parse_qs(parsed.query)
            target = qs.get("uddg", [None])[0]
            if target:
                return urllib.parse.unquote(target)
    return url


@dataclass
class SearchWebTool:
    """Search the public web via DuckDuckGo's HTML endpoint. Returns
    title + URL + snippet for the top results.

    When `allowed_hosts` is supplied (typically `character.
    fetch_url_allowed_hosts`), results from those hosts float to the
    top of the output with a `[allowlisted]` marker; non-allowlisted
    hits stay visible below with `[external]`. Pairs with FetchUrlTool
    on a scoped corpus: the agent sees what's out there, but the
    sources it can actually fetch land first. None (default) means
    no reranking and no markers — the unbounded-web behavior for
    general-purpose characters."""

    default_max_results: int = 5
    timeout_s: int = _DEFAULT_TIMEOUT_S
    allowed_hosts: frozenset[str] | None = None
    # When set, every query gets `site:<filter> ` prepended unless
    # the query already includes a `site:` operator. Lets scholar-
    # style characters scope their default search to one host
    # (airton_f → scholar.google.com) without forcing the model to
    # remember the operator on every call. Override path: the model
    # writes `query site:<other>` and the prepend is skipped.
    default_site_filter: str | None = None

    @property
    def spec(self) -> ToolSpec:
        scope_note = ""
        if self.default_site_filter:
            scope_note = (
                f" This character's default search scope is "
                f"`site:{self.default_site_filter}` — every query is "
                f"prepended with that operator unless your query already "
                f"contains a `site:` clause. To search outside that "
                f"scope, include a different `site:` operator (e.g. "
                f"`site:arxiv.org`) directly in the query string."
            )
        return ToolSpec(
            name="search_web",
            description=(
                "Search the public web for a query via DuckDuckGo. "
                "Returns a numbered list of `TITLE — URL — SNIPPET` "
                "triples. Use when the user asks a factual question "
                "you can't answer from memory or the workspace. "
                "Prefer a broad fetch (default 5 results) so you can "
                "compare sources and pick the best candidates, then "
                "follow up with fetch_url on the most promising one "
                "or two URLs when the snippets aren't enough. Do not "
                "set max_results to 1 unless the user explicitly asked "
                "for a single top result." + scope_note
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'MLX Qwen 32B quantization'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Number of results to return. Default 5. Cap around 10 "
                            "to keep the tool output readable. Only set this when "
                            "the user named a specific count — otherwise omit it "
                            "and take the default broad fetch."
                        ),
                    },
                },
                "required": ["query"],
            },
            tier="read",
            display_name="Search web",
            high_noise=True,
        )

    def call(self, *, query: str, max_results: int | None = None) -> str:
        if not query.strip():
            raise ValueError("query must not be empty")
        n = max_results if max_results is not None else self.default_max_results
        if n <= 0:
            raise ValueError("max_results must be > 0")

        # Auto-prepend `site:<filter>` for scholar-style characters
        # whose default search is scoped to one host. Skipped when the
        # caller already named a `site:` operator — that's the
        # override path. Case-insensitive match on the operator only;
        # the host portion is preserved verbatim.
        effective_query = query
        if self.default_site_filter and not _SITE_OPERATOR_RE.search(query):
            effective_query = f"site:{self.default_site_filter} {query}"

        data = urllib.parse.urlencode({"q": effective_query}).encode()
        req = urllib.request.Request(  # noqa: S310 — https only, validated literal host
            _HTML_ENDPOINT,
            data=data,
            headers={"User-Agent": _USER_AGENT},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                body = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return f"(search failed: {type(exc).__name__}: {exc})"

        titles = _TITLE_RE.findall(body)
        snippets = _SNIPPET_RE.findall(body)
        if not titles:
            return f"(no results for {query!r})"

        # Unwrap + classify every result BEFORE slicing. If we sliced
        # first, an allowlisted hit at position 8 would be cut off
        # and the agent would never see a source it can fetch — even
        # though one exists. Reranking-then-slicing surfaces them.
        classified: list[_Result] = []
        for idx, (url, title_html) in enumerate(titles):
            url_clean = _unwrap_ddg_redirect(url)
            host = (urllib.parse.urlparse(url_clean).hostname or "").lower()
            is_allowlisted = self.allowed_hosts is not None and _host_matches_allowlist(
                host, self.allowed_hosts
            )
            snippet = _strip_tags(snippets[idx]) if idx < len(snippets) else ""
            classified.append(
                _Result(
                    is_allowlisted=is_allowlisted,
                    original_idx=idx,
                    url=url_clean,
                    title=_strip_tags(title_html),
                    snippet=snippet,
                )
            )

        if self.allowed_hosts is not None:
            # Stable sort: allowlisted hits float to the top while
            # preserving DDG's intra-group ranking. Python's sort is
            # stable so the secondary key (original_idx) only matters
            # when two results share is_allowlisted.
            classified.sort(key=lambda r: (not r.is_allowlisted, r.original_idx))

        lines: list[str] = []
        for display_idx, result in enumerate(classified[:n], start=1):
            marker = ""
            if self.allowed_hosts is not None:
                marker = " [allowlisted]" if result.is_allowlisted else " [external]"
            entry = f"{display_idx}.{marker} {result.title} — {result.url}"
            if result.snippet:
                entry += f"\n   {result.snippet}"
            lines.append(entry)
        return "\n".join(lines)


@dataclass(frozen=True)
class _Result:
    """Parsed search result plus allowlist classification. Internal —
    keeps the call() body readable + the sort key correct (Python
    can't sort dicts; named-tuple / dataclass is the clean answer)."""

    is_allowlisted: bool
    original_idx: int
    url: str
    title: str
    snippet: str


def _host_matches_allowlist(host: str, allowed_hosts: frozenset[str]) -> bool:
    """Match `host` (lowercased netloc) against `allowed_hosts` using
    exact match OR registrable-suffix match. `en.wikipedia.org`
    declared in the allowlist matches `en.wikipedia.org` exactly;
    `arxiv.org` matches `arxiv.org` AND `www.arxiv.org` (a common
    DDG-result variant). Empty host → never matches."""
    if not host:
        return False
    for allowed in allowed_hosts:
        a = allowed.lower()
        if host == a or host.endswith("." + a):
            return True
    return False
