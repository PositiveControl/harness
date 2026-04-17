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
    title + URL + snippet for the top results."""

    default_max_results: int = 5
    timeout_s: int = _DEFAULT_TIMEOUT_S

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_web",
            description=(
                "Search the public web for a query via DuckDuckGo. "
                "Returns a numbered list of `TITLE — URL — SNIPPET` "
                "triples. Use when the user asks a factual question "
                "you can't answer from memory or the workspace. "
                "Follow up with fetch_url if you need the page "
                "contents (when that tool is available)."
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
                            "to keep the tool output readable."
                        ),
                    },
                },
                "required": ["query"],
            },
            tier="read",
            display_name="Search web",
        )

    def call(self, *, query: str, max_results: int | None = None) -> str:
        if not query.strip():
            raise ValueError("query must not be empty")
        n = max_results if max_results is not None else self.default_max_results
        if n <= 0:
            raise ValueError("max_results must be > 0")

        data = urllib.parse.urlencode({"q": query}).encode()
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

        lines: list[str] = []
        for i, (url, title_html) in enumerate(titles[:n], start=1):
            url_clean = _unwrap_ddg_redirect(url)
            title = _strip_tags(title_html)
            snippet = ""
            if i - 1 < len(snippets):
                snippet = _strip_tags(snippets[i - 1])
            entry = f"{i}. {title} — {url_clean}"
            if snippet:
                entry += f"\n   {snippet}"
            lines.append(entry)
        return "\n".join(lines)
