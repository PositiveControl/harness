"""Read-tier tool: HTTPS GET + boilerplate-stripped text extraction.

Lets the agent pull documentation pages, GitHub threads, release notes
etc. when `search_web` gave it a URL but the snippet isn't enough.
Stdlib only — no new dependencies — at the cost of a heuristic
extractor rather than a battle-tested one (trafilatura /
readability-lxml). If extraction quality becomes a bottleneck, swap
in a library behind an optional extra.

Privacy note: like `search_web`, this tool crosses the Tailscale
trust boundary. The CLI keeps it out of the `core` / `minimal`
profiles so sessions that want local-only stay local-only.

Security controls (harness-3zs):
- HTTPS only. HTTP URLs are rejected with a clear error — plaintext
  traffic is almost never what the user wants and encourages MITM.
- Response body capped at `max_response_bytes` (default 2 MB). We
  stream-read in chunks so a 5 GB misconfigured endpoint can't
  exhaust RAM.
- Non-HTML content types are rejected: extracting text from PDFs,
  binary blobs, or JSON APIs needs tools built for that, not a
  generic tag stripper.
- Timeout defaults to 10 s, capped at 30 s so a slow server can't
  stall the whole turn.

Extraction strategy (when `extract_text=True`, the default):
- Parse with `html.parser.HTMLParser` (stdlib).
- Skip `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`,
  `<aside>`, `<form>`, `<noscript>` entirely — navigation chrome and
  style blocks aren't content.
- Prefer the text inside the first `<article>` / `<main>` element
  when one exists; otherwise fall back to the whole body minus the
  skip list.
- Collapse consecutive whitespace to single spaces; split paragraphs
  on block-level opens.
- Truncate the result to `max_chars` characters (default 16k) with a
  `…[truncated]` marker so the summarizer hook (sota punch #3) has a
  manageable chunk to compress.
"""

from __future__ import annotations

import contextlib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser

from harness.store.fetch_denylist import FetchDenylistStore
from harness.tools.base import ToolSpec

_DENYLIST_BLOCKED_STATUSES: frozenset[int] = frozenset({401, 403})

_DEFAULT_TIMEOUT_S = 10
_MAX_TIMEOUT_S = 30
_DEFAULT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB
_DEFAULT_MAX_CHARS = 16_384

_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 harness-fetch"

# Tags whose content never contributes to "what this page says".
# Case-insensitive match on the tag name; HTMLParser lowercases.
_SKIP_TAGS: frozenset[str] = frozenset(
    {
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "noscript",
        "svg",
        "iframe",
    }
)

# Block-level tags — a closing one emits a paragraph break so the
# extracted text keeps readable structure instead of becoming one
# long blob.
_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
    }
)

# Primary-content tags: when present, we prefer the text inside the
# FIRST such element over the whole body. Most modern sites wrap the
# article in one of these; hobbyist HTML does not, so we fall back.
_PRIMARY_TAGS: frozenset[str] = frozenset({"article", "main"})


class _TextExtractor(HTMLParser):
    """Walk the DOM, collecting text outside the skip set. When we
    enter a `<article>` or `<main>`, capture that subtree's text
    separately so the caller can prefer it over the full-body fallback."""

    def __init__(self) -> None:
        # convert_charrefs=True turns &amp; etc into their literal chars
        # before they reach `handle_data`, which is what we want — we
        # don't need to see entity refs ourselves.
        super().__init__(convert_charrefs=True)
        self._body_parts: list[str] = []
        self._primary_parts: list[str] = []
        self._skip_depth = 0
        self._primary_depth = 0
        self._title_parts: list[str] = []
        self._in_title = False

    @property
    def title(self) -> str:
        return "".join(self._title_parts).strip()

    def best_text(self) -> str:
        """Return the primary-content text if we captured any, else
        the whole-body text. Whitespace already collapsed per-chunk
        during parsing; do a final pass here to join paragraphs."""
        source = self._primary_parts if self._primary_parts else self._body_parts
        # Compact consecutive paragraph breaks into single blank lines.
        out: list[str] = []
        last_blank = True
        for chunk in source:
            is_blank = not chunk.strip()
            if is_blank and last_blank:
                continue
            out.append(chunk)
            last_blank = is_blank
        return "\n".join(out).strip()

    # ---- HTMLParser callbacks ----

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _PRIMARY_TAGS:
            self._primary_depth += 1
        # Opening a block-level tag starts a new paragraph if we have
        # content already.
        if tag in _BLOCK_TAGS:
            self._emit_break()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _PRIMARY_TAGS and self._primary_depth > 0:
            self._primary_depth -= 1
        if tag in _BLOCK_TAGS:
            self._emit_break()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closing tags. The only one that matters for readability
        # is <br />, handled via _BLOCK_TAGS.
        if tag in _BLOCK_TAGS:
            self._emit_break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        # Collapse whitespace inside the chunk but preserve at least
        # one space when the chunk wasn't already blank.
        stripped = " ".join(data.split())
        if not stripped:
            return
        self._body_parts.append(stripped)
        if self._primary_depth > 0:
            self._primary_parts.append(stripped)

    # ---- helpers ----

    def _emit_break(self) -> None:
        """Mark a paragraph boundary. `best_text` collapses runs of
        breaks, so repeated calls are safe."""
        self._body_parts.append("")
        if self._primary_depth > 0:
            self._primary_parts.append("")


@dataclass
class FetchUrlTool:
    """Read-tier HTTPS GET + text extraction. See module docstring
    for the full policy + extraction rationale.

    Instance-level knobs exist for tests; production wiring uses the
    defaults. `request_opener` is injectable so the test suite can
    monkeypatch the network call without patching `urllib` globally."""

    default_timeout_s: int = _DEFAULT_TIMEOUT_S
    max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES
    max_chars: int = _DEFAULT_MAX_CHARS
    user_agent: str = _USER_AGENT
    # Optional host allowlist. None (default) = no restriction: fetch
    # any HTTPS URL. A non-None frozenset restricts the tool to URLs
    # whose netloc is an exact member. Caller-supplied per-character:
    # atc (airton_c) ships with an aviation-source allowlist so
    # fetch_url can't wander off into arbitrary web content. Kept as
    # a tool-level knob rather than a hook so the refusal happens
    # before the network call (atc-3 / harness-xbk.3).
    allowed_hosts: frozenset[str] | None = None
    # Persistent denylist (harness-4dgm). When supplied, hosts that
    # have returned 401/403 within the store's TTL window short-circuit
    # before the network call, and fresh 401/403 responses are recorded.
    # None disables the feature — useful for tests and for the echo-
    # adapter dry run where there's no character DB.
    denylist: FetchDenylistStore | None = None
    # Allow tests (or future callers) to supply their own opener —
    # the production path uses urllib.request.urlopen through the
    # module-level alias below. Keep the type wide (`object`) so the
    # dataclass stays importable without pulling urllib typing into
    # the public surface.
    request_opener: object | None = field(default=None, repr=False)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fetch_url",
            description=(
                "Fetch an HTTPS URL and return its readable text "
                "(boilerplate stripped). Use after search_web when the "
                "snippet isn't enough to answer the question — e.g. to "
                "read a GitHub issue thread, a docs page, or a blog "
                "post. HTTP URLs, non-HTML payloads, and responses "
                "over ~2 MB are refused. Extracted text is truncated "
                "to ~16 KB characters; let the summarizer compress if "
                "that's still too much."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "HTTPS URL to fetch. Must start with https:// — "
                            "plain http:// is rejected."
                        ),
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": (
                            "Strip HTML chrome + extract the readable text (default True). "
                            "Set False to get the raw HTML (still size-capped)."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": (
                            "Per-request timeout in seconds. Default 10, capped at 30."
                        ),
                    },
                },
                "required": ["url"],
            },
            tier="read",
            display_name="Fetch URL",
            high_noise=True,
        )

    def call(
        self,
        *,
        url: str,
        extract_text: bool = True,
        timeout_seconds: int | None = None,
    ) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            return (
                f"fetch_url error: only https:// is allowed (got scheme "
                f"{parsed.scheme!r}). Plain http is rejected to avoid MITM."
            )
        if not parsed.netloc:
            return f"fetch_url error: URL {url!r} has no host"
        # Strip user:pass@ and :port for both the allowlist check and
        # the denylist record. DNS is case-insensitive, so the lowered
        # hostname is the canonical key.
        host = (parsed.hostname or parsed.netloc).lower()
        if self.allowed_hosts is not None and host not in self.allowed_hosts:
            return (
                f"fetch_url error: host {host!r} is not in this "
                "character's fetch allowlist. Allowed hosts: "
                f"{sorted(self.allowed_hosts)}"
            )

        if self.denylist is not None:
            blocked = self.denylist.is_blocked(host)
            if blocked is not None:
                return (
                    f"fetch_url error: host {host!r} is on the denylist "
                    f"(HTTP {blocked.last_status} {blocked.last_reason}, "
                    f"first seen {blocked.first_seen_at.date().isoformat()}, "
                    f"{blocked.count} hit{'s' if blocked.count != 1 else ''}). "
                    "Skipping the fetch. Try a different source, or run "
                    f"`harness denylist clear --host {host}` to retry."
                )

        timeout_s = self.default_timeout_s if timeout_seconds is None else int(timeout_seconds)
        if timeout_s <= 0:
            return "fetch_url error: timeout_seconds must be > 0"
        timeout_s = min(timeout_s, _MAX_TIMEOUT_S)

        opener = self.request_opener or urllib.request.urlopen
        # We've already validated that the URL's scheme is https above;
        # S310 would be right to complain about an unsanitized Request
        # target but here the guard above makes it unreachable.
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})  # noqa: S310
        try:
            response = opener(req, timeout=timeout_s)  # type: ignore[operator]
        except urllib.error.HTTPError as exc:
            if self.denylist is not None and exc.code in _DENYLIST_BLOCKED_STATUSES:
                self.denylist.record(
                    host=host,
                    status=exc.code,
                    reason=str(exc.reason),
                    url=url,
                )
            return f"fetch_url error: HTTP {exc.code} {exc.reason} for {url}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return f"fetch_url error: network failure for {url}: {exc}"

        # Introspect headers BEFORE reading the body so we can reject
        # oversized / wrong-type payloads without paying the network
        # cost first.
        headers = getattr(response, "headers", None)
        try:
            content_type = (
                str(headers.get("Content-Type", "") if headers is not None else "")
                .split(";")[0]
                .strip()
                .lower()
            )
            content_length_header = headers.get("Content-Length") if headers is not None else None
        except Exception:
            content_type = ""
            content_length_header = None

        if content_type and not (
            content_type.startswith("text/") or content_type == "application/xhtml+xml"
        ):
            return (
                f"fetch_url error: content-type {content_type!r} is not HTML/text. "
                "fetch_url handles only HTML pages; use a domain-specific tool for "
                "PDFs / JSON APIs / binary payloads."
            )

        if content_length_header is not None:
            try:
                declared = int(content_length_header)
            except ValueError:
                declared = 0
            if declared > self.max_response_bytes:
                return (
                    f"fetch_url error: response declares {declared:,} bytes, "
                    f"cap is {self.max_response_bytes:,}. Page too large to fetch."
                )

        final_url = getattr(response, "url", url)
        try:
            raw_bytes = response.read(self.max_response_bytes + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return f"fetch_url error: read failure for {url}: {exc}"
        finally:
            # Best-effort close; a double-close or already-GC'd
            # response isn't fatal for this tool.
            with contextlib.suppress(Exception):
                response.close()

        truncated_at_cap = len(raw_bytes) > self.max_response_bytes
        if truncated_at_cap:
            raw_bytes = raw_bytes[: self.max_response_bytes]

        # Decoding: take the charset from the content-type when
        # declared; fall back to UTF-8 with replacement so a mangled
        # byte doesn't kill the extraction.
        charset = "utf-8"
        if content_type:
            # headers sometimes carry "text/html; charset=UTF-8"; we
            # already split off the "text/html" prefix, so re-parse.
            try:
                ct_full = str(headers.get("Content-Type", "")) if headers is not None else ""
                for part in ct_full.split(";"):
                    if "charset=" in part:
                        charset = part.split("charset=", 1)[1].strip().strip('"').lower() or "utf-8"
                        break
            except Exception:
                charset = "utf-8"
        try:
            raw_text = raw_bytes.decode(charset, errors="replace")
        except LookupError:
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        if not extract_text:
            body = raw_text[: self.max_chars]
            return _format_output(
                final_url=final_url,
                title="",
                body=body,
                truncated_body=len(raw_text) > self.max_chars or truncated_at_cap,
                mode="raw html",
            )

        extractor = _TextExtractor()
        try:
            extractor.feed(raw_text)
            extractor.close()
        except Exception as exc:
            # html.parser is permissive but can fail on severely
            # malformed input. Fall back to raw (truncated) text so
            # the caller still gets SOMETHING to read.
            return _format_output(
                final_url=final_url,
                title="",
                body=raw_text[: self.max_chars],
                truncated_body=len(raw_text) > self.max_chars or truncated_at_cap,
                mode=f"raw (parser error: {type(exc).__name__})",
            )

        body = extractor.best_text()
        body_truncated = len(body) > self.max_chars or truncated_at_cap
        if len(body) > self.max_chars:
            body = body[: self.max_chars]
        return _format_output(
            final_url=final_url,
            title=extractor.title,
            body=body,
            truncated_body=body_truncated,
            mode="extracted text",
        )


def _format_output(
    *,
    final_url: str,
    title: str,
    body: str,
    truncated_body: bool,
    mode: str,
) -> str:
    """Compose the tool's return string. The header line gives the
    model provenance (URL / title / size) so it can cite correctly
    without re-quoting a long page."""
    header_bits = [f"url: {final_url}"]
    if title:
        header_bits.append(f"title: {title}")
    header_bits.append(f"mode: {mode}")
    header_bits.append(f"chars: {len(body)}")
    if truncated_body:
        header_bits.append("truncated: yes")
    header = "[" + " · ".join(header_bits) + "]"
    if truncated_body and not body.endswith("…[truncated]"):
        body = body.rstrip() + "\n…[truncated]"
    return f"{header}\n\n{body}"
