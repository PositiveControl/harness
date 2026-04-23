"""Tests for FetchUrlTool (harness-3zs).

The tool wraps stdlib urllib, so tests inject a fake `request_opener`
callable to avoid real network — same shape as urllib.request.urlopen:
takes `(request, timeout=...)` and returns an object with `.read()`,
`.headers`, `.url`, `.close()`.
"""

from __future__ import annotations

import io
import urllib.error
from dataclasses import dataclass, field
from email.message import Message

from harness.tools.fetch_url import FetchUrlTool

# ---------- fake urllib response ----------


def _headers(**kv: str) -> Message:
    """Build an email.message.Message with the supplied headers —
    urllib.response objects expose their headers via this interface."""
    m = Message()
    for key, value in kv.items():
        m[key.replace("_", "-")] = value
    return m


@dataclass
class _FakeResponse:
    body: bytes
    headers: Message
    url: str
    closed: bool = False

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            return self.body
        return self.body[:n]

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeOpener:
    """Captures the last call so tests can assert URL + timeout; returns
    a queued response or raises a queued exception."""

    response: _FakeResponse | None = None
    raise_exc: Exception | None = None
    calls: list[tuple[str, int]] = field(default_factory=list)

    def __call__(self, request: object, *, timeout: int) -> _FakeResponse:
        url = getattr(request, "full_url", None) or getattr(request, "url", "")
        self.calls.append((str(url), timeout))
        if self.raise_exc is not None:
            raise self.raise_exc
        assert self.response is not None, "no response queued"
        return self.response


# ---------- spec surface ----------


def test_spec_is_read_tier_and_high_noise() -> None:
    tool = FetchUrlTool()
    assert tool.spec.name == "fetch_url"
    assert tool.spec.tier == "read"
    assert tool.spec.high_noise is True
    required = tool.spec.parameters.get("required", [])
    assert "url" in required


# ---------- URL validation ----------


def test_rejects_plain_http() -> None:
    tool = FetchUrlTool(request_opener=_FakeOpener())
    out = tool.call(url="http://example.com/thing")
    assert "only https:// is allowed" in out


def test_rejects_url_without_host() -> None:
    tool = FetchUrlTool(request_opener=_FakeOpener())
    out = tool.call(url="https:///no-host")
    assert "has no host" in out


def test_rejects_nonpositive_timeout() -> None:
    tool = FetchUrlTool(request_opener=_FakeOpener())
    out = tool.call(url="https://example.com/", timeout_seconds=0)
    assert "timeout_seconds must be > 0" in out


# ---------- host allowlist (harness-xbk.3) ----------


def test_allowed_hosts_none_means_unrestricted() -> None:
    """Default (allowed_hosts=None): any https host reaches the opener.
    Back-compat with every caller that existed before atc."""
    opener = _FakeOpener(
        response=_FakeResponse(
            b"<html><body>ok</body></html>",
            _headers(Content_Type="text/html"),
            "https://example.com/",
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    tool.call(url="https://example.com/")
    assert opener.calls, "opener should have been invoked"


def test_allowlist_rejects_non_matching_host_before_network() -> None:
    """allowed_hosts restricts to exact netloc membership. Refusal
    happens pre-network so no opener call is made — important so a
    misdirected URL never exfiltrates query params to an off-list
    server."""
    opener = _FakeOpener()  # no response queued — would fail if called
    tool = FetchUrlTool(
        request_opener=opener,
        allowed_hosts=frozenset({"aviationweather.gov"}),
    )
    out = tool.call(url="https://example.com/weather")
    assert "not in this character's fetch allowlist" in out
    assert "example.com" in out
    assert opener.calls == []


def test_allowlist_accepts_member_host() -> None:
    opener = _FakeOpener(
        response=_FakeResponse(
            b"<html><body>METAR KPAO</body></html>",
            _headers(Content_Type="text/html"),
            "https://aviationweather.gov/metar",
        )
    )
    tool = FetchUrlTool(
        request_opener=opener,
        allowed_hosts=frozenset({"aviationweather.gov"}),
    )
    out = tool.call(url="https://aviationweather.gov/metar")
    assert "METAR KPAO" in out
    assert opener.calls, "allowed host should reach the opener"


def test_allowlist_is_case_insensitive_on_host() -> None:
    """DNS is case-insensitive; a user typing AviationWeather.Gov should
    not bypass or be rejected by the allowlist for casing reasons."""
    opener = _FakeOpener(
        response=_FakeResponse(
            b"<html><body>ok</body></html>",
            _headers(Content_Type="text/html"),
            "https://AviationWeather.Gov/metar",
        )
    )
    tool = FetchUrlTool(
        request_opener=opener,
        allowed_hosts=frozenset({"aviationweather.gov"}),
    )
    out = tool.call(url="https://AviationWeather.Gov/metar")
    assert "not in" not in out
    assert opener.calls


def test_allowlist_ignores_port_and_userinfo() -> None:
    """parsed.hostname strips user:pass@ and :port so the allowlist
    check keys on the bare host — a user:password or :port injection
    in the URL can't mask the netloc."""
    opener = _FakeOpener(
        response=_FakeResponse(
            b"<html><body>ok</body></html>",
            _headers(Content_Type="text/html"),
            "https://user:pass@aviationweather.gov:443/metar",
        )
    )
    tool = FetchUrlTool(
        request_opener=opener,
        allowed_hosts=frozenset({"aviationweather.gov"}),
    )
    out = tool.call(url="https://user:pass@aviationweather.gov:443/metar")
    assert "not in" not in out
    assert opener.calls


def test_clamps_timeout_to_cap() -> None:
    opener = _FakeOpener(
        response=_FakeResponse(
            b"<html><body>ok</body></html>",
            _headers(Content_Type="text/html"),
            "https://example.com/",
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    tool.call(url="https://example.com/", timeout_seconds=9999)
    assert opener.calls[0][1] == 30  # _MAX_TIMEOUT_S


# ---------- content-type + size gates ----------


def test_rejects_non_html_content_type() -> None:
    opener = _FakeOpener(
        response=_FakeResponse(
            b"{}", _headers(Content_Type="application/json"), "https://api.example.com/"
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://api.example.com/")
    assert "not HTML/text" in out
    assert "application/json" in out


def test_rejects_oversized_declared_length() -> None:
    huge = str(10 * 1024 * 1024)  # 10 MB
    opener = _FakeOpener(
        response=_FakeResponse(
            b"x",
            _headers(Content_Type="text/html", Content_Length=huge),
            "https://example.com/huge",
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://example.com/huge")
    assert "Page too large to fetch" in out


def test_reads_up_to_byte_cap_when_length_undeclared() -> None:
    """Endpoints without Content-Length can still dump 5 GB at us.
    The tool must stream-bound the read rather than trust the header."""
    big_body = ("<html><body>" + ("A" * 10_000) + "</body></html>").encode()
    opener = _FakeOpener(
        response=_FakeResponse(
            big_body, _headers(Content_Type="text/html"), "https://example.com/"
        ),
    )
    tool = FetchUrlTool(
        request_opener=opener,
        max_response_bytes=1_024,
        max_chars=2_048,
    )
    out = tool.call(url="https://example.com/")
    assert "truncated: yes" in out


# ---------- extraction ----------


_SAMPLE_HTML = """
<!DOCTYPE html>
<html>
  <head>
    <title>Sample Page · Example</title>
    <style>.nav { color: red }</style>
    <script>alert(1);</script>
  </head>
  <body>
    <nav><a href="/home">Home</a><a href="/about">About</a></nav>
    <header>Site header chrome</header>
    <article>
      <h1>Main Title</h1>
      <p>First paragraph with identifier <code>BeadsAdapter.get_focus</code>.</p>
      <p>Second paragraph.</p>
    </article>
    <aside>Ad goes here</aside>
    <footer>© 2026</footer>
  </body>
</html>
"""


def test_extracts_title_and_article_body() -> None:
    opener = _FakeOpener(
        response=_FakeResponse(
            _SAMPLE_HTML.encode(),
            _headers(Content_Type="text/html; charset=utf-8"),
            "https://example.com/post",
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://example.com/post")
    # Header carries provenance.
    assert "url: https://example.com/post" in out
    assert "title: Sample Page · Example" in out
    # Article body survives.
    assert "Main Title" in out
    assert "BeadsAdapter.get_focus" in out
    assert "First paragraph" in out
    assert "Second paragraph" in out
    # Script / style / nav / aside / footer / header chrome all dropped.
    assert "alert(1)" not in out
    assert ".nav" not in out  # stylesheet content
    assert "Home" not in out  # nav link text
    assert "About" not in out
    assert "Ad goes here" not in out
    assert "© 2026" not in out
    assert "Site header chrome" not in out


def test_falls_back_to_body_when_no_article() -> None:
    html = "<html><body><p>Just a paragraph.</p><p>Another one.</p></body></html>"
    opener = _FakeOpener(
        response=_FakeResponse(
            html.encode(), _headers(Content_Type="text/html"), "https://example.com/"
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://example.com/")
    assert "Just a paragraph" in out
    assert "Another one" in out


def test_extract_false_returns_raw_html() -> None:
    html = "<html><body><article>x</article></body></html>"
    opener = _FakeOpener(
        response=_FakeResponse(
            html.encode(), _headers(Content_Type="text/html"), "https://example.com/"
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://example.com/", extract_text=False)
    assert "<article>" in out
    assert "mode: raw html" in out


def test_truncates_long_body_with_marker() -> None:
    long_body = "<html><body><article>" + ("word " * 5_000) + "</article></body></html>"
    opener = _FakeOpener(
        response=_FakeResponse(
            long_body.encode(), _headers(Content_Type="text/html"), "https://example.com/"
        )
    )
    tool = FetchUrlTool(request_opener=opener, max_chars=200)
    out = tool.call(url="https://example.com/")
    assert "truncated: yes" in out
    assert "…[truncated]" in out


def test_decodes_declared_charset() -> None:
    body = "<html><body><article>£ sign</article></body></html>".encode("latin-1")
    opener = _FakeOpener(
        response=_FakeResponse(
            body,
            _headers(Content_Type="text/html; charset=latin-1"),
            "https://example.com/",
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://example.com/")
    assert "£ sign" in out


def test_final_url_from_response_reflects_redirects() -> None:
    """urllib returns the post-redirect URL on .url — pass it through
    to the header so the model can cite correctly."""
    opener = _FakeOpener(
        response=_FakeResponse(
            b"<html><body><p>ok</p></body></html>",
            _headers(Content_Type="text/html"),
            "https://www.example.com/final",  # different from request URL
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://example.com/start")
    assert "url: https://www.example.com/final" in out


# ---------- error paths ----------


def test_surfaces_http_error() -> None:
    # HTTPError's `hdrs` param wants a Message; pass an empty one to
    # avoid a mypy complaint without leaning on a type: ignore.
    opener = _FakeOpener(
        raise_exc=urllib.error.HTTPError(
            "https://example.com/404",
            404,
            "Not Found",
            Message(),
            io.BytesIO(b""),
        )
    )
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://example.com/404")
    assert "HTTP 404" in out


def test_surfaces_network_error() -> None:
    opener = _FakeOpener(raise_exc=urllib.error.URLError("dns fail"))
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://no-such-host.example/")
    assert "network failure" in out
    assert "dns fail" in out


def test_surfaces_timeout() -> None:
    opener = _FakeOpener(raise_exc=TimeoutError("deadline exceeded"))
    tool = FetchUrlTool(request_opener=opener)
    out = tool.call(url="https://example.com/slow")
    assert "network failure" in out
    assert "deadline exceeded" in out


# ---------- profile membership ----------


def test_fetch_url_is_in_research_and_coding() -> None:
    from harness.tools.profiles import TOOL_PROFILES

    assert "fetch_url" in TOOL_PROFILES["research"]
    assert "fetch_url" in TOOL_PROFILES["coding"]
    assert "fetch_url" in TOOL_PROFILES["full"]


def test_fetch_url_not_in_minimal_or_core() -> None:
    from harness.tools.profiles import TOOL_PROFILES

    assert "fetch_url" not in TOOL_PROFILES["minimal"]
    assert "fetch_url" not in TOOL_PROFILES["core"]
