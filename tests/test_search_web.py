"""Tests for SearchWebTool. We don't hit the live network — instead we
monkeypatch urllib.request.urlopen with a canned HTML response shaped
like DuckDuckGo's html endpoint. Keeps the suite fast + offline-safe."""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any

import pytest

from harness.tools.search_web import SearchWebTool, _strip_tags, _unwrap_ddg_redirect

# Minimal canned response — two results in DDG's html format. Pure ASCII
# so the bytes literal stays valid.
_DDG_WRAP = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fml-explore%2Fmlx"
_CANNED = (
    b"<html><body>"
    b'<div class="result">'
    b'<a class="result__a" href="https://ml-explore.github.io/mlx/">MLX Documentation</a>'
    b'<a class="result__snippet">Apple array framework for ML on Apple silicon.</a>'
    b"</div>"
    b'<div class="result">'
    b'<a class="result__a" href="' + _DDG_WRAP.encode() + b'">mlx on GitHub</a>'
    b'<a class="result__snippet">MLX repo -- models &amp; utilities.</a>'
    b"</div>"
    b"</body></html>"
)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _fake_urlopen(_req: Any, timeout: int = 10) -> _FakeResponse:
    return _FakeResponse(_CANNED)


def test_returns_formatted_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    out = SearchWebTool().call(query="MLX")
    assert "1." in out
    assert "MLX Documentation" in out
    assert "https://ml-explore.github.io/mlx/" in out
    assert "Apple array framework" in out
    # Second result — the DDG redirect should be unwrapped.
    assert "2." in out
    assert "https://github.com/ml-explore/mlx" in out


def test_respects_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    out = SearchWebTool().call(query="MLX", max_results=1)
    assert "1. MLX Documentation" in out
    assert "2." not in out


def test_empty_query_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SearchWebTool().call(query="")


def test_zero_max_results_rejected() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        SearchWebTool().call(query="hi", max_results=0)


# --- harness-q7kn: snippet-thin results → fetch_url hint -----------------


_CANNED_NO_SNIPPETS = (
    b"<html><body>"
    b'<div class="result">'
    b'<a class="result__a" href="https://example.com/a">Title A</a>'
    b"</div>"
    b'<div class="result">'
    b'<a class="result__a" href="https://example.com/b">Title B</a>'
    b"</div>"
    b"</body></html>"
)


def test_results_with_no_snippets_get_fetch_url_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mark's q7kn repro: search_web returns titles + URLs but no
    snippets. Without a hint, the agent fabricated bracket placeholders
    ('[Check the link for the latest temperature]') as if it had data.
    The tool now appends an explicit fetch_url chain hint when the
    result set carries titles but zero snippet content."""

    def _no_snippet_response(_req: Any, timeout: int = 10) -> _FakeResponse:
        return _FakeResponse(_CANNED_NO_SNIPPETS)

    monkeypatch.setattr(urllib.request, "urlopen", _no_snippet_response)
    out = SearchWebTool().call(query="anything")
    assert "Title A" in out
    assert "Title B" in out
    assert "Next step: call `fetch_url" in out


def test_results_with_any_snippet_do_not_get_fetch_url_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hint must NOT appear when at least one result has a snippet
    — those are content-rich and the agent has enough to summarize."""
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    out = SearchWebTool().call(query="MLX")
    assert "Next step: call `fetch_url" not in out


def test_network_failure_returns_graceful_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_req: Any, timeout: int = 10) -> _FakeResponse:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    out = SearchWebTool().call(query="anything")
    assert "search failed" in out


def test_empty_body_returns_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def _empty(_req: Any, timeout: int = 10) -> _FakeResponse:
        return _FakeResponse(b"<html></html>")

    monkeypatch.setattr(urllib.request, "urlopen", _empty)
    out = SearchWebTool().call(query="zzz")
    assert "no results" in out


def test_strip_tags_handles_entities() -> None:
    assert _strip_tags("<b>Hello &amp; World</b>") == "Hello & World"
    assert _strip_tags("  multi   space  ") == "multi space"


def test_unwrap_ddg_redirect_extracts_target() -> None:
    url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage"
    assert _unwrap_ddg_redirect(url) == "https://example.com/page"


def test_unwrap_ddg_leaves_direct_urls_alone() -> None:
    assert _unwrap_ddg_redirect("https://example.com/") == "https://example.com/"


def test_spec_is_read_tier() -> None:
    assert SearchWebTool().spec.tier == "read"


@pytest.fixture
def _offline_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safety fixture: if a test forgets to stub urlopen, blow up loudly
    instead of silently hitting the network from CI."""

    def _forbid(*_: Any, **__: Any) -> Any:
        raise RuntimeError("test tried to call urllib.request.urlopen without stubbing it")

    monkeypatch.setattr(urllib.request, "urlopen", _forbid)


# ---------- allowlist rerank (scholar pairing with fetch_url) ----------


def _mixed_results_body() -> bytes:
    """Canned DDG-shape HTML with 4 results: external, allowlisted,
    external, allowlisted. Confirms the rerank surfaces allowlisted
    sources even when they're past the first slot."""
    return (
        b"<html><body>"
        # Position 1: external (crypto.stackexchange.com)
        b'<div class="result">'
        b'<a class="result__a" href="https://crypto.stackexchange.com/q/77">'
        b"DH in TLS handshake</a>"
        b'<a class="result__snippet">Q&amp;A about DH alternatives.</a>'
        b"</div>"
        # Position 2: allowlisted (arxiv.org)
        b'<div class="result">'
        b'<a class="result__a" href="https://arxiv.org/abs/2401.12345">'
        b"PQDH preprint</a>"
        b'<a class="result__snippet">Quantum-safe replacement for DH.</a>'
        b"</div>"
        # Position 3: external (researchgate.net)
        b'<div class="result">'
        b'<a class="result__a" href="https://www.researchgate.net/publication/X">'
        b"Alternative DH Protocol</a>"
        b'<a class="result__snippet">Floating-point variant.</a>'
        b"</div>"
        # Position 4: allowlisted (scholar.google.com)
        b'<div class="result">'
        b'<a class="result__a" href="https://scholar.google.com/scholar?q=DH">'
        b"Scholar results for Diffie-Hellman</a>"
        b'<a class="result__snippet">Citation index.</a>'
        b"</div>"
        b"</body></html>"
    )


def _stub_urlopen_for(body: bytes) -> Any:
    """Build a urlopen stub returning the given body. Inlines the
    FakeResponse pattern used elsewhere in this file."""

    def _stub(_req: Any, timeout: int = 10) -> _FakeResponse:
        return _FakeResponse(body)

    return _stub


def test_allowlist_floats_allowlisted_results_to_top(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two allowlisted hits (arxiv.org pos2, scholar.google.com
    pos4) become positions 1 and 2; the two external hits become
    positions 3 and 4. Stable-sort preserves intra-group ranking."""
    monkeypatch.setattr(urllib.request, "urlopen", _stub_urlopen_for(_mixed_results_body()))
    tool = SearchWebTool(allowed_hosts=frozenset({"arxiv.org", "scholar.google.com"}))
    out = tool.call(query="DH alternatives")
    lines = out.splitlines()
    # Allowlisted at positions 1 and 2.
    assert lines[0].startswith("1. [allowlisted] PQDH preprint")
    # Position 2 is the snippet line for result 1 (4-space indent).
    # Find the next numbered entry.
    numbered = [line for line in lines if line[:2] in ("1.", "2.", "3.", "4.")]
    assert numbered[0].startswith("1. [allowlisted] PQDH preprint")
    assert numbered[1].startswith("2. [allowlisted] Scholar results for Diffie-Hellman")
    assert numbered[2].startswith("3. [external] DH in TLS handshake")
    assert numbered[3].startswith("4. [external] Alternative DH Protocol")


def test_allowlist_subdomain_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """`arxiv.org` in the allowlist matches `www.arxiv.org` results too —
    DDG often returns the `www.`-prefixed host."""
    body = (
        b"<html><body>"
        b'<div class="result">'
        b'<a class="result__a" href="https://www.arxiv.org/abs/2401">paper</a>'
        b'<a class="result__snippet">Preprint.</a>'
        b"</div></body></html>"
    )
    monkeypatch.setattr(urllib.request, "urlopen", _stub_urlopen_for(body))
    tool = SearchWebTool(allowed_hosts=frozenset({"arxiv.org"}))
    out = tool.call(query="anything")
    assert "[allowlisted] paper" in out


def test_no_allowlist_preserves_unmarked_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `allowed_hosts` is None (default), no markers + no rerank
    — the open-web behavior for general-purpose characters."""
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    out = SearchWebTool().call(query="MLX")
    assert "[allowlisted]" not in out
    assert "[external]" not in out
    # Original order: ml-explore docs first, github wrap second.
    lines = [line for line in out.splitlines() if line[:2] in ("1.", "2.")]
    assert lines[0].startswith("1. MLX Documentation")
    assert lines[1].startswith("2. mlx on GitHub")


# ---------- default_site_filter (scholar-as-default-search) ----------


def _capture_urlopen_query() -> tuple[list[str], Any]:
    """Build a urlopen stub that records the urlencoded query body it
    was invoked with. Lets tests assert the actual outgoing query
    string after the default_site_filter prepend."""
    captured: list[str] = []

    def _stub(req: Any, timeout: int = 10) -> _FakeResponse:
        # urllib.request.Request.data is the urlencoded form body.
        raw = req.data if isinstance(req.data, bytes) else req.data.encode()
        parsed = urllib.parse.parse_qs(raw.decode())
        captured.extend(parsed.get("q", []))
        # Return an empty results body so the test exits cleanly.
        return _FakeResponse(b"<html></html>")

    return captured, _stub


def test_default_site_filter_prepends_when_query_has_no_site_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """airton_f's spec: scholar.google.com is the default scope.
    A bare query gets `site:scholar.google.com ` prepended before
    DDG sees it."""
    captured, stub = _capture_urlopen_query()
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    tool = SearchWebTool(default_site_filter="scholar.google.com")
    tool.call(query="diffie-hellman alternatives")
    assert captured == ["site:scholar.google.com diffie-hellman alternatives"]


def test_default_site_filter_skipped_when_query_already_has_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override path: when the model wants to search elsewhere
    (arxiv, wikipedia, etc.), it includes `site:<other>` in its query
    and the default-filter prepend is skipped. The model's named
    scope wins."""
    captured, stub = _capture_urlopen_query()
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    tool = SearchWebTool(default_site_filter="scholar.google.com")
    tool.call(query="key exchange site:arxiv.org")
    # Default scholar filter NOT prepended; the arxiv site: stays.
    assert captured == ["key exchange site:arxiv.org"]


def test_default_site_filter_case_insensitive_site_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uppercase / mixed-case `SITE:` and `Site:` are detected as the
    override path. DDG normalizes case server-side; the tool should
    too rather than double-prepending."""
    captured, stub = _capture_urlopen_query()
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    tool = SearchWebTool(default_site_filter="scholar.google.com")
    tool.call(query="topic SITE:arxiv.org")
    assert captured == ["topic SITE:arxiv.org"]


def test_default_site_filter_word_boundary_not_substring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`website:foo` (no boundary before `site`) is NOT the override
    path — that's a search FOR the word 'website:'. Default scope
    still applies."""
    captured, stub = _capture_urlopen_query()
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    tool = SearchWebTool(default_site_filter="scholar.google.com")
    tool.call(query="my favorite website: example.com")
    assert captured == ["site:scholar.google.com my favorite website: example.com"]


def test_default_site_filter_none_leaves_query_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the character doesn't declare a default scope (the open-web
    default), the query passes through unmodified."""
    captured, stub = _capture_urlopen_query()
    monkeypatch.setattr(urllib.request, "urlopen", stub)
    tool = SearchWebTool()
    tool.call(query="anything")
    assert captured == ["anything"]


def test_default_site_filter_reflected_in_tool_description() -> None:
    """The tool spec's description must surface the default scope so
    the model knows what's happening. Without the note, the model
    might write its own `site:` clause assuming the default is the
    open web."""
    tool = SearchWebTool(default_site_filter="scholar.google.com")
    assert "site:scholar.google.com" in tool.spec.description
    assert "default search scope" in tool.spec.description.lower()


def test_default_site_filter_absent_in_description_when_none() -> None:
    """The scope note only appears when default_site_filter is set;
    non-scholar characters get the original unscoped description."""
    tool = SearchWebTool()
    assert "site:" not in tool.spec.description


def test_allowlist_surfaces_late_hits_within_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5-result fetch where the only allowlisted hit is at position
    4 of the raw DDG output: rerank floats it to position 1 in the
    sliced output. Without rerank-then-slice the agent would never
    see it under max_results=3."""
    body = (
        b"<html><body>"
        b'<div class="result"><a class="result__a" href="https://example.com/a">A</a>'
        b'<a class="result__snippet">snip-a</a></div>'
        b'<div class="result"><a class="result__a" href="https://example.com/b">B</a>'
        b'<a class="result__snippet">snip-b</a></div>'
        b'<div class="result"><a class="result__a" href="https://example.com/c">C</a>'
        b'<a class="result__snippet">snip-c</a></div>'
        b'<div class="result"><a class="result__a" href="https://arxiv.org/abs/X">D</a>'
        b'<a class="result__snippet">snip-d</a></div>'
        b'<div class="result"><a class="result__a" href="https://example.com/e">E</a>'
        b'<a class="result__snippet">snip-e</a></div>'
        b"</body></html>"
    )
    monkeypatch.setattr(urllib.request, "urlopen", _stub_urlopen_for(body))
    tool = SearchWebTool(allowed_hosts=frozenset({"arxiv.org"}))
    out = tool.call(query="something", max_results=3)
    # Position 1 is the allowlisted hit (originally 4th).
    assert out.startswith("1. [allowlisted] D — https://arxiv.org/abs/X")
    # Only one allowlisted; positions 2 and 3 are externals (A, B).
    assert "2. [external] A" in out
    assert "3. [external] B" in out
    # E shouldn't appear (sliced off).
    assert "snip-e" not in out
