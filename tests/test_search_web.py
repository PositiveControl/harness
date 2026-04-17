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
