"""Tests for SearchScholarTool. Mocks both Semantic Scholar + OpenAlex
HTTP endpoints with canned JSON shaped like their real APIs. Keeps the
suite fast + offline-safe."""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from harness.tools.search_scholar import (
    SearchScholarTool,
    _dedupe_key,
    _format_authors,
    _merge_papers,
    _normalize_title,
    _Paper,
    _parse_openalex,
    _parse_s2,
    _reconstruct_openalex_abstract,
)

# ---------- canned API responses ----------


def _s2_response(papers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"total": len(papers), "offset": 0, "data": papers}


def _openalex_response(works: list[dict[str, Any]]) -> dict[str, Any]:
    return {"meta": {"count": len(works)}, "results": works}


_IJEPA_S2 = {
    "paperId": "abc123",
    "title": "I-JEPA: Self-Supervised Learning with a Joint-Embedding Predictive Architecture",
    "abstract": "We introduce I-JEPA, the Image-based Joint-Embedding Predictive Architecture, "
    "for non-generative self-supervised learning. ",
    "year": 2023,
    "authors": [
        {"authorId": "1", "name": "Mahmoud Assran"},
        {"authorId": "2", "name": "Quentin Duval"},
        {"authorId": "3", "name": "Ishan Misra"},
        {"authorId": "4", "name": "Piotr Bojanowski"},
        {"authorId": "5", "name": "Yann LeCun"},
    ],
    "citationCount": 412,
    "externalIds": {"DOI": "10.1109/CVPR52729.2023.01499", "ArXiv": "2301.08243"},
    "openAccessPdf": {"url": "https://arxiv.org/pdf/2301.08243.pdf"},
}


_IJEPA_OPENALEX = {
    "id": "https://openalex.org/W123",
    "doi": "https://doi.org/10.1109/CVPR52729.2023.01499",
    "title": "I-JEPA: Self-Supervised Learning with a Joint-Embedding Predictive Architecture",
    "publication_year": 2023,
    "cited_by_count": 420,  # slightly higher than S2 — OpenAlex tracks more sources
    "authorships": [
        {"author": {"display_name": "Mahmoud Assran"}},
        {"author": {"display_name": "Quentin Duval"}},
    ],
    "abstract_inverted_index": {
        "We": [0],
        "introduce": [1],
        "I-JEPA": [2],
    },
    "open_access": {"is_oa": True, "oa_url": "https://arxiv.org/pdf/2301.08243.pdf"},
    "ids": {"doi": "10.1109/CVPR52729.2023.01499", "openalex": "W123"},
}


_VJEPA_OPENALEX = {
    "id": "https://openalex.org/W456",
    "doi": "https://doi.org/10.x/vjepa",
    "title": "V-JEPA: Video Joint-Embedding Predictive Architecture",
    "publication_year": 2024,
    "cited_by_count": 87,
    "authorships": [{"author": {"display_name": "Adrien Bardes"}}],
    "abstract_inverted_index": {"V-JEPA": [0], "extends": [1], "I-JEPA": [2]},
    "open_access": {"is_oa": False, "oa_url": None},
    "ids": {"doi": "10.x/vjepa"},
}


# ---------- helpers ----------


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _stub_openers(
    *, s2_payload: dict[str, Any] | None = None, openalex_payload: dict[str, Any] | None = None
) -> Any:
    """Return a urlopen-compatible callable that branches on URL host.
    Either payload can be None to simulate an empty response from
    that API; raise OSError-shaped to simulate a network failure."""

    def _opener(req: Any, timeout: int = 10) -> _FakeResponse:
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "semanticscholar.org" in url:
            if s2_payload is None:
                return _FakeResponse(b"{}")
            return _FakeResponse(json.dumps(s2_payload).encode())
        if "openalex.org" in url:
            if openalex_payload is None:
                return _FakeResponse(b"{}")
            return _FakeResponse(json.dumps(openalex_payload).encode())
        raise RuntimeError(f"unexpected host in test: {url}")

    return _opener


# ---------- _normalize_title ----------


def test_normalize_title_strips_punct_and_lowercases() -> None:
    assert _normalize_title("I-JEPA: A Title!") == "i jepa a title"
    assert _normalize_title("  Multi  Space\t Title  ") == "multi space title"


# ---------- _dedupe_key ----------


def test_dedupe_key_prefers_doi() -> None:
    p = _Paper(
        title="T",
        authors=(),
        year=None,
        citation_count=None,
        abstract="",
        doi="10.x/y",
        arxiv_id="2301.08243",
        open_access_pdf=None,
        source_url="",
    )
    assert _dedupe_key(p) == "doi:10.x/y"


def test_dedupe_key_falls_back_to_arxiv() -> None:
    p = _Paper(
        title="T",
        authors=(),
        year=None,
        citation_count=None,
        abstract="",
        doi=None,
        arxiv_id="2301.08243",
        open_access_pdf=None,
        source_url="",
    )
    assert _dedupe_key(p) == "arxiv:2301.08243"


def test_dedupe_key_falls_back_to_normalized_title() -> None:
    p = _Paper(
        title="A Strange Title!",
        authors=(),
        year=None,
        citation_count=None,
        abstract="",
        doi=None,
        arxiv_id=None,
        open_access_pdf=None,
        source_url="",
    )
    assert _dedupe_key(p) == "title:a strange title"


# ---------- _reconstruct_openalex_abstract ----------


def test_reconstruct_openalex_abstract_orders_by_position() -> None:
    inverted = {"world": [1], "hello": [0]}
    assert _reconstruct_openalex_abstract(inverted) == "hello world"


def test_reconstruct_openalex_abstract_handles_empty_input() -> None:
    assert _reconstruct_openalex_abstract(None) == ""
    assert _reconstruct_openalex_abstract({}) == ""


# ---------- _format_authors ----------


def test_format_authors_inlines_three_or_fewer() -> None:
    assert _format_authors(("A", "B", "C")) == "A, B, C"
    assert _format_authors(("A",)) == "A"
    assert _format_authors(()) == "(unknown authors)"


def test_format_authors_collapses_to_et_al_at_four_plus() -> None:
    assert _format_authors(("A", "B", "C", "D")) == "A et al"


# ---------- _parse_s2 ----------


def test_parse_s2_extracts_paper_fields() -> None:
    papers = _parse_s2(_s2_response([_IJEPA_S2]))
    assert len(papers) == 1
    p = papers[0]
    assert p.title.startswith("I-JEPA")
    assert p.year == 2023
    assert p.citation_count == 412
    assert p.doi == "10.1109/CVPR52729.2023.01499"
    assert p.arxiv_id == "2301.08243"
    assert p.sources == frozenset({"s2"})
    assert p.source_url == "https://doi.org/10.1109/CVPR52729.2023.01499"
    assert "Yann LeCun" in p.authors


def test_parse_s2_tolerates_missing_fields() -> None:
    minimal = {"title": "Minimal", "paperId": "x1"}
    papers = _parse_s2(_s2_response([minimal]))
    assert len(papers) == 1
    assert papers[0].year is None
    assert papers[0].citation_count is None
    assert papers[0].doi is None
    assert papers[0].source_url.startswith("https://www.semanticscholar.org/")


def test_parse_s2_skips_rows_without_title() -> None:
    papers = _parse_s2(_s2_response([{"paperId": "x"}, {"title": ""}, {"title": "Real"}]))
    assert len(papers) == 1
    assert papers[0].title == "Real"


def test_parse_s2_returns_empty_on_bad_payload() -> None:
    assert _parse_s2(None) == []
    assert _parse_s2({}) == []
    assert _parse_s2({"data": "not a list"}) == []


# ---------- _parse_openalex ----------


def test_parse_openalex_extracts_fields_and_reconstructs_abstract() -> None:
    papers = _parse_openalex(_openalex_response([_IJEPA_OPENALEX]))
    assert len(papers) == 1
    p = papers[0]
    assert p.title.startswith("I-JEPA")
    assert p.year == 2023
    assert p.citation_count == 420
    assert p.doi == "10.1109/CVPR52729.2023.01499"
    assert p.abstract == "We introduce I-JEPA"
    assert p.sources == frozenset({"openalex"})


def test_parse_openalex_strips_doi_url_prefix() -> None:
    """OpenAlex returns DOI as `https://doi.org/10.x/y`; the bare
    `10.x/y` form is what S2 uses and what _dedupe_key expects."""
    papers = _parse_openalex(_openalex_response([_IJEPA_OPENALEX]))
    assert papers[0].doi == "10.1109/CVPR52729.2023.01499"
    assert not papers[0].doi.startswith("https://")


def test_parse_openalex_returns_empty_on_bad_payload() -> None:
    assert _parse_openalex(None) == []
    assert _parse_openalex({"results": "not a list"}) == []


# ---------- _merge_papers ----------


def test_merge_dedupes_by_doi_and_records_both_sources() -> None:
    s2 = _parse_s2(_s2_response([_IJEPA_S2]))
    oa = _parse_openalex(_openalex_response([_IJEPA_OPENALEX]))
    merged = _merge_papers(s2, oa, max_results=10)
    # Same paper in both APIs → 1 result with sources from both.
    assert len(merged) == 1
    assert merged[0].sources == frozenset({"s2", "openalex"})
    # S2's fields win on conflict (S2 had citation_count=412; OA had 420).
    assert merged[0].citation_count == 412


def test_merge_keeps_unique_papers_from_each_api() -> None:
    s2 = _parse_s2(_s2_response([_IJEPA_S2]))
    oa = _parse_openalex(_openalex_response([_VJEPA_OPENALEX]))
    merged = _merge_papers(s2, oa, max_results=10)
    assert len(merged) == 2
    titles = {p.title for p in merged}
    assert any("I-JEPA" in t for t in titles)
    assert any("V-JEPA" in t for t in titles)


def test_merge_ranks_both_api_papers_first() -> None:
    """Papers found by BOTH APIs are stronger relevance signals than
    single-API hits — they float to the top regardless of citation
    count."""
    # V-JEPA has 87 citations (only OpenAlex); I-JEPA has 412 (in both).
    # I-JEPA should rank above V-JEPA via the sources tier.
    s2 = _parse_s2(_s2_response([_IJEPA_S2]))
    oa = _parse_openalex(_openalex_response([_IJEPA_OPENALEX, _VJEPA_OPENALEX]))
    merged = _merge_papers(s2, oa, max_results=10)
    assert "I-JEPA" in merged[0].title  # Both-API hit first
    assert "V-JEPA" in merged[1].title


def test_merge_ranks_by_citation_count_within_tier() -> None:
    """Within the same source-tier, higher citation count wins."""
    paper_a = _Paper(
        title="A",
        authors=(),
        year=2020,
        citation_count=100,
        abstract="",
        doi="a/1",
        arxiv_id=None,
        open_access_pdf=None,
        source_url="",
        sources=frozenset({"s2"}),
    )
    paper_b = _Paper(
        title="B",
        authors=(),
        year=2020,
        citation_count=500,
        abstract="",
        doi="b/1",
        arxiv_id=None,
        open_access_pdf=None,
        source_url="",
        sources=frozenset({"s2"}),
    )
    merged = _merge_papers([paper_a, paper_b], [], max_results=10)
    assert merged[0].title == "B"


def test_merge_respects_max_results() -> None:
    papers = [
        _Paper(
            title=f"Paper {i}",
            authors=(),
            year=2020,
            citation_count=i,
            abstract="",
            doi=f"x/{i}",
            arxiv_id=None,
            open_access_pdf=None,
            source_url="",
            sources=frozenset({"s2"}),
        )
        for i in range(10)
    ]
    merged = _merge_papers(papers, [], max_results=3)
    assert len(merged) == 3


# ---------- SearchScholarTool.call (end-to-end with mocked HTTP) ----------


def test_call_fetches_both_apis_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: both APIs return I-JEPA + V-JEPA, merge dedupes,
    output contains the title, author, year, citation count, source
    badge, and URL."""
    opener = _stub_openers(
        s2_payload=_s2_response([_IJEPA_S2]),
        openalex_payload=_openalex_response([_IJEPA_OPENALEX, _VJEPA_OPENALEX]),
    )
    tool = SearchScholarTool(request_opener=opener)
    out = tool.call(query="JEPA")
    # Both papers appear; I-JEPA first (cross-validated).
    assert "I-JEPA" in out
    assert "V-JEPA" in out
    assert "[2023]" in out
    assert "Mahmoud Assran" in out
    assert "cited 412" in out
    assert "[openalex+s2]" in out or "[s2+openalex]" in out  # alphabetical join
    # Source URL present.
    assert "doi.org/10.1109/CVPR52729.2023.01499" in out


def test_call_handles_one_api_failure_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Semantic Scholar errors out, OpenAlex results still flow."""

    def _opener(req: Any, timeout: int = 10) -> _FakeResponse:
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "semanticscholar.org" in url:
            raise urllib.error.URLError("S2 down")
        return _FakeResponse(json.dumps(_openalex_response([_IJEPA_OPENALEX])).encode())

    tool = SearchScholarTool(request_opener=_opener)
    out = tool.call(query="JEPA")
    assert "I-JEPA" in out
    assert "[openalex]" in out
    assert "[s2]" not in out


def test_call_returns_no_results_message_when_both_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = _stub_openers(s2_payload=_s2_response([]), openalex_payload=_openalex_response([]))
    tool = SearchScholarTool(request_opener=opener)
    out = tool.call(query="zzzz nonsense xyz")
    assert "no scholar results" in out
    assert "zzzz" in out


def test_call_respects_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_results caps the final output, not the per-API request
    (each API still asks for the same count; merge truncates)."""
    extra_papers = [
        {
            "paperId": f"id{i}",
            "title": f"Paper {i}",
            "year": 2020,
            "citationCount": 100 - i,  # descending so order is predictable
            "externalIds": {"DOI": f"x/{i}"},
        }
        for i in range(10)
    ]
    opener = _stub_openers(
        s2_payload=_s2_response(extra_papers),
        openalex_payload=_openalex_response([]),
    )
    tool = SearchScholarTool(request_opener=opener)
    out = tool.call(query="anything", max_results=3)
    assert out.startswith("3 result(s) for")
    assert "Paper 0" in out
    assert "Paper 2" in out
    assert "Paper 5" not in out


def test_call_rejects_empty_query() -> None:
    tool = SearchScholarTool()
    with pytest.raises(ValueError, match="must not be empty"):
        tool.call(query="")


def test_call_rejects_zero_max_results() -> None:
    tool = SearchScholarTool()
    with pytest.raises(ValueError, match="must be > 0"):
        tool.call(query="anything", max_results=0)


def test_spec_is_read_tier_and_describes_search_scholar() -> None:
    tool = SearchScholarTool()
    assert tool.spec.name == "search_scholar"
    assert tool.spec.tier == "read"
    assert "Semantic Scholar" in tool.spec.description
    assert "OpenAlex" in tool.spec.description


def test_call_uses_env_mailto_for_openalex(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAlex polite-pool email comes from HARNESS_OPENALEX_MAILTO
    when set. Verifies the env-var override path."""
    monkeypatch.setenv("HARNESS_OPENALEX_MAILTO", "mark@example.com")
    captured: list[str] = []

    def _opener(req: Any, timeout: int = 10) -> _FakeResponse:
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        captured.append(url)
        return _FakeResponse(b"{}")

    tool = SearchScholarTool(request_opener=_opener)
    tool.call(query="anything")
    openalex_call = next(url for url in captured if "openalex.org" in url)
    assert "mailto=mark%40example.com" in openalex_call
