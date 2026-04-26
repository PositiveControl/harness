from __future__ import annotations

from harness.tools.citations import CITATION_RE, extract_citations


def test_primary_three_segment_form() -> None:
    assert extract_citations("see §4-1-1 for details") == frozenset({"§4-1-1"})


def test_broader_two_segment_form() -> None:
    # §9-6 — entire 'Unmanned Free Balloons' section, no paragraph.
    assert extract_citations("cf. §9-6") == frozenset({"§9-6"})


def test_multiple_citations_union() -> None:
    text = "compare §4-1-1 and §13-1-2, plus TBL 4-1-2 and Figure 5-5-1"
    assert extract_citations(text) == frozenset({"§4-1-1", "§13-1-2", "TBL 4-1-2", "FIGURE 5-5-1"})


def test_en_dash_canonicalised_to_ascii() -> None:
    # Corpus-common variant: §4–1–1 with en-dashes.  # noqa: RUF003
    assert extract_citations("see §4–1–1") == frozenset({"§4-1-1"})  # noqa: RUF001


def test_unicode_minus_canonicalised_to_ascii() -> None:
    assert extract_citations("see §4−1−1") == frozenset({"§4-1-1"})  # noqa: RUF001


def test_space_after_section_sign_tolerated() -> None:
    assert extract_citations("see § 4-1-1") == frozenset({"§4-1-1"})


def test_tbl_and_table_forms_equivalent_after_canonicalise() -> None:
    assert extract_citations("TBL 4-1-2") == frozenset({"TBL 4-1-2"})
    # "Table" normalises to "TABLE" — case is upper-cased by the
    # canonicaliser so consumers can rely on one form.
    assert extract_citations("Table 4-1-2") == frozenset({"TABLE 4-1-2"})


def test_no_citations_returns_empty_frozenset() -> None:
    assert extract_citations("nothing to see here") == frozenset()
    assert extract_citations("") == frozenset()


def test_cfr_style_citations_are_not_grounded() -> None:
    # §91.113 is CFR-shape, not JO-shape. CITATION_RE (intentionally)
    # doesn't match it — this task's scope is JO 7110.65 / AIM section
    # anchors. If CFR ever needs grounding we add a separate extractor.
    assert extract_citations("per §91.113(b)") == frozenset()


def test_citation_re_is_case_insensitive_on_labels() -> None:
    # The regex accepts any case on tbl/table/fig/figure; the
    # canonicaliser upper-cases the label. The underlying pattern is
    # still IGNORECASE so lowercase input matches.
    assert CITATION_RE.search("see tbl 4-1-2") is not None
