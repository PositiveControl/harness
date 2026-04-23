"""Tests for src/harness/evals/atc.py (harness-xbk.7).

Covers the pure scoring / loading paths with in-memory fixtures and a
scripted run_turn callback. The CLI wires the real persona + retrieval
stack; these tests verify the contract, not the integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness.evals.atc import (
    AtcFixtureRow,
    default_fixture_path,
    load_fixture,
    run_atc_eval,
)

# ---------- default fixture path ----------


def test_default_fixture_path_under_character_dir(tmp_path: Path) -> None:
    assert default_fixture_path(tmp_path) == tmp_path / "atc_eval.yaml"


# ---------- load_fixture ----------


def _write_yaml(tmp_path: Path, data: object, name: str = "atc_eval.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_fixture_accepts_top_level_list(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        [
            {
                "id": "ppl_001",
                "audience": "ppl",
                "question": "What's the VFR cloud clearance above 10,000 MSL?",
                "expected_citations": ["91.155"],
                "expected_keywords": ["1,000 feet below", "1,000 feet above"],
                "min_keyword_hits": 1,
            },
        ],
    )
    rows = load_fixture(path)
    assert len(rows) == 1
    row = rows[0]
    assert row.id == "ppl_001"
    assert row.audience == "ppl"
    assert row.expected_citations == ("91.155",)
    # Scalar strings promote to single-alternate tuples — backward
    # compatible with pre-lane-A fixtures (harness-099).
    assert row.expected_keywords == (("1,000 feet below",), ("1,000 feet above",))
    assert row.min_keyword_hits == 1


def test_load_fixture_accepts_cases_key_wrapper(tmp_path: Path) -> None:
    """The atc-1 scaffold created an `atc_eval.yaml` shape of
    `{version, purpose, cases: []}`. The loader accepts that too."""
    path = _write_yaml(
        tmp_path,
        {
            "version": 1,
            "cases": [
                {
                    "id": "ifr_001",
                    "audience": "ifr",
                    "question": "Can I cancel IFR in the clouds?",
                    "expected_citations": ["91.155"],
                }
            ],
        },
    )
    rows = load_fixture(path)
    assert len(rows) == 1
    assert rows[0].audience == "ifr"


def test_load_fixture_empty_cases_returns_nothing(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, {"cases": []})
    assert load_fixture(path) == ()


def test_load_fixture_rejects_non_list_top_level(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "just a string")
    with pytest.raises(ValueError, match="is not a list"):
        load_fixture(path)


def test_load_fixture_rejects_missing_id(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, [{"audience": "ppl", "question": "x"}])
    with pytest.raises(ValueError, match="missing or empty 'id'"):
        load_fixture(path)


def test_load_fixture_rejects_empty_question(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, [{"id": "x", "audience": "ppl", "question": "   "}])
    with pytest.raises(ValueError, match="missing or empty 'question'"):
        load_fixture(path)


def test_load_fixture_rejects_min_hits_exceeding_keywords(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        [
            {
                "id": "x",
                "audience": "ppl",
                "question": "q",
                "expected_keywords": ["a", "b"],
                "min_keyword_hits": 5,
            }
        ],
    )
    with pytest.raises(ValueError, match="exceeds"):
        load_fixture(path)


def test_load_fixture_rejects_negative_min_hits(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        [
            {
                "id": "x",
                "audience": "ppl",
                "question": "q",
                "expected_keywords": ["a"],
                "min_keyword_hits": -1,
            }
        ],
    )
    with pytest.raises(ValueError, match="must be >= 0"):
        load_fixture(path)


# ---------- scoring ----------


def _row(**kw: object) -> AtcFixtureRow:
    base: dict[str, object] = {
        "id": "t",
        "audience": "ppl",
        "question": "?",
        "expected_citations": (),
        "expected_keywords": (),
        "min_keyword_hits": 0,
    }
    base.update(kw)
    return AtcFixtureRow(**base)  # type: ignore[arg-type]


def test_case_passes_when_citation_and_keywords_present() -> None:
    row = _row(
        expected_citations=("91.155",),
        expected_keywords=(("1,000 feet below",), ("1 statute mile",)),
        min_keyword_hits=2,
    )
    reply = (
        "Per 14 CFR 91.155 you need 1,000 feet below, 1,000 feet above, "
        "and 1 statute mile of horizontal separation from clouds."
    )
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    assert result.pass_rate == 1.0
    assert result.cases[0].citations_pass
    assert result.cases[0].keywords_pass
    assert result.cases[0].passed


def test_case_fails_on_missing_citation() -> None:
    row = _row(expected_citations=("91.155",))
    reply = "Cloud clearance rules are in the AIM."  # no section cited
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    assert result.pass_rate == 0.0
    case = result.cases[0]
    assert case.missing_citations == ("91.155",)
    assert not case.citations_pass


def test_case_fails_when_keyword_hits_below_min() -> None:
    row = _row(
        expected_citations=("91.155",),
        expected_keywords=(
            ("1,000 feet below",),
            ("1 statute mile",),
            ("5 statute miles",),
        ),
        min_keyword_hits=2,
    )
    reply = "See 14 CFR 91.155 — the table sets visibility and clearance."
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    assert result.cases[0].citations_pass  # citation present
    assert result.cases[0].keyword_hits == 0
    assert not result.cases[0].keywords_pass
    assert not result.cases[0].passed


def test_keyword_matching_is_case_insensitive() -> None:
    row = _row(
        expected_keywords=(("1,000 FEET BELOW",),),
        min_keyword_hits=1,
    )
    reply = "You need 1,000 feet below any cloud."
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    assert result.cases[0].matched_keywords == ("1,000 FEET BELOW",)


def test_citation_matching_is_case_insensitive() -> None:
    row = _row(expected_citations=("91.155",))
    reply = "Per 14 cfr 91.155 the rule holds."
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    assert result.cases[0].citations_pass


def test_empty_expected_citations_auto_passes_that_half() -> None:
    """A case with no expected citations (maybe a procedural 'what
    happens if' question) just checks keywords. Citations pass by
    default — no 'missing' set."""
    row = _row(
        expected_keywords=(("climb",), ("maintain",)),
        min_keyword_hits=1,
    )
    reply = "You climb and maintain 3,000."
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    assert result.cases[0].passed


# ---------- keyword alternates (harness-099, lane A) ----------


def test_keyword_alternates_count_once_when_multiple_match() -> None:
    """When an entry lists multiple alternate phrasings, a reply that
    contains more than one alternate still counts as ONE hit — the
    alternates share a concept. First-match-wins determines which
    string lands in matched_keywords so debug output shows what the
    model actually produced."""
    row = _row(
        expected_keywords=(("1,000 feet below", "1000 feet below", "1,000 ft below"),),
        min_keyword_hits=1,
    )
    reply = "You need 1,000 feet below and 1000 feet below any cloud."
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    case = result.cases[0]
    assert case.keyword_hits == 1
    # First alternate in the entry wins when multiple match.
    assert case.matched_keywords == ("1,000 feet below",)


def test_keyword_alternates_accept_any_phrasing() -> None:
    """This is the lane-A lift: the rubric accepts semantically-
    equivalent phrasings without needing a fixture edit for every
    grammatical variant."""
    row = _row(
        expected_keywords=(
            ("last assigned", "last ATC clearance", "last clearance"),
        ),
        min_keyword_hits=1,
    )
    reply = "Fly the last ATC clearance until two-way radio is restored."
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    assert result.cases[0].keyword_hits == 1
    assert result.cases[0].matched_keywords == ("last ATC clearance",)


def test_keyword_alternates_still_fail_when_no_alternate_matches() -> None:
    """Loose rubric must still fail a reply that misses the concept."""
    row = _row(
        expected_keywords=(
            ("1,000 feet below", "1000 feet below"),
            ("1 statute mile", "1 SM"),
        ),
        min_keyword_hits=2,
    )
    reply = "Cloud clearance rules are complicated."
    result = run_atc_eval([row], run_turn=lambda _q: reply)
    assert result.cases[0].keyword_hits == 0
    assert not result.cases[0].keywords_pass


def test_load_fixture_accepts_list_of_alternates_in_yaml(tmp_path: Path) -> None:
    """YAML-side: a keyword entry can be either a scalar string (the
    common case) or a list of alternate phrasings."""
    path = _write_yaml(
        tmp_path,
        [
            {
                "id": "x",
                "audience": "ppl",
                "question": "q",
                "expected_keywords": [
                    "scalar keyword",
                    ["alt one", "alt two", "alt three"],
                ],
                "min_keyword_hits": 2,
            }
        ],
    )
    rows = load_fixture(path)
    assert rows[0].expected_keywords == (
        ("scalar keyword",),
        ("alt one", "alt two", "alt three"),
    )


def test_load_fixture_rejects_empty_alternate_list(tmp_path: Path) -> None:
    """An empty list of alternates can never match — fixture bug."""
    path = _write_yaml(
        tmp_path,
        [
            {
                "id": "x",
                "audience": "ppl",
                "question": "q",
                "expected_keywords": [[]],
                "min_keyword_hits": 1,
            }
        ],
    )
    with pytest.raises(ValueError, match="empty list"):
        load_fixture(path)


def test_load_fixture_rejects_non_string_non_list_keyword(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        [
            {
                "id": "x",
                "audience": "ppl",
                "question": "q",
                "expected_keywords": [42],
                "min_keyword_hits": 1,
            }
        ],
    )
    with pytest.raises(ValueError, match="must be str or list"):
        load_fixture(path)


# ---------- aggregate reporting ----------


def test_pass_rate_by_audience_splits_ppl_ifr() -> None:
    rows = [
        _row(id="p1", audience="ppl", expected_citations=("A",)),
        _row(id="p2", audience="ppl", expected_citations=("B",)),
        _row(id="i1", audience="ifr", expected_citations=("C",)),
        _row(id="i2", audience="ifr", expected_citations=("D",)),
    ]
    # ppl replies cite A (pass) and nothing (fail); ifr replies both cite.
    replies = {
        "p1": "see A",
        "p2": "no citation here",
        "i1": "the rule is C",
        "i2": "per D, the answer is X",
    }
    fixture_id_by_question: dict[str, str] = {row.question + row.id: row.id for row in rows}
    # Questions are "?" for all — disambiguate via a stateful counter.
    counter = iter(["p1", "p2", "i1", "i2"])

    def scripted(_question: str) -> str:
        return replies[next(counter)]

    _ = fixture_id_by_question  # silence unused warning; kept for clarity
    result = run_atc_eval(rows, run_turn=scripted)
    rates = result.pass_rate_by_audience()
    assert rates == {"ppl": 0.5, "ifr": 1.0}


def test_failures_returns_only_failing_cases() -> None:
    rows = [
        _row(id="pass", expected_citations=("A",)),
        _row(id="fail", expected_citations=("B",)),
    ]
    replies = iter(["see A", "nothing here"])
    result = run_atc_eval(rows, run_turn=lambda _q: next(replies))
    failed_ids = [c.id for c in result.failures()]
    assert failed_ids == ["fail"]
