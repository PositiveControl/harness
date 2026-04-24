"""Tests for harness.retrieval.query_expander (harness-ajn)."""

from __future__ import annotations

from pathlib import Path

from harness.retrieval.query_expander import (
    NullQueryExpander,
    QueryExpander,
    default_synonyms_path,
    expand_many,
    load_query_expander,
)


def _exp(sections: dict[str, list[str]]) -> QueryExpander:
    return QueryExpander({sec: tuple(variants) for sec, variants in sections.items()})


# ---------- QueryExpander.triggered_sections ----------


def test_triggered_sections_single_match() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    assert exp.triggered_sections("what is the shortest distance on approach") == ("3-10-3",)


def test_triggered_sections_case_insensitive() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    assert exp.triggered_sections("SHORTEST DISTANCE ON APPROACH rule") == ("3-10-3",)


def test_triggered_sections_multiple_sections() -> None:
    exp = _exp(
        {
            "3-10-3": ["shortest distance on approach"],
            "3-9-6": ["shortest distance before takeoff"],
        }
    )
    result = exp.triggered_sections("compare shortest distance on approach vs takeoff spacing")
    assert result == ("3-10-3",)


def test_triggered_sections_dedupes_multiple_terms_per_section() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach", "minimum spacing on final"]})
    result = exp.triggered_sections("shortest distance on approach and minimum spacing on final")
    # One section, hit twice by different terms — appears once.
    assert result == ("3-10-3",)


def test_triggered_sections_no_match_returns_empty() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    assert exp.triggered_sections("completely unrelated query about emergencies") == ()


def test_triggered_sections_empty_query_returns_empty() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    assert exp.triggered_sections("") == ()


def test_triggered_sections_matches_paraphrase_with_content_token_subset() -> None:
    """Content-token-subset matching: lay term 'shortest distance on
    approach' triggers on the fixture paraphrase 'what is the shortest
    distance an aircraft is allowed to be next to another aircraft on
    approach to the runway' even though the verbatim phrase doesn't
    appear. The lay term's content tokens {shortest, distance, approach}
    are all present in the query — substring matching (the original
    strategy) would have missed this."""
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    paraphrase = (
        "what is the shortest distance an aircraft is allowed "
        "to be next to another aircraft on approach to the runway"
    )
    assert exp.triggered_sections(paraphrase) == ("3-10-3",)


def test_triggered_sections_rejects_partial_content_overlap() -> None:
    """The subset rule is strict: if the query has SOME but not ALL of
    the lay term's content tokens, it doesn't trigger. Protects against
    a shared topical token (e.g. 'approach') attracting every section
    with 'approach' in a lay entry."""
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    # Has 'approach' but lacks 'shortest' and 'distance'.
    assert exp.triggered_sections("when do you clear for the approach") == ()


def test_triggered_sections_stopwords_dont_have_to_appear() -> None:
    """Stopwords in the lay term are ignored on both sides — so a
    query that drops 'on' still triggers the 'shortest distance on
    approach' lay term."""
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    assert exp.triggered_sections("shortest distance for approach") == ("3-10-3",)


# ---------- QueryExpander.expand ----------


def test_expand_appends_related_block_on_match() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach", "minimum spacing on final"]})
    out = exp.expand("how close can aircraft be — shortest distance on approach?")
    assert "[related:" in out
    assert "§3-10-3" in out
    assert "minimum spacing on final" in out
    assert "shortest distance on approach" in out


def test_expand_identity_when_no_match() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    q = "completely unrelated query"
    assert exp.expand(q) == q


def test_expand_identity_on_empty_query() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    assert exp.expand("") == ""


def test_expand_multi_section_separates_with_pipe() -> None:
    exp = _exp(
        {
            "3-10-3": ["shortest distance on approach"],
            "3-9-6": ["shortest distance before takeoff"],
        }
    )
    out = exp.expand(
        "need both: shortest distance on approach AND shortest distance before takeoff"
    )
    assert "§3-10-3" in out
    assert "§3-9-6" in out
    assert "|" in out  # chunk separator


# ---------- NullQueryExpander ----------


def test_null_expander_returns_query_unchanged() -> None:
    null = NullQueryExpander()
    assert null.is_empty
    assert null.expand("anything at all") == "anything at all"
    assert null.triggered_sections("anything at all") == ()


# ---------- load_query_expander ----------


def test_load_returns_null_when_path_is_none() -> None:
    exp = load_query_expander(None)
    assert isinstance(exp, NullQueryExpander)


def test_load_returns_null_when_file_missing(tmp_path: Path) -> None:
    exp = load_query_expander(tmp_path / "nonexistent.yaml")
    assert isinstance(exp, NullQueryExpander)


def test_load_returns_null_on_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(":::not yaml:::")
    exp = load_query_expander(path)
    assert isinstance(exp, NullQueryExpander)


def test_load_returns_null_without_sections_key(tmp_path: Path) -> None:
    path = tmp_path / "no_sections.yaml"
    path.write_text("version: 1\nother_key: []\n")
    exp = load_query_expander(path)
    assert isinstance(exp, NullQueryExpander)


def test_load_parses_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "synonyms.yaml"
    path.write_text(
        "version: 1\n"
        "sections:\n"
        '  "3-10-3":\n'
        "    - shortest distance on approach\n"
        "    - minimum spacing on final\n"
        '  "3-9-6":\n'
        "    - takeoff spacing\n"
    )
    exp = load_query_expander(path)
    assert not exp.is_empty
    assert exp.triggered_sections("shortest distance on approach") == ("3-10-3",)
    assert exp.triggered_sections("takeoff spacing") == ("3-9-6",)


def test_load_skips_non_string_variants(tmp_path: Path) -> None:
    """Variants that aren't strings silently drop — keeps the loader
    tolerant of hand-edits without blowing up chat bootstrap."""
    path = tmp_path / "synonyms.yaml"
    path.write_text(
        "version: 1\n"
        "sections:\n"
        '  "3-10-3":\n'
        "    - shortest distance on approach\n"
        "    - 42\n"
        "    - null\n"
    )
    exp = load_query_expander(path)
    # The string entry survives; the int and null are dropped.
    assert exp.triggered_sections("shortest distance on approach") == ("3-10-3",)


# ---------- default_synonyms_path ----------


def test_default_synonyms_path_matches_ingest_convention(tmp_path: Path) -> None:
    assert default_synonyms_path(tmp_path) == tmp_path / "corpus" / "synonyms.yaml"


# ---------- expand_many ----------


def test_expand_many_batches_correctly() -> None:
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    results = expand_many(exp, ["first query", "shortest distance on approach query"])
    assert results[0] == "first query"
    assert "[related:" in results[1]


def test_expand_many_with_null_expander_is_identity() -> None:
    null = NullQueryExpander()
    queries = ("a", "b", "c")
    assert expand_many(null, queries) == queries


# ---------- query_only_path merging (harness-ajn split) ----------


def test_load_merges_shared_and_query_only_files(tmp_path: Path) -> None:
    """Entries in `query_synonyms.yaml` augment entries in `synonyms.yaml`
    on the expander side but leave the ingest-consumed file unchanged.
    Same section in both: variants union (shared first, then query-only,
    duplicates deduped)."""
    shared = tmp_path / "synonyms.yaml"
    shared.write_text(
        "version: 1\n"
        "sections:\n"
        '  "3-10-3":\n'
        "    - shortest distance on approach\n"
    )
    query_only = tmp_path / "query_synonyms.yaml"
    query_only.write_text(
        "version: 1\n"
        "sections:\n"
        '  "3-10-3":\n'
        "    - soon after plane lands\n"
        '  "2-1-19":\n'
        "    - small plane behind big jet\n"
    )
    exp = load_query_expander(shared, query_only_path=query_only)
    assert not exp.is_empty
    # Section 3-10-3 picked up entries from both files.
    assert exp.triggered_sections("shortest distance on approach") == ("3-10-3",)
    assert exp.triggered_sections("soon after plane lands") == ("3-10-3",)
    # Section 2-1-19 appears only because query_only supplied it.
    assert exp.triggered_sections("small plane behind big jet") == ("2-1-19",)


def test_load_with_only_query_only_file(tmp_path: Path) -> None:
    """A character with no shared synonyms.yaml but a query_synonyms.yaml
    still gets an expander — the shared-file branch is optional."""
    query_only = tmp_path / "query_synonyms.yaml"
    query_only.write_text(
        "version: 1\nsections:\n  \"2-1-19\":\n    - small plane behind big jet\n"
    )
    exp = load_query_expander(tmp_path / "synonyms.yaml", query_only_path=query_only)
    assert not exp.is_empty
    assert exp.triggered_sections("small plane behind big jet") == ("2-1-19",)


def test_load_default_query_only_path_is_corpus_sibling(tmp_path: Path) -> None:
    """Convention: query_synonyms.yaml lives next to synonyms.yaml under
    character/<name>/corpus/. Pinned so the CLI, tests, and any
    downstream tool agree on where to look."""
    from harness.retrieval.query_expander import default_query_only_synonyms_path

    assert default_query_only_synonyms_path(tmp_path) == tmp_path / "corpus" / "query_synonyms.yaml"
