"""Tests for harness.retrieval.query_expander (harness-ajn + harness-hvu1)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from harness.model.adapter import ChatMessage
from harness.retrieval.query_expander import (
    LLMQueryExpander,
    NullQueryExpander,
    QueryExpander,
    _parse_rewrites,
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
    shared.write_text('version: 1\nsections:\n  "3-10-3":\n    - shortest distance on approach\n')
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
    query_only.write_text('version: 1\nsections:\n  "2-1-19":\n    - small plane behind big jet\n')
    exp = load_query_expander(tmp_path / "synonyms.yaml", query_only_path=query_only)
    assert not exp.is_empty
    assert exp.triggered_sections("small plane behind big jet") == ("2-1-19",)


def test_load_default_query_only_path_is_corpus_sibling(tmp_path: Path) -> None:
    """Convention: query_synonyms.yaml lives next to synonyms.yaml under
    character/<name>/corpus/. Pinned so the CLI, tests, and any
    downstream tool agree on where to look."""
    from harness.retrieval.query_expander import default_query_only_synonyms_path

    assert default_query_only_synonyms_path(tmp_path) == tmp_path / "corpus" / "query_synonyms.yaml"


# ---------- LLMQueryExpander (harness-hvu1) ----------


class _ScriptedAdapter:
    """Stand-in for a real ModelAdapter — returns a queued reply per
    `complete()` call. Tests assert on the captured prompt + the
    expander's behavior given a known model output."""

    def __init__(self, replies: list[str], *, raises: bool = False) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []
        self.raises = raises

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(list(messages))
        if self.raises:
            raise RuntimeError("scripted adapter failure")
        return self.replies.pop(0)


def test_parse_rewrites_strips_numbering_and_bullets() -> None:
    raw = (
        "1. minimum same-runway separation\n2) landing behind category\n- arrival spacing minima\n"
    )
    parsed = _parse_rewrites(raw, max_rewrites=5)
    assert parsed == (
        "minimum same-runway separation",
        "landing behind category",
        "arrival spacing minima",
    )


def test_parse_rewrites_strips_quotes_around_phrases() -> None:
    raw = "\"minimum same-runway separation\"\n'landing behind category'\n"
    parsed = _parse_rewrites(raw, max_rewrites=5)
    assert parsed == ("minimum same-runway separation", "landing behind category")


def test_parse_rewrites_caps_at_max() -> None:
    raw = "phrase one\nphrase two\nphrase three\nphrase four\nphrase five\nphrase six\n"
    parsed = _parse_rewrites(raw, max_rewrites=3)
    assert parsed == ("phrase one", "phrase two", "phrase three")


def test_parse_rewrites_dedupes_repeats() -> None:
    raw = "Same Runway\nsame runway\nlanding behind\nLANDING BEHIND\n"
    parsed = _parse_rewrites(raw, max_rewrites=5)
    # Dedupe is case-insensitive; first occurrence's casing preserved.
    assert parsed == ("Same Runway", "landing behind")


def test_parse_rewrites_drops_short_header_echoes() -> None:
    """Small models sometimes echo the prompt's `Phrases:` line back."""
    raw = "Phrases:\nminimum same-runway separation\nOutput:\nlanding behind category\n"
    parsed = _parse_rewrites(raw, max_rewrites=5)
    assert parsed == ("minimum same-runway separation", "landing behind category")


def test_parse_rewrites_empty_when_input_blank() -> None:
    assert _parse_rewrites("", max_rewrites=5) == ()
    assert _parse_rewrites("   \n  \n", max_rewrites=5) == ()


def test_llm_expander_appends_paraphrases_to_query() -> None:
    adapter = _ScriptedAdapter(
        replies=["minimum same-runway separation\nlanding behind category\n"],
    )
    exp = LLMQueryExpander(adapter)
    out = exp.expand("shortest distance next to aircraft on approach")
    assert out == (
        "shortest distance next to aircraft on approach "
        "[paraphrases: minimum same-runway separation; landing behind category]"
    )
    # Adapter received exactly one call with one user-role message.
    assert len(adapter.calls) == 1
    assert len(adapter.calls[0]) == 1


def test_llm_expander_returns_query_unchanged_when_no_rewrites() -> None:
    adapter = _ScriptedAdapter(replies=["\n   \n"])
    exp = LLMQueryExpander(adapter)
    assert exp.expand("the query") == "the query"


def test_llm_expander_falls_through_to_chained_static_expander() -> None:
    """LLM rewrite + static synonym chain compose: LLM emits its
    paraphrases, then the static expander layers any section-tagged
    synonyms on top."""
    adapter = _ScriptedAdapter(replies=["minimum runway separation\n"])
    static = QueryExpander({"3-10-3": ("same runway separation",)})
    exp = LLMQueryExpander(adapter, chain_to=static)
    out = exp.expand("same runway separation")
    # Static triggers on the augmented query and adds its [related: ...] tag.
    assert "[paraphrases: minimum runway separation]" in out
    assert "[related: §3-10-3:" in out


def test_llm_expander_swallows_adapter_exceptions() -> None:
    """Adapter failure must not crash the chat — fall through to the
    chained expander on the original query."""
    adapter = _ScriptedAdapter(replies=[], raises=True)
    static = QueryExpander({"3-10-3": ("same runway separation",)})
    exp = LLMQueryExpander(adapter, chain_to=static)
    # No paraphrases tag (LLM failed), but static fired on the literal query.
    out = exp.expand("same runway separation")
    assert "[paraphrases:" not in out
    assert "[related: §3-10-3:" in out


def test_llm_expander_passes_max_tokens_and_temperature() -> None:
    """Wiring sanity: the configured budget reaches the adapter call."""

    class _Capturing:
        def __init__(self) -> None:
            self.kw: dict[str, object] = {}

        def complete(self, messages: Iterable[ChatMessage], **kw: object) -> str:
            self.kw = dict(kw)
            return "x\n"

    adapter = _Capturing()
    exp = LLMQueryExpander(adapter, max_tokens=64, temperature=0.3)
    exp.expand("anything")
    assert adapter.kw == {"max_tokens": 64, "temperature": 0.3}


def test_llm_expander_is_empty_returns_false() -> None:
    """Even with no synonyms-table component, the LLM expander always
    has something to contribute — so `is_empty` returns False to keep
    callers from prematurely skipping the build."""
    adapter = _ScriptedAdapter(replies=["x\n"])
    assert LLMQueryExpander(adapter).is_empty is False


def test_llm_expander_uses_default_prompt_template_when_unspecified() -> None:
    """Default template names the corpus context hint and expects the
    `{query}` substitution. Captured prompt should embed the user's
    question verbatim and reference the JO context."""
    adapter = _ScriptedAdapter(replies=["x\n"])
    exp = LLMQueryExpander(adapter)
    exp.expand("what is the purpose of 7110.65?")
    sent = adapter.calls[0]
    assert len(sent) == 1
    content = sent[0].content
    assert "what is the purpose of 7110.65?" in content
    assert "JO 7110.65" in content


def test_llm_expander_respects_custom_prompt_template() -> None:
    """Per-character prompt override path: pass any string with the
    {context_hint} + {query} placeholders and the expander uses it."""
    adapter = _ScriptedAdapter(replies=["x\n"])
    template = "Context: {context_hint}\nQuestion: {query}\nKeywords:"
    exp = LLMQueryExpander(adapter, prompt_template=template, context_hint="ATC")
    exp.expand("how do I separate aircraft?")
    sent = adapter.calls[0]
    content = sent[0].content
    assert "Context: ATC" in content
    assert "Question: how do I separate aircraft?" in content


def test_default_llm_expand_prompt_path_is_under_character_dir(tmp_path: Path) -> None:
    """Convention pin so CLI + eval agree on where the override lives."""
    from harness.retrieval.query_expander import default_llm_expand_prompt_path

    assert (
        default_llm_expand_prompt_path(tmp_path)
        == tmp_path / "retrieval_prompts" / "query_expansion.md"
    )


def test_load_llm_expand_prompt_returns_default_when_path_missing(tmp_path: Path) -> None:
    from harness.retrieval.query_expander import (
        _DEFAULT_LLM_EXPAND_PROMPT,
        _load_llm_expand_prompt,
    )

    assert _load_llm_expand_prompt(None) == _DEFAULT_LLM_EXPAND_PROMPT
    assert _load_llm_expand_prompt(tmp_path / "missing.md") == _DEFAULT_LLM_EXPAND_PROMPT


def test_load_llm_expand_prompt_reads_file_when_present(tmp_path: Path) -> None:
    from harness.retrieval.query_expander import _load_llm_expand_prompt

    override = tmp_path / "prompt.md"
    override.write_text("Custom: {query}", encoding="utf-8")
    assert _load_llm_expand_prompt(override) == "Custom: {query}"


def test_load_llm_expand_prompt_falls_back_when_file_empty(tmp_path: Path) -> None:
    """An empty / whitespace-only override is treated as 'use default' —
    cleaner failure mode than silently embedding an empty prompt."""
    from harness.retrieval.query_expander import (
        _DEFAULT_LLM_EXPAND_PROMPT,
        _load_llm_expand_prompt,
    )

    override = tmp_path / "prompt.md"
    override.write_text("   \n\n   ", encoding="utf-8")
    assert _load_llm_expand_prompt(override) == _DEFAULT_LLM_EXPAND_PROMPT


# ---------- harness-m78r: prefix parameterization + topics: schema ----------


def test_query_expander_topic_prefix_defaults_to_section_anchor() -> None:
    """Constructor default preserves airton_c1's `§` prefix — back-compat
    contract. New characters can pass `topic_prefix=""` for a clean
    glossary-id output."""
    exp = QueryExpander({"3-10-3": ("shortest distance on approach",)})
    out = exp.expand("shortest distance on approach")
    assert "[related: §3-10-3:" in out


def test_query_expander_topic_prefix_can_be_overridden() -> None:
    exp = QueryExpander(
        {"E11.9": ("high blood sugar",)},
        topic_prefix="ICD-10 ",
    )
    out = exp.expand("patient has high blood sugar")
    assert "[related: ICD-10 E11.9:" in out


def test_query_expander_empty_prefix_produces_bare_topic_id() -> None:
    """No prefix at all — the expanded query carries the raw topic key."""
    exp = QueryExpander(
        {"frost-prevention": ("cover plants",)},
        topic_prefix="",
    )
    out = exp.expand("how do I cover plants for frost")
    assert "[related: frost-prevention:" in out


def test_load_query_expander_reads_topics_yaml_with_empty_default_prefix(
    tmp_path: Path,
) -> None:
    """New `topics:` schema (no `sections:`) defaults to empty prefix —
    glossary content travels as-is. Characters carrying ATC-style
    section anchors keep their existing `sections:` shape and the
    legacy `§` default."""
    path = tmp_path / "glossary.yaml"
    path.write_text(
        "version: 1\n"
        "topics:\n"
        "  refund-policy:\n"
        "    - send back damaged item\n"
        "    - return for refund\n"
    )
    exp = load_query_expander(path)
    out = exp.expand("can I send back a damaged item")
    assert "[related: refund-policy:" in out


def test_load_query_expander_sections_yaml_keeps_legacy_section_prefix(
    tmp_path: Path,
) -> None:
    """Legacy `sections:` shape — `§` prefix preserved so airton_c1
    deployment behavior is unchanged."""
    path = tmp_path / "synonyms.yaml"
    path.write_text('version: 1\nsections:\n  "3-10-3":\n    - shortest distance on approach\n')
    exp = load_query_expander(path)
    out = exp.expand("shortest distance on approach")
    assert "[related: §3-10-3:" in out


def test_load_query_expander_explicit_prefix_in_yaml_wins(tmp_path: Path) -> None:
    """An explicit `prefix:` in YAML overrides both the legacy default
    and the new-schema default."""
    path = tmp_path / "glossary.yaml"
    path.write_text('version: 1\nprefix: "ICD-10 "\ntopics:\n  E11.9:\n    - high blood sugar\n')
    exp = load_query_expander(path)
    out = exp.expand("patient has high blood sugar")
    assert "[related: ICD-10 E11.9:" in out


def test_load_query_expander_explicit_prefix_overrides_legacy_default(
    tmp_path: Path,
) -> None:
    """A legacy `sections:` shape that also declares `prefix:` honors
    the explicit declaration — even when the explicit value is empty."""
    path = tmp_path / "synonyms.yaml"
    path.write_text(
        'version: 1\nprefix: ""\nsections:\n  "3-10-3":\n    - shortest distance on approach\n'
    )
    exp = load_query_expander(path)
    out = exp.expand("shortest distance on approach")
    # No `§` because the YAML explicitly cleared the prefix.
    assert "[related: 3-10-3:" in out
    assert "§3-10-3" not in out


def test_load_query_expander_prefers_topics_when_both_keys_present(
    tmp_path: Path,
) -> None:
    """If a YAML carries both `topics:` and `sections:`, the new
    convention wins (and the legacy default-`§` doesn't apply)."""
    path = tmp_path / "glossary.yaml"
    path.write_text(
        "version: 1\n"
        "topics:\n"
        "  new-thing:\n"
        "    - new lay phrase\n"
        "sections:\n"
        '  "3-10-3":\n'
        "    - shortest distance on approach\n"
    )
    exp = load_query_expander(path)
    # Only the topics-keyed entry surfaces.
    assert exp.triggered_topics("new lay phrase") == ("new-thing",)
    assert exp.triggered_topics("shortest distance on approach") == ()
    out = exp.expand("new lay phrase")
    # New-schema default prefix is empty.
    assert "[related: new-thing:" in out


def test_triggered_topics_is_alias_of_triggered_sections() -> None:
    """Back-compat: existing callers using `triggered_sections()` keep
    working; new code should prefer `triggered_topics()`."""
    exp = _exp({"3-10-3": ["shortest distance on approach"]})
    q = "shortest distance on approach"
    assert exp.triggered_topics(q) == exp.triggered_sections(q) == ("3-10-3",)
