"""Tests for harness.retrieval.synonym_workflow (harness-rup2).

Pure-function tests — no MLX adapter, no real store. The suggester
side uses a scripted adapter; the tester side uses synthetic
baseline + ranks dicts. End-to-end integration against a real
airton_c1 store is exercised by running scripts/test_synonyms.py
against the live YAML files (manual smoke), not from these tests."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from harness.evals.atc import AtcFixtureRow
from harness.model.adapter import ChatMessage
from harness.retrieval.synonym_workflow import (
    CandidateMiss,
    CaseRankChange,
    SectionVerdict,
    Verdict,
    baseline_ranks,
    build_suggestion_prompt,
    cases_for_section,
    classify_section,
    emit_suggestion_doc,
    emit_suggestion_doc_header,
    emit_suggestion_yaml_block,
    expander_with,
    format_verdict_report,
    load_proposed_yaml,
    merge_proposed,
    select_candidates,
    suggest_for_case,
)


def _row(
    case_id: str,
    *,
    expected: tuple[tuple[str, ...], ...] = (("3-10-3",),),
    question: str = "test question",
) -> AtcFixtureRow:
    return AtcFixtureRow(
        id=case_id,
        audience="controller",
        question=question,
        expected_citations=expected,
        expected_keywords=(),
        min_keyword_hits=0,
    )


# ---------- load_proposed_yaml ----------


def test_load_proposed_yaml_returns_dict_for_real_shape(tmp_path: Path) -> None:
    path = tmp_path / "proposed.yaml"
    path.write_text(
        "version: 1\n"
        "sections:\n"
        '  "5-10-11":\n'
        "    - go around at last second\n"
        "    - missed approach already told pilot\n"
        '  "5-4-5":\n'
        "    - changing altitude during handoff\n"
    )
    result = load_proposed_yaml(path)
    assert "5-10-11" in result
    assert "5-4-5" in result
    assert "go around at last second" in result["5-10-11"]


def test_load_proposed_yaml_empty_for_missing_file(tmp_path: Path) -> None:
    result = load_proposed_yaml(tmp_path / "nope.yaml")
    assert result == {}


# ---------- merge_proposed ----------


def test_merge_proposed_unions_disjoint_sections() -> None:
    base = {"3-10-3": ("a", "b")}
    proposed = {"5-4-5": ("x",)}
    merged = merge_proposed(base, proposed)
    assert merged == {"3-10-3": ("a", "b"), "5-4-5": ("x",)}


def test_merge_proposed_appends_to_shared_section() -> None:
    base = {"3-10-3": ("a", "b")}
    proposed = {"3-10-3": ("c",)}
    merged = merge_proposed(base, proposed)
    assert merged["3-10-3"] == ("a", "b", "c")


def test_merge_proposed_dedupes_repeats() -> None:
    base = {"3-10-3": ("a", "b")}
    proposed = {"3-10-3": ("a", "c")}
    merged = merge_proposed(base, proposed)
    # 'a' from base survives once; 'c' from proposed appends.
    assert merged["3-10-3"] == ("a", "b", "c")


# ---------- expander_with ----------


def test_expander_with_layers_proposed_on_top(tmp_path: Path) -> None:
    shared = tmp_path / "synonyms.yaml"
    shared.write_text('version: 1\nsections:\n  "3-10-3":\n    - shortest distance on approach\n')
    proposed = {"3-10-3": ("layered phrase",)}
    exp = expander_with(shared_path=shared, query_only_path=None, proposed=proposed)
    assert exp.triggered_sections("shortest distance on approach") == ("3-10-3",)
    assert exp.triggered_sections("layered phrase") == ("3-10-3",)


def test_expander_with_no_proposed_matches_load_query_expander_baseline(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "synonyms.yaml"
    shared.write_text('version: 1\nsections:\n  "3-10-3":\n    - shortest distance on approach\n')
    exp = expander_with(shared_path=shared, query_only_path=None, proposed=None)
    assert exp.triggered_sections("shortest distance on approach") == ("3-10-3",)


# ---------- cases_for_section ----------


def test_cases_for_section_finds_matches() -> None:
    fixture = (
        _row("a", expected=(("3-10-3",),)),
        _row("b", expected=(("3-9-6",),)),
        _row("c", expected=(("3-10-3", "3-12-3"),)),  # alternate-list shape
    )
    assert cases_for_section(fixture, "3-10-3") == ("a", "c")
    assert cases_for_section(fixture, "3-9-6") == ("b",)
    assert cases_for_section(fixture, "9-9-9") == ()


def test_cases_for_section_walks_multi_position_expectations() -> None:
    fixture = (_row("multi", expected=(("3-10-3",), ("4-1-1",))),)
    # Both positions count — section §4-1-1 listed in second position
    # still maps the case to its serving anchors.
    assert cases_for_section(fixture, "4-1-1") == ("multi",)


# ---------- classify_section ----------


def test_classify_section_lift_when_target_improves_others_stable() -> None:
    before = {"target_a": None, "other_b": 0}
    after = {"target_a": 1, "other_b": 0}
    v = classify_section(
        section="5-10-11",
        proposed_variants=("phrase",),
        target_case_ids=("target_a",),
        before_ranks=before,
        after_ranks=after,
    )
    assert v.verdict == Verdict.LIFT


def test_classify_section_dead_when_nothing_moves() -> None:
    before = {"target_a": 0, "other_b": 1}
    after = {"target_a": 0, "other_b": 1}
    v = classify_section(
        section="5-10-11",
        proposed_variants=("phrase",),
        target_case_ids=("target_a",),
        before_ranks=before,
        after_ranks=after,
    )
    assert v.verdict == Verdict.DEAD


def test_classify_section_mixed_when_target_lifts_other_regresses() -> None:
    before = {"target_a": None, "other_b": 0}
    after = {"target_a": 1, "other_b": 3}
    v = classify_section(
        section="5-10-11",
        proposed_variants=("phrase",),
        target_case_ids=("target_a",),
        before_ranks=before,
        after_ranks=after,
    )
    assert v.verdict == Verdict.MIXED


def test_classify_section_harmful_when_target_unchanged_other_regresses() -> None:
    before = {"target_a": 0, "other_b": 0}
    after = {"target_a": 0, "other_b": 5}
    v = classify_section(
        section="5-10-11",
        proposed_variants=("phrase",),
        target_case_ids=("target_a",),
        before_ranks=before,
        after_ranks=after,
    )
    assert v.verdict == Verdict.HARMFUL


def test_classify_section_harmful_when_target_regresses() -> None:
    """The wake_turb cautionary case from the policy doc: §2-1-19
    promoted to shared synonyms moved the lay query rank up but the
    JARGON sibling case down. Tester must flag this."""
    before = {"jargon_a": 0, "lay_a": None}
    after = {"jargon_a": 9, "lay_a": 0}
    v = classify_section(
        section="2-1-19",
        proposed_variants=("phrase",),
        target_case_ids=("lay_a",),
        before_ranks=before,
        after_ranks=after,
    )
    # Target lifted (lay_a None→0) BUT collateral regressed
    # (jargon_a 0→9). MIXED captures the real wake_turb signature.
    assert v.verdict == Verdict.MIXED
    assert any(c.case_id == "jargon_a" for c in v.collateral_changes)
    assert any(c.case_id == "lay_a" for c in v.target_changes)


def test_classify_section_records_target_changes_only_for_target_ids() -> None:
    before = {"a": None, "b": None, "c": 5}
    after = {"a": 0, "b": 0, "c": 5}
    v = classify_section(
        section="5-10-11",
        proposed_variants=("phrase",),
        target_case_ids=("a",),
        before_ranks=before,
        after_ranks=after,
    )
    target_ids = {c.case_id for c in v.target_changes}
    collateral_ids = {c.case_id for c in v.collateral_changes}
    assert target_ids == {"a"}
    assert collateral_ids == {"b"}  # 'c' didn't move


# ---------- baseline_ranks ----------


def test_baseline_ranks_parses_saved_envelope() -> None:
    baseline = {
        "cases": [
            {"id": "a", "rank_of_first_expected": 0},
            {"id": "b", "rank_of_first_expected": None},
            {"id": "c", "rank_of_first_expected": 4},
        ]
    }
    ranks = baseline_ranks(baseline)
    assert ranks == {"a": 0, "b": None, "c": 4}


def test_baseline_ranks_tolerates_missing_optional_fields() -> None:
    baseline = {
        "cases": [
            {"id": "a", "rank_of_first_expected": 0},
            {"id": "b"},  # missing rank
            {"some-other-shape": True},  # malformed row
        ]
    }
    ranks = baseline_ranks(baseline)
    # Malformed rows skipped; 'b' lands as None (default).
    assert "a" in ranks
    assert ranks["a"] == 0
    assert "b" in ranks
    assert ranks["b"] is None


# ---------- select_candidates ----------


def test_select_candidates_picks_hard_miss_and_above_threshold() -> None:
    fixture = (
        _row("safe_a", expected=(("3-10-3",),)),
        _row("low_b", expected=(("3-9-6",),)),
        _row("miss_c", expected=(("4-1-1",),)),
    )
    baseline = {
        "cases": [
            {"id": "safe_a", "rank_of_first_expected": 0},
            {"id": "low_b", "rank_of_first_expected": 5},
            {"id": "miss_c", "rank_of_first_expected": None},
        ]
    }
    candidates = select_candidates(fixture=fixture, baseline=baseline, rank_threshold=3)
    ids = {c.case_id for c in candidates}
    assert ids == {"low_b", "miss_c"}


def test_select_candidates_skips_cases_with_no_expected_anchors() -> None:
    """A fixture row with empty expected_citations can't be helped by
    a synonym; the suggester would have no section to attach phrases
    to. Skip rather than emit a malformed candidate."""
    fixture = (
        _row("no_anchor", expected=()),
        _row("real_miss", expected=(("4-1-1",),)),
    )
    baseline = {
        "cases": [
            {"id": "no_anchor", "rank_of_first_expected": None},
            {"id": "real_miss", "rank_of_first_expected": None},
        ]
    }
    candidates = select_candidates(fixture=fixture, baseline=baseline)
    ids = {c.case_id for c in candidates}
    assert ids == {"real_miss"}


def test_select_candidates_uses_first_anchor_in_alternate_list() -> None:
    fixture = (_row("multi", expected=(("3-10-3", "3-12-3"),)),)
    baseline = {"cases": [{"id": "multi", "rank_of_first_expected": None}]}
    candidates = select_candidates(fixture=fixture, baseline=baseline)
    assert candidates[0].expected_section == "3-10-3"


# ---------- build_suggestion_prompt ----------


def test_build_suggestion_prompt_embeds_query_section_and_body() -> None:
    prompt = build_suggestion_prompt(
        query="lay query",
        section="5-10-11",
        body="real body content",
    )
    assert "lay query" in prompt
    assert "5-10-11" in prompt
    assert "real body content" in prompt


def test_build_suggestion_prompt_truncates_body_at_cap() -> None:
    long_body = "x" * 5000
    prompt = build_suggestion_prompt(query="q", section="3-10-3", body=long_body, body_cap=200)
    # Truncation marker present; full long body NOT.
    assert "..." in prompt
    assert long_body not in prompt


def test_build_suggestion_prompt_uses_context_hint() -> None:
    prompt = build_suggestion_prompt(
        query="q",
        section="5-10-11",
        body="b",
        context_hint="JO 7110.65 Appendix Z",
    )
    assert "JO 7110.65 Appendix Z" in prompt


# ---------- suggest_for_case ----------


class _ScriptedAdapter:
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


def test_suggest_for_case_returns_parsed_variants() -> None:
    adapter = _ScriptedAdapter(
        replies=[
            "go around at last second\nmissed approach already told pilot\n"
            "tell pilot missed approach before descent\n"
        ]
    )
    variants = suggest_for_case(
        adapter=adapter,
        query="If a plane has to go around at the last second...",
        section="5-10-11",
        body="Before an aircraft starts final descent...",
    )
    assert variants == (
        "go around at last second",
        "missed approach already told pilot",
        "tell pilot missed approach before descent",
    )


def test_suggest_for_case_returns_empty_on_adapter_failure() -> None:
    adapter = _ScriptedAdapter(replies=[], raises=True)
    variants = suggest_for_case(adapter=adapter, query="q", section="5-10-11", body="b")
    assert variants == ()


def test_suggest_for_case_caps_at_max_rewrites() -> None:
    adapter = _ScriptedAdapter(replies=["a\nb\nc\nd\ne\nf\ng\n"])
    variants = suggest_for_case(
        adapter=adapter,
        query="q",
        section="5-10-11",
        body="b",
        max_rewrites=3,
    )
    assert variants == ("a", "b", "c")


# ---------- emit_suggestion_yaml_block ----------


def test_emit_suggestion_yaml_block_renders_variants() -> None:
    block = emit_suggestion_yaml_block(
        section="5-10-11",
        variants=("go around", "missed approach"),
        case_id="controller_missed_approach_lay",
        current_rank=None,
    )
    assert '"5-10-11":' in block
    assert "controller_missed_approach_lay" in block
    assert "hard-miss" in block
    assert "go around" in block
    assert "missed approach" in block


def test_emit_suggestion_yaml_block_handles_empty_variants() -> None:
    block = emit_suggestion_yaml_block(
        section="5-10-11",
        variants=(),
        case_id="controller_missed_approach_lay",
        current_rank=8,
    )
    assert "rank 8" in block
    assert "[]" in block


def test_emit_suggestion_yaml_block_uses_rank_for_audit_trail() -> None:
    block = emit_suggestion_yaml_block(
        section="5-10-11",
        variants=("phrase",),
        case_id="case_a",
        current_rank=4,
    )
    assert "rank 4" in block


# ---------- emit_suggestion_doc / header ----------


def test_emit_suggestion_doc_header_starts_with_review_warning() -> None:
    header = emit_suggestion_doc_header()
    assert "REVIEW" in header
    assert "version: 1" in header
    assert "sections:" in header


def test_emit_suggestion_doc_concatenates_blocks() -> None:
    blocks = [
        emit_suggestion_yaml_block(
            section="5-10-11",
            variants=("a",),
            case_id="case_a",
            current_rank=None,
        ),
        emit_suggestion_yaml_block(
            section="5-4-5",
            variants=("b",),
            case_id="case_b",
            current_rank=4,
        ),
    ]
    doc = emit_suggestion_doc(blocks)
    assert '"5-10-11":' in doc
    assert '"5-4-5":' in doc
    # Header above content.
    assert doc.index("REVIEW") < doc.index('"5-10-11":')


# ---------- format_verdict_report ----------


def test_format_verdict_report_includes_section_and_verdict() -> None:
    verdict = SectionVerdict(
        section="5-10-11",
        proposed_variants=("a",),
        target_changes=(CaseRankChange(case_id="case_a", before=None, after=0),),
        collateral_changes=(),
        verdict=Verdict.LIFT,
        rationale="lift OK",
    )
    report = format_verdict_report([verdict])
    assert "§5-10-11" in report
    assert "LIFT" in report
    assert "case_a" in report
    assert "rank — → 0" in report  # None renders as —


def test_format_verdict_report_separates_target_and_collateral() -> None:
    verdict = SectionVerdict(
        section="2-1-19",
        proposed_variants=("p",),
        target_changes=(CaseRankChange(case_id="lay", before=None, after=0),),
        collateral_changes=(CaseRankChange(case_id="jargon", before=0, after=9),),
        verdict=Verdict.MIXED,
        rationale="trade",
    )
    report = format_verdict_report([verdict])
    assert "target cases:" in report
    assert "collateral cases:" in report
    # Order: target before collateral.
    assert report.index("target cases") < report.index("collateral cases")


def test_format_verdict_report_handles_no_movement() -> None:
    verdict = SectionVerdict(
        section="5-10-11",
        proposed_variants=("a",),
        target_changes=(),
        collateral_changes=(),
        verdict=Verdict.DEAD,
        rationale="nothing",
    )
    report = format_verdict_report([verdict])
    assert "(no movement)" in report


# ---------- CandidateMiss ----------


def test_candidate_miss_carries_optional_chunk_body() -> None:
    """The selector emits CandidateMiss without a body; the suggester
    fills it in once it's fetched the chunk. Default to empty string
    so callers can test for `if cand.chunk_body` cleanly."""
    cand = CandidateMiss(
        case_id="a",
        query="q",
        expected_section="5-10-11",
        current_rank=None,
    )
    assert cand.chunk_body == ""
