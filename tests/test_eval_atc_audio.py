"""Tests for src/harness/evals/atc_audio.py (harness-gy5z).

Pin the WER tokenizer + edit-distance, the utt/*.jsonl fixture loader
(verified-only filter, dedup, section normalization), the two-pass
scoring including verdict_shift, the skip-noisy stub path, the
event-tag breakdown, and the baseline comparator. The lint pipeline
itself is stubbed via a deterministic LintFn so the scorer stays
pure.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from harness.evals.atc_audio import (
    AudioEvalCase,
    AudioFixtureRow,
    _levenshtein,
    _normalize_for_wer,
    compare_baselines,
    compute_wer,
    iter_utt_rows,
    load_fixture,
    run_audio_eval,
)
from harness.tools.phraseology_lint import PhraseologyVerdict, Verdict

# ---------- WER ----------


def test_normalize_for_wer_lowercases_and_strips_edge_punct() -> None:
    assert _normalize_for_wer("  Hello, World!  ") == ["hello", "world"]


def test_normalize_for_wer_preserves_internal_punct() -> None:
    """Apostrophes inside tokens (don't, F.A.A.) must survive so
    contractions and abbreviations align cleanly token-for-token."""
    assert _normalize_for_wer("Don't taxi via F.A.A. ramp") == [
        "don't",
        "taxi",
        "via",
        "f.a.a",
        "ramp",
    ]


def test_normalize_for_wer_empty_returns_empty() -> None:
    assert _normalize_for_wer("") == []
    assert _normalize_for_wer("   ") == []


def test_levenshtein_identical_zero() -> None:
    assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0


def test_levenshtein_substitution() -> None:
    assert _levenshtein(["a", "b", "c"], ["a", "X", "c"]) == 1


def test_levenshtein_insertion_deletion() -> None:
    assert _levenshtein(["a", "b"], ["a", "b", "c"]) == 1
    assert _levenshtein(["a", "b", "c"], ["a", "b"]) == 1


def test_compute_wer_perfect_match() -> None:
    assert compute_wer("november 478", "November 478.") == 0.0


def test_compute_wer_one_substitution_in_three_words() -> None:
    assert compute_wer("alpha bravo charlie", "alpha delta charlie") == pytest.approx(1 / 3)


def test_compute_wer_empty_reference_returns_hyp_word_count() -> None:
    """Empty reference + non-empty hyp: every hyp token is an
    insertion. Returning the token count gives a usable signal for
    averaging without producing inf."""
    assert compute_wer("", "hello world") == pytest.approx(2.0)


def test_compute_wer_empty_hypothesis_is_total_loss() -> None:
    assert compute_wer("hello world", "") == pytest.approx(1.0)


def test_compute_wer_both_empty_is_zero() -> None:
    assert compute_wer("", "") == 0.0


# ---------- fixture loader ----------


def _row(
    *,
    clip_id: str = "abc123",
    utt_index: int = 0,
    speaker: str = "controller",
    transcript: str = "cleared for takeoff",
    seed: str = "cleard for take off",
    verdict: str = "ok",
    section: str | None = "3-9-10",
    phraseology: str | None = "CLEARED FOR TAKEOFF",
    mismatch: str | None = None,
    verified: bool = True,
    event_tag: str | None = "departure",
) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "utt_index": utt_index,
        "speaker_role": speaker,
        "facility": "KATL",
        "position": "Twr",
        "event_tag": event_tag,
        "transcript_text": transcript,
        "transcript_seed": seed,
        "expected_verdict": verdict,
        "expected_section": section,
        "expected_phraseology": phraseology,
        "mismatch": mismatch,
        "human_verified": verified,
    }


def _write_utt(target: Path, clip_id: str, rows: list[dict[str, object]]) -> None:
    utt_dir = target / "utt"
    utt_dir.mkdir(parents=True, exist_ok=True)
    path = utt_dir / f"{clip_id}.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def test_iter_utt_rows_skips_blank_and_corrupt_lines(tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    (target / "utt").mkdir(parents=True)
    (target / "utt" / "x.jsonl").write_text(
        json.dumps({"a": 1}) + "\n\n{not json\n" + json.dumps({"a": 2}) + "\n",
        encoding="utf-8",
    )
    rows = list(iter_utt_rows(target))
    assert rows == [{"a": 1}, {"a": 2}]


def test_iter_utt_rows_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert list(iter_utt_rows(tmp_path)) == []


def test_load_fixture_drops_unverified_by_default(tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    _write_utt(
        target,
        "cid1",
        [_row(utt_index=0), _row(utt_index=1, verified=False), _row(utt_index=2)],
    )
    rows = load_fixture(target)
    assert {r.utt_index for r in rows} == {0, 2}


def test_load_fixture_includes_unverified_when_flagged(tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    _write_utt(target, "cid1", [_row(utt_index=0, verified=False)])
    rows = load_fixture(target, only_verified=False)
    assert len(rows) == 1


def test_load_fixture_filters_by_clip_id(tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    _write_utt(target, "cid1", [_row(clip_id="cid1", utt_index=0)])
    _write_utt(target, "cid2", [_row(clip_id="cid2", utt_index=0)])
    rows = load_fixture(target, only_clip_ids=("cid1",))
    assert {r.clip_id for r in rows} == {"cid1"}


def test_load_fixture_dedups_by_clip_and_utt_index(tmp_path: Path) -> None:
    """Append-only writes can produce duplicate rows on resume edge
    cases. Loader keeps the first occurrence and drops repeats."""
    target = tmp_path / "atc_audio"
    _write_utt(
        target,
        "cid1",
        [
            _row(utt_index=0, transcript="first"),
            _row(utt_index=0, transcript="second-write"),
        ],
    )
    rows = load_fixture(target)
    assert len(rows) == 1
    assert rows[0].transcript_text == "first"


def test_load_fixture_normalizes_section_form(tmp_path: Path) -> None:
    """Section anchors might come in as `§3-9-10`, `3−9−10` (U+2212),
    or plain — eval gates need them comparable."""
    target = tmp_path / "atc_audio"
    _write_utt(target, "cid1", [_row(utt_index=0, section="§3−9−10")])  # noqa: RUF001
    rows = load_fixture(target)
    assert rows[0].expected_section == "3-9-10"


def test_load_fixture_drops_invalid_verdict(tmp_path: Path) -> None:
    target = tmp_path / "atc_audio"
    _write_utt(target, "cid1", [_row(utt_index=0, verdict="maybe")])
    assert load_fixture(target) == ()


# ---------- run_audio_eval ----------


def _stub_lint(
    expected: dict[str, PhraseologyVerdict],
) -> Callable[[str, str | None], PhraseologyVerdict]:
    """Return a LintFn that maps utterance → verdict via lookup. Falls
    back to OOS when the input isn't recognised so the harness can
    cleanly evaluate "predicted OOS" branches."""

    def fn(utterance: str, _scenario: str | None) -> PhraseologyVerdict:
        return expected.get(
            utterance.strip(),
            PhraseologyVerdict(
                verdict="out_of_scope",
                expected_section=None,
                expected_phraseology=None,
                mismatch=None,
                citation_quote=None,
            ),
        )

    return fn


def _make_row(
    transcript: str,
    seed: str,
    verdict: str,
    section: str | None,
    *,
    clip_id: str = "cid1",
    utt_index: int = 0,
    event_tag: str = "departure",
) -> AudioFixtureRow:
    return AudioFixtureRow(
        clip_id=clip_id,
        utt_index=utt_index,
        speaker_role="controller",
        facility="KATL",
        position="Twr",
        event_tag=event_tag,
        transcript_text=transcript,
        transcript_seed=seed,
        expected_verdict=verdict,
        expected_section=section,
        expected_phraseology=None,
        mismatch=None,
    )


def test_run_audio_eval_clean_pass_correct() -> None:
    fixture = (_make_row("alpha", "alpha", "ok", "3-9-10"),)
    lint = _stub_lint(
        {
            "alpha": PhraseologyVerdict(
                verdict="ok",
                expected_section="3-9-10",
                expected_phraseology=None,
                mismatch=None,
                citation_quote=None,
            )
        }
    )
    result = run_audio_eval(fixture, lint)
    assert result.case_count == 1
    assert result.clean_verdict_accuracy == 1.0
    assert result.noisy_verdict_accuracy == 1.0
    assert result.mean_wer == 0.0
    assert result.verdict_shift_rate == 0.0


def test_run_audio_eval_verdict_shift_when_noisy_breaks() -> None:
    """The headline shift case: clean transcript → ok, but whisper
    seed mangled the verb to something the lint can't classify, so
    noisy comes back OOS. verdict_shift_rate must move."""
    clean = "cleared for takeoff"
    noisy = "cleared four touchdown"
    fixture = (_make_row(clean, noisy, "ok", "3-9-10"),)
    lint = _stub_lint(
        {
            clean: PhraseologyVerdict(
                verdict="ok",
                expected_section="3-9-10",
                expected_phraseology=None,
                mismatch=None,
                citation_quote=None,
            ),
            # noisy text isn't in the map → falls back to OOS.
        }
    )
    result = run_audio_eval(fixture, lint)
    assert result.clean_verdict_accuracy == 1.0
    assert result.noisy_verdict_accuracy == 0.0
    assert result.verdict_shift_rate == 1.0
    assert result.mean_wer > 0.0


def test_run_audio_eval_skip_noisy_zeros_wer_and_uses_oos_stub() -> None:
    fixture = (_make_row("alpha", "alpha-mangled", "ok", "3-9-10"),)
    lint = _stub_lint(
        {
            "alpha": PhraseologyVerdict(
                verdict="ok",
                expected_section="3-9-10",
                expected_phraseology=None,
                mismatch=None,
                citation_quote=None,
            )
        }
    )
    result = run_audio_eval(fixture, lint, skip_noisy=True)
    case = result.cases[0]
    assert case.clean_verdict_pass is True
    assert case.noisy_actual.verdict == "out_of_scope"  # stubbed
    assert case.wer == 0.0
    assert result.mean_wer == 0.0


def test_run_audio_eval_oos_citation_passes_on_null_match() -> None:
    """Pilot-side / chatter rows expect oos + null section. The lint
    tool must also return null section to count as a citation pass."""
    fixture = (_make_row("rando chatter", "rando chatter", "out_of_scope", None),)
    lint = _stub_lint({})  # everything → OOS via fallback
    result = run_audio_eval(fixture, lint)
    case = result.cases[0]
    assert case.clean_verdict_pass is True
    assert case.clean_citation_pass is True
    assert case.clean_passed is True


def test_run_audio_eval_by_event_tag_partition() -> None:
    fixture = (
        _make_row("a1", "a1", "ok", "3-9-10", clip_id="dep1", event_tag="departure"),
        _make_row("a2", "a2", "ok", "3-9-10", clip_id="dep2", event_tag="departure"),
        _make_row("e1", "e1", "ok", "3-10-1", clip_id="emer1", event_tag="emergency"),
    )
    lint = _stub_lint(
        {
            "a1": PhraseologyVerdict(
                verdict="ok",
                expected_section="3-9-10",
                expected_phraseology=None,
                mismatch=None,
                citation_quote=None,
            ),
            "a2": PhraseologyVerdict(
                verdict="wrong",
                expected_section="3-9-10",
                expected_phraseology=None,
                mismatch="x",
                citation_quote=None,
            ),
            "e1": PhraseologyVerdict(
                verdict="ok",
                expected_section="3-10-1",
                expected_phraseology=None,
                mismatch=None,
                citation_quote=None,
            ),
        }
    )
    result = run_audio_eval(fixture, lint)
    by_tag = result.by_event_tag()
    assert by_tag["departure"]["count"] == 2
    assert by_tag["departure"]["clean_verdict_accuracy"] == 0.5
    assert by_tag["emergency"]["count"] == 1
    assert by_tag["emergency"]["clean_verdict_accuracy"] == 1.0


# ---------- AudioEvalCase / case_id ----------


def test_audio_fixture_row_case_id_pads_utt_index() -> None:
    row = _make_row("a", "a", "ok", "3-9-10", clip_id="abcd", utt_index=7)
    assert row.case_id == "abcd:0007"


def test_audio_fixture_row_case_id_unique_within_fixture() -> None:
    rows = [_make_row("a", "a", "ok", "3-9-10", clip_id="A", utt_index=i) for i in range(3)]
    assert len({r.case_id for r in rows}) == 3


# ---------- baseline diff ----------


def _verdict(verdict: str = "ok", section: str | None = "3-9-10") -> PhraseologyVerdict:
    return PhraseologyVerdict(
        verdict=cast("Verdict", verdict),
        expected_section=section,
        expected_phraseology=None,
        mismatch=None,
        citation_quote=None,
    )


def _make_case(
    case_id: str,
    *,
    expected: str = "ok",
    expected_section: str | None = "3-9-10",
    clean: PhraseologyVerdict | None = None,
    noisy: PhraseologyVerdict | None = None,
    wer: float = 0.0,
) -> AudioEvalCase:
    clip_id, _, utt = case_id.partition(":")
    row = _make_row(
        "cleaned",
        "noisy",
        expected,
        expected_section,
        clip_id=clip_id,
        utt_index=int(utt),
    )
    return AudioEvalCase(
        row=row,
        clean_actual=clean or _verdict(expected, expected_section),
        noisy_actual=noisy or _verdict(expected, expected_section),
        wer=wer,
    )


def test_compare_baselines_no_change_zero_regressions() -> None:
    cases = (_make_case("A:0000"),)
    from harness.evals.atc_audio import AudioEvalResult

    result = AudioEvalResult(cases=cases)
    baseline = {
        "clean_verdict_accuracy": 1.0,
        "noisy_verdict_accuracy": 1.0,
        "clean_citation_accuracy": 1.0,
        "noisy_citation_accuracy": 1.0,
        "clean_combined_accuracy": 1.0,
        "noisy_combined_accuracy": 1.0,
        "mean_wer": 0.0,
        "cases": [
            {
                "case_id": "A:0000",
                "clean_verdict_pass": True,
                "noisy_verdict_pass": True,
                "clean_citation_pass": True,
                "noisy_citation_pass": True,
            }
        ],
    }
    diff = compare_baselines(baseline, result)
    assert diff.case_regressions == ()
    assert diff.aggregate_regressions == ()
    assert diff.has_regression() is False


def test_compare_baselines_aggregate_regression_blocks_gate() -> None:
    cases = (_make_case("A:0000", clean=_verdict("wrong", "3-9-10")),)
    from harness.evals.atc_audio import AudioEvalResult

    result = AudioEvalResult(cases=cases)
    baseline = {
        "clean_verdict_accuracy": 1.0,  # was 100%, now 0%
        "noisy_verdict_accuracy": 1.0,
        "clean_citation_accuracy": 1.0,
        "noisy_citation_accuracy": 1.0,
        "clean_combined_accuracy": 1.0,
        "noisy_combined_accuracy": 1.0,
        "mean_wer": 0.0,
        "cases": [
            {
                "case_id": "A:0000",
                "clean_verdict_pass": True,
                "noisy_verdict_pass": True,
                "clean_citation_pass": True,
                "noisy_citation_pass": True,
            }
        ],
    }
    diff = compare_baselines(baseline, result)
    assert diff.has_regression() is True
    assert any(d.metric == "clean_verdict_accuracy" for d in diff.aggregate_regressions)


def test_compare_baselines_wer_rise_is_regression() -> None:
    """WER's polarity is inverted — higher = worse. The gate must
    classify a WER rise as a regression even though every other
    aggregate is monotonic the other way."""
    cases = (_make_case("A:0000", wer=0.5),)
    from harness.evals.atc_audio import AudioEvalResult

    result = AudioEvalResult(cases=cases)
    baseline = {
        "clean_verdict_accuracy": 1.0,
        "noisy_verdict_accuracy": 1.0,
        "clean_citation_accuracy": 1.0,
        "noisy_citation_accuracy": 1.0,
        "clean_combined_accuracy": 1.0,
        "noisy_combined_accuracy": 1.0,
        "mean_wer": 0.2,
        "cases": [
            {
                "case_id": "A:0000",
                "clean_verdict_pass": True,
                "noisy_verdict_pass": True,
                "clean_citation_pass": True,
                "noisy_citation_pass": True,
            }
        ],
    }
    diff = compare_baselines(baseline, result)
    assert any(d.metric == "mean_wer" and d.is_regression for d in diff.aggregate_regressions)


def test_compare_baselines_per_case_flip_under_budget() -> None:
    """A single case flip with budget=1 must NOT trip the gate as long
    as aggregate metrics still hold. Mirrors the phraseology eval's
    regression-budget escape hatch — useful when fixture growth
    naturally moves a borderline case."""
    cases = (
        _make_case("A:0000", clean=_verdict("wrong", "3-9-10")),
        _make_case("B:0000"),
    )
    from harness.evals.atc_audio import AudioEvalResult

    result = AudioEvalResult(cases=cases)
    baseline = {
        # Aggregates set high enough that 1 flip doesn't drop us
        # below the saved value (50% old, 50% new).
        "clean_verdict_accuracy": 0.5,
        "noisy_verdict_accuracy": 1.0,
        "clean_citation_accuracy": 1.0,
        "noisy_citation_accuracy": 1.0,
        "clean_combined_accuracy": 0.5,
        "noisy_combined_accuracy": 1.0,
        "mean_wer": 0.0,
        "cases": [
            {
                "case_id": "A:0000",
                "clean_verdict_pass": True,  # was passing, now failing
                "noisy_verdict_pass": True,
                "clean_citation_pass": True,
                "noisy_citation_pass": True,
            },
            {
                "case_id": "B:0000",
                "clean_verdict_pass": False,  # was failing, now passing
                "noisy_verdict_pass": True,
                "clean_citation_pass": True,
                "noisy_citation_pass": True,
            },
        ],
    }
    diff = compare_baselines(baseline, result)
    assert len(diff.case_regressions) == 1
    assert len(diff.case_improvements) == 1
    # Default budget=0 fails; budget=1 passes.
    assert diff.has_regression() is True
    assert diff.has_regression(regression_budget=1) is False


def test_compare_baselines_new_and_dropped_cases_are_listed() -> None:
    cases = (_make_case("NEW:0000"),)
    from harness.evals.atc_audio import AudioEvalResult

    result = AudioEvalResult(cases=cases)
    baseline = {
        "cases": [
            {
                "case_id": "OLD:0000",
                "clean_verdict_pass": True,
                "noisy_verdict_pass": True,
                "clean_citation_pass": True,
                "noisy_citation_pass": True,
            }
        ]
    }
    diff = compare_baselines(baseline, result)
    assert diff.new_cases == ("NEW:0000",)
    assert diff.dropped_cases == ("OLD:0000",)
