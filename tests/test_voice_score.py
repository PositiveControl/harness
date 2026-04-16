from __future__ import annotations

from harness.evals.voice_score import score_actual_against_gold


def test_perfect_match_scores_one() -> None:
    gold = "Don't know. Two paths: (a) read source, (b) write a repro."
    score = score_actual_against_gold(actual=gold, gold=gold)
    assert score.aggregate == 1.0
    assert score.length_match == 1.0
    assert score.no_banned_openers == 1.0
    assert score.bullet_discipline == 1.0


def test_banned_opener_zeroes_that_axis() -> None:
    gold = "No. Branch has two others on it."
    actual = "That's a solid question. No, branch has others on it."
    score = score_actual_against_gold(actual=actual, gold=gold)
    assert score.no_banned_openers == 0.0
    assert "banned_opener" in "".join(score.notes)


def test_case_insensitive_opener_match() -> None:
    gold = "No."
    actual = "that's a great question. here's what I'd do..."
    score = score_actual_against_gold(actual=actual, gold=gold)
    assert score.no_banned_openers == 0.0


def test_code_fence_opener_allowed() -> None:
    gold = "```\ncommit\n```"
    actual = "```\nalso a commit\n```"
    score = score_actual_against_gold(actual=actual, gold=gold)
    assert score.no_banned_openers == 1.0


def test_unexpected_numbered_list_zeroes_bullet_discipline() -> None:
    gold = "Don't know. Two paths: read source, or write a repro."
    actual = "Several causes:\n1. Caching\n2. Compaction bug\n3. Index staleness"
    score = score_actual_against_gold(actual=actual, gold=gold)
    assert score.bullet_discipline == 0.0


def test_matching_numbered_lists_pass() -> None:
    gold = "Steps:\n1. Check A\n2. Verify B"
    actual = "Try:\n1. Inspect A\n2. Validate B"
    score = score_actual_against_gold(actual=actual, gold=gold)
    assert score.bullet_discipline == 1.0


def test_length_score_is_symmetric_in_log() -> None:
    gold = "short"  # 5 chars
    twice = "shortshort"  # 10 chars - ratio 2
    half = "sh"  # 2 chars - ratio 0.4
    s_twice = score_actual_against_gold(twice, gold).length_match
    s_half = score_actual_against_gold(half, gold).length_match
    # Both are worse than 1.0; the log-symmetric metric treats 2x and 0.5x
    # as similar penalties.
    assert 0.5 < s_twice < 1.0
    assert 0.5 < s_half < 1.0


def test_length_score_punishes_long_drift() -> None:
    gold = "short"  # 5 chars
    long = "x" * 500  # 100x
    score = score_actual_against_gold(long, gold)
    assert score.length_match < 0.3


def test_aggregate_is_mean_of_three() -> None:
    gold = "Don't know."
    # Banned opener + bullets + long = all three should hit
    actual = "That's a great question. Here are the causes:\n1. One\n2. Two" + " extra" * 50
    score = score_actual_against_gold(actual, gold)
    expected = (score.length_match + score.no_banned_openers + score.bullet_discipline) / 3.0
    assert score.aggregate == round(expected, 3)
