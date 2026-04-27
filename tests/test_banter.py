"""Tests for harness.persona.banter (harness-jmts).

Detector + corpus + ladder. The actual hook integration (router intent
in harness-q7ff and the streak-tracker / pre-model intercept in
harness-vadq) is tested separately when those land — this file pins
only the primitives.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from harness.persona.banter import (
    REDIRECT_LADDER,
    BanterCorpus,
    BanterStreakTracker,
    JokeEntry,
    is_banter_prompt,
    load_default_tracker,
)

# ---------- detector positives ----------


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
        "\n\n",
        "test",
        "test.",
        "TEST",
        "ping",
        "ping?",
        "this page intentionally left blank",
        "This page is intentionally left blank.",
        "page intentionally left blank",  # no leading "this"
        "are you there",
        "Are you alive?",
        "are you awake",
        "lorem ipsum dolor sit amet",
        "foo bar",
        "hello world",
        "aaaa",
        "aaaaaa",
        ".....",
        "!!!!",
    ],
)
def test_detector_fires_on_smartass_prompts(prompt: str) -> None:
    assert is_banter_prompt(prompt) is True


def test_detector_fires_on_repeat_of_prior_prompt() -> None:
    prior = "what is the squawk for hijacking?"
    assert is_banter_prompt(prior, prior_prompt=prior) is True


def test_detector_fires_on_short_prompt_without_anchor() -> None:
    # 3 tokens, no domain anchor.
    assert is_banter_prompt("ok cool whatever") is True


# ---------- detector negatives ----------


@pytest.mark.parametrize(
    "prompt",
    [
        "what does §4-5-1 say about vertical separation",
        "explain the wake turbulence minima for a heavy followed by a light",
        "controller phraseology for missed approach clearance",
        "what is the squawk code for hijacking",
        "tell me about RVSM and altitude assignments",
        "how does the pilot read back a hold short instruction",
        "section 3-10-4 same runway separation",
        "§10-2-5",
    ],
)
def test_detector_passes_real_domain_questions(prompt: str) -> None:
    assert is_banter_prompt(prompt) is False


@pytest.mark.parametrize(
    "prompt",
    [
        "hi",
        "hey",
        "hey!",
        "hello",
        "hello.",
        "morning",
        "afternoon",
        "evening",
        "sup",
        "yo",
        "howdy",
        "heyyy",
    ],
)
def test_detector_passes_greetings(prompt: str) -> None:
    # Greetings get a separate path, not a joke.
    assert is_banter_prompt(prompt) is False


def test_detector_does_not_fire_on_short_prompt_with_anchor() -> None:
    # 4 tokens but contains a domain anchor → real question, exit clean.
    assert is_banter_prompt("controller separation minima please") is False


def test_detector_does_not_fire_on_section_only_prompt() -> None:
    assert is_banter_prompt("§4-5-1") is False


def test_detector_repeat_check_only_on_exact_match() -> None:
    # Different prompt — should fall through to other rules. This one
    # is a real domain question, so False.
    assert (
        is_banter_prompt(
            "what does the controller say next",
            prior_prompt="what is a hold short clearance",
        )
        is False
    )


# ---------- corpus ----------


@pytest.fixture
def corpus_path() -> Path:
    repo_root = Path(__file__).parent.parent
    return repo_root / "character" / "airton_c1" / "jokes.yaml"


def test_corpus_loads_from_airton_c1(corpus_path: Path) -> None:
    corpus = BanterCorpus.load(corpus_path)
    assert len(corpus.jokes) == 20
    for joke in corpus.jokes:
        assert isinstance(joke, JokeEntry)
        assert joke.id
        assert joke.text
        # ≤140 chars target — soft cap, hard fail at 200.
        assert len(joke.text) < 200, f"{joke.id} too long: {len(joke.text)} chars"
        assert isinstance(joke.tags, frozenset)


def test_corpus_ids_are_unique(corpus_path: Path) -> None:
    corpus = BanterCorpus.load(corpus_path)
    ids = [j.id for j in corpus.jokes]
    assert len(ids) == len(set(ids))


def test_pick_excludes_seen(corpus_path: Path) -> None:
    corpus = BanterCorpus.load(corpus_path)
    rng = random.Random(42)
    seen = {corpus.jokes[0].id, corpus.jokes[1].id}
    picked = corpus.pick(seen, rng=rng)
    assert picked.id not in seen


def test_pick_reshuffles_when_corpus_exhausted(corpus_path: Path) -> None:
    corpus = BanterCorpus.load(corpus_path)
    rng = random.Random(42)
    seen = {j.id for j in corpus.jokes}
    # All seen — must still return a joke (from the full corpus).
    picked = corpus.pick(seen, rng=rng)
    assert picked.id in seen  # one of the (now-reshuffled) corpus


def test_pick_with_empty_seen_picks_anything(corpus_path: Path) -> None:
    corpus = BanterCorpus.load(corpus_path)
    rng = random.Random(42)
    picked = corpus.pick(set(), rng=rng)
    assert picked in corpus.jokes


def test_pick_on_empty_corpus_raises() -> None:
    corpus = BanterCorpus(jokes=())
    with pytest.raises(ValueError, match="empty"):
        corpus.pick(set())


def test_load_round_trip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "jokes.yaml"
    yaml_path.write_text(
        "- id: one\n"
        "  text: short joke one\n"
        "  tags: [approach, radio]\n"
        "- id: two\n"
        "  text: short joke two\n"
        "  tags: []\n",
        encoding="utf-8",
    )
    corpus = BanterCorpus.load(yaml_path)
    assert len(corpus.jokes) == 2
    assert corpus.jokes[0].id == "one"
    assert corpus.jokes[0].tags == frozenset({"approach", "radio"})
    assert corpus.jokes[1].tags == frozenset()


def test_load_handles_missing_tags_field(tmp_path: Path) -> None:
    yaml_path = tmp_path / "jokes.yaml"
    yaml_path.write_text("- id: solo\n  text: just a joke\n", encoding="utf-8")
    corpus = BanterCorpus.load(yaml_path)
    assert corpus.jokes[0].tags == frozenset()


# ---------- ladder ----------


def test_redirect_ladder_has_three_tiers() -> None:
    assert len(REDIRECT_LADDER) == 3
    for line in REDIRECT_LADDER:
        assert isinstance(line, str)
        assert line  # non-empty


# ---------- streak tracker ----------


def _two_joke_corpus() -> BanterCorpus:
    return BanterCorpus(
        jokes=(
            JokeEntry(id="alpha", text="alpha joke"),
            JokeEntry(id="beta", text="beta joke"),
        )
    )


def test_tracker_first_turn_returns_a_joke() -> None:
    tracker = BanterStreakTracker(corpus=_two_joke_corpus(), rng=random.Random(42))
    out = tracker.consume()
    assert out in {"alpha joke", "beta joke"}
    assert tracker.streak == 1


def test_tracker_cycles_joke_then_three_redirects() -> None:
    tracker = BanterStreakTracker(corpus=_two_joke_corpus(), rng=random.Random(42))
    turn1 = tracker.consume()
    turn2 = tracker.consume()
    turn3 = tracker.consume()
    turn4 = tracker.consume()
    assert turn1 in {"alpha joke", "beta joke"}
    assert turn2 == REDIRECT_LADDER[0]
    assert turn3 == REDIRECT_LADDER[1]
    assert turn4 == REDIRECT_LADDER[2]


def test_tracker_fifth_turn_returns_another_joke() -> None:
    tracker = BanterStreakTracker(corpus=_two_joke_corpus(), rng=random.Random(42))
    for _ in range(4):
        tracker.consume()
    turn5 = tracker.consume()
    assert turn5 in {"alpha joke", "beta joke"}


def test_tracker_real_prompt_resets_cycle() -> None:
    tracker = BanterStreakTracker(corpus=_two_joke_corpus(), rng=random.Random(42))
    tracker.consume()  # joke
    tracker.consume()  # redirect tier 1
    assert tracker.streak == 2
    tracker.note_real_prompt()
    assert tracker.streak == 0
    # Next banter consume → joke again, not redirect tier 2.
    out = tracker.consume()
    assert out in {"alpha joke", "beta joke"}


def test_tracker_does_not_repeat_joke_within_session() -> None:
    tracker = BanterStreakTracker(corpus=_two_joke_corpus(), rng=random.Random(42))
    first = tracker.consume()
    # Burn through three redirects to land on next joke turn.
    for _ in range(3):
        tracker.consume()
    second = tracker.consume()
    # Two-joke corpus: the first and second jokes must differ — that's
    # the whole point of the no-repeat-with-reshuffle pick.
    assert first != second


def test_tracker_reshuffles_after_corpus_exhausted() -> None:
    tracker = BanterStreakTracker(corpus=_two_joke_corpus(), rng=random.Random(42))
    seen_jokes = set()
    for _ in range(3):  # 3 joke cycles → consumes both + reshuffles
        seen_jokes.add(tracker.consume())
        for _ in range(3):
            tracker.consume()
    # After exhausting the 2-joke corpus once, the tracker still
    # produces jokes — never raises, never goes silent.
    assert len(seen_jokes) >= 2  # both unique jokes appeared


def test_load_default_tracker_returns_none_when_no_jokes_yaml(tmp_path: Path) -> None:
    # tmp_path is empty — no jokes.yaml.
    assert load_default_tracker(tmp_path) is None


def test_load_default_tracker_loads_when_jokes_yaml_present(tmp_path: Path) -> None:
    (tmp_path / "jokes.yaml").write_text(
        "- id: solo\n  text: only joke\n",
        encoding="utf-8",
    )
    tracker = load_default_tracker(tmp_path)
    assert tracker is not None
    assert tracker.streak == 0
    assert len(tracker.corpus.jokes) == 1


def test_load_default_tracker_loads_airton_c1_corpus() -> None:
    repo_root = Path(__file__).parent.parent
    tracker = load_default_tracker(repo_root / "character" / "airton_c1")
    assert tracker is not None
    assert len(tracker.corpus.jokes) == 20
