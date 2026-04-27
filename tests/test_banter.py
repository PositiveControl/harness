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
    JokeEntry,
    is_banter_prompt,
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
