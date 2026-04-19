from __future__ import annotations

from pathlib import Path

from harness.character import load_character

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON = REPO_ROOT / "character" / "airton"


def test_load_airton_shape() -> None:
    character = load_character(AIRTON)

    assert character.name == "airton"
    assert character.pronouns == "it"
    assert len(character.values) == 5
    assert len(character.taboos) >= 6
    assert len(character.seed_memories) == 5
    assert len(character.voice_samples) >= 20
    assert all(s.principle for s in character.seed_memories)


def test_voice_samples_have_unique_ids() -> None:
    character = load_character(AIRTON)
    ids = [s.id for s in character.voice_samples]
    assert len(ids) == len(set(ids))


def test_voice_sample_counts_sum_to_total(tmp_path: Path) -> None:
    """harness-46z: the introspect tool needs the canonical/captured
    split. Load Airton as-is (no captured corpus in the repo by default)
    and assert the invariant canonical + captured == len(voice_samples).
    Then synthesize a character dir with a captured.yaml and assert the
    split is picked up."""
    character = load_character(AIRTON)
    assert character.canonical_voice_count + character.captured_voice_count == len(
        character.voice_samples
    )
    # Repo carries canonical only by default — bootstrapping sanity.
    assert character.canonical_voice_count >= 20


def test_captured_samples_counted_separately(tmp_path: Path) -> None:
    """Synthesize a mini character directory with both canonical and
    captured samples; assert the counts line up and the merged tuple
    puts canonical first."""
    import shutil

    fake = tmp_path / "airton"
    shutil.copytree(AIRTON, fake)
    captured_path = fake / "voice" / "captured.yaml"
    captured_path.write_text(
        "samples:\n"
        "  - id: captured-1\n"
        "    prompt: test prompt\n"
        "    gold: test gold\n"
        "  - id: captured-2\n"
        "    prompt: another prompt\n"
        "    gold: another gold\n"
    )

    character = load_character(fake)
    assert character.captured_voice_count == 2
    assert character.canonical_voice_count + 2 == len(character.voice_samples)
    # Merged order: canonical first, captured appended.
    tail_ids = [s.id for s in character.voice_samples[-2:]]
    assert tail_ids == ["captured-1", "captured-2"]


def test_self_awareness_is_explicit() -> None:
    character = load_character(AIRTON)
    assert "program on your Mac" in character.self_awareness


def test_self_reference_voice_sample_present() -> None:
    character = load_character(AIRTON)
    ids = {s.id for s in character.voice_samples}
    assert "self_reference" in ids


def test_system_prompt_includes_values_and_taboos() -> None:
    character = load_character(AIRTON)
    prompt = character.system_prompt()

    for v in character.values:
        assert v.rule in prompt
    for t in character.taboos:
        assert t in prompt
    assert character.premise in prompt


def test_system_prompt_embeds_voice_examples() -> None:
    character = load_character(AIRTON)
    prompt = character.system_prompt()

    for sample in character.voice_samples:
        assert sample.prompt in prompt
        assert sample.gold.strip() in prompt
    assert "Voice examples" in prompt
    assert "How you speak" in prompt


def test_thought_graph_loaded() -> None:
    """harness-9qw: ab's thought-graph workflow rules live in
    core.yaml and are surfaced on the loaded Character."""
    character = load_character(AIRTON)
    assert character.thought_graph is not None
    assert "Query bd before" in character.thought_graph.query_first
    assert "3 ab-owned beads" in character.thought_graph.budgets
    assert "thought:" in character.thought_graph.thought_labels


def test_system_prompt_includes_thought_graph_guidance() -> None:
    """harness-9qw: the rendered system prompt must carry the three
    workflow guardrails so the model sees them without a runtime
    lookup."""
    character = load_character(AIRTON)
    prompt = character.system_prompt()

    assert "Thought-graph workflow" in prompt
    assert "Query bd before" in prompt
    assert "3 ab-owned beads" in prompt
    assert "thought:hypothesis" in prompt


def test_system_prompt_excludes_named_voice_examples() -> None:
    character = load_character(AIRTON)
    excluded = frozenset({"self_reference"})
    prompt = character.system_prompt(exclude_example_ids=excluded)

    excluded_sample = next(s for s in character.voice_samples if s.id == "self_reference")
    assert excluded_sample.gold.strip() not in prompt

    for sample in character.voice_samples:
        if sample.id in excluded:
            continue
        assert sample.gold.strip() in prompt
