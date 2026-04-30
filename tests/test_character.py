from __future__ import annotations

from pathlib import Path

from harness.character import load_character

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRTON = REPO_ROOT / "character" / "airton"
AIRTON_B = REPO_ROOT / "character" / "airton_b"
AIRTON_C = REPO_ROOT / "character" / "airton_c"
AIRTON_C1 = REPO_ROOT / "character" / "airton_c1"


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


def test_ablated_samples_loaded_separately(tmp_path: Path) -> None:
    """Synthesize a character dir with a voice/ablation.yaml and assert
    the ablated samples load into `ablated_voice_samples` and DO NOT
    appear in `voice_samples` (the tuple the VoiceRetriever reads).
    harness-w49p: carving samples out of canonical for generalization
    eval must make them invisible to retrieval. Renamed from
    "holdout" 2026-04-26 to avoid ATC-phraseology collision."""
    import shutil

    fake = tmp_path / "airton"
    shutil.copytree(AIRTON, fake)
    ablation_path = fake / "voice" / "ablation.yaml"
    ablation_path.write_text(
        "samples:\n  - id: ablated-1\n    prompt: ablated prompt\n    gold: ablated gold\n"
    )

    character = load_character(fake)
    assert len(character.ablated_voice_samples) == 1
    assert character.ablated_voice_samples[0].id == "ablated-1"
    # Invariant: ablated samples are NOT in voice_samples.
    voice_ids = {s.id for s in character.voice_samples}
    assert "ablated-1" not in voice_ids


def test_no_ablation_file_empty_tuple(tmp_path: Path) -> None:
    """Characters without voice/ablation.yaml have an empty ablated
    tuple. Null-safe default for existing personas."""
    character = load_character(AIRTON)
    assert character.ablated_voice_samples == ()


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
    core.yaml and are surfaced on the loaded Character. They live
    on airton_b (the character that actually activates the ab
    adapter at runtime), not on airton proper."""
    character = load_character(AIRTON_B)
    assert character.thought_graph is not None
    assert "Query bd before" in character.thought_graph.query_first
    assert "3 ab-owned beads" in character.thought_graph.budgets
    assert "thought:" in character.thought_graph.thought_labels


def test_system_prompt_includes_thought_graph_guidance() -> None:
    """harness-9qw: the rendered system prompt must carry the three
    workflow guardrails so the model sees them without a runtime
    lookup. Checked on airton_b — airton itself keeps a lighter
    prompt since ab-ops never run there."""
    character = load_character(AIRTON_B)
    prompt = character.system_prompt()

    assert "Thought-graph workflow" in prompt
    assert "Query bd before" in prompt
    assert "3 ab-owned beads" in prompt
    assert "thought:hypothesis" in prompt


def test_system_prompt_injects_today_when_date_supplied() -> None:
    """harness-2o2: model has no temporal anchor unless system_prompt
    carries today's date. Passing `now=` must surface the ISO date and
    weekday near the top so even a small model can judge
    'is this deadline close or months away?'."""
    from datetime import date

    character = load_character(AIRTON)
    prompt = character.system_prompt(now=date(2026, 4, 18))  # a Saturday
    assert "2026-04-18" in prompt
    assert "Saturday" in prompt


def test_system_prompt_omits_date_when_not_supplied() -> None:
    """harness-2o2: date injection is opt-in. Callers that don't pass
    `now` (e.g. voice-eval snapshots) must see no date line, so fixtures
    stay deterministic across days."""
    character = load_character(AIRTON)
    prompt = character.system_prompt()
    assert "Today:" not in prompt


def test_require_search_memory_defaults_to_false() -> None:
    """harness-3uh: the forced-search-memory flag is opt-in. Every
    character without it set in core.yaml should report False so
    their orchestrator path is unchanged."""
    airton = load_character(AIRTON)
    assert airton.require_search_memory is False
    airton_b = load_character(AIRTON_B)
    assert airton_b.require_search_memory is False


def test_require_search_memory_true_on_airton_c1() -> None:
    """harness-3uh: airton_c1 opts in because lay-language JO 7110.65
    queries score below the passive-retrieval floor. The flag drives
    the tool-loop's forced-search-memory injection at CLI + TUI call
    sites."""
    airton_c1 = load_character(AIRTON_C1)
    assert airton_c1.require_search_memory is True


# ---------- harness-a2sa: data-driven branch fields ----------


def test_voice_rewriter_default_persona() -> None:
    """Characters without an explicit voice.rewriter declaration
    default to 'persona' (Airton's two-pass voice rewrite)."""
    assert load_character(AIRTON).voice_rewriter == "persona"
    assert load_character(AIRTON_C1).voice_rewriter == "persona"


def test_voice_rewriter_caveman_on_airton_b() -> None:
    """ab declares voice.rewriter=caveman in core.yaml so the CLI
    wraps its adapter in CavemanRewriter instead of PersonaAdapter."""
    assert load_character(AIRTON_B).voice_rewriter == "caveman"


def test_bd_assignee_loaded_from_core() -> None:
    """airton + airton_b both ship `bd.assignee: airton_b` in core.yaml
    because they share the project bd dir; the assignee tags ab-owned
    writes for budget caps. atc-family characters omit bd: → None."""
    assert load_character(AIRTON).bd_assignee == "airton_b"
    assert load_character(AIRTON_B).bd_assignee == "airton_b"
    assert load_character(AIRTON_C1).bd_assignee is None


def test_bd_exclude_assignee_loaded_from_core() -> None:
    """The same shared-dir characters set exclude_assignee=airton_b
    so default bd reads hide ab's thought-graph rows. atc-family
    characters that own a private bd dir leave it None."""
    assert load_character(AIRTON).bd_exclude_assignee == "airton_b"
    assert load_character(AIRTON_B).bd_exclude_assignee == "airton_b"
    assert load_character(AIRTON_C1).bd_exclude_assignee is None


def test_bd_scope_allowlist_only_on_airton_b() -> None:
    """airton_b narrows reads to professional/personal scopes; every
    other character leaves the allowlist empty (no filter)."""
    assert load_character(AIRTON_B).bd_scope_allowlist == ("professional", "personal")
    assert load_character(AIRTON).bd_scope_allowlist == ()
    assert load_character(AIRTON_C1).bd_scope_allowlist == ()


def test_fetch_url_allowed_hosts_on_atc_family() -> None:
    """airton_c + airton_c1 both ship the same aviation-source
    allowlist; airton + airton_b leave it empty (FetchUrlTool
    unrestricted)."""
    hosts = load_character(AIRTON_C).fetch_url_allowed_hosts
    assert "aviationweather.gov" in hosts
    assert "faa.gov" in hosts
    assert load_character(AIRTON_C1).fetch_url_allowed_hosts == hosts
    assert load_character(AIRTON).fetch_url_allowed_hosts == ()
    assert load_character(AIRTON_B).fetch_url_allowed_hosts == ()


def test_voice_rewriter_rejects_unknown_value(tmp_path: Path) -> None:
    """An unknown rewriter value in core.yaml raises rather than
    silently defaulting — typos surface immediately."""
    import shutil

    import pytest

    char_dir = tmp_path / "broken"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(core_path.read_text() + "\nvoice:\n  rewriter: nonsense\n")
    with pytest.raises(ValueError, match=r"voice.rewriter must be one of"):
        load_character(char_dir)


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
