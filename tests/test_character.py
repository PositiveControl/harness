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


def test_system_prompt_include_style_rules_default_true() -> None:
    """Default behavior: style rules + voice examples are present. The
    chat / voice eval paths rely on this — they want the persona to
    sound consistent across turns."""
    character = load_character(AIRTON)
    prompt = character.system_prompt()
    # Style rules
    assert "Prose by default" in prompt
    assert "1-4 sentences" in prompt
    assert "Don't know" in prompt
    # Voice examples header
    assert "Voice examples" in prompt


def test_system_prompt_include_style_rules_false_strips_chat_register() -> None:
    """harness-d6ak: drive executor passes include_style_rules=False
    so the chat-shaped style rules + voice examples don't bias the
    model toward narration in an agent context. Identity, values,
    taboos, directives, constitution — all preserved as the model's
    generative anchor; only the chat register is gone."""
    character = load_character(AIRTON)
    prompt = character.system_prompt(include_style_rules=False)
    # Style rules gone
    assert "Prose by default" not in prompt
    assert "1-4 sentences" not in prompt
    assert '"Don\'t know"' not in prompt
    assert "How you speak" not in prompt
    # Voice examples gone too (they prime the same register the style
    # rules described — keeping them without the rules would leak the
    # same chat-shaped bias into drive turns).
    assert "Voice examples" not in prompt
    # Identity preserved — the model still has an anchor.
    assert f"You are {character.name}" in prompt
    assert character.premise in prompt
    # Values / taboos / directives preserved — agent still needs to
    # respect the character's hard constraints.
    assert "Values (always defended):" in prompt
    assert "Taboos (always refused):" in prompt
    assert "Directives:" in prompt


def test_require_search_memory_defaults_to_false() -> None:
    """harness-3uh: the forced-search-memory flag is opt-in. Every
    character without it set in core.yaml should report False so
    their orchestrator path is unchanged."""
    airton = load_character(AIRTON)
    assert airton.require_search_memory is False
    airton_b = load_character(AIRTON_B)
    assert airton_b.require_search_memory is False


def test_airton_c1_uses_forced_assemble_context() -> None:
    """harness-j5cs: airton_c1 migrated from forced search_memory to
    forced assemble_context. The contract orchestrator is now the
    primary retrieval path; the flat-chunk search_memory path is
    superseded by tree-shaped contract slots. UngroundedCitationHook
    still gets a tools_ran signal because assemble_context is in
    _GROUNDING_TOOLS."""
    airton_c1 = load_character(AIRTON_C1)
    assert airton_c1.require_search_memory is False
    assert airton_c1.require_assemble_context is True
    assert airton_c1.default_contract_role == "airton_c1"


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


def test_citation_grammar_loaded_on_atc_family() -> None:
    """airton_c + airton_c1 ship a citation_grammar block; airton +
    airton_b leave it None (harness-jaqe)."""
    c1 = load_character(AIRTON_C1)
    assert c1.citation_grammar is not None
    assert c1.citation_grammar.document_name == "JO 7110.65"
    assert c1.citation_grammar.example_anchor == "§1-1-1"
    # The four FAA surface patterns compiled and load-bearing.
    assert len(c1.citation_grammar.surface_patterns) == 4
    # AIM, CFR, JO, AC — at least one of each shape matches.
    text = "Per AIM 4-4-7, 14 CFR §91.155, JO 7110.65 §2-6-4, and AC 90-66B."
    matches = [p for p in c1.citation_grammar.surface_patterns if p.search(text)]
    assert len(matches) == 4

    c = load_character(AIRTON_C)
    assert c.citation_grammar is not None
    assert c.citation_grammar.document_name == "JO 7110.65"


def test_citation_grammar_absent_for_airton_and_ab() -> None:
    """Non-citation-disciplined characters report None — citation
    hooks short-circuit, rewriter no-ops."""
    assert load_character(AIRTON).citation_grammar is None
    assert load_character(AIRTON_B).citation_grammar is None


def test_citation_grammar_rejects_invalid_regex(tmp_path: Path) -> None:
    """Malformed regex in core.yaml citation_grammar surfaces at load
    time with a path-anchored error rather than crashing later."""
    import shutil

    import pytest

    char_dir = tmp_path / "broken"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text()
        + "\ncitation_grammar:\n"
        + "  surface_patterns: ['['] \n"  # unbalanced bracket
        + "  anchor_pattern: '\\d+'\n"
        + "  document_reference: 'foo'\n"
        + "  document_name: 'Foo'\n"
        + "  example_anchor: '§1'\n"
    )
    with pytest.raises(ValueError, match="surface_patterns"):
        load_character(char_dir)


# ---------- harness-qvwq: opt-in domain catchers ----------


def test_catchers_default_empty_for_airton() -> None:
    """Default dev character carries no opt-in catchers — non-corpus
    replies don't need ATC scope checks or ab fabrication detectors."""
    assert load_character(AIRTON).catchers == ()


def test_catchers_ab_fabrication_on_airton_b() -> None:
    """ab opts into ab_fabrication so its ops-receipt imitation
    shapes (Captured. / Remembered: / Bead id:) get caught even
    though they self-gate on tools_ran=False."""
    assert load_character(AIRTON_B).catchers == ("ab_fabrication",)


def test_catchers_atc_roster_on_atc_family() -> None:
    """airton_c + airton_c1 ship the three FAA-shape catchers:
    ambiguous_context, scope_redirect, reserved_squawk_code."""
    assert load_character(AIRTON_C).catchers == (
        "ambiguous_context",
        "scope_redirect",
        "reserved_squawk_code",
    )
    assert load_character(AIRTON_C1).catchers == (
        "ambiguous_context",
        "scope_redirect",
        "reserved_squawk_code",
    )


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


# ---------- document_trees (harness-ejxn) ----------


def test_document_trees_default_empty_for_non_opted_characters() -> None:
    """Characters that don't ship `document_trees:` keep an empty
    tuple. Null-safe default — no migration required for any persona
    that doesn't opt in. airton_c1 (harness-j5cs) and airton_c
    (harness-a22f) opted in and are tested directly below."""
    for char_path in (AIRTON, AIRTON_B):
        character = load_character(char_path)
        assert character.document_trees == (), f"{char_path.name} should default to empty"


def test_airton_c1_ships_jsonl_tree_spec() -> None:
    """harness-j5cs: airton_c1 declares its JO 7110.65 JSONL corpus
    via document_trees:. The CLI session bootstrap builds the per-
    character DocumentTreeStore from this spec."""
    character = load_character(AIRTON_C1)
    assert len(character.document_trees) == 1
    spec = character.document_trees[0]
    assert spec.name == "jo_7110_65"
    assert spec.source_format == "jsonl"
    assert spec.jsonl_depth_fields == ("chapter", "parent_section", "section")
    assert spec.jsonl_heading_prefixes == ("Chapter ", "§", "")
    assert spec.jsonl_leaf_heading_field == "title"
    if not spec.source_path.exists():
        # FAA corpus JSONL is not checked into the repo (heavy +
        # licensed). Spec wiring still verified; skip the on-disk
        # check when the corpus hasn't been materialized locally.
        import pytest

        pytest.skip(f"corpus not materialized: {spec.source_path.name}")


def test_airton_c_ships_five_jsonl_tree_specs() -> None:
    """harness-a22f: airton_c declares 5 hierarchical corpora via
    document_trees: (JO, AIM, CFR Vol 1, CFR Vol 2, PCG). PHAK is the
    sixth source but stays in episodic per the harness-by1j hybrid
    decision."""
    character = load_character(AIRTON_C)
    names = [spec.name for spec in character.document_trees]
    assert names == ["JO_7110.65", "AIM", "CFR_14_Vol1", "CFR_14_Vol2", "PCG"]
    # All five share the same jsonl shape except PCG which is 2-level
    # (chapter == letter, section == glossary entry).
    by_name = {spec.name: spec for spec in character.document_trees}
    assert by_name["JO_7110.65"].jsonl_depth_fields == (
        "chapter",
        "parent_section",
        "section",
    )
    assert by_name["PCG"].jsonl_depth_fields == ("chapter", "section")
    # The leaf heading comes from the `title` field for every source.
    for spec in character.document_trees:
        assert spec.jsonl_leaf_heading_field == "title"
        if not spec.source_path.exists():
            import pytest

            pytest.skip(f"corpus not materialized: {spec.source_path.name}")
    # Forced-call flags wired.
    assert character.require_search_memory is False
    assert character.require_assemble_context is True
    assert character.default_contract_role == "airton_c"


def test_document_trees_loads_markdown_spec(tmp_path: Path) -> None:
    """Synthesize a character dir with a `document_trees:` block and
    assert the tuple shape, resolved path, and format."""
    import shutil

    char_dir = tmp_path / "with_trees"
    shutil.copytree(AIRTON, char_dir)
    seed_dir = char_dir / "seed_documents"
    seed_dir.mkdir()
    (seed_dir / "manual.md").write_text("# Title\n\nbody\n", encoding="utf-8")
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n"
        "  - name: manual\n"
        "    description: Test manual\n"
        "    source: seed_documents/manual.md\n"
    )

    character = load_character(char_dir)
    assert len(character.document_trees) == 1
    spec = character.document_trees[0]
    assert spec.name == "manual"
    assert spec.description == "Test manual"
    # Path is resolved against the character dir.
    assert spec.source_path == (char_dir / "seed_documents" / "manual.md").resolve()
    # Format defaults to markdown.
    assert spec.source_format == "markdown"


def test_document_trees_accepts_jsonl_format(tmp_path: Path) -> None:
    """harness-k38k: JSONL specs now require a `jsonl:` block with at
    least `depth_fields`. The format itself is still accepted at the
    top-level; the per-corpus config lives nested."""
    import shutil

    char_dir = tmp_path / "jsonl_char"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n"
        "  - name: corpus\n"
        "    description: chunked corpus\n"
        "    source: corpus/chunks/source.jsonl\n"
        "    format: jsonl\n"
        "    jsonl:\n"
        "      depth_fields: [chapter, section]\n"
    )

    character = load_character(char_dir)
    assert character.document_trees[0].source_format == "jsonl"
    assert character.document_trees[0].jsonl_depth_fields == ("chapter", "section")


def test_document_trees_rejects_unknown_format(tmp_path: Path) -> None:
    import shutil

    import pytest

    char_dir = tmp_path / "bad_format"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n"
        "  - name: m\n"
        "    description: d\n"
        "    source: x.md\n"
        "    format: pdf\n"
    )
    with pytest.raises(ValueError, match=r"document_trees\[0\] format must be one of"):
        load_character(char_dir)


def test_document_trees_rejects_missing_fields(tmp_path: Path) -> None:
    import shutil

    import pytest

    char_dir = tmp_path / "missing_fields"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n  - description: missing name + source\n"
    )
    with pytest.raises(ValueError, match=r"document_trees\[0\] needs `name` \+ `source`"):
        load_character(char_dir)


def test_document_trees_rejects_duplicate_name(tmp_path: Path) -> None:
    """Two specs with the same `name:` would collide in the underlying
    DocumentTreeStore (one document per name). Catch it at load time
    so the conflict surfaces in core.yaml, not in a runtime SQL error."""
    import shutil

    import pytest

    char_dir = tmp_path / "dupe_names"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n"
        "  - name: same\n"
        "    description: first\n"
        "    source: a.md\n"
        "  - name: same\n"
        "    description: second\n"
        "    source: b.md\n"
    )
    with pytest.raises(ValueError, match=r"document_trees\[1\] duplicate name 'same'"):
        load_character(char_dir)


def test_document_trees_rejects_non_list(tmp_path: Path) -> None:
    import shutil

    import pytest

    char_dir = tmp_path / "wrong_type"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(core_path.read_text() + "\ndocument_trees: not a list\n")
    with pytest.raises(ValueError, match=r"`document_trees` must be a list"):
        load_character(char_dir)


# ---------- document_trees JSONL config (harness-k38k) ----------


def test_document_trees_jsonl_spec_loads_full_config(tmp_path: Path) -> None:
    """JSONL spec with full jsonl: block populates all the per-corpus
    knobs on DocumentTreeSpec — depth_fields, heading_prefixes, the
    leaf-heading field. The character bootstrap can then dispatch to
    iter_jsonl_nodes without code-level config."""
    import shutil

    char_dir = tmp_path / "jsonl_full"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n"
        "  - name: jo_7110_65\n"
        "    description: ATC corpus\n"
        "    source: corpus/jo.jsonl\n"
        "    format: jsonl\n"
        "    jsonl:\n"
        "      depth_fields: [chapter, parent_section, section]\n"
        "      heading_prefixes: ['Chapter ', '§', '']\n"
        "      leaf_heading_field: title\n"
    )

    character = load_character(char_dir)
    spec = character.document_trees[0]
    assert spec.source_format == "jsonl"
    assert spec.jsonl_depth_fields == ("chapter", "parent_section", "section")
    assert spec.jsonl_heading_prefixes == ("Chapter ", "§", "")
    assert spec.jsonl_leaf_heading_field == "title"
    # Defaults preserved when caller doesn't override.
    assert spec.jsonl_body_field == "body"
    assert spec.jsonl_chunk_index_field == "chunk_index"


def test_document_trees_jsonl_requires_depth_fields(tmp_path: Path) -> None:
    """JSONL format without a `jsonl:` block (or with no depth_fields)
    raises at YAML load time so the misconfig surfaces immediately
    rather than at session-start ingest."""
    import shutil

    import pytest

    char_dir = tmp_path / "jsonl_bad"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n"
        "  - name: corpus\n"
        "    description: missing config\n"
        "    source: corpus/source.jsonl\n"
        "    format: jsonl\n"
    )
    with pytest.raises(ValueError, match=r"jsonl format requires"):
        load_character(char_dir)


def test_document_trees_jsonl_rejects_empty_depth_fields(tmp_path: Path) -> None:
    import shutil

    import pytest

    char_dir = tmp_path / "jsonl_empty_df"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n"
        "  - name: corpus\n"
        "    description: bad config\n"
        "    source: corpus/source.jsonl\n"
        "    format: jsonl\n"
        "    jsonl:\n"
        "      depth_fields: []\n"
    )
    with pytest.raises(ValueError, match=r"jsonl.depth_fields must be a non-empty list of strings"):
        load_character(char_dir)


def test_require_assemble_context_defaults_false_for_non_opted() -> None:
    """harness-jkmk: every character without an explicit opt-in keeps
    the default False so its orchestrator path is unchanged. airton_c1
    (harness-j5cs) and airton_c (harness-a22f) opted in and are covered
    by dedicated tests."""
    for char_path in (AIRTON, AIRTON_B):
        character = load_character(char_path)
        assert character.require_assemble_context is False
        assert character.default_contract_role is None


def test_require_assemble_context_loads_with_role(tmp_path: Path) -> None:
    """Setting both flags loads them onto the Character. The
    orchestrator wires the role into the forced-call argument."""
    import shutil

    char_dir = tmp_path / "char_with_forced_ctx"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text()
        + "\nrequire_assemble_context: true\n"
        + "default_contract_role: my_role\n"
    )
    character = load_character(char_dir)
    assert character.require_assemble_context is True
    assert character.default_contract_role == "my_role"


def test_require_assemble_context_without_role_raises(tmp_path: Path) -> None:
    """harness-jkmk: setting the flag without naming a role surfaces
    at YAML load time. No silent no-op at runtime."""
    import shutil

    import pytest

    char_dir = tmp_path / "char_bad_forced_ctx"
    shutil.copytree(AIRTON, char_dir)
    core_path = char_dir / "core.yaml"
    core_path.write_text(core_path.read_text() + "\nrequire_assemble_context: true\n")
    with pytest.raises(ValueError, match=r"needs `default_contract_role`"):
        load_character(char_dir)


def test_document_trees_markdown_ignores_jsonl_block(tmp_path: Path) -> None:
    """A jsonl: block under a markdown spec is silently allowed (no-op).
    Lets a user keep a stale block around when they flip a corpus from
    jsonl back to markdown without forcing a YAML cleanup."""
    import shutil

    char_dir = tmp_path / "md_with_jsonl_block"
    shutil.copytree(AIRTON, char_dir)
    (char_dir / "seed_documents").mkdir()
    (char_dir / "seed_documents" / "doc.md").write_text("# Title\n\nbody\n", encoding="utf-8")
    core_path = char_dir / "core.yaml"
    core_path.write_text(
        core_path.read_text() + "\ndocument_trees:\n"
        "  - name: doc\n"
        "    description: markdown\n"
        "    source: seed_documents/doc.md\n"
        "    format: markdown\n"
        "    jsonl:\n"
        "      depth_fields: [stale]\n"
    )
    character = load_character(char_dir)
    spec = character.document_trees[0]
    assert spec.source_format == "markdown"
    # The jsonl_depth_fields field reflects what was declared (parser
    # parses-but-doesn't-enforce-for-markdown). That's OK — markdown
    # adapter doesn't read it.
    assert spec.jsonl_depth_fields == ("stale",)
