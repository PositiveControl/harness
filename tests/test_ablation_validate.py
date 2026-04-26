"""Tests for harness.persona.ablation_validate (harness-wrzz, plan #8).

Pure-function tests. The validator owns the structural drift checks
between voice/canonical.yaml and voice/ablation.yaml; the live
end-to-end check happens via scripts/check_ablation_manifest.py
against the airton_c1 character on each push (when the YAML files
appear in the staged diff)."""

from __future__ import annotations

from pathlib import Path

from harness.persona.ablation_validate import (
    AblationProblem,
    AblationProblemKind,
    collect_canonical_ids,
    format_problems,
    validate_ablation,
)


def _write_canonical(path: Path, samples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["version: 2", "samples:"]
    for s in samples:
        lines.append(f"  - id: {s['id']}")
        lines.append(f'    prompt: "{s["prompt"]}"')
        lines.append("    gold: |")
        for body_line in s["gold"].splitlines():
            lines.append(f"      {body_line}")
    path.write_text("\n".join(lines) + "\n")


def _write_ablation(path: Path, samples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["version: 2", "samples:"]
    for s in samples:
        if "id" in s:
            lines.append(f"  - id: {s['id']}")
        else:
            lines.append("  - {}")
            continue
        if "prompt" in s:
            lines.append(f'    prompt: "{s["prompt"]}"')
        if "gold" in s:
            lines.append("    gold: |")
            for body_line in s["gold"].splitlines():
                lines.append(f"      {body_line}")
    path.write_text("\n".join(lines) + "\n")


# ---------- happy path ----------


def test_no_ablation_file_is_clean(tmp_path: Path) -> None:
    """File absent = nothing to validate. Returns empty list."""
    canonical = tmp_path / "canonical.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "p", "gold": "g"}])
    problems = validate_ablation(
        canonical_path=canonical,
        ablation_path=tmp_path / "missing.yaml",
    )
    assert problems == []


def test_clean_manifest_matches_canonical(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(
        canonical,
        [
            {"id": "a", "prompt": "prompt a", "gold": "gold a"},
            {"id": "b", "prompt": "prompt b", "gold": "gold b"},
        ],
    )
    _write_ablation(
        ablation,
        [{"id": "a", "prompt": "prompt a", "gold": "gold a"}],
    )
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert problems == []


def test_clean_manifest_id_only_entry(tmp_path: Path) -> None:
    """ID-only ablation entries (no prompt/gold) are legal — drift
    check skips fields that aren't carried."""
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "p", "gold": "g"}])
    _write_ablation(ablation, [{"id": "a"}])
    assert validate_ablation(canonical_path=canonical, ablation_path=ablation) == []


# ---------- structural drift ----------


def test_missing_id_in_canonical_flagged(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "p", "gold": "g"}])
    _write_ablation(ablation, [{"id": "ghost", "prompt": "p", "gold": "g"}])
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert len(problems) == 1
    assert problems[0].kind == AblationProblemKind.MISSING_ID
    assert problems[0].sample_id == "ghost"


def test_gold_drift_flagged(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "p", "gold": "new gold"}])
    _write_ablation(ablation, [{"id": "a", "prompt": "p", "gold": "old gold"}])
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert len(problems) == 1
    assert problems[0].kind == AblationProblemKind.GOLD_DRIFT
    assert problems[0].sample_id == "a"


def test_prompt_drift_flagged(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "new prompt", "gold": "g"}])
    _write_ablation(ablation, [{"id": "a", "prompt": "old prompt", "gold": "g"}])
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert len(problems) == 1
    assert problems[0].kind == AblationProblemKind.PROMPT_DRIFT


def test_multiple_problems_reported_at_once(tmp_path: Path) -> None:
    """Validator surfaces every structural issue per run — one fix-all
    cycle instead of fix-and-iterate."""
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(
        canonical,
        [{"id": "a", "prompt": "p", "gold": "current gold"}],
    )
    _write_ablation(
        ablation,
        [
            {"id": "ghost", "prompt": "p", "gold": "g"},  # missing in canonical
            {"id": "a", "prompt": "p", "gold": "stale gold"},  # gold drift
        ],
    )
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    kinds = {p.kind for p in problems}
    assert AblationProblemKind.MISSING_ID in kinds
    assert AblationProblemKind.GOLD_DRIFT in kinds


# ---------- empty / parse-error edge cases ----------


def test_empty_manifest_flagged(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "p", "gold": "g"}])
    ablation.write_text("version: 2\nsamples: []\n")
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert any(p.kind == AblationProblemKind.EMPTY_MANIFEST for p in problems)


def test_malformed_yaml_flagged(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "p", "gold": "g"}])
    # Unbalanced flow sequence is a parse error in PyYAML.
    ablation.write_text("[unclosed list\n  - x")
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert any(p.kind == AblationProblemKind.PARSE_ERROR for p in problems)


def test_top_level_not_a_mapping_flagged(tmp_path: Path) -> None:
    """`:::not yaml at all:::` parses as a scalar string in PyYAML —
    no parse exception, but the top-level isn't the dict shape we
    need. Validator still flags as PARSE_ERROR with a clear message."""
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "p", "gold": "g"}])
    ablation.write_text("just a top-level string\n")
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert any(
        p.kind == AblationProblemKind.PARSE_ERROR and "mapping" in p.message for p in problems
    )


def test_entry_without_id_flagged(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    ablation = tmp_path / "ablation.yaml"
    _write_canonical(canonical, [{"id": "a", "prompt": "p", "gold": "g"}])
    ablation.write_text("version: 2\nsamples:\n  - prompt: orphan\n    gold: orphan\n")
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert any(p.kind == AblationProblemKind.MISSING_ID and not p.sample_id for p in problems)


# ---------- collect_canonical_ids ----------


def test_collect_canonical_ids_returns_frozenset(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.yaml"
    _write_canonical(
        canonical,
        [
            {"id": "a", "prompt": "p", "gold": "g"},
            {"id": "b", "prompt": "p", "gold": "g"},
        ],
    )
    ids = collect_canonical_ids(canonical)
    assert isinstance(ids, frozenset)
    assert ids == frozenset({"a", "b"})


def test_collect_canonical_ids_empty_for_missing_file(tmp_path: Path) -> None:
    assert collect_canonical_ids(tmp_path / "absent.yaml") == frozenset()


# ---------- live airton_c1 check ----------


def test_live_airton_c1_manifest_is_clean() -> None:
    """The committed airton_c1 voice/ablation.yaml must validate
    against canonical.yaml — guards against PR-time drift between
    the two files (the same check the pre-push hook runs)."""
    repo = Path(__file__).resolve().parents[1]
    canonical = repo / "character" / "airton_c1" / "voice" / "canonical.yaml"
    ablation = repo / "character" / "airton_c1" / "voice" / "ablation.yaml"
    if not ablation.exists():
        # Manifest absent on this character is the no-op case; the
        # validator treats it as clean. Test still passes.
        problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
        assert problems == []
        return
    problems = validate_ablation(canonical_path=canonical, ablation_path=ablation)
    assert problems == [], format_problems(problems)


# ---------- format_problems ----------


def test_format_problems_handles_empty_list() -> None:
    assert format_problems([]) == ""


def test_format_problems_renders_header_and_body() -> None:
    problems = [
        AblationProblem(
            kind=AblationProblemKind.MISSING_ID,
            sample_id="ghost",
            message="ID not found",
        )
    ]
    out = format_problems(problems)
    assert "missing_id" in out
    assert "ghost" in out
    assert "ID not found" in out


def test_format_problems_omits_id_when_empty() -> None:
    problems = [
        AblationProblem(
            kind=AblationProblemKind.PARSE_ERROR,
            sample_id="",
            message="YAML parse error",
        )
    ]
    out = format_problems(problems)
    assert "[parse_error]:" in out  # No ID slot before the colon.
    assert "YAML parse error" in out
