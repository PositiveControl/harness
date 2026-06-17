"""Tests for the cross-workspace premise check (harness-y0hu5).

The guard parks a bead whose text references files that exist nowhere
under the workspace (the harness-rorj mis-wire: a snake_game.py bead
under the GTAII epic). Every test here pins one of the conservative
guards — the check must bias toward driving, never park legitimate
new-file or partially-grounded work.
"""

from __future__ import annotations

from pathlib import Path

from harness.driver.premise_guard import (
    _extract_path_tokens,
    already_implemented_markers,
    flag_blocked_names_own_deliverable,
    flag_blocked_names_withheld_tool,
    referenced_missing_files,
)

_EDITORS = ("edit_file", "write_file", "stream_edit")

_RORJ_TEXT = (
    "Add score tracking and on-screen score display. PLAN.md Phase 3.2 "
    "calls for score tracking but snake_game.py has no score state."
)


def _gta_workspace(tmp_path: Path) -> Path:
    (tmp_path / "game.js").write_text("// gta\n")
    (tmp_path / "index.html").write_text("<canvas></canvas>\n")
    return tmp_path


def test_trips_when_all_referenced_files_absent(tmp_path: Path) -> None:
    """The rorj scenario: two foreign-file references, populated
    workspace, zero matches → missing list returned."""
    ws = _gta_workspace(tmp_path)
    missing = referenced_missing_files(_RORJ_TEXT, ws)
    assert missing == ["PLAN.md", "snake_game.py"]


def test_drives_when_any_referenced_file_exists(tmp_path: Path) -> None:
    """ALL-absent is required — one grounded reference means the bead
    belongs here even if a sibling token is missing."""
    ws = _gta_workspace(tmp_path)
    text = "Add drawCop to game.js, mirroring the pattern in cop_ai.js"
    assert referenced_missing_files(text, ws) is None


def test_drives_on_single_token(tmp_path: Path) -> None:
    """A creation bead names exactly its one deliverable — a single
    absent token must not park."""
    ws = _gta_workspace(tmp_path)
    assert referenced_missing_files("Create utils.js with vector helpers", ws) is None


def test_drives_on_from_scratch_workspace(tmp_path: Path) -> None:
    """No source files yet → the named-but-absent files ARE the work
    (a §1-style skeleton bead), not a mis-wire."""
    assert referenced_missing_files("Create index.html + game.js skeleton", tmp_path) is None


def test_drives_when_no_path_tokens(tmp_path: Path) -> None:
    ws = _gta_workspace(tmp_path)
    assert referenced_missing_files("Tune cop chase speed and wanted decay", ws) is None


def test_basename_match_anywhere_counts_as_present(tmp_path: Path) -> None:
    """A reference like `src/engine/physics.py` is grounded if
    physics.py exists anywhere under the workspace — bead text often
    drops or rewrites directory prefixes."""
    ws = _gta_workspace(tmp_path)
    (ws / "lib").mkdir()
    (ws / "lib" / "physics.py").write_text("pass\n")
    text = "Fix the integrator in src/engine/physics.py per NOTES.md"
    assert referenced_missing_files(text, ws) is None


def test_hidden_dirs_do_not_satisfy_premise(tmp_path: Path) -> None:
    """Driver scratch under .harness/ must not ground a reference."""
    ws = _gta_workspace(tmp_path)
    scratch = ws / ".harness" / "loop_runs"
    scratch.mkdir(parents=True)
    (scratch / "snake_game.py").write_text("pass\n")
    (scratch / "PLAN.md").write_text("plan\n")
    missing = referenced_missing_files(_RORJ_TEXT, ws)
    assert missing == ["PLAN.md", "snake_game.py"]


def test_extract_skips_urls_and_dedupes() -> None:
    text = (
        "See https://example.com/docs/page.html — update game.js and game.js again, plus cop_ai.js"
    )
    assert _extract_path_tokens(text) == ["game.js", "cop_ai.js"]


def test_extract_caps_token_count() -> None:
    text = " ".join(f"file{i}.py" for i in range(20))
    assert len(_extract_path_tokens(text)) == 8


# --- mid-turn flag_blocked deliverable check (harness-0t2f9) ----------

_PED_DELIVERABLE = (
    "Spawn pedestrians in game.js\n"
    "Add pedestrian spawning and wandering logic so peds appear on the "
    "sidewalk and walk around.\n"
    "Acceptance: pedestrians spawn periodically in game.js and wander."
)


def test_flag_blocked_rejected_when_missing_is_own_deliverable() -> None:
    """harness-4s2bb: a flag whose `missing` restates the bead's §6a
    acceptance criteria is the deliverable, not an upstream block."""
    assert flag_blocked_names_own_deliverable(
        "pedestrian spawning and wandering logic in game.js",
        _PED_DELIVERABLE,
    )


def test_flag_blocked_honored_for_genuine_upstream_precondition() -> None:
    """A concrete foreign symbol the bead presupposes shares few
    deliverable tokens → flag stands, the bead still parks PREMISE_UNMET."""
    siren_deliverable = (
        "Add police siren audio near the player\n"
        "When a cop is within range, play a looping siren.\n"
        "Acceptance: siren is audible as a cop approaches."
    )
    assert not flag_blocked_names_own_deliverable(
        "drawCop() render function — cops are never drawn, nothing to attach audio to",
        siren_deliverable,
    )


def test_flag_blocked_single_token_missing_never_trips() -> None:
    """One content word carries too little signal — bias toward parking
    (honoring the flag) rather than rejecting on a bare symbol name."""
    assert not flag_blocked_names_own_deliverable("spawnPed", _PED_DELIVERABLE)


def test_withheld_tool_detected_on_vsv_shape() -> None:
    # loop_run=065ff3c1, harness-vsv verbatim.
    missing = "edit_file tool not available — I cannot execute the edit_file tool"
    assert flag_blocked_names_withheld_tool(missing, _EDITORS) == "edit_file"


def test_withheld_tool_detects_each_editor() -> None:
    assert flag_blocked_names_withheld_tool("write_file is missing", _EDITORS) == "write_file"
    assert flag_blocked_names_withheld_tool("no stream_edit here", _EDITORS) == "stream_edit"


def test_withheld_tool_none_for_genuine_premise() -> None:
    # A real upstream precondition naming no editor tool stays honored.
    assert (
        flag_blocked_names_withheld_tool(
            "drawCop() render function from cop.js — never defined", _EDITORS
        )
        is None
    )


def test_withheld_tool_word_boundary_no_false_positive() -> None:
    # "edit the foo config file" must not match the editor token edit_file.
    assert flag_blocked_names_withheld_tool("edit the foo config file first", _EDITORS) is None


def test_flag_blocked_empty_deliverable_text_never_trips() -> None:
    """No bead text supplied → check disabled, flag honored."""
    assert not flag_blocked_names_own_deliverable("pedestrian spawning logic in game.js", "")


# --- already-implemented premise (harness-8k6lp) ----------------------------

# The l3tgq REOPENED-note shape: prose that asserts specific dotted-member
# assignments are missing, when game.js already has them.
_L3TGQ_NOTES = (
    "§11b Traffic — NPC AI. REOPENED: audit vs game.js. THREE gaps: "
    "1. NO position integration — there is no car.x += Math.cos(car.angle)*"
    "car.speed*dt (nor car.y). Cars accelerate but never move. "
    "2. car.intent is READ (angleDiff = car.intent - car.angle) but NEVER "
    "assigned anywhere (no '.intent =' in the file)."
)


def _traffic_workspace(tmp_path: Path, *, with_intent: bool = True) -> Path:
    """A game.js that integrates position and (optionally) assigns intent —
    the spacing differs from the bead prose on purpose."""
    intent_line = "      car.intent = chosenDirection;\n" if with_intent else ""
    (tmp_path / "game.js").write_text(
        "for (let i = 0; i < traffic.length; i++) {\n"
        "  const car = traffic[i];\n"
        "  car.x  +=  Math.cos(car.angle) * car.speed * dt;\n"
        "  car.y += Math.sin(car.angle) * car.speed * dt;\n"
        f"{intent_line}"
        "}\n"
    )
    return tmp_path


def test_parks_when_asserted_missing_code_is_present(tmp_path: Path) -> None:
    """Both markers the notes say are missing (car.x +=, .intent =) exist in
    game.js — premise already met, park."""
    ws = _traffic_workspace(tmp_path)
    markers = already_implemented_markers(_L3TGQ_NOTES, ws)
    assert markers is not None
    stripped = {m.replace(" ", "") for m in markers}
    assert "car.x+=" in stripped
    assert ".intent=" in stripped


def test_drives_when_one_marker_still_absent(tmp_path: Path) -> None:
    """Only position integration landed; .intent = is still missing → the
    second gap is real work, so drive (don't park)."""
    ws = _traffic_workspace(tmp_path, with_intent=False)
    assert already_implemented_markers(_L3TGQ_NOTES, ws) is None


def test_drives_when_single_marker(tmp_path: Path) -> None:
    """One negated marker is too little signal — never parks on it alone."""
    ws = _traffic_workspace(tmp_path)
    text = "there is no car.x += cos(angle)*speed*dt"
    assert already_implemented_markers(text, ws) is None


def test_drives_on_spec_without_negation(tmp_path: Path) -> None:
    """A bead that SPECS the assignments without asserting they're missing
    must not park even though the code is present — no negation cue, so the
    markers aren't treated as gap claims (this would be an extend/verify bead)."""
    ws = _traffic_workspace(tmp_path)
    text = "Integrate motion: set car.x += vx and assign car.intent = dir at junctions."
    assert already_implemented_markers(text, ws) is None


def test_drives_on_empty_workspace(tmp_path: Path) -> None:
    """No source files → blob empty → drive (the from-scratch case)."""
    assert already_implemented_markers(_L3TGQ_NOTES, tmp_path) is None


def test_comparison_operators_not_treated_as_assignment(tmp_path: Path) -> None:
    """`a.b == c` / `a.b >= c` reads are not assignment markers, so a note
    full of comparisons against present code does not park."""
    (tmp_path / "game.js").write_text("if (car.speed >= maxSpeed && car.x == 0) {}\n")
    text = "bug: no car.speed >= maxSpeed clamp and no car.x == 0 reset"
    assert already_implemented_markers(text, tmp_path) is None


def test_hidden_dir_source_does_not_satisfy_premise(tmp_path: Path) -> None:
    """Code living only under a hidden dir (.harness scratch) must not count
    as the workspace having implemented the markers."""
    hidden = tmp_path / ".harness" / "scratch"
    hidden.mkdir(parents=True)
    (hidden / "game.js").write_text("car.x += Math.cos(a)*s*dt;\ncar.intent = dir;\n")
    assert already_implemented_markers(_L3TGQ_NOTES, tmp_path) is None
