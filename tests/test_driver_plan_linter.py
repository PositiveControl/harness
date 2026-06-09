"""Tests for driver/plan_linter.py — over-scoped-bead detection
(harness-bpix)."""

from __future__ import annotations

from pathlib import Path

from harness.driver.plan_linter import (
    SUBSECTION_THRESHOLD,
    BeadComplexity,
    is_structural_bead,
    score_bead,
)

# Shaped like §7 Police: many sub-sections, long, bullet-heavy → the
# kind that walled the 32B coder.
_OVERSCOPED = """Implement police per §7:
  - §7.1 state per cop: maxSpeed 200, acceleration 160, turnRate 2.0
  - §7.2 spawn rule: alive cop count == wanted; spawn >=200px from player
  - §7.3 chase AI: steer via angleDiff, throttle by distance, building collision
  - §7.4 visual: dark-navy body, light bar alternating per §3.3
  - §7.5 siren: bonus, skip
  - extra detail line one
  - extra detail line two
"""
_OVERSCOPED_ACCEPT = (
    "Police spawn within ~5s of wanted ticking up; alive count tracks wanted. "
    "Each cop chases the player. Cops bounce off B tiles. Ramming drains health. "
    "Visual is dark-navy with a flashing light bar."
)

# Shaped like §7a (a sub-bead after splitting): one concern, short.
_RIGHTSIZED = """Add a cop-state factory and spawn function.
Spawn cops at road tiles >=200px from the player, never in a building.
"""
_RIGHTSIZED_ACCEPT = "With wanted>0, that many cops spawn >=200px from player; loads clean."


def test_overscoped_bead_is_flagged_with_reasons() -> None:
    c = score_bead("harness-x", "§7 Police", _OVERSCOPED, _OVERSCOPED_ACCEPT)
    assert c.flagged is True
    assert c.subsection_refs >= SUBSECTION_THRESHOLD
    assert c.reasons  # non-empty, explains why
    assert any("sub-section" in r for r in c.reasons)


def test_rightsized_bead_not_flagged() -> None:
    c = score_bead("harness-y", "§7a Police spawn", _RIGHTSIZED, _RIGHTSIZED_ACCEPT)
    assert c.flagged is False
    assert c.reasons == ()


def test_subsection_refs_deduped() -> None:
    # §7.1 cited twice counts once.
    text = "do §7.1 then §7.1 again, and §7.2, and §7.3"
    c = score_bead("z", "t", text)
    assert c.subsection_refs == 3


def test_cross_section_refs_excludes_subsections() -> None:
    # §3 and §4 are cross-section coupling; §7.1/§4.5 are sub-sections.
    c = score_bead("z", "t", "reuse §3 drawing per §3.3, collide per §4.5, see §4")
    assert c.cross_section_refs == 2  # §3, §4
    assert c.subsection_refs == 2  # §3.3, §4.5


def test_length_alone_flags() -> None:
    long_desc = "x" * 1600  # no sub-sections, but over the char threshold
    c = score_bead("z", "t", long_desc)
    assert c.flagged is True
    assert any("char description" in r for r in c.reasons)


def test_acceptance_clause_count() -> None:
    acc = (
        "Cars spawn at startup; they cruise toward maxSpeed. "
        "They turn at intersections. They stop behind other cars AND bounce off buildings."
    )
    c = score_bead("z", "t", "short body", acc)
    assert c.acceptance_clauses >= 4
    assert c.flagged is True


def test_bullet_heavy_short_subsection_flags_on_bullets() -> None:
    desc = "Do many things:\n" + "\n".join(f"- concern {i}" for i in range(7))
    c = score_bead("z", "t", desc)
    assert c.bullets == 7
    assert c.flagged is True
    assert any("bullet" in r for r in c.reasons)


# --- is_structural_bead (harness-1kd9t: bypass-TDD signal) --------------


def test_skeleton_bead_is_structural() -> None:
    # The §1/1kd9t shape: a project-shape skeleton. Acceptance mentions
    # "every frame" (behavioral wording) but the bead IS a skeleton — the
    # marker carries it; no behavioral veto.
    title = "§1 Project shape — index.html + game.js skeleton"
    desc = "Bootstrap the two source files. index.html is the exact skeleton in §1.5."
    accept = "index.html matches the §1.5 skeleton verbatim. gameLoop fires every frame."
    assert is_structural_bead(title, desc, accept) is True


def test_declaration_at_defaults_is_structural() -> None:
    # The §6c-i/bie shape: declare shared state at documented defaults.
    title = "§6c-i Pedestrians — declare shared run-over state at defaults"
    desc = "Declare the shared state variables at the §12.1 documented defaults: wanted = 0."
    assert is_structural_bead(title, desc) is True


def test_scaffold_and_stub_markers_are_structural() -> None:
    assert is_structural_bead("§13a HUD — screen-space scaffold + score text", "") is True
    assert is_structural_bead("Add a render stub for the splat", "") is True
    assert is_structural_bead("Boilerplate config loader", "") is True


def test_behavioral_bead_is_not_structural() -> None:
    # The §7a-iii/nwf shape: continuous tracking, no scaffold marker. Must
    # NOT be classified structural — it has a real gap to TDD.
    title = "§7a-iii Police — count tracks wanted (0..6, spawn within ~5s of increase)"
    desc = "Each frame in update(), spawn police so police.length climbs toward wanted."
    accept = "update() spawns police while police.length < wanted (capped 6)."
    assert is_structural_bead(title, desc, accept) is False


def test_plain_feature_bead_is_not_structural() -> None:
    assert is_structural_bead("Add score tracking and on-screen score display", "") is False
    # "default" without a declaration verb nearby must not trip the marker.
    assert is_structural_bead("Use the default road tile when none specified", "") is False


# ---- harness-pnkvp: behavioral-acceptance (assertability) signal -------


def test_behavioral_acceptance_flags_otherwise_rightsized_bead() -> None:
    """A single-subsection, short, low-clause bead — every SIZE signal
    quiet — still flags when its acceptance is runtime behavior the
    executor can't reduce to one failing test (§7a/lsna2 shape)."""
    c = score_bead(
        "harness-lsna2",
        "§7a Police spawn-by-wanted",
        "Spawn police as the wanted level rises.",
        "A police car spawns within 5s of wanted hitting 1; the count tracks wanted.",
    )
    assert c.behavioral_acceptance is True
    assert c.flagged is True
    assert any("one failing test" in r for r in c.reasons)


def test_code_presence_acceptance_not_behavioral() -> None:
    """The assertable sub-bead form (literal tokens/constants) must NOT
    trip the behavioral signal — that's the target granularity."""
    c = score_bead(
        "harness-4po",
        "§7a-i Police state decl",
        "Declare the police array and a spawn factory.",
        "game.js declares a police array and a spawnPoliceCar() whose object "
        "carries maxSpeed 200, acceleration 160, turnRate 2.0.",
    )
    assert c.behavioral_acceptance is False
    assert c.flagged is False


def test_behavioral_signal_falls_back_to_description_when_no_acceptance() -> None:
    c = score_bead("z", "t", "The camera follows the foot pose each frame.")
    assert c.behavioral_acceptance is True


def test_returns_beadcomplexity_shape() -> None:
    c = score_bead("harness-abc", "title", "body")
    assert isinstance(c, BeadComplexity)
    assert c.bead_id == "harness-abc"
    assert c.title == "title"
    assert c.behavioral_acceptance is False


# ---- harness-yzg8: wrap-long-function heuristic ------------------------


_BIG_JS_FUNCTION = "function {name}(now) {{\n" + ("    let x = 1;\n" * 120) + "}}\n"
_SHORT_JS_FUNCTION = "function {name}(now) {{\n" + ("    let x = 1;\n" * 5) + "}}\n"


def test_wrap_heuristic_flags_long_function_referenced_in_text(tmp_path: Path) -> None:
    """harness-yzg8: when the bead text matches a wrap-pattern phrase
    AND the referenced function is >=LONG_FUNCTION_THRESHOLD lines in
    the workspace, surface a whole-file-rewrite hint."""
    (tmp_path / "game.js").write_text(_BIG_JS_FUNCTION.format(name="gameLoop"))
    c = score_bead(
        "harness-yd3m",
        "§15b Controls — pause toggle",
        "let paused=false; Escape toggles paused. `gameLoop`: when paused, "
        "SKIP the update step but still render + draw PAUSED overlay.",
        workspace=tmp_path,
    )
    assert c.flagged is True
    assert any("gameLoop" in r and "whole-file rewrite" in r for r in c.reasons)


def test_wrap_heuristic_quiet_when_function_is_short(tmp_path: Path) -> None:
    """Wrap-pattern wording on a short function is NOT flagged — the
    pattern alone is too noisy; both signals required."""
    (tmp_path / "game.js").write_text(_SHORT_JS_FUNCTION.format(name="gameLoop"))
    c = score_bead(
        "harness-x",
        "trivial wrap",
        "Wrap `gameLoop` to skip the update step when paused.",
        workspace=tmp_path,
    )
    assert not any("whole-file rewrite" in r for r in c.reasons)


def test_wrap_heuristic_quiet_without_wrap_pattern(tmp_path: Path) -> None:
    """A long function alone (no wrap-pattern wording) is fine — beads
    that ADD a new function or REPLACE one don't need the warning."""
    (tmp_path / "game.js").write_text(_BIG_JS_FUNCTION.format(name="gameLoop"))
    c = score_bead(
        "harness-x",
        "add render call",
        "Call `gameLoop` after init() in main.",
        workspace=tmp_path,
    )
    assert not any("whole-file rewrite" in r for r in c.reasons)


def test_wrap_heuristic_quiet_when_function_absent(tmp_path: Path) -> None:
    """Wrap-pattern wording but no matching function in the workspace —
    silent; the lint is about coding strategy, not bead-text quality."""
    (tmp_path / "game.js").write_text(_SHORT_JS_FUNCTION.format(name="otherThing"))
    c = score_bead(
        "harness-x",
        "wrap mystery",
        "Wrap `gameLoop` to skip the update step.",
        workspace=tmp_path,
    )
    assert not any("whole-file rewrite" in r for r in c.reasons)


def test_wrap_heuristic_disabled_via_flag(tmp_path: Path) -> None:
    """`check_long_functions=False` skips the workspace scan entirely
    (opt-out per bead acceptance)."""
    (tmp_path / "game.js").write_text(_BIG_JS_FUNCTION.format(name="gameLoop"))
    c = score_bead(
        "harness-x",
        "pause toggle",
        "Wrap `gameLoop` to skip the update step when paused.",
        workspace=tmp_path,
        check_long_functions=False,
    )
    assert not any("whole-file rewrite" in r for r in c.reasons)


def test_wrap_heuristic_handles_python_def(tmp_path: Path) -> None:
    """Python def is measured by indent-drop, not braces."""
    body = "def process(data):\n" + ("    x = 1\n" * 120)
    (tmp_path / "thing.py").write_text(body)
    c = score_bead(
        "harness-x",
        "freeze processing",
        "Wrap `process` to freeze the update step on pause.",
        workspace=tmp_path,
    )
    assert any("process" in r and "whole-file rewrite" in r for r in c.reasons)


def test_wrap_heuristic_skips_excluded_dirs(tmp_path: Path) -> None:
    """node_modules / .git / .harness etc. are skipped so vendored
    code never triggers the warning."""
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "vendor.js").write_text(_BIG_JS_FUNCTION.format(name="gameLoop"))
    c = score_bead(
        "harness-x",
        "pause toggle",
        "Wrap `gameLoop` to skip the update step.",
        workspace=tmp_path,
    )
    assert not any("whole-file rewrite" in r for r in c.reasons)


def test_wrap_heuristic_workspace_none_is_noop(tmp_path: Path) -> None:
    """Callers that don't want filesystem access pass workspace=None
    (the legacy text-only signature). All existing tests use this
    shape — must stay a pure-text behavior."""
    c = score_bead(
        "harness-x",
        "pause toggle",
        "Wrap `gameLoop` to skip the update step.",
        workspace=None,
    )
    assert not any("whole-file rewrite" in r for r in c.reasons)
