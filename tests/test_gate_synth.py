"""Source-text gate synthesis — harness-l3tgq (loop_run=77f684f9).

Run 77f684f9 parked harness-l3tgq at WRITE_TEST across 4 attempts: the
executor re-authored the same structurally-blind gate (body-slice / eval)
every time, so nothing was ever submitted. The gap was real and the
assessment named the code shapes that close it. gate_synth turns those named
shapes into a structure-safe source-text gate; these tests pin the pure
synthesizer and the adopt/discard orchestration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.driver.fsm_turn import _adopt_synthesized_gate
from harness.driver.gate_synth import build_source_text_gate

_L3TGQ_ASSESSMENT = {
    "approach": (
        "Implement position integration in the traffic loop using "
        "car.x += Math.cos(car.angle)*car.speed*dt and "
        "car.y += Math.sin(car.angle)*car.speed*dt, plus intent assignment at "
        "intersections with nextDecisionAt timing."
    ),
    "gap": "No position integration - cars accelerate and rotate but never move.",
    "current_state": (
        "The traffic update loop in game.js rotates angle toward intent but lacks x/y updates."
    ),
}


# --- build_source_text_gate (pure) ----------------------------------


def test_synthesizes_gate_from_absent_named_shapes(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text(
        "const canvas = document.getElementById('c');\n"
        "let traffic = [];\nfunction update(dt) { car.angle += 0.1; }\n"
    )

    gate = build_source_text_gate(_L3TGQ_ASSESSMENT, tmp_path, is_browser_js=True)

    assert gate is not None
    assert gate.test_filename == "test_synth_game.js"
    assert gate.test_cmd == "node test_synth_game.js"
    # The position-integration shapes the assessment named, absent from source.
    assert r"/car\.x\s*\+=/" in gate.content
    assert r"/car\.y\s*\+=/" in gate.content
    assert "readFileSync('game.js'" in gate.content
    # Structure-safe: never eval, never a body slice.
    assert "eval(" not in gate.content
    assert "{[^}]" not in gate.content


def test_drops_already_present_shapes(tmp_path: Path) -> None:
    """A shape already in the source adds no red-now signal — only ABSENT
    shapes are gated. With car.x/car.y already present, the only remaining
    absent shape (nextDecisionAt) is below the 2-shape floor → declines."""
    (tmp_path / "game.js").write_text(
        "const window = {};\nlet traffic = [];\nfunction update(dt) { car.x += 1; car.y += 1; }\n"
    )

    assert build_source_text_gate(_L3TGQ_ASSESSMENT, tmp_path, is_browser_js=True) is None


def test_declines_below_two_absent_shapes(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("const canvas = document.body;\nlet x = 0;\n")
    thin = {"approach": "set car.x += speed", "gap": "", "current_state": ""}

    assert build_source_text_gate(thin, tmp_path, is_browser_js=True) is None


def test_declines_when_not_browser_js(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("let traffic = [];\nfunction update(dt) {}\n")

    assert build_source_text_gate(_L3TGQ_ASSESSMENT, tmp_path, is_browser_js=False) is None


def test_declines_without_assessment_or_source(tmp_path: Path) -> None:
    assert build_source_text_gate(None, tmp_path, is_browser_js=True) is None
    # Assessment present but no source file in the workspace.
    assert build_source_text_gate(_L3TGQ_ASSESSMENT, tmp_path, is_browser_js=True) is None


def test_resolves_source_named_in_assessment_over_largest(tmp_path: Path) -> None:
    """When the assessment names a file, prefer it even if another .js is
    larger. The named game.js is smaller than engine.js here."""
    (tmp_path / "game.js").write_text(
        "const canvas = document.body;\nfunction update(dt) { car.angle += 1; }\n"
    )
    (tmp_path / "engine.js").write_text("// padding\n" * 500 + "const window = {};\n")

    gate = build_source_text_gate(_L3TGQ_ASSESSMENT, tmp_path, is_browser_js=True)

    assert gate is not None
    assert gate.test_filename == "test_synth_game.js"
    assert "readFileSync('game.js'" in gate.content


def test_skips_test_files_when_resolving_source(tmp_path: Path) -> None:
    """The largest .js is a test file; the resolver must skip it and pick the
    real source so the gate reads the implementation, not a test."""
    (tmp_path / "game.js").write_text(
        "const canvas = document.body;\nfunction update(dt) { car.angle += 1; }\n"
    )
    (tmp_path / "test_huge.js").write_text("// padding\n" * 800)
    # No field names a source file → resolver falls back to the largest
    # non-test .js, which must skip test_huge.js and land on game.js.
    assessment = {k: v.replace("game.js", "the loop") for k, v in _L3TGQ_ASSESSMENT.items()}

    gate = build_source_text_gate(assessment, tmp_path, is_browser_js=True)

    assert gate is not None
    assert "test_huge" not in gate.content
    assert "readFileSync('game.js'" in gate.content


# --- _adopt_synthesized_gate (orchestration) ------------------------


def test_adopts_red_runnable_synthesized_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A synthesized gate that runs red + clean is adopted as a failing test,
    and its file is written to the workspace for VERIFY to re-run."""
    (tmp_path / "game.js").write_text(
        "const canvas = document.body;\nfunction update(dt) { car.angle += 1; }\n"
    )
    monkeypatch.setattr(
        "harness.driver.fsm_turn._exec_test_cmd",
        lambda cmd, ws: (1, "FAIL: source missing required shapes: /car\\.x/, /car\\.y/"),
    )

    outcome = _adopt_synthesized_gate(tmp_path, _L3TGQ_ASSESSMENT, True)

    assert outcome is not None
    assert outcome.kind == "failing_test_submitted"
    assert outcome.payload["test_path"] == "test_synth_game.js"
    assert (tmp_path / "test_synth_game.js").is_file()


def test_discards_synthesized_gate_that_runs_green(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the synthesized gate somehow runs green (nothing to prove), discard
    it AND remove its file so it can't become adopt-bait next attempt."""
    (tmp_path / "game.js").write_text(
        "const canvas = document.body;\nfunction update(dt) { car.angle += 1; }\n"
    )
    monkeypatch.setattr(
        "harness.driver.fsm_turn._exec_test_cmd",
        lambda cmd, ws: (0, "OK: all required shapes present"),
    )

    outcome = _adopt_synthesized_gate(tmp_path, _L3TGQ_ASSESSMENT, True)

    assert outcome is None
    assert not (tmp_path / "test_synth_game.js").exists()


def test_adopt_returns_none_when_synthesis_declines(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("let traffic = [];\nfunction update(dt) {}\n")
    # Not browser-JS → synthesis declines → no file written, no outcome.
    assert _adopt_synthesized_gate(tmp_path, _L3TGQ_ASSESSMENT, False) is None
    assert not (tmp_path / "test_synth_game.js").exists()
