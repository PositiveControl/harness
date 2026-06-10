"""Tests for the cross-workspace premise check (harness-y0hu5).

The guard parks a bead whose text references files that exist nowhere
under the workspace (the harness-rorj mis-wire: a snake_game.py bead
under the GTAII epic). Every test here pins one of the conservative
guards — the check must bias toward driving, never park legitimate
new-file or partially-grounded work.
"""

from __future__ import annotations

from pathlib import Path

from harness.driver.premise_guard import _extract_path_tokens, referenced_missing_files

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
