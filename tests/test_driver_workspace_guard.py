"""Tests for driver/workspace_guard.py — regression guard + scratch
hygiene helpers (harness-16w6, harness-ul5z). Pure filesystem; real
``tmp_path`` workspaces, no mocking."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.driver.workspace_guard import (
    DEFAULT_SCRATCH_PATTERNS,
    WorkspaceTooBigError,
    archive_workspace,
    detect_regression,
    extract_symbols,
    is_scratch,
    list_workspace_files,
    restore_workspace,
    sweep_scratch,
    workspace_changed,
)

_CAP = 100 * 1024 * 1024


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


# --- file census -----------------------------------------------------


def test_list_workspace_files_relative_and_prunes_excluded(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.js").write_text("x")
    (ws / "sub").mkdir()
    (ws / "sub" / "a.py").write_text("y")
    (ws / ".harness").mkdir()
    (ws / ".harness" / "state.json").write_text("{}")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "dep.js").write_text("z")

    files = list_workspace_files(ws)
    assert files == {"game.js", "sub/a.py"}


# --- snapshot + restore ----------------------------------------------


def test_archive_then_restore_roundtrip_reverts_edits(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.js").write_text("original\n")
    snap = archive_workspace(ws, tmp_path / "snap.tar.gz", size_cap_bytes=_CAP)

    (ws / "game.js").write_text("mutated\n")
    restored, removed = restore_workspace(ws, snap)

    assert (ws / "game.js").read_text() == "original\n"
    assert restored == 1
    assert removed == []


def test_restore_removes_files_added_after_snapshot(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.js").write_text("keep\n")
    snap = archive_workspace(ws, tmp_path / "snap.tar.gz", size_cap_bytes=_CAP)

    # Scratch the issue created after the snapshot.
    (ws / "temp_plan.md").write_text("notes")
    (ws / "game.js").write_text("broken\n")
    _, removed = restore_workspace(ws, snap)

    assert (ws / "game.js").read_text() == "keep\n"
    assert not (ws / "temp_plan.md").exists()
    assert removed == ["temp_plan.md"]


def test_restore_recreates_a_deleted_file(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.js").write_text("body\n")
    snap = archive_workspace(ws, tmp_path / "snap.tar.gz", size_cap_bytes=_CAP)
    (ws / "game.js").unlink()

    restore_workspace(ws, snap)
    assert (ws / "game.js").read_text() == "body\n"


def test_archive_over_cap_raises(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "big.js").write_text("x" * 2048)
    with pytest.raises(WorkspaceTooBigError):
        archive_workspace(ws, tmp_path / "snap.tar.gz", size_cap_bytes=1024)


# --- symbol extraction + regression ----------------------------------


def test_extract_symbols_js_and_py() -> None:
    js = "function spawnCop(){}\nconst drawCar = (c) => {}\nlet update = function(){}"
    assert extract_symbols(js) == {"spawnCop", "drawCar", "update"}
    py = "def handle():\n    pass\nclass Engine:\n    pass\n"
    assert extract_symbols(py) == {"handle", "Engine"}


def test_detect_regression_none_when_code_grows(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.js").write_text("function a(){}\nfunction b(){}\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    # Add a function — adding code must never trip the guard.
    (ws / "game.js").write_text("function a(){}\nfunction b(){}\nfunction c(){}\n")
    assert detect_regression(ws, snap) is None


def test_detect_regression_flags_lost_symbols(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.js").write_text("function a(){}\nfunction b(){}\nfunction c(){}\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    # Stub-rewrite: only `a` survives.
    (ws / "game.js").write_text("function a(){}\n")
    reason = detect_regression(ws, snap)
    assert reason is not None
    assert "b" in reason
    assert "c" in reason
    assert "rewrite" in reason.lower()


def test_detect_regression_flags_deleted_file(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.js").write_text("function a(){}\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    (ws / "game.js").unlink()
    reason = detect_regression(ws, snap)
    assert reason is not None
    assert "deleted" in reason


def test_detect_regression_flags_dramatic_shrink(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    # 50 lines, no symbols, so the shrink branch (not symbol-loss) fires.
    (ws / "data.js").write_text("\n".join(f"// line {i}" for i in range(50)) + "\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    (ws / "data.js").write_text("// line 0\n// line 1\n")
    reason = detect_regression(ws, snap)
    assert reason is not None
    assert "shrank" in reason


def test_detect_regression_ignores_small_file_trim(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "tiny.js").write_text("// a\n// b\n// c\n")  # 3 lines < min_lines
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    (ws / "tiny.js").write_text("// a\n")
    assert detect_regression(ws, snap) is None


# --- scratch hygiene -------------------------------------------------


def test_is_scratch_patterns() -> None:
    assert is_scratch("police_plan.md", DEFAULT_SCRATCH_PATTERNS)
    assert is_scratch("temp_grid.js", DEFAULT_SCRATCH_PATTERNS)
    assert is_scratch("validate_gamejs.js", DEFAULT_SCRATCH_PATTERNS)
    assert is_scratch("game_backup.js", DEFAULT_SCRATCH_PATTERNS)
    assert not is_scratch("game.js", DEFAULT_SCRATCH_PATTERNS)
    assert not is_scratch("index.html", DEFAULT_SCRATCH_PATTERNS)


def test_sweep_archives_new_scratch_only(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.js").write_text("deliverable")
    baseline = list_workspace_files(ws)  # census before the issue

    # Issue creates scratch + edits the deliverable.
    (ws / "police_plan.md").write_text("plan")
    (ws / "temp_x.js").write_text("scratch")
    (ws / "game.js").write_text("edited deliverable")

    archive = ws / ".harness" / "loop_runs" / "abc_scratch"
    moved = sweep_scratch(ws, baseline, patterns=DEFAULT_SCRATCH_PATTERNS, archive_dir=archive)

    assert moved == ["police_plan.md", "temp_x.js"]
    assert not (ws / "police_plan.md").exists()
    assert (archive / "police_plan.md").read_text() == "plan"
    # Deliverable + its edits untouched.
    assert (ws / "game.js").read_text() == "edited deliverable"


def test_sweep_leaves_preexisting_scratch_named_file(tmp_path: Path) -> None:
    # A scratch-named file that existed BEFORE the issue is not the
    # issue's doing — only files created during the issue are swept.
    ws = _ws(tmp_path)
    (ws / "temp_legacy.js").write_text("old")
    baseline = list_workspace_files(ws)  # temp_legacy.js already present

    archive = ws / ".harness" / "scratch"
    moved = sweep_scratch(ws, baseline, patterns=DEFAULT_SCRATCH_PATTERNS, archive_dir=archive)

    assert moved == []
    assert (ws / "temp_legacy.js").exists()


def test_detect_regression_flags_deleted_index_html(tmp_path: Path) -> None:
    # harness-9ugc: .html is now a tracked source suffix, so deleting the
    # entry index.html trips the regression guard.
    ws = _ws(tmp_path)
    (ws / "index.html").write_text("<script src='game.js'></script>\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    (ws / "index.html").unlink()
    reason = detect_regression(ws, snap)
    assert reason is not None
    assert "index.html" in reason


# --- workspace_changed (harness-82r1v) -------------------------------


def test_workspace_changed_false_when_identical(tmp_path: Path) -> None:
    # No edits since the snapshot → no work done this attempt.
    ws = _ws(tmp_path)
    (ws / "game.py").write_text("def main():\n    return 1\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    assert workspace_changed(ws, snap) is False


def test_workspace_changed_true_when_source_content_differs(tmp_path: Path) -> None:
    # The model edited a tracked source file → real work this attempt.
    ws = _ws(tmp_path)
    (ws / "game.py").write_text("def main():\n    return 1\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    (ws / "game.py").write_text("def main():\n    score = 0\n    return score\n")
    assert workspace_changed(ws, snap) is True


def test_workspace_changed_true_when_source_file_added(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.py").write_text("def main():\n    return 1\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    (ws / "score.py").write_text("SCORE = 0\n")
    assert workspace_changed(ws, snap) is True


def test_workspace_changed_true_when_source_file_removed(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws / "game.py").write_text("def main():\n    return 1\n")
    (ws / "score.py").write_text("SCORE = 0\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    (ws / "score.py").unlink()
    assert workspace_changed(ws, snap) is True


def test_workspace_changed_ignores_non_source_files(tmp_path: Path) -> None:
    # A scratch/data file changing is not issue work — only source
    # suffixes count, matching detect_regression's scope.
    ws = _ws(tmp_path)
    (ws / "game.py").write_text("def main():\n    return 1\n")
    (ws / "notes.txt").write_text("scratch\n")
    snap = archive_workspace(ws, tmp_path / "g.tar.gz", size_cap_bytes=_CAP)
    (ws / "notes.txt").write_text("different scratch\n")
    (ws / "data.json").write_text("{}\n")
    assert workspace_changed(ws, snap) is False
