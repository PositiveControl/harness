"""Tests for src/harness/tools/edit_file.py — focused on the
harness-w0gw error-message augmentation (file contents inline on
old_string mismatch / no-op) and the EditFileTool's existing
contract (path validation, sandbox escape, append mode, replace_all)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.edit_file import EditFileTool, _format_file_contents


def _tool(workspace: Path) -> EditFileTool:
    return EditFileTool(root=workspace)


# --- happy paths ----------------------------------------------------


def test_replace_single_match_writes_file(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello world")
    result = _tool(tmp_path).call(path="f.txt", old_string="world", new_string="harness")
    assert "edited f.txt" in result
    assert (tmp_path / "f.txt").read_text() == "hello harness"


def test_append_mode_with_empty_old_string(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("line 1\n")
    result = _tool(tmp_path).call(path="f.txt", old_string="", new_string="line 2\n")
    assert "appended to f.txt" in result
    assert (tmp_path / "f.txt").read_text() == "line 1\nline 2\n"


def test_replace_all_multiple_matches(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("foo bar foo baz foo")
    result = _tool(tmp_path).call(
        path="f.txt", old_string="foo", new_string="qux", replace_all=True
    )
    assert "3 replacement(s)" in result
    assert (tmp_path / "f.txt").read_text() == "qux bar qux baz qux"


# --- error: old_string not found inlines file contents (harness-w0gw) ----


def test_old_string_not_found_inlines_current_contents(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("the actual contents\nline 2\n")
    with pytest.raises(ValueError, match=r"(not found|no-op|matches)") as exc_info:
        _tool(tmp_path).call(
            path="f.txt",
            old_string="hallucinated content the model invented",
            new_string="replacement",
        )
    msg = str(exc_info.value)
    # Original guidance preserved.
    assert "old_string not found in f.txt" in msg
    # NEW: file contents inlined so the model has ground truth in the
    # same response. No "go re-read" punt.
    assert "--- CURRENT CONTENTS OF f.txt (BEGIN) ---" in msg
    assert "the actual contents" in msg
    assert "line 2" in msg
    assert "--- END f.txt ---" in msg


def test_no_op_edit_inlines_current_contents(tmp_path: Path) -> None:
    """When old_string == new_string the edit is a no-op. Inline file
    contents so the model can see what's actually there and pick a real
    diff to make."""
    (tmp_path / "f.txt").write_text("// some real content\nlet x = 1;\n")
    with pytest.raises(ValueError, match=r"(not found|no-op|matches)") as exc_info:
        _tool(tmp_path).call(path="f.txt", old_string="let x = 1;", new_string="let x = 1;")
    msg = str(exc_info.value)
    assert "edit is a no-op" in msg
    assert "--- CURRENT CONTENTS OF f.txt (BEGIN) ---" in msg
    assert "let x = 1;" in msg


def test_ambiguous_old_string_match_inlines_contents(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("foo\nfoo\nfoo\n")
    with pytest.raises(ValueError, match=r"(not found|no-op|matches)") as exc_info:
        _tool(tmp_path).call(path="f.txt", old_string="foo", new_string="bar")
    msg = str(exc_info.value)
    assert "matches 3 places" in msg
    # Contents inlined so the model can pick distinguishing context.
    assert "--- CURRENT CONTENTS OF f.txt (BEGIN) ---" in msg


def test_format_file_contents_truncates_large_files() -> None:
    """For pathologically large files, inline content is capped so the
    error doesn't balloon the model's context."""
    huge = "x" * (1024 * 1024 + 100)
    rendered = _format_file_contents("big.txt", huge)
    assert "truncated" in rendered
    assert "1048576" in rendered or "1024" in rendered
    # The truncation marker is bounded — total render shouldn't exceed
    # cap by more than the wrapper overhead.
    assert len(rendered) < 1024 * 1024 + 500


# --- preserved behavior ---------------------------------------------


def test_path_escape_rejected(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("contents")
    with pytest.raises(ValueError, match="escapes workspace root"):
        _tool(tmp_path).call(path="../outside.txt", old_string="x", new_string="y")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _tool(tmp_path).call(path="nope.txt", old_string="x", new_string="y")


def test_both_empty_strings_rejected(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("contents")
    with pytest.raises(ValueError, match="nothing to do"):
        _tool(tmp_path).call(path="f.txt", old_string="", new_string="")
