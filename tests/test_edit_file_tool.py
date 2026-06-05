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


def test_no_op_edit_when_old_string_is_in_file(tmp_path: Path) -> None:
    """When old_string == new_string AND old_string IS in the file, the
    'edit is a no-op' error fires and the fix is 'pick a different
    new_string'. The error does NOT inline file contents (harness-6la9)
    — showing the file again is noise when the model needs to change
    new_string, not find a matching old_string."""
    (tmp_path / "f.txt").write_text("// some real content\nlet x = 1;\n")
    with pytest.raises(ValueError, match=r"(not found|no-op|matches)") as exc_info:
        _tool(tmp_path).call(path="f.txt", old_string="let x = 1;", new_string="let x = 1;")
    msg = str(exc_info.value)
    assert "edit is a no-op" in msg
    # Inlined contents are NOT present — the no-op error is a model-
    # output bug, not a 'help me find content' bug.
    assert "--- CURRENT CONTENTS OF f.txt (BEGIN) ---" not in msg


def test_no_op_check_does_not_mask_not_found_for_hallucinated_string(
    tmp_path: Path,
) -> None:
    """harness-6la9 regression: when old_string == new_string AND
    old_string is NOT in the file, the 'not found' error must fire
    (not the 'no-op' error). The previous ordering told the model
    'your edit is a no-op' when the real problem was 'X isn't in the
    file' — driving a loop of (X, X) → 'no-op' → try (X', X') → 'no-op'
    forever because the model never learned its strings were
    hallucinated."""
    (tmp_path / "f.txt").write_text(
        "// actual file contents the model never sees because hallucinated\n"
    )
    hallucinated = "if (keys.KeyR && !prevKeys.KeyR && !player.alive) {"
    with pytest.raises(ValueError, match=r"(not found|no-op|matches)") as exc_info:
        _tool(tmp_path).call(
            path="f.txt",
            old_string=hallucinated,
            new_string=hallucinated,
        )
    msg = str(exc_info.value)
    # The 'not found' diagnosis wins so the model gets the file ground
    # truth, not the misleading 'no-op' message.
    assert "old_string not found in f.txt" in msg
    assert "edit is a no-op" not in msg
    # And the inlined contents are present — that's the whole point of
    # the not-found path (give the model the file so it can construct a
    # matching old_string).
    assert "--- CURRENT CONTENTS OF f.txt (BEGIN) ---" in msg


def test_ambiguous_old_string_match_inlines_contents(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("foo\nfoo\nfoo\n")
    with pytest.raises(ValueError, match=r"(not found|no-op|matches)") as exc_info:
        _tool(tmp_path).call(path="f.txt", old_string="foo", new_string="bar")
    msg = str(exc_info.value)
    assert "matches 3 places" in msg
    # Contents inlined so the model can pick distinguishing context.
    assert "--- CURRENT CONTENTS OF f.txt (BEGIN) ---" in msg


def test_format_file_contents_truncates_large_files() -> None:
    """File content embedded in error messages is capped (harness-kpx8).
    On large files an unbounded dump stacked across edit_file retries
    blew the model's context window (drive-loop halt 2026-05-23 on a
    50 KB game.js against Qwen2.5-Coder 32B). Cap at 4 KB, point the
    model at read_file(offset, limit) for further inspection."""
    huge = "x" * (8 * 1024)
    rendered = _format_file_contents("big.txt", huge)
    assert "truncated" in rendered
    assert "4096" in rendered  # cap reported in the marker
    # Marker steers the model toward a bounded recovery action.
    assert "read_file" in rendered
    # The render stays bounded — wrapper + marker overhead is small.
    assert len(rendered) < 4 * 1024 + 500


def test_format_file_contents_passes_short_files_through() -> None:
    """Files at or under the 4 KB cap (harness-kpx8) embed in full,
    with no truncation marker — the common case for edit-target files."""
    short = "line\n" * 100  # ~500 bytes, well under cap
    rendered = _format_file_contents("small.py", short)
    assert "truncated" not in rendered
    assert short in rendered


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


# --- typed-args boundary (harness-4fa0m) ----------------------------
#
# Run d45fd2f7: the model repeatedly emitted edit_file with args={} and
# got back a raw `TypeError: EditFileTool.call() missing 3 required
# keyword-only arguments` — no field names it could act on, so it
# re-emitted the same malformed call for rounds. edit_file was the one
# file-op tool left off the harness-5cjj9 typed-args pass; these tests
# pin the args_model boundary.


def test_registry_empty_args_returns_structured_validation_error(tmp_path: Path) -> None:
    from harness.tools.base import ToolRegistry

    registry = ToolRegistry()
    registry.register(_tool(tmp_path))

    result = registry.call("edit_file", {})

    assert not result.success
    assert (result.error or "") == "validation_error"
    # The d45fd2f7 failure shape must be gone…
    assert "TypeError" not in (result.error or "")
    assert "keyword-only" not in result.output
    # …replaced by every missing field named for the model to act on.
    for field in ("path", "old_string", "new_string"):
        assert field in result.output


def test_registry_unknown_arg_keeps_harness_d7e_hint(tmp_path: Path) -> None:
    """extra_forbidden still routes through the unknown-kwarg rewrite
    that lists accepted fields."""
    from harness.tools.base import ToolRegistry

    registry = ToolRegistry()
    registry.register(_tool(tmp_path))
    (tmp_path / "f.txt").write_text("hello world")

    result = registry.call(
        "edit_file",
        {"path": "f.txt", "old_string": "hello", "new_string": "hi", "mode": "fast"},
    )

    assert not result.success
    assert (result.error or "").startswith("unknown_kwarg:mode")
    assert "replace_all" in result.output


def test_registry_valid_call_still_dispatches(tmp_path: Path) -> None:
    """The args_model is transparent for well-formed calls."""
    from harness.tools.base import ToolRegistry

    registry = ToolRegistry()
    registry.register(_tool(tmp_path))
    (tmp_path / "f.txt").write_text("hello world")

    result = registry.call(
        "edit_file", {"path": "f.txt", "old_string": "world", "new_string": "harness"}
    )

    assert result.success
    assert (tmp_path / "f.txt").read_text() == "hello harness"


def test_schema_shape_unchanged_by_args_model() -> None:
    """tool_schema_from_model output must keep the same required set +
    property names the hand-written schema had — the drive prompts and
    router fixtures key off them."""
    spec = EditFileTool(root=Path(".")).spec
    assert set(spec.parameters["required"]) == {"path", "old_string", "new_string"}
    assert set(spec.parameters["properties"].keys()) == {
        "path",
        "old_string",
        "new_string",
        "replace_all",
    }
    assert spec.parameters["properties"]["replace_all"]["type"] == "boolean"
