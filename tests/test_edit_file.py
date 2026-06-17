from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.tools.edit_file import EditFileArgs, EditFileTool

_requires_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="parse-check JS integration requires `node` on PATH",
)


def _tool(tmp_path: Path) -> EditFileTool:
    return EditFileTool(root=tmp_path)


def test_replaces_unique_match(tmp_path: Path) -> None:
    target = tmp_path / "hello.py"
    target.write_text("print('hello, world')\n")
    result = _tool(tmp_path).call(
        path="hello.py",
        old_string="hello, world",
        new_string="hello, Airton",
    )
    assert target.read_text() == "print('hello, Airton')\n"
    assert "1 replacement" in result
    assert "+1 bytes" in result  # "hello, Airton" is one char longer


def test_replaces_only_first_by_default(tmp_path: Path) -> None:
    target = tmp_path / "many.txt"
    target.write_text("foo\nfoo\n")
    with pytest.raises(ValueError, match="matches 2 places"):
        _tool(tmp_path).call(path="many.txt", old_string="foo", new_string="bar")


def test_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "many.txt"
    target.write_text("foo\nfoo\nfoo\n")
    result = _tool(tmp_path).call(
        path="many.txt",
        old_string="foo",
        new_string="bar",
        replace_all=True,
    )
    assert target.read_text() == "bar\nbar\nbar\n"
    assert "3 replacement" in result


def test_missing_match_raises(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello\n")
    with pytest.raises(ValueError, match="not found"):
        _tool(tmp_path).call(path="doc.md", old_string="goodbye", new_string="bye")


def test_empty_old_string_appends(tmp_path: Path) -> None:
    """Regression: 'add X to .gitignore' — the model reaches for
    edit_file with empty old_string. We treat that as append."""
    target = tmp_path / ".gitignore"
    target.write_text(".venv/\ndata/\n")
    result = _tool(tmp_path).call(
        path=".gitignore",
        old_string="",
        new_string="scratch/\n",
    )
    assert target.read_text() == ".venv/\ndata/\nscratch/\n"
    assert "appended" in result
    assert "+9 bytes" in result


def test_append_to_file_without_trailing_newline(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("alpha")
    _tool(tmp_path).call(path="notes.md", old_string="", new_string="\nbeta")
    assert target.read_text() == "alpha\nbeta"


def test_both_empty_rejected(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello\n")
    with pytest.raises(ValueError, match="nothing to do"):
        _tool(tmp_path).call(path="doc.md", old_string="", new_string="")


def test_noop_rejected(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello\n")
    with pytest.raises(ValueError, match="no-op"):
        _tool(tmp_path).call(path="doc.md", old_string="hello", new_string="hello")


def test_batch_edits_shape_rejected_with_corrective_message() -> None:
    """loop_run=467233ea: models invent a batch API,
    edits=[{"action": "replace", "content": "..."}], which omits the
    required old_string/new_string and otherwise lands in the generic
    field-error path. The before-validator must reject it with a message
    that names the real one-location-per-call contract."""
    with pytest.raises(ValidationError, match="does not take an `edits` array") as exc:
        EditFileArgs.model_validate(
            {"path": "game.js", "edits": '[{"action": "replace", "content": "x"}]'}
        )
    msg = str(exc.value)
    assert "old_string" in msg
    assert "new_string" in msg


def test_normal_args_still_validate() -> None:
    """The batch-shape guard must not disturb a well-formed call."""
    args = EditFileArgs.model_validate({"path": "game.js", "old_string": "a", "new_string": "b"})
    assert (args.old_string, args.new_string, args.replace_all) == ("a", "b", False)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _tool(tmp_path).call(path="missing.txt", old_string="x", new_string="y")


def test_escape_workspace_rejected(tmp_path: Path) -> None:
    # Sibling file outside the workspace root — even though it exists, the
    # path must be rejected before any read happens.
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret\n")
    try:
        with pytest.raises(ValueError, match="escapes workspace root"):
            _tool(tmp_path).call(
                path="../outside.txt",
                old_string="secret",
                new_string="public",
            )
    finally:
        outside.unlink(missing_ok=True)


def test_directory_rejected(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    with pytest.raises(IsADirectoryError):
        _tool(tmp_path).call(path="subdir", old_string="x", new_string="y")


def test_preserves_surrounding_content(tmp_path: Path) -> None:
    """Surgical edit — only the matched span changes."""
    target = tmp_path / "module.py"
    target.write_text(
        "def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 3\n"
    )
    _tool(tmp_path).call(
        path="module.py",
        old_string="def b():\n    return 2",
        new_string="def b():\n    return 42",
    )
    assert target.read_text() == (
        "def a():\n    return 1\n\ndef b():\n    return 42\n\ndef c():\n    return 3\n"
    )


def test_spec_metadata(tmp_path: Path) -> None:
    spec = _tool(tmp_path).spec
    assert spec.name == "edit_file"
    assert spec.tier == "write"
    required = spec.parameters["required"]
    assert set(required) == {"path", "old_string", "new_string"}
    assert "replace_all" in spec.parameters["properties"]


def test_binary_file_error(tmp_path: Path) -> None:
    target = tmp_path / "img.bin"
    target.write_bytes(b"\x00\x01\x02\xff\xfe")
    with pytest.raises(ValueError, match="not a UTF-8 text file"):
        _tool(tmp_path).call(path="img.bin", old_string="x", new_string="y")


# --- harness-h6wa: post-write parse gate ---------------------------


@_requires_node
def test_edit_that_breaks_js_parse_raises_with_parser_output(tmp_path: Path) -> None:
    """harness-h6wa: replicates loop run 26c39558's exact failure mode.
    A successful string-splice that introduces a duplicate `const` in
    the same block scope is valid as a `replace` operation but produces
    invalid JS. The post-write parse-check must convert the success
    into a failure that carries the parser's stderr — so the model
    sees the SyntaxError on the same turn it wrote the bad code."""
    target = tmp_path / "game.js"
    target.write_text("function f() {\n  const tileX = 1;\n}\n")
    with pytest.raises(ValueError, match="no longer parses"):
        _tool(tmp_path).call(
            path="game.js",
            old_string="const tileX = 1;",
            new_string="const tileX = 1;\n  const tileX = 2;",
        )


@_requires_node
def test_edit_landed_state_persists_after_parse_failure(tmp_path: Path) -> None:
    """Surface-only semantics: when the parse-check rejects, the file
    stays on disk in the broken state rather than getting rolled back.
    Rolling back would force the model to re-derive the entire edit;
    keeping the broken state gives the next round a concrete target
    to repair (read_file shows the dup, the parser error names the
    line)."""
    target = tmp_path / "game.js"
    target.write_text("function f() {\n  const x = 1;\n}\n")
    with pytest.raises(ValueError, match="no longer parses"):
        _tool(tmp_path).call(
            path="game.js",
            old_string="const x = 1;",
            new_string="const x = 1;\n  const x = 2;",
        )
    # File reflects the bad edit — model can read it and target the fix.
    assert "const x = 2" in target.read_text()


@_requires_node
def test_append_mode_also_parse_checks(tmp_path: Path) -> None:
    """Append-mode writes go through the same gate. A bare `}` appended
    to a valid file produces a SyntaxError and must be rejected, same
    as a replace-mode bad edit."""
    target = tmp_path / "ok.js"
    target.write_text("const x = 1;\n")
    with pytest.raises(ValueError, match="no longer parses"):
        _tool(tmp_path).call(path="ok.js", old_string="", new_string="}\n")


@_requires_node
def test_valid_js_edit_succeeds(tmp_path: Path) -> None:
    """Sanity check: a syntactically valid edit on a JS file passes
    the gate untouched. The parse-check is invisible when the edit
    is clean."""
    target = tmp_path / "ok.js"
    target.write_text("const x = 1;\n")
    result = _tool(tmp_path).call(
        path="ok.js", old_string="const x = 1;", new_string="const x = 42;"
    )
    assert "1 replacement" in result
    assert target.read_text() == "const x = 42;\n"


# --- harness-b2xa: context block in success message ---------------


def _long_text_file(tmp_path: Path, *, n_lines: int) -> Path:
    """Helper: create a tmp_path file whose content is N numbered
    plain-text lines. Plain text keeps the parse-check silent (no
    extension in `_PARSERS`) so the context-block tests don't fight
    with the parse gate."""
    target = tmp_path / "long.txt"
    target.write_text("\n".join(f"line {i}" for i in range(1, n_lines + 1)) + "\n")
    return target


def test_context_block_omitted_for_short_files(tmp_path: Path) -> None:
    """File of ≤25 lines: context block omitted. The model can
    `read_file` the whole thing cheaply, so an excerpt is noise."""
    target = tmp_path / "short.txt"
    target.write_text("a\nb\nc\nd\n")
    result = _tool(tmp_path).call(path="short.txt", old_string="b", new_string="B")
    assert "context" not in result
    assert target.read_text() == "a\nB\nc\nd\n"


def test_context_block_shows_splice_with_margin_for_small_edit(tmp_path: Path) -> None:
    """Long file, single-line edit: the context block surfaces the
    splice line ± a small margin so the model can verify what landed
    instead of trusting the byte count."""
    _long_text_file(tmp_path, n_lines=50)
    result = _tool(tmp_path).call(path="long.txt", old_string="line 20", new_string="LINE TWENTY")
    assert "context" in result
    # Splice landed on line 20; with 2-line margin, lines 18-22 appear
    # with 1-based line-number prefixes.
    assert "20: LINE TWENTY" in result
    assert "18: line 18" in result
    assert "22: line 22" in result
    # Header reports the splice's position and the total line count.
    assert "splice spans 20-20" in result
    assert "of 50" in result


def test_context_block_truncates_long_splice_with_head_and_tail(tmp_path: Path) -> None:
    """Long file, multi-line splice exceeding the full-show threshold
    (25 lines): the block emits only the first and last edge-window
    lines (10 each by default) with an omitted-line hint pointing the
    model at re-read to inspect the middle. Pins the loop run 26c39558
    failure mode — a 150-line new_string would have produced a
    truncated context with the head visible, the tail visible, and
    the middle (where the duplicate tileX hid) flagged for re-read."""
    _long_text_file(tmp_path, n_lines=200)
    # Replace line 50 with a 40-line block — exceeds full threshold.
    new_block = "\n".join(f"NEW_{i}" for i in range(1, 41))
    result = _tool(tmp_path).call(path="long.txt", old_string="line 50", new_string=new_block)
    assert "context" in result
    # Head (first 10 of inserted block) and tail (last 10) appear.
    assert "NEW_1" in result
    assert "NEW_10" in result
    assert "NEW_31" in result
    assert "NEW_40" in result
    # Middle 20 lines are omitted with the hint pointing at the file.
    assert "lines omitted" in result
    assert "long.txt" in result
    # Splice spans 40 lines starting at line 50.
    assert "splice spans lines 50-89" in result
    # harness-0tni: hint now references the real read_file API with
    # offset+limit so the model can pull the full splice region in one
    # call instead of guessing how to address the range.
    assert "read_file(path='long.txt', offset=50, limit=40)" in result


def test_context_block_omitted_for_replace_all(tmp_path: Path) -> None:
    """replace_all skips the context block — multi-splice excerpts
    are noisy and the model can re-read the file if it cares."""
    target = tmp_path / "multi.txt"
    target.write_text("foo\n" * 40)
    result = _tool(tmp_path).call(
        path="multi.txt",
        old_string="foo",
        new_string="bar",
        replace_all=True,
    )
    assert "context" not in result
    assert "40 replacement" in result


def test_context_block_omitted_for_append(tmp_path: Path) -> None:
    """Append mode skips the context block — the model just wrote
    the content and knows where it landed (end of file)."""
    target = tmp_path / "long.txt"
    target.write_text("\n".join(f"line {i}" for i in range(1, 51)) + "\n")
    result = _tool(tmp_path).call(path="long.txt", old_string="", new_string="appended line\n")
    assert "context" not in result
    assert "appended" in result
