from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.edit_file import EditFileTool


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
