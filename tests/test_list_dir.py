from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.list_dir import ListDirTool


def _populate(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x\n")
    (tmp_path / "src" / "b.py").write_text("xx\n")
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "src" / "nested" / "c.py").write_text("xxx\n")
    (tmp_path / "README.md").write_text("# readme\n")
    (tmp_path / ".gitignore").write_text("x\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("\n")


def test_top_level_skips_noise_and_hidden(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = ListDirTool(root=tmp_path).call()
    assert "src/" in out
    assert "README.md" in out
    assert ".git" not in out  # skipped by _DEFAULT_SKIP
    assert ".gitignore" not in out  # skipped because hidden
    assert "__pycache__" not in out


def test_include_hidden(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = ListDirTool(root=tmp_path).call(include_hidden=True)
    assert ".gitignore" in out
    # .git directory is still skipped via the skip_dirs list, even with
    # include_hidden=True — it's always noise.
    assert ".git/" not in out


def test_recursive_walks(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = ListDirTool(root=tmp_path).call(recursive=True)
    assert "src/nested/c.py" in out
    assert "src/a.py" in out


def test_subdirectory(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = ListDirTool(root=tmp_path).call(path="src")
    assert "src/a.py" in out
    assert "src/nested/" in out
    # Not recursive — c.py in nested subdir should not appear.
    assert "src/nested/c.py" not in out


def test_path_traversal_rejected(tmp_path: Path) -> None:
    _populate(tmp_path)
    with pytest.raises(ValueError, match="escapes workspace root"):
        ListDirTool(root=tmp_path).call(path="../etc")


def test_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ListDirTool(root=tmp_path).call(path="nope")


def test_file_path_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(NotADirectoryError):
        ListDirTool(root=tmp_path).call(path="a.txt")


def test_truncation(tmp_path: Path) -> None:
    for i in range(20):
        (tmp_path / f"f{i:02d}.txt").write_text("x")
    tool = ListDirTool(root=tmp_path, max_entries=5)
    out = tool.call()
    assert "[truncated at 5 entries]" in out


def test_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    out = ListDirTool(root=tmp_path).call(path="empty")
    assert "empty directory" in out


def test_spec_is_read_tier(tmp_path: Path) -> None:
    assert ListDirTool(root=tmp_path).spec.tier == "read"
