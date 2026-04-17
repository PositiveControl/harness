from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.glob import GlobTool


def _populate(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x")
    (tmp_path / "src" / "b.py").write_text("x")
    (tmp_path / "src" / "sub").mkdir()
    (tmp_path / "src" / "sub" / "c.py").write_text("x")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("x")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("x")


def test_top_level_glob(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GlobTool(root=tmp_path).call(pattern="*.md")
    assert "README.md" in out
    assert "docs/guide.md" not in out  # * doesn't recurse


def test_recursive_glob(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GlobTool(root=tmp_path).call(pattern="**/*.py")
    assert "src/a.py" in out
    assert "src/sub/c.py" in out


def test_subpath_glob(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GlobTool(root=tmp_path).call(pattern="*.md", path="docs")
    assert "docs/guide.md" in out
    assert "README.md" not in out


def test_noise_dirs_skipped(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GlobTool(root=tmp_path).call(pattern="**/*")
    assert "__pycache__" not in out


def test_no_matches(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GlobTool(root=tmp_path).call(pattern="*.rs")
    assert "no files matching" in out


def test_empty_pattern_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GlobTool(root=tmp_path).call(pattern="")


def test_path_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes workspace root"):
        GlobTool(root=tmp_path).call(pattern="*", path="..")


def test_missing_base_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GlobTool(root=tmp_path).call(pattern="*", path="missing")


def test_spec_is_read_tier(tmp_path: Path) -> None:
    assert GlobTool(root=tmp_path).spec.tier == "read"
