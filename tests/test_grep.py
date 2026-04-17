from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.grep import GrepTool


def _populate(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("import os\nprint('hello')\n")
    (tmp_path / "src" / "b.py").write_text("from typing import Any\nprint('HELLO')\n")
    (tmp_path / "README.md").write_text("# title\nhello world\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hooks.py").write_text("hello from git\n")


def test_matches_across_files(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GrepTool(root=tmp_path).call(pattern="hello")
    assert "src/a.py:2:print('hello')" in out
    assert "README.md:2:hello world" in out
    # Skip dir ignored.
    assert ".git/hooks.py" not in out


def test_case_insensitive(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GrepTool(root=tmp_path).call(pattern="HELLO", case_insensitive=True)
    assert "src/a.py" in out
    assert "src/b.py" in out


def test_glob_filter(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GrepTool(root=tmp_path).call(pattern="hello", glob="**/*.md")
    assert "README.md" in out
    assert "src/a.py" not in out


def test_regex_metacharacters(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("import re\nimport os\nimport json\n")
    out = GrepTool(root=tmp_path).call(pattern=r"^import (re|os)$")
    assert "m.py:1:import re" in out
    assert "m.py:2:import os" in out
    assert "m.py:3" not in out


def test_no_matches(tmp_path: Path) -> None:
    _populate(tmp_path)
    out = GrepTool(root=tmp_path).call(pattern="nonexistent")
    assert "no matches" in out


def test_max_results_truncation(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("\n".join("needle" for _ in range(30)))
    out = GrepTool(root=tmp_path).call(pattern="needle", max_results=5)
    assert out.count("big.txt:") == 5
    assert "truncated at 5" in out


def test_empty_pattern_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GrepTool(root=tmp_path).call(pattern="")


def test_invalid_regex(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid regex"):
        GrepTool(root=tmp_path).call(pattern="[")


def test_skips_binary_large_files(tmp_path: Path) -> None:
    big = tmp_path / "giant.bin"
    big.write_bytes(b"x" * 2_000_000)
    tool = GrepTool(root=tmp_path, max_file_bytes=100)
    out = tool.call(pattern="x")
    # Didn't blow up; just returned no matches.
    assert "no matches" in out


def test_spec_is_read_tier(tmp_path: Path) -> None:
    assert GrepTool(root=tmp_path).spec.tier == "read"
