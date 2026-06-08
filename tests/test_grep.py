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


def test_skips_dot_harness_driver_artifacts(tmp_path: Path) -> None:
    # The driver's own .harness/loop_runs traces embed entire prior
    # prompts; a workspace-root grep that walked them slurped a
    # 162k-token line into the next prompt and crashed the run.
    (tmp_path / "game.js").write_text("needle in source\n")
    runs = tmp_path / ".harness" / "loop_runs"
    runs.mkdir(parents=True)
    (runs / "abc.vllm_trace.jsonl").write_text('{"prompt": "needle in a trace record"}\n')
    out = GrepTool(root=tmp_path).call(pattern="needle")
    assert "game.js:1:needle in source" in out
    assert ".harness" not in out


def test_clamps_runaway_matched_line(tmp_path: Path) -> None:
    # A single matched line (minified JS, a JSONL record) must not
    # dominate the result. Clamp it to max_line_chars.
    (tmp_path / "min.js").write_text("needle" + "x" * 50_000 + "\n")
    tool = GrepTool(root=tmp_path, max_line_chars=100)
    out = tool.call(pattern="needle")
    assert "line truncated" in out
    # The emitted hit line is bounded — not the full 50k.
    assert len(out) < 1_000


def test_total_output_byte_cap(tmp_path: Path) -> None:
    # 1000 matched lines of ~200 chars each would be ~200KB; the
    # total-char ceiling stops the result well before that, even
    # though the match-count cap (100) hasn't been reached per file.
    lines = "\n".join(f"needle line {i} " + "y" * 180 for i in range(1000))
    (tmp_path / "big.txt").write_text(lines)
    tool = GrepTool(root=tmp_path, max_total_chars=5_000, default_max_results=10_000)
    out = tool.call(pattern="needle")
    assert "truncated" in out
    assert len(out) < 6_000


def test_grep_accepts_single_file_path(tmp_path: Path) -> None:
    """harness-373e: when `path` resolves to a file, grep that single
    file instead of raising NotADirectoryError. Models routinely reach
    for the Unix-grep / ripgrep pattern of `grep PATTERN file.js` —
    the prior behavior dropped them into a DuplicateCallHook dedup
    loop on every drive run."""
    _populate(tmp_path)
    out = GrepTool(root=tmp_path).call(pattern="hello", path="src/a.py")
    # Hit only inside the target file.
    assert "src/a.py:2:print('hello')" in out
    # Files outside the target aren't searched.
    assert "README.md" not in out
    assert "src/b.py" not in out


def test_grep_single_file_path_respects_skip_dirs(tmp_path: Path) -> None:
    """A model that explicitly names a file inside a skip-dir is being
    intentional — the skip list is for recursive walks, not explicit
    targets. Confirm we honor the explicit pick over the skip filter."""
    _populate(tmp_path)
    # `.git/hooks.py` exists in _populate; it's normally skipped by
    # the recursive walk, but explicit targeting should still work.
    out = GrepTool(root=tmp_path).call(pattern="hello", path=".git/hooks.py")
    assert ".git/hooks.py:1:hello from git" in out


def test_grep_file_path_with_no_matches(tmp_path: Path) -> None:
    """A single-file grep with no hits returns the same '(no matches…)'
    sentinel as a directory walk."""
    _populate(tmp_path)
    out = GrepTool(root=tmp_path).call(pattern="nonexistent", path="src/a.py")
    assert "no matches" in out


def test_grep_missing_path_still_raises(tmp_path: Path) -> None:
    """The single-file-path support doesn't loosen the missing-path
    check — the old contract holds for typoed targets so the model
    sees a clear error instead of an empty-result silent-pass."""
    _populate(tmp_path)
    with pytest.raises(FileNotFoundError):
        GrepTool(root=tmp_path).call(pattern="x", path="src/missing.py")
