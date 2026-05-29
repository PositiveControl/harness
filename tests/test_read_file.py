"""Unit tests for tools/read_file.py.

Focus: harness-296bh numeric-arg coercion. Small models routinely emit
`offset`/`limit` as JSON strings (`offset='20'`); the tool used to
reject the call and burn a round + retry (run b085854e turn 1). These
pin that a digit string is coerced to the same result as the integer,
that whitespace is tolerated, and that genuinely non-numeric input still
errors clearly. The line-slicing / symbol paths are exercised elsewhere
(test_tool_loop.py, test_outline_tool.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.read_file import ReadFileTool

_SAMPLE = "line1\nline2\nline3\nline4\nline5\n"


def _tool(tmp_path: Path) -> ReadFileTool:
    (tmp_path / "f.txt").write_text(_SAMPLE, encoding="utf-8")
    return ReadFileTool(root=tmp_path)


def test_string_offset_limit_match_integer_args(tmp_path: Path) -> None:
    """offset='2', limit='2' must produce the identical slice as the
    integer form — coercion happens before any slicing logic."""
    tool = _tool(tmp_path)
    as_str = tool.call(path="f.txt", offset="2", limit="2")
    as_int = tool.call(path="f.txt", offset=2, limit=2)
    assert as_str == as_int
    assert "line2\nline3\n" in as_str
    assert "[showing lines 2-3 of 5]" in as_str


def test_whitespace_padded_numeric_string_is_coerced(tmp_path: Path) -> None:
    """Tolerate stray whitespace around the digits — small models emit
    ` 2 ` as readily as `2`."""
    tool = _tool(tmp_path)
    assert tool.call(path="f.txt", offset=" 2 ", limit=" 2 ") == tool.call(
        path="f.txt", offset=2, limit=2
    )


def test_non_numeric_offset_still_errors_clearly(tmp_path: Path) -> None:
    """A genuinely non-numeric value is a real mistake — fail with a
    message that names the arg and shows what was passed."""
    tool = _tool(tmp_path)
    with pytest.raises(ValueError, match="offset must be an integer"):
        tool.call(path="f.txt", offset="abc")


def test_string_offset_below_one_still_rejected(tmp_path: Path) -> None:
    """Coercion feeds the existing >=1 guard: '0' coerces to 0, which
    the 1-based check rejects — the bound is enforced post-coercion."""
    tool = _tool(tmp_path)
    with pytest.raises(ValueError, match="offset must be >= 1"):
        tool.call(path="f.txt", offset="0")
