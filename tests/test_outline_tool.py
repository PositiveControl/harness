"""Unit tests for tools/outline.py — harness-ywhf.

Pins the skeleton rendering (indented signatures + line ranges, bodies
elided), max_depth, the workspace-escape guard, and the
degrade-to-a-note paths (unsupported extension, no symbols). Parsing
tests skip when the [code] extra is absent; the guard + degraded paths
need no parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.outline import OutlineTool

try:
    import tree_sitter_language_pack  # noqa: F401

    _HAS_CODE = True
except ImportError:  # pragma: no cover - exercised only on lean installs
    _HAS_CODE = False

_requires_code = pytest.mark.skipif(not _HAS_CODE, reason="requires the [code] extra")

_PY_MODULE = """import os


class Foo:
    def bar(self, x):
        return x + 1

    def baz(self):
        return 2


def top(a, b):
    return a + b
"""


def _line(result: str, needle: str) -> str:
    """Return the first output line containing `needle`."""
    for line in result.splitlines():
        if needle in line:
            return line
    raise AssertionError(f"no line containing {needle!r} in:\n{result}")


@_requires_code
def test_outline_renders_skeleton_with_ranges(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(_PY_MODULE)
    tool = OutlineTool(root=tmp_path)
    result = tool.call(path="m.py")

    assert result.startswith("m.py: 4 symbols")
    assert "[L4-9]" in _line(result, "class Foo:")
    assert "[L5-6]" in _line(result, "def bar")
    assert "[L12-13]" in _line(result, "def top")
    # bodies elided: the actual statements never appear.
    assert "return x + 1" not in result
    assert "return a + b" not in result


@_requires_code
def test_outline_indents_nested_symbols(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(_PY_MODULE)
    tool = OutlineTool(root=tmp_path)
    result = tool.call(path="m.py")

    # method is indented under its class; the top-level class is not.
    assert _line(result, "class Foo:").startswith("class Foo:")
    assert _line(result, "def bar").startswith("  def bar")
    # leaf with a body gets an elision marker; the parent class does not.
    assert "…" in _line(result, "def bar")
    assert "…" not in _line(result, "class Foo:")


@_requires_code
def test_outline_max_depth_limits_nesting(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(_PY_MODULE)
    tool = OutlineTool(root=tmp_path)
    result = tool.call(path="m.py", max_depth=1)

    assert "class Foo:" in result
    assert "def top" in result
    # methods (depth 1) are suppressed at max_depth=1.
    assert "def bar" not in result
    assert "def baz" not in result


@_requires_code
def test_outline_javascript(tmp_path: Path) -> None:
    (tmp_path / "w.js").write_text(
        "class Widget {\n"
        "  render() {\n"
        "    return 1;\n"
        "  }\n"
        "}\n"
        "\n"
        "function helper(x) {\n"
        "  return x;\n"
        "}\n"
    )
    tool = OutlineTool(root=tmp_path)
    result = tool.call(path="w.js")
    assert "class Widget" in result
    assert _line(result, "render").startswith("  ")  # nested
    assert "function helper" in result


def test_outline_max_depth_zero_rejected(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def f():\n    pass\n")
    tool = OutlineTool(root=tmp_path)
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        tool.call(path="m.py", max_depth=0)


def test_outline_refuses_path_traversal(tmp_path: Path) -> None:
    tool = OutlineTool(root=tmp_path)
    with pytest.raises(ValueError, match="escapes workspace root"):
        tool.call(path="../secret.py")


def test_outline_missing_file_raises(tmp_path: Path) -> None:
    tool = OutlineTool(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        tool.call(path="nope.py")


def test_outline_unsupported_extension_degrades(tmp_path: Path) -> None:
    """No grammar for the extension → a guidance note, not a raise, and
    no [code] extra needed (the check precedes the lazy import)."""
    (tmp_path / "notes.txt").write_text("just prose\n")
    tool = OutlineTool(root=tmp_path)
    result = tool.call(path="notes.txt")
    assert "outline unavailable" in result
    assert "read_file" in result


@_requires_code
def test_outline_no_symbols_note(tmp_path: Path) -> None:
    (tmp_path / "empty.py").write_text("x = 1\ny = 2\n")
    tool = OutlineTool(root=tmp_path)
    result = tool.call(path="empty.py")
    assert "no symbols" in result
