"""Unit tests for tools/parse_check.py — harness-h6wa.

The helper dispatches on file extension and runs the appropriate parser.
These tests pin the dispatch table, the skip-on-missing-binary policy,
and the (ok, detail) return contract that EditFileTool and WriteFileTool
depend on. Extension-specific behavior tested via the actual `node` /
`python` binaries when available; tests skip otherwise so the suite
stays portable."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.tools.parse_check import parse_check

# --- skip helpers -------------------------------------------------

_HAS_NODE = shutil.which("node") is not None
_HAS_PYTHON = shutil.which("python") is not None

_requires_node = pytest.mark.skipif(
    not _HAS_NODE, reason="parse_check JS path requires `node` on PATH"
)
_requires_python = pytest.mark.skipif(
    not _HAS_PYTHON, reason="parse_check Python path requires `python` on PATH"
)


# --- dispatch ------------------------------------------------------


def test_unsupported_extension_returns_ok(tmp_path: Path) -> None:
    """No parser configured for `.md` (or `.txt`, `.yaml`, etc.) → the
    gate emits no signal. Returning ok=True keeps the write pipeline
    non-fatal for file types the gate doesn't know how to check."""
    target = tmp_path / "notes.md"
    target.write_text("# anything\n")
    ok, detail = parse_check(target)
    assert ok is True
    assert detail == ""


def test_missing_binary_returns_ok(tmp_path: Path) -> None:
    """If the configured parser binary isn't installed (e.g. `node` in
    a Python-only container), the gate skips rather than failing the
    write. Mocked via shutil.which so the test runs even on a box
    that does have node."""
    target = tmp_path / "any.js"
    target.write_text("const x = 1;\n")
    with patch("harness.tools.parse_check.shutil.which", return_value=None):
        ok, detail = parse_check(target)
    assert ok is True
    assert detail == ""


# --- JavaScript (.js) ---------------------------------------------


@_requires_node
def test_valid_js_returns_ok(tmp_path: Path) -> None:
    target = tmp_path / "ok.js"
    target.write_text("const x = 1;\nfunction f() { return x; }\n")
    ok, detail = parse_check(target)
    assert ok is True
    assert detail == ""


@_requires_node
def test_invalid_js_returns_failure_with_stderr(tmp_path: Path) -> None:
    """The exact failure mode from loop run 26c39558 — duplicate
    `const` in the same block scope is a parse-time SyntaxError. The
    gate must surface the parser's text verbatim so the model can
    target the actual line."""
    target = tmp_path / "broken.js"
    target.write_text("function f() {\n  const tileX = 1;\n  const tileX = 2;\n}\n")
    ok, detail = parse_check(target)
    assert ok is False
    # node --check writes a SyntaxError trace; pin the salient strings
    # without over-fitting to node's exact phrasing across versions.
    assert "SyntaxError" in detail or "already been declared" in detail.lower()
    assert "tileX" in detail


@_requires_node
def test_mjs_extension_routed_to_node(tmp_path: Path) -> None:
    """.mjs / .cjs share the JS parser. Pin that the dispatch table
    covers both so an ESM file with the same duplicate-const bug still
    gets caught."""
    target = tmp_path / "broken.mjs"
    target.write_text("export const x = 1;\nexport const x = 2;\n")
    ok, _ = parse_check(target)
    assert ok is False


# --- Python (.py) -------------------------------------------------


@_requires_python
def test_valid_python_returns_ok(tmp_path: Path) -> None:
    target = tmp_path / "ok.py"
    target.write_text("def f():\n    return 42\n")
    ok, detail = parse_check(target)
    assert ok is True
    assert detail == ""


@_requires_python
def test_invalid_python_returns_failure(tmp_path: Path) -> None:
    """py_compile rejects bad syntax with a SyntaxError trace. Pin
    that the gate surfaces it as ok=False."""
    target = tmp_path / "broken.py"
    target.write_text("def f(:\n    return 1\n")  # missing identifier
    ok, detail = parse_check(target)
    assert ok is False
    assert "SyntaxError" in detail or "invalid syntax" in detail.lower()


# --- JSON (.json) -------------------------------------------------


@_requires_python
def test_valid_json_returns_ok(tmp_path: Path) -> None:
    target = tmp_path / "ok.json"
    target.write_text('{"a": 1, "b": [2, 3]}\n')
    ok, detail = parse_check(target)
    assert ok is True
    assert detail == ""


@_requires_python
def test_invalid_json_returns_failure(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text('{"a": 1, "b": ,\n')  # trailing comma + missing value
    ok, detail = parse_check(target)
    assert ok is False
    # Python's JSONDecodeError text includes "Expecting" or "JSON".
    assert "JSON" in detail or "Expecting" in detail


# --- harness-0tni: duplicate-decl enrichment ----------------------


@_requires_node
def test_duplicate_decl_enrichment_lists_all_locations(tmp_path: Path) -> None:
    """harness-0tni: when node flags `Identifier 'X' has already
    been declared`, parse_check appends a grep of WHERE the
    conflicting identifier is declared elsewhere in the file. This
    saves the model a search round on the loop run's most expensive
    failure mode (block-scope blindness)."""
    # Both decls in the same (module-top) scope so node actually
    # flags a duplicate. Function-body redeclaration would be valid JS.
    target = tmp_path / "broken.js"
    target.write_text("let paused = false;\n// some comment\nlet paused = true;\n")
    ok, detail = parse_check(target)
    assert ok is False
    # Original parser output is still there.
    assert "already been declared" in detail
    # Enrichment: both declaration sites listed.
    assert "All declarations of `paused`" in detail
    assert "line 1:" in detail
    assert "line 3:" in detail
    # Helpful resolution hint.
    assert "Remove or rename" in detail


@_requires_node
def test_duplicate_decl_enrichment_skipped_when_single_decl_visible(
    tmp_path: Path,
) -> None:
    """When the parser's own error already points at the conflict
    AND only one matching decl appears in the file (because the
    other is in a different scope visible only to node's symbol
    table), don't bother appending — the parser's message is the
    better signal alone."""
    # Trigger an "already been declared" error where only one decl
    # is grep-visible: declaration inside a destructure pattern that
    # the regex won't match.
    target = tmp_path / "broken.js"
    target.write_text(
        # `const {x} = obj` declares x but our keyword-prefix regex
        # only matches `const x` shapes, so the destructured one
        # isn't found by grep.
        "const x = 1;\nconst {x} = {x: 2};\n"
    )
    ok, detail = parse_check(target)
    assert ok is False
    # Either no enrichment block, or it was suppressed because only
    # one match was found. Pin the no-enrichment branch.
    assert "All declarations of" not in detail


@_requires_node
def test_duplicate_decl_enrichment_no_op_for_non_dup_errors(tmp_path: Path) -> None:
    """A regular SyntaxError (not a redeclaration) shouldn't trigger
    the enrichment — it would just be confusing noise."""
    target = tmp_path / "broken.js"
    target.write_text("function f() {\n  return\n}\n)\n")  # stray paren
    ok, detail = parse_check(target)
    assert ok is False
    assert "All declarations of" not in detail
