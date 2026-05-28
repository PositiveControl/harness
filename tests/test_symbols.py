"""Unit tests for tools/_symbols.py — harness-2aes.

Pins the three primitives (outline / enclosing_symbol / resolve_symbol),
the qualified-name + nesting-depth contract, multi-language coverage
(Python + JavaScript + Go), the ambiguity-is-a-list contract that the
read tools depend on, and the graceful-degradation paths
(SymbolsUnavailableError on unsupported extension + on a missing [code]
extra). Tests parse real grammars via tree-sitter-language-pack and skip
if the extra is not installed, so the suite stays portable."""

from __future__ import annotations

import builtins

import pytest

from harness.tools._symbols import (
    SymbolsUnavailableError,
    enclosing_symbol,
    language_for,
    outline,
    resolve_symbol,
)

# Skip the parsing tests when the [code] extra is absent — the degraded
# paths are tested separately and never need a parser.
try:
    import tree_sitter_language_pack  # noqa: F401

    _HAS_CODE = True
except ImportError:  # pragma: no cover - exercised only on lean installs
    _HAS_CODE = False

_requires_code = pytest.mark.skipif(not _HAS_CODE, reason="requires the [code] extra")


PY_SRC = """import os


class Foo:
    def bar(self, x):
        return x + 1

    def baz(self):
        return 2


def top(a, b):
    return a + b
"""


# --- language detection -------------------------------------------


def test_language_for_known_and_unknown() -> None:
    assert language_for("a.py") == "python"
    assert language_for("WIDGET.JS") == "javascript"  # case-insensitive
    assert language_for("main.go") == "go"
    assert language_for("notes.txt") is None
    assert language_for("Makefile") is None


# --- outline (Python) ---------------------------------------------


@_requires_code
def test_outline_python_names_kinds_and_qualified_paths() -> None:
    syms = outline(PY_SRC, filename="m.py")
    by_qual = {s.qualified_name: s for s in syms}
    assert set(by_qual) == {"Foo", "Foo.bar", "Foo.baz", "top"}

    foo = by_qual["Foo"]
    assert foo.kind == "class_definition"
    assert foo.depth == 0

    bar = by_qual["Foo.bar"]
    assert bar.name == "bar"
    assert bar.depth == 1
    assert bar.kind == "function_definition"

    top = by_qual["top"]
    assert top.depth == 0
    assert top.name == "top"


@_requires_code
def test_outline_is_document_order() -> None:
    quals = [s.qualified_name for s in outline(PY_SRC, filename="m.py")]
    assert quals == ["Foo", "Foo.bar", "Foo.baz", "top"]


@_requires_code
def test_outline_line_ranges_are_one_based_inclusive() -> None:
    by_qual = {s.qualified_name: s for s in outline(PY_SRC, filename="m.py")}
    bar = by_qual["Foo.bar"]
    # `def bar` is on line 5, its body return on line 6.
    assert bar.start_line == 5
    assert bar.end_line == 6
    # body_start_line points at the `return` (block start), not the def.
    assert bar.body_start_line == 6


@_requires_code
def test_outline_handles_multibyte_source() -> None:
    # A non-ASCII identifier + string body: byte offsets must index the
    # UTF-8 bytes correctly or the name comes back garbled.
    src = 'def naïve():\n    return "café"\n'
    syms = outline(src, filename="m.py")
    assert [s.name for s in syms] == ["naïve"]


# --- enclosing_symbol ---------------------------------------------


@_requires_code
def test_enclosing_symbol_picks_innermost() -> None:
    # Line 6 is `return x + 1` inside Foo.bar (nested in Foo).
    inner = enclosing_symbol(PY_SRC, filename="m.py", line=6)
    assert inner is not None
    assert inner.qualified_name == "Foo.bar"


@_requires_code
def test_enclosing_symbol_outside_any_def_is_none() -> None:
    # Line 1 is the import — outside every definition.
    assert enclosing_symbol(PY_SRC, filename="m.py", line=1) is None


# --- resolve_symbol -----------------------------------------------


@_requires_code
def test_resolve_symbol_bare_name() -> None:
    matches = resolve_symbol(PY_SRC, filename="m.py", name="bar")
    assert [s.qualified_name for s in matches] == ["Foo.bar"]


@_requires_code
def test_resolve_symbol_qualified_wins_outright() -> None:
    matches = resolve_symbol(PY_SRC, filename="m.py", name="Foo.baz")
    assert [s.qualified_name for s in matches] == ["Foo.baz"]


@_requires_code
def test_resolve_symbol_ambiguity_returns_all() -> None:
    # Two classes, each with an __init__ — a bare reference is ambiguous
    # and must return both, not silently pick one.
    src = (
        "class A:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "\n"
        "class B:\n"
        "    def __init__(self):\n"
        "        pass\n"
    )
    matches = resolve_symbol(src, filename="m.py", name="__init__")
    quals = sorted(s.qualified_name for s in matches)
    assert quals == ["A.__init__", "B.__init__"]


@_requires_code
def test_resolve_symbol_no_match_is_empty() -> None:
    assert resolve_symbol(PY_SRC, filename="m.py", name="does_not_exist") == []


# --- non-Python coverage ------------------------------------------


@_requires_code
def test_outline_javascript() -> None:
    js = (
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
    by_qual = {s.qualified_name: s for s in outline(js, filename="w.js")}
    assert "Widget" in by_qual
    assert "Widget.render" in by_qual
    assert "helper" in by_qual
    assert by_qual["Widget.render"].depth == 1


@_requires_code
def test_outline_go() -> None:
    go = (
        "package main\n"
        "\n"
        "func Add(a int, b int) int {\n"
        "\treturn a + b\n"
        "}\n"
        "\n"
        "func (s *Server) Handle() {\n"
        "}\n"
    )
    names = {s.name for s in outline(go, filename="s.go")}
    assert "Add" in names
    assert "Handle" in names


# --- graceful degradation -----------------------------------------


def test_unsupported_extension_raises_symbols_unavailable() -> None:
    with pytest.raises(SymbolsUnavailableError, match="unsupported extension"):
        outline("plain text\n", filename="notes.txt")


def test_missing_code_extra_raises_symbols_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a lean install where tree-sitter is not importable: the
    # extension is known (so we get past language detection), but the
    # lazy import inside _get_parser fails.
    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "tree_sitter_language_pack":
            raise ImportError("simulated missing [code] extra")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(SymbolsUnavailableError, match="\\[code\\] extra"):
        outline("def f():\n    pass\n", filename="m.py")
