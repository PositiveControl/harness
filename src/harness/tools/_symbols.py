"""Language-agnostic symbol indexing over source text — harness-2aes.

The foundation for symbol-addressed reads (harness-2kob) and the
outline-and-expand tool (harness-ywhf), under epic harness-umvg. The
problem it solves: `read_file` addresses code by line offset/limit, so
the model gets handed half-functions and over-reads whole files. A
symbol layer lets the read tools say "give me the whole span named
`Foo.bar`" or "show me a signatures-only skeleton" instead.

Design invariants:

  * Text stays source of truth. This module is a *derived* index built
    on demand from source the caller already holds — there is no
    parallel code store, no serializer, no round-trip. The rejected
    alternative (JSON/AST-as-truth, i.e. projectional editing) is why.
  * Stateless + IO-free. Callers (the file-ops tools) own the
    workspace-escape guard and the byte cap; they read the file and
    hand us `(source, filename)`. We never touch the filesystem, so the
    sandbox lives in exactly one place.
  * Graceful degradation. tree-sitter is gated behind the `[code]`
    extra and imported lazily. When the extra is absent, the extension
    is unknown, or the language is not configured, we raise
    `SymbolsUnavailableError` — the tools catch it and fall back to
    line-based reads. Same skip-policy spirit as `parse_check`.

Language coverage rides on `tree-sitter-language-pack`, which bundles
~165 precompiled grammars in one wheel — that is what makes this
multi-language instead of Python-only. We map a conservative,
*reliably-named* set of definition node kinds per language; nameless
constructs (e.g. Rust `impl` blocks, Go type specs) are skipped rather
than guessed, and can be added later behind tests.

Binding note: the parser shipped by tree-sitter-language-pack exposes
node accessors as **methods**, not properties — `node.kind()`,
`node.start_byte()`, `node.start_position().row()`. Byte offsets are
UTF-8 byte offsets, so spans are sliced from `source.encode("utf-8")`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SymbolsUnavailableError(Exception):
    """Symbol extraction could not run for this input. Raised when the
    `[code]` extra is not installed, the file extension maps to no
    grammar, or the language has no configured definition kinds. Callers
    treat it as "degrade to line-based reads", never as a hard error."""


@dataclass(frozen=True)
class Symbol:
    """One definition (function / class / method / …) found in a file.

    Line numbers are 1-based and inclusive — the same convention
    `read_file` uses for its `[showing lines X-Y]` marker, so a tool can
    feed `start_line`/`end_line` straight into a line slice.
    """

    name: str  # bare identifier, e.g. "bar"
    kind: str  # tree-sitter node kind, e.g. "function_definition"
    qualified_name: str  # dotted scope chain, e.g. "Foo.bar"
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    depth: int  # nesting depth: 0 = top-level, 1 = one def deep, …
    # 1-based line where this symbol's body block begins. The
    # outline-and-expand tool (harness-ywhf) uses it to elide bodies:
    # the signature is roughly start_line..body_start_line-1. Falls back
    # to start_line for one-liners / bodyless declarations.
    body_start_line: int


# Extension → language token understood by tree-sitter-language-pack's
# get_parser(). Lowercased suffixes; the same token keys _DEF_KINDS.
_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".php": "php",
}

# Per-language node kinds we treat as addressable symbols. Deliberately
# conservative: every kind here exposes a reliable name (via a `name`
# field or an identifier-ish first child). Constructs whose name lives
# deeper (Rust `impl_item`, Go `type_declaration`) are omitted until a
# test-backed extractor exists — better to miss a symbol than to emit a
# nameless or wrongly-named one.
_DEF_KINDS: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition", "class_definition"}),
    "javascript": frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
        }
    ),
    "typescript": frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
            "interface_declaration",
            "enum_declaration",
            "type_alias_declaration",
        }
    ),
    "tsx": frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
            "interface_declaration",
            "enum_declaration",
            "type_alias_declaration",
        }
    ),
    "go": frozenset({"function_declaration", "method_declaration"}),
    "rust": frozenset({"function_item", "struct_item", "enum_item", "trait_item", "mod_item"}),
    "ruby": frozenset({"method", "singleton_method", "class", "module"}),
    "java": frozenset(
        {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "method_declaration",
            "constructor_declaration",
        }
    ),
    "c": frozenset({"function_definition"}),
    "cpp": frozenset({"function_definition", "class_specifier", "struct_specifier"}),
    "csharp": frozenset(
        {
            "class_declaration",
            "struct_declaration",
            "interface_declaration",
            "enum_declaration",
            "method_declaration",
        }
    ),
    "php": frozenset(
        {
            "function_definition",
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
        }
    ),
}

# Node kinds that, as the first named child of a definition, carry the
# definition's name when there is no `name` field. Covers Ruby
# (constant/identifier), Java/JS (identifier), Rust (type_identifier).
_NAME_NODE_KINDS: frozenset[str] = frozenset(
    {
        "identifier",
        "type_identifier",
        "constant",
        "field_identifier",
        "scoped_identifier",
        "property_identifier",
        "name",
    }
)

# Node kinds that hold a definition's body. Used to locate where the
# signature ends and the body begins (body_start_line).
_BODY_KINDS: frozenset[str] = frozenset(
    {
        "block",
        "statement_block",
        "class_body",
        "declaration_list",
        "field_declaration_list",
        "enum_body",
        "interface_body",
        "body",
        "compound_statement",
    }
)


def language_for(filename: str) -> str | None:
    """Return the grammar token for `filename`'s extension, or None when
    we have no grammar mapping for it. Case-insensitive on the suffix."""
    return _EXT_LANG.get(Path(filename).suffix.lower())


def _get_parser(language: str) -> Any:
    """Lazily build a parser for `language`. Raises SymbolsUnavailableError
    when the `[code]` extra is missing or the grammar is not bundled —
    the single place the optional dependency is touched."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:  # extra not installed
        raise SymbolsUnavailableError(
            "symbol indexing needs the [code] extra "
            "(uv sync --extra all) — tree-sitter is not importable"
        ) from exc
    try:
        return get_parser(language)
    except LookupError as exc:  # grammar token unknown to the pack
        raise SymbolsUnavailableError(f"no tree-sitter grammar bundled for {language!r}") from exc


def _name_of(node: Any, source_bytes: bytes) -> str | None:
    """Extract a definition node's name: the `name` field if present,
    else the first identifier-ish named child. None when neither exists
    (caller skips the node rather than emitting a nameless symbol)."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        for i in range(node.named_child_count()):
            child = node.named_child(i)
            if child.kind() in _NAME_NODE_KINDS:
                name_node = child
                break
    if name_node is None:
        return None
    return source_bytes[name_node.start_byte() : name_node.end_byte()].decode("utf-8", "replace")


def _body_start(node: Any) -> int:
    """1-based line where `node`'s body block starts, or its own start
    line when no body child is found (one-liners, bodyless decls)."""
    for i in range(node.named_child_count()):
        child = node.named_child(i)
        if child.kind() in _BODY_KINDS:
            return int(child.start_position().row) + 1
    return int(node.start_position().row) + 1


def _walk(
    node: Any,
    def_kinds: frozenset[str],
    scope: tuple[str, ...],
    source_bytes: bytes,
    out: list[Symbol],
) -> None:
    """Pre-order DFS collecting definition nodes in document order. Only
    definition names extend the scope chain, so a method inside a class
    becomes `Class.method` while a function nested in an `if` block keeps
    its own bare name."""
    child_scope = scope
    if node.kind() in def_kinds:
        name = _name_of(node, source_bytes)
        if name is not None:
            qualified = ".".join((*scope, name))
            out.append(
                Symbol(
                    name=name,
                    kind=node.kind(),
                    qualified_name=qualified,
                    start_line=int(node.start_position().row) + 1,
                    end_line=int(node.end_position().row) + 1,
                    depth=len(scope),
                    body_start_line=_body_start(node),
                )
            )
            child_scope = (*scope, name)
    for i in range(node.named_child_count()):
        _walk(node.named_child(i), def_kinds, child_scope, source_bytes, out)


def outline(source: str, *, filename: str) -> list[Symbol]:
    """Parse `source` (whose name/extension is `filename`) and return its
    definition symbols in document order, each with its qualified name,
    line range, and nesting depth.

    Raises SymbolsUnavailableError when the extension maps to no grammar, the
    language has no configured definition kinds, or the `[code]` extra is
    not installed.
    """
    language = language_for(filename)
    if language is None:
        raise SymbolsUnavailableError(f"no symbol grammar for {filename!r} (unsupported extension)")
    def_kinds = _DEF_KINDS.get(language)
    if def_kinds is None:
        raise SymbolsUnavailableError(f"symbol extraction not configured for language {language!r}")
    parser = _get_parser(language)
    # This binding wants `str` (it rejects `bytes`); it encodes UTF-8
    # internally, so the byte offsets it reports index the UTF-8 bytes.
    tree = parser.parse(source)
    root = tree.root_node()
    out: list[Symbol] = []
    _walk(root, def_kinds, (), source.encode("utf-8"), out)
    return out


def enclosing_symbol(source: str, *, filename: str, line: int) -> Symbol | None:
    """Return the innermost symbol whose line range contains `line`
    (1-based), or None when `line` falls outside every definition. Ties
    resolve to the most deeply nested, then latest-starting, symbol."""
    containing = [
        s for s in outline(source, filename=filename) if s.start_line <= line <= s.end_line
    ]
    if not containing:
        return None
    return max(containing, key=lambda s: (s.depth, s.start_line))


def resolve_symbol(source: str, *, filename: str, name: str) -> list[Symbol]:
    """Resolve a symbol reference to every matching definition.

    Matching, in order of preference:
      1. Exact qualified-name match (`Foo.bar`) — wins outright when any
         symbol matches, so a fully-qualified reference is unambiguous.
      2. Otherwise bare-name or qualified-suffix match (`bar` matches
         both a top-level `bar` and `Foo.bar`).

    Returns a list, not a scalar: ambiguity (overloads, same-named
    methods on different classes) is first-class, and the read tool
    surfaces the candidate list so the model can re-ask with a qualified
    name rather than silently getting the wrong span.
    """
    target = name.strip()
    symbols = outline(source, filename=filename)
    exact = [s for s in symbols if s.qualified_name == target]
    if exact:
        return exact
    suffix = "." + target
    return [s for s in symbols if s.name == target or s.qualified_name.endswith(suffix)]
