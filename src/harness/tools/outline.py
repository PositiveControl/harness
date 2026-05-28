"""Outline tool — a signatures-only skeleton of a source file (harness-ywhf).

The navigation half of outline-and-expand, under epic harness-umvg.
`read_file` either dumps a whole file (token-expensive, and large results
stall MLX decoding around ~32 KB — the file-ops bench's limitation #1) or
slices arbitrary line ranges (which hands the model half-functions). The
outline tool gives the cheap middle path: every definition's signature
with its body elided and its line range shown, so the model can map a
2,000-line file for a few hundred tokens, then expand the one or two
symbols it actually needs via `read_file path symbol=...`.

Read-tier and workspace-sandboxed, exactly like `read_file`. Built on the
tree-sitter symbol index (harness-2aes); degrades to a guidance note when
the `[code]` extra is absent or the language is unsupported, never a hard
error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec

# Soft ceiling on rendered symbols — a skeleton is meant to fit in a few
# hundred tokens. A file with thousands of defs is pathological; show the
# first chunk and tell the model the rest were elided so it can narrow by
# reading a subtree instead.
_MAX_SYMBOLS = 600


@dataclass
class OutlineTool:
    """Render a file's definition skeleton: indented signatures with line
    ranges, bodies elided. Read-tier — safe for the default coding flow."""

    root: Path
    max_bytes: int = 200_000

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="outline",
            description=(
                "Show a signatures-only skeleton of a source file: every "
                "function/class/method, indented by nesting, with its line "
                "range and body elided. Use it to navigate a large file "
                "cheaply, then read a specific symbol in full with "
                "`read_file path symbol=NAME`. Path is relative to the "
                "workspace root. Optional `max_depth` limits nesting shown "
                "(1 = top-level only). Needs a supported language; returns a "
                "note for plain text or unsupported files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": (
                            "Max nesting level to show (1 = top-level "
                            "definitions only). Default: all levels. "
                            "Values <1 are rejected."
                        ),
                    },
                },
                "required": ["path"],
            },
            tier="read",
            display_name="Outline",
        )

    def call(self, *, path: str, max_depth: int | None = None) -> str:
        root = self.root.resolve()
        target = (self.root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path {path!r} escapes workspace root") from exc
        if not target.exists():
            raise FileNotFoundError(f"{path} not found")
        if not target.is_file():
            raise IsADirectoryError(f"{path} is not a regular file")
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1; got {max_depth}")

        # Lazy import so the [code] extra stays optional.
        from harness.tools._symbols import SymbolsUnavailableError, outline

        text = target.read_bytes().decode("utf-8", errors="replace")
        try:
            symbols = outline(text, filename=path)
        except SymbolsUnavailableError as exc:
            return (
                f"[outline unavailable: {exc}] Read {path!r} with read_file (offset/limit) instead."
            )

        if max_depth is not None:
            symbols = [s for s in symbols if s.depth < max_depth]
        if not symbols:
            return f"[no symbols in {path}] It may be empty, all top-level code, or comments."

        truncated = len(symbols) > _MAX_SYMBOLS
        shown = symbols[:_MAX_SYMBOLS]
        lines = text.splitlines(keepends=True)
        rows = [f"{path}: {len(symbols)} symbols"]
        for i, sym in enumerate(shown):
            indent = "  " * sym.depth
            signature = _signature(lines, sym)
            # A symbol with no nested definition shown beneath it and a
            # real body gets an explicit elision marker; one whose children
            # are listed below it does not (the nesting already shows it).
            has_child = i + 1 < len(shown) and shown[i + 1].depth > sym.depth
            elision = " …" if not has_child and sym.end_line > sym.start_line else ""
            rows.append(f"{indent}{signature}{elision}  [L{sym.start_line}-{sym.end_line}]")
        if truncated:
            rows.append(
                f"… [{len(symbols) - _MAX_SYMBOLS} more symbols elided — "
                f"outline a subdirectory or narrow with max_depth]"
            )
        return "\n".join(rows)


def _signature(lines: list[str], sym: object) -> str:
    """Collapse a definition's header (everything before its body block)
    into one normalized line. Multi-line signatures fold to single-space-
    separated text so the skeleton stays one row per symbol."""
    # `sym` is a _symbols.Symbol; typed as object to avoid importing it at
    # module load (the import is lazy inside call()).
    start = sym.start_line  # type: ignore[attr-defined]
    body = sym.body_start_line  # type: ignore[attr-defined]
    header = lines[start - 1 : body - 1] if body > start else lines[start - 1 : start]
    return " ".join(" ".join(part.split()) for part in header).strip()
