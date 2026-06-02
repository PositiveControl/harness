from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from harness.tools.base import ToolSpec, tool_schema_from_model


class ReadFileArgs(BaseModel):
    """Typed arguments for read_file (harness-5cjj9). Numeric strings
    coerce to int by default in pydantic's lax mode, so `offset='20'`
    still works (harness-296bh) without the hand-rolled coercion."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path relative to the workspace root")
    offset: int | None = Field(
        default=None,
        description=(
            "1-based line number to start reading from. Default 1 (start "
            "of file). Values <1 are rejected."
        ),
    )
    limit: int | None = Field(
        default=None,
        description=(
            "Maximum number of lines to return starting at `offset`. "
            "Default: read to end of file. Values <1 are rejected."
        ),
    )
    symbol: str | None = Field(
        default=None,
        description=(
            "Name of a function/class/method to read whole. Bare (`bar`) "
            "or qualified (`Foo.bar`). Returns the entire enclosing span "
            "plus a `[symbol NAME, lines X-Y of N]` marker. If the name is "
            "ambiguous, the response lists the candidate qualified names "
            "instead. Mutually exclusive with offset/limit."
        ),
    )


def _coerce_line_arg(value: int | str | None, *, name: str) -> int | None:
    """harness-296bh: small models routinely emit numeric tool args as
    JSON strings (e.g. ``offset='20'``). The schema says integer, so the
    raw call used to fail and burn a round + a retry (run b085854e turn
    1: ``read_file offset='20' limit='20'`` → tool_call_failed). Accept a
    digit string and coerce it; reject only genuinely non-numeric input
    with a clear message. ``None`` passes through unchanged."""
    if value is None or isinstance(value, int):
        return value
    text = value.strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc


@dataclass
class ReadFileTool:
    """Read a text file from within the workspace root. Safe for
    read-tier — can't escape the root via `..`, truncates at a cap so
    a 10 MB file can't fill the context window.

    Supports line-range slicing via `offset` (1-based start line) and
    `limit` (max lines to return), matching the standard tool prior
    most chat models carry (harness-0tni). A trailing
    `[showing lines X-Y of N]` marker tells the model what slice
    landed and how much of the file remains.

    Also supports symbol addressing via `symbol` (harness-2kob): ask for
    a function/class/method by name (bare `bar` or qualified `Foo.bar`)
    and get the whole enclosing span back, resolved at call time via
    tree-sitter — no guessing offsets, no half-functions. Mutually
    exclusive with offset/limit. Degrades to a guidance note when the
    `[code]` extra is absent or the language is unsupported."""

    root: Path
    max_bytes: int = 200_000

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_file",
            description=(
                "Read a text file from the workspace. Returns the file's "
                "contents (truncated to ~200 KB). Path is relative to the "
                "workspace root; cannot escape it. Optional `offset` "
                "(1-based start line, default=1) and `limit` (max lines, "
                "default=all) let you read a slice; the response appends "
                "`[showing lines X-Y of N]` so you know where you are. "
                "To read a whole function/class/method by name instead of "
                "guessing line numbers, pass `symbol` (e.g. `Foo.bar`) — it "
                "returns the entire enclosing span. `symbol` cannot be "
                "combined with `offset`/`limit`."
            ),
            parameters=tool_schema_from_model(ReadFileArgs),
            args_model=ReadFileArgs,
            tier="read",
            display_name="Read file",
        )

    def call(
        self,
        *,
        path: str,
        offset: int | str | None = None,
        limit: int | str | None = None,
        symbol: str | None = None,
    ) -> str:
        offset = _coerce_line_arg(offset, name="offset")
        limit = _coerce_line_arg(limit, name="limit")
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
        if symbol is not None:
            if offset is not None or limit is not None:
                raise ValueError(
                    "symbol cannot be combined with offset/limit — "
                    "pick one addressing mode (a symbol span, or a line slice)"
                )
            return self._read_symbol(target.read_bytes(), path=path, symbol=symbol)
        if offset is not None and offset < 1:
            raise ValueError(f"offset must be >= 1 (1-based line number); got {offset}")
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be >= 1; got {limit}")
        data = target.read_bytes()
        byte_truncated = len(data) > self.max_bytes
        text = data[: self.max_bytes].decode("utf-8", errors="replace")

        # No slicing requested — preserve the legacy whole-file behavior.
        if offset is None and limit is None:
            if byte_truncated:
                text += f"\n\n… [truncated at {self.max_bytes} bytes of {len(data)}]"
            return text

        # Slice by lines. splitlines(keepends=True) preserves trailing
        # newlines so the joined output round-trips faithfully — important
        # for `edit_file`'s old_string matching when the model is iterating
        # on a fresh slice.
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        start_idx = (offset - 1) if offset is not None else 0
        # When offset is past EOF, return empty body + an explanatory note
        # instead of an opaque empty string. Keeps the model from re-issuing
        # the same call thinking it failed silently.
        if start_idx >= total_lines:
            return (
                f"[no lines in range — file has {total_lines} lines, "
                f"requested offset={offset or 1}]"
            )
        end_idx = (start_idx + limit) if limit is not None else total_lines
        sliced = lines[start_idx:end_idx]
        actual_first = start_idx + 1
        actual_last = start_idx + len(sliced)
        body = "".join(sliced)
        marker = f"\n\n[showing lines {actual_first}-{actual_last} of {total_lines}]"
        if byte_truncated:
            marker += f" [file also byte-truncated at {self.max_bytes} of {len(data)}]"
        return body + marker

    def _read_symbol(self, data: bytes, *, path: str, symbol: str) -> str:
        """Resolve `symbol` against the file and return the whole enclosing
        span, a candidate list when ambiguous, or a guidance note when the
        symbol is absent or symbol indexing is unavailable. The full file is
        parsed (not the byte-capped slice) so resolution sees every
        definition; only the returned span is subject to the byte cap.
        """
        # Lazy import so the [code] extra stays optional: a base install
        # that never passes `symbol` never imports tree-sitter.
        from harness.tools._symbols import SymbolsUnavailableError, resolve_symbol

        full_text = data.decode("utf-8", errors="replace")
        try:
            matches = resolve_symbol(full_text, filename=path, name=symbol)
        except SymbolsUnavailableError as exc:
            return f"[symbol read unavailable: {exc}] Re-read {path!r} with offset/limit instead."
        if not matches:
            return (
                f"[no symbol named {symbol!r} in {path}] "
                f"Read the file without `symbol` to see what it defines."
            )
        if len(matches) > 1:
            head = (
                f"[{len(matches)} symbols match {symbol!r} in {path} — "
                f"re-ask with a qualified name]"
            )
            rows = [f"  {s.qualified_name}  [lines {s.start_line}-{s.end_line}]" for s in matches]
            return "\n".join([head, *rows])

        sym = matches[0]
        lines = full_text.splitlines(keepends=True)
        total_lines = len(lines)
        span = "".join(lines[sym.start_line - 1 : sym.end_line])
        marker = (
            f"\n\n[symbol {sym.qualified_name}, "
            f"lines {sym.start_line}-{sym.end_line} of {total_lines}]"
        )
        span_bytes = span.encode("utf-8")
        if len(span_bytes) > self.max_bytes:
            span = span_bytes[: self.max_bytes].decode("utf-8", errors="replace")
            marker += f" [span byte-truncated at {self.max_bytes} of {len(span_bytes)}]"
        return span + marker
