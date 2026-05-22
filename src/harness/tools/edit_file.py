from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec
from harness.tools.parse_check import parse_check

# 1 MB cap on inlined file contents in error messages. For 200-line
# game.js (~6 KB) this is trivially under; for a million-line monolith
# we truncate rather than blow out the model's context. Constrained-
# model design: better than nothing, bounded against pathological size.
_INLINE_CONTENTS_CAP_BYTES = 1024 * 1024


def _format_file_contents(path: str, contents: str) -> str:
    """Render `contents` as a delimited block the model can quote
    verbatim. Cap at `_INLINE_CONTENTS_CAP_BYTES` so a runaway file
    doesn't balloon the response. Bracketed with explicit BEGIN/END
    markers so the model knows where the live data is. Public for
    tests."""
    if len(contents) > _INLINE_CONTENTS_CAP_BYTES:
        head = contents[:_INLINE_CONTENTS_CAP_BYTES]
        suffix = (
            f"\n... [truncated; file is {len(contents)} bytes, "
            f"capped at {_INLINE_CONTENTS_CAP_BYTES} for inline display]"
        )
        contents = head + suffix
    return f"--- CURRENT CONTENTS OF {path} (BEGIN) ---\n{contents}\n--- END {path} ---"


# Line counts that shape the context block. These thresholds make a
# small, medium, and large edit each produce a predictably-sized
# excerpt — context fits a few hundred tokens at most.
_CONTEXT_FULL_THRESHOLD_LINES = 25
_CONTEXT_EDGE_WINDOW_LINES = 10
_CONTEXT_MARGIN_LINES = 2


def _format_replace_context(
    updated: str,
    splice_offset: int,
    new_string: str,
    path: str,
) -> str:
    """harness-b2xa: render a context block showing the lines the
    splice landed on. Appended to the success message so the model
    sees the actual landed region instead of only a byte count —
    closing the 'edit_file made the model edit blind' gap that
    surfaced in loop run 26c39558.

    Strategy: count lines up to the splice, slice the file at that
    region with a small margin, and present them with 1-based line
    numbers. For splices longer than `_CONTEXT_FULL_THRESHOLD_LINES`
    show only the head and tail edge windows with an omitted-line
    hint — so a 2 KB block insert doesn't dump 100 lines into the
    tool result. Returns the empty string when the file is short
    enough that the model could trivially `read_file` the whole
    thing (≤25 total lines)."""
    lines = updated.splitlines()
    total = len(lines)
    if total <= _CONTEXT_FULL_THRESHOLD_LINES:
        return ""

    # `splice_offset` is the byte index where new_string begins in
    # `updated`. Count newlines BEFORE that offset to get the 1-based
    # starting line; new_string's own newline count gives the span.
    start_line = updated.count("\n", 0, splice_offset) + 1
    splice_line_count = max(new_string.count("\n") + (1 if new_string else 0), 1)
    end_line = start_line + splice_line_count - 1

    def _render_range(start_1based: int, end_1based: int) -> str:
        return "\n".join(
            f"  {i}: {lines[i - 1]}" for i in range(start_1based, min(end_1based, total) + 1)
        )

    if splice_line_count <= _CONTEXT_FULL_THRESHOLD_LINES:
        window_start = max(1, start_line - _CONTEXT_MARGIN_LINES)
        window_end = min(total, end_line + _CONTEXT_MARGIN_LINES)
        body = _render_range(window_start, window_end)
        return (
            f"\n\ncontext (lines {window_start}-{window_end} of {total}, "
            f"splice spans {start_line}-{end_line}):\n{body}"
        )

    # Long splice — show only the edges so the result stays bounded.
    head_end = start_line + _CONTEXT_EDGE_WINDOW_LINES - 1
    tail_start = end_line - _CONTEXT_EDGE_WINDOW_LINES + 1
    omitted = max(tail_start - head_end - 1, 0)
    head_body = _render_range(start_line, head_end)
    tail_body = _render_range(tail_start, end_line)
    hint = (
        f"  ... {omitted} lines omitted; "
        f"read_file(path={path!r}, offset={start_line}, limit={splice_line_count}) "
        f"to inspect the full splice region ..."
    )
    return (
        f"\n\ncontext (splice spans lines {start_line}-{end_line}, "
        f"{splice_line_count} lines, of {total} total):\n"
        f"{head_body}\n{hint}\n{tail_body}"
    )


def _enforce_parse_check(target: Path, path: str) -> None:
    """harness-h6wa: run the extension-specific syntax check after a
    successful write. On parser-rejected output, raise a ValueError so
    the registry surfaces a ToolResult(success=False) carrying the
    parser's stderr. The file is intentionally left on disk in the
    broken state — rolling back would force the model to re-derive the
    entire edit; surfacing the parser output lets it target the actual
    syntax error in the next round."""
    ok, detail = parse_check(target)
    if ok:
        return
    raise ValueError(
        f"edit landed on disk but {path} no longer parses. Read the "
        f"file and either fix the syntax error or back out the bad "
        f"edit. Parser output:\n{detail}"
    )


@dataclass
class EditFileTool:
    """Edit an existing workspace file: either replace an exact-string
    occurrence, or append new text to the end of the file. Write-tier
    — requires user confirmation. Biggest win over `write_file`: edits
    don't re-emit the whole file, so the model spends ~hundreds of
    tokens instead of thousands and can't silently drop content when
    its output truncates.

    Three modes:
      1. Replace (default). `old_string` appears in the file; we
         substitute `new_string`. Must be unique unless `replace_all`.
      2. Append. `old_string` is empty; `new_string` is added to the
         end of the file. Common case: 'add a line to .gitignore'.
      3. No-op rejected. If old and new are both empty, or equal, the
         tool errors so the model doesn't hallucinate success."""

    root: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="edit_file",
            description=(
                "Edit an EXISTING file in the workspace. Use this "
                "(not write_file) for every change to a file that "
                "already exists — it's cheaper in tokens and never "
                "destroys content you didn't touch.\n\n"
                "Two modes:\n"
                "  • APPEND (most common for 'add X to Y'): leave "
                "`old_string` EMPTY and `new_string` is added to the "
                "end of the file. Example — 'add scratch to "
                '.gitignore\' → edit_file(path=".gitignore", '
                'old_string="", new_string="scratch\\n"). '
                "Remember the trailing newline so the next entry "
                "lands on its own line.\n"
                "  • REPLACE: set `old_string` to the exact text to "
                "find (including whitespace) and `new_string` to the "
                "replacement. `old_string` must be unique in the file "
                "unless `replace_all=true`. Include surrounding lines "
                "when the literal target repeats.\n\n"
                "Returns a summary of what changed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact text to replace (including whitespace). "
                            "Empty string = append `new_string` to end of file."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": (
                            "Replacement or appended text. Must differ from "
                            "old_string when replacing."
                        ),
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "If true, replace every occurrence. Default false — "
                            "requires old_string to be unique in the file. "
                            "Ignored when old_string is empty (append mode)."
                        ),
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
            tier="write",
            display_name="Edit file",
        )

    def call(
        self,
        *,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        if not old_string and not new_string:
            raise ValueError("old_string and new_string are both empty — nothing to do")

        # Path validation + read happens FIRST so the helpful error
        # messages below can include current file contents. The model
        # routinely hallucinates an old_string that doesn't match the
        # actual file; without ground truth in the error response the
        # next round emits a near-identical broken call (harness-w0gw).
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

        try:
            original = target.read_text()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path} is not a UTF-8 text file") from exc

        if old_string and old_string == new_string:
            raise ValueError(
                "old_string and new_string are identical — edit is a no-op. "
                "Your edit doesn't change anything. Pick a different "
                "new_string, or revise old_string to point at code you "
                "actually intend to change.\n" + _format_file_contents(path, original)
            )

        # Append mode: empty old_string → new_string goes at the end.
        if not old_string:
            updated = original + new_string
            target.write_text(updated)
            _enforce_parse_check(target, path)
            return f"appended to {path}: +{len(new_string)} bytes"

        count = original.count(old_string)
        if count == 0:
            raise ValueError(
                f"old_string not found in {path}. Use the file contents "
                f"below to construct an old_string that matches verbatim "
                f"(including indentation and trailing whitespace). Do NOT "
                f"re-emit the same edit — that won't help.\n"
                + _format_file_contents(path, original)
            )
        if count > 1 and not replace_all:
            raise ValueError(
                f"old_string matches {count} places in {path}. Add "
                f"surrounding context to make it unique, or pass "
                f"replace_all=true to replace every match.\n"
                + _format_file_contents(path, original)
            )

        # Capture the splice offset BEFORE the replace runs — for a
        # single-match edit this equals `original.find(old_string)`,
        # which (because the offset is the same in `updated`) tells
        # us where new_string lands. Used to render the context
        # block (harness-b2xa). `replace_all` skips context entirely
        # because multi-splice excerpts would just be noise.
        splice_offset = original.find(old_string) if not replace_all else -1

        updated = (
            original.replace(old_string, new_string)
            if replace_all
            else original.replace(old_string, new_string, 1)
        )
        target.write_text(updated)
        _enforce_parse_check(target, path)

        applied = count if replace_all else 1
        delta = len(updated) - len(original)
        sign = "+" if delta >= 0 else ""
        message = f"edited {path}: {applied} replacement(s), {sign}{delta} bytes"
        if not replace_all:
            message += _format_replace_context(updated, splice_offset, new_string, path)
        return message
