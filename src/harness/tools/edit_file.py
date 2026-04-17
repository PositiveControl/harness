from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec


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
                "Edit a file in the workspace. Prefer this over "
                "write_file for changes to existing files — it's "
                "cheaper in tokens and safer against truncation.\n\n"
                "Two modes:\n"
                "  • REPLACE: set `old_string` to the exact text to "
                "find (including whitespace) and `new_string` to the "
                "replacement. `old_string` must be unique in the file "
                "unless `replace_all=true`. Include surrounding lines "
                "when the literal target repeats.\n"
                "  • APPEND: leave `old_string` empty and `new_string` "
                "is added to the end of the file. Use this for "
                "'add a line to .gitignore' / 'append an entry' style "
                "requests.\n\n"
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
        if old_string and old_string == new_string:
            raise ValueError("old_string and new_string are identical — edit is a no-op")

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

        # Append mode: empty old_string → new_string goes at the end.
        if not old_string:
            updated = original + new_string
            target.write_text(updated)
            return f"appended to {path}: +{len(new_string)} bytes"

        count = original.count(old_string)
        if count == 0:
            raise ValueError(
                f"old_string not found in {path}. Re-read the file and match "
                f"it exactly, including indentation and trailing whitespace."
            )
        if count > 1 and not replace_all:
            raise ValueError(
                f"old_string matches {count} places in {path}. Add surrounding "
                f"context to make it unique, or pass replace_all=true to "
                f"replace every match."
            )

        updated = (
            original.replace(old_string, new_string)
            if replace_all
            else original.replace(old_string, new_string, 1)
        )
        target.write_text(updated)

        applied = count if replace_all else 1
        delta = len(updated) - len(original)
        sign = "+" if delta >= 0 else ""
        return f"edited {path}: {applied} replacement(s), {sign}{delta} bytes"
