from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec


@dataclass
class EditFileTool:
    """Replace exact-string occurrences inside a workspace file.
    Write-tier — requires user confirmation. Biggest win over
    `write_file`: edits don't re-emit the whole file, so the model
    spends ~hundreds of tokens instead of thousands and can't silently
    drop content when its output truncates.

    Contract: `old_string` must appear in the file. By default it must
    appear exactly once (the model should include enough surrounding
    context to disambiguate) — pass `replace_all=true` to replace every
    occurrence. A no-op (old == new) is an error: if the model thinks
    it's editing but isn't, we want the tool loop to see it and try
    again rather than declare success."""

    root: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="edit_file",
            description=(
                "Edit a file in the workspace by replacing an exact "
                "string. Prefer this over write_file for changes to "
                "existing files — it's cheaper in tokens and safer "
                "against truncation. `old_string` must match the "
                "current file contents exactly (whitespace included) "
                "and must be unique unless `replace_all=true`. Include "
                "surrounding lines to disambiguate when the literal "
                "target repeats in the file. Returns a summary of how "
                "many replacements were applied."
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
                            "Exact text to replace. Must appear in the file "
                            "verbatim, whitespace and newlines included."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text. Must differ from old_string.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "If true, replace every occurrence. Default false — "
                            "requires old_string to be unique in the file."
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
        if not old_string:
            raise ValueError("old_string must not be empty; use write_file to create a file")
        if old_string == new_string:
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
