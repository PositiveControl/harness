from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec


@dataclass
class ReadFileTool:
    """Read a text file from within the workspace root. Safe for
    read-tier — can't escape the root via `..`, truncates at a cap so
    a 10 MB file can't fill the context window.

    Supports line-range slicing via `offset` (1-based start line) and
    `limit` (max lines to return), matching the standard tool prior
    most chat models carry (harness-0tni). A trailing
    `[showing lines X-Y of N]` marker tells the model what slice
    landed and how much of the file remains."""

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
                "`[showing lines X-Y of N]` so you know where you are."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root",
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "1-based line number to start reading from. "
                            "Default 1 (start of file). Values <1 are rejected."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of lines to return starting at "
                            "`offset`. Default: read to end of file. Values "
                            "<1 are rejected."
                        ),
                    },
                },
                "required": ["path"],
            },
            tier="read",
            display_name="Read file",
        )

    def call(
        self,
        *,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
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
