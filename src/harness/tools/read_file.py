from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec


@dataclass
class ReadFileTool:
    """Read a text file from within the workspace root. Safe for
    read-tier — can't escape the root via `..`, truncates at a cap so
    a 10 MB file can't fill the context window."""

    root: Path
    max_bytes: int = 200_000

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_file",
            description=(
                "Read a text file from the workspace. Returns the file's "
                "contents (truncated to ~200 KB). Path is relative to the "
                "workspace root; cannot escape it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root",
                    }
                },
                "required": ["path"],
            },
            tier="read",
        )

    def call(self, *, path: str) -> str:
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
        data = target.read_bytes()
        truncated = len(data) > self.max_bytes
        text = data[: self.max_bytes].decode("utf-8", errors="replace")
        if truncated:
            text += f"\n\n… [truncated at {self.max_bytes} bytes of {len(data)}]"
        return text
