from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec


@dataclass
class WriteFileTool:
    """Write text to a file within the workspace root. Write-tier —
    requires user confirmation before each call (until the user
    approves this tool for the session). Creates parent dirs;
    overwrites existing files."""

    root: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write_file",
            description=(
                "Write text to a file in the workspace. Creates parent "
                "directories if needed. Overwrites existing files. Path "
                "is relative to the workspace root; cannot escape it. "
                "User confirmation is required for this tool."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace root",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write",
                    },
                },
                "required": ["path", "content"],
            },
            tier="write",
        )

    def call(self, *, path: str, content: str) -> str:
        root = self.root.resolve()
        target = (self.root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path {path!r} escapes workspace root") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"
