from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec


@dataclass
class WriteFileTool:
    """Write text to a file within the workspace root. Write-tier —
    requires user confirmation before each call (until the user
    approves this tool for the session). Creates parent dirs. Refuses
    to overwrite an existing file unless `overwrite=True` is passed
    explicitly — that guardrail exists because models reach for
    `write_file` when they mean 'modify', and the previous always-
    overwrite behavior silently destroyed user content."""

    root: Path

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write_file",
            description=(
                "Create a NEW file in the workspace. Use `edit_file` "
                "for changes to an existing file — `write_file` always "
                "writes the full content you pass and will refuse to "
                "overwrite an existing file unless you explicitly set "
                "`overwrite=true` (which destroys the previous "
                "content entirely). Creates parent directories as "
                "needed. Path is relative to the workspace root; "
                "cannot escape it. User confirmation is required."
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
                        "description": "Full file content to write",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "If true, replace an existing file's entire "
                            "content with `content`. Default false. "
                            "Prefer edit_file for partial changes."
                        ),
                    },
                },
                "required": ["path", "content"],
            },
            tier="write",
            display_name="Write file",
        )

    def call(self, *, path: str, content: str, overwrite: bool = False) -> str:
        root = self.root.resolve()
        target = (self.root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path {path!r} escapes workspace root") from exc
        pre_existed = target.exists()
        if pre_existed and not overwrite:
            raise ValueError(
                f"{path} already exists. Use edit_file for partial changes, "
                f"or pass overwrite=true to replace the entire file "
                f"(destroys previous content)."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        action = "overwrote" if pre_existed else "wrote"
        return f"{action} {len(content)} chars to {path}"
