from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolSpec
from harness.tools.parse_check import parse_check


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
                "Create a NEW file in the workspace. Do NOT use this "
                "tool to add a line to an existing file, edit an "
                "existing file, or append — use `edit_file` for all "
                "of those (edit_file with empty old_string appends). "
                "`write_file` always writes the full content you pass "
                "and refuses to overwrite an existing file unless you "
                "explicitly set `overwrite=true` (which destroys the "
                "previous content entirely). Only use overwrite when "
                "the user explicitly asks to regenerate the file from "
                "scratch. Creates parent directories as needed. Path "
                "is relative to the workspace root; cannot escape it."
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
        if pre_existed and overwrite:
            # Sanity check: if the new content is much smaller than the
            # existing file, this is almost certainly the 'model meant to
            # append but reached for overwrite' mistake (see harness-2tq).
            # Refuse the shrink and redirect to edit_file. Threshold: new
            # content is both under half the existing size AND under 1KB —
            # the 1KB floor avoids blocking legitimate regenerations of
            # small configs where the new version happens to be smaller.
            existing_size = target.stat().st_size
            if len(content) < existing_size // 2 and len(content) < 1024:
                raise ValueError(
                    f"refusing to overwrite {path}: new content is "
                    f"{len(content)} bytes but the existing file is "
                    f"{existing_size} bytes. This looks like you meant to "
                    f"append or edit, not replace. Use "
                    f"edit_file(path={path!r}, old_string='', "
                    f"new_string=<line to append>) to append, or set an "
                    f"explicit old_string to replace a specific section."
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        # harness-h6wa: post-write syntax gate. Failed parse → raise so
        # the registry returns ToolResult(success=False). File stays on
        # disk in the broken state; the model fixes or backs out next
        # round (see edit_file._enforce_parse_check for the rationale).
        ok, detail = parse_check(target)
        if not ok:
            raise ValueError(
                f"file written but {path} no longer parses. Read the "
                f"file and either fix the syntax error or pass a "
                f"different content. Parser output:\n{detail}"
            )
        action = "overwrote" if pre_existed else "wrote"
        return f"{action} {len(content)} chars to {path}"
