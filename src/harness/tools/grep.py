from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.tools.base import ToolSpec

# Same skip set as list_dir — a grep that dives into node_modules or
# __pycache__ is almost never what the model wanted.
_DEFAULT_SKIP: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".beads",
        ".idea",
        ".vscode",
        ".claude",
    }
)


@dataclass
class GrepTool:
    """Search file contents inside the workspace. Read-tier. Pure
    Python — no ripgrep dependency. We deliberately walk + regex match
    ourselves so the tool works on every Mac out of the box; ripgrep
    would be faster but the sub-second difference doesn't matter when
    tool-call turns already pay a ~1s model step."""

    root: Path
    skip_dirs: frozenset[str] = field(default_factory=lambda: _DEFAULT_SKIP)
    default_max_results: int = 100
    max_file_bytes: int = 1_000_000

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="grep",
            description=(
                "Search file contents in the workspace for a regex "
                "pattern. Returns matching lines formatted as "
                "`PATH:LINE:TEXT`, capped at 100 hits by default. "
                "Restrict which files are searched with `glob` (e.g. "
                "'src/**/*.py'). Noise dirs are skipped. Pattern is a "
                "Python regex; escape literals as needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regex to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory (relative to workspace root) to search "
                            "in. Default is the workspace root."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": (
                            "Optional glob to filter filenames, e.g. "
                            "'**/*.py' or 'docs/**/*.md'. Default: all files."
                        ),
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive match. Default false.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Cap on hits returned. Default 100.",
                    },
                },
                "required": ["pattern"],
            },
            tier="read",
            display_name="Grep",
            high_noise=True,
        )

    def call(
        self,
        *,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        case_insensitive: bool = False,
        max_results: int | None = None,
    ) -> str:
        if not pattern:
            raise ValueError("pattern must not be empty")
        try:
            regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc

        root = self.root.resolve()
        target = (self.root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path {path!r} escapes workspace root") from exc
        if not target.exists():
            raise FileNotFoundError(f"{path} not found")
        if not target.is_dir():
            raise NotADirectoryError(f"{path} is not a directory")

        cap = max_results if max_results is not None else self.default_max_results
        hits: list[str] = []
        truncated = False

        iterator = target.rglob(glob) if glob else target.rglob("*")
        for p in iterator:
            if not p.is_file():
                continue
            rel_parts = p.relative_to(root).parts
            if any(part in self.skip_dirs for part in rel_parts):
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > self.max_file_bytes:
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            rel = p.relative_to(root).as_posix()
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel}:{lineno}:{line.rstrip()}")
                    if len(hits) >= cap:
                        truncated = True
                        break
            if truncated:
                break

        if not hits:
            return f"(no matches for {pattern!r})"
        body = "\n".join(hits)
        if truncated:
            body += f"\n… [truncated at {cap} matches]"
        return body
