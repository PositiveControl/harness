from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from harness.tools.base import ToolSpec

# Directories we never want to enumerate — dominated by generated / vendored
# content that burns tokens and tells the model nothing useful.
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
class ListDirTool:
    """Enumerate a directory inside the workspace. Read-tier — safe
    for the default profile. Replaces the common 'shell(ls)' pattern,
    which burned a write-tier confirmation for a read-only intent."""

    root: Path
    skip_dirs: frozenset[str] = field(default_factory=lambda: _DEFAULT_SKIP)
    max_entries: int = 200

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list_dir",
            description=(
                "List files and subdirectories inside the workspace. "
                "Returns one entry per line as `TYPE  SIZE  PATH` where "
                "TYPE is `d` (directory) or `f` (file). Noise dirs "
                "(.git, .venv, node_modules, __pycache__, etc.) are "
                "skipped by default. Use `recursive=true` for a full "
                "walk. Path is relative to the workspace root."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory relative to workspace root. "
                            "Default is the workspace root itself."
                        ),
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Descend into subdirectories. Default false.",
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include dotfiles. Default false.",
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="List directory",
        )

    def call(
        self,
        *,
        path: str = ".",
        recursive: bool = False,
        include_hidden: bool = False,
    ) -> str:
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

        lines: list[str] = []
        truncated = False

        def _emit(entry: Path) -> bool:
            """Append one line; returns False if we hit max_entries."""
            rel = entry.relative_to(root).as_posix()
            if entry.is_dir():
                lines.append(f"d  {'-':>8}  {rel}/")
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"f  {size:>8}  {rel}")
            return len(lines) < self.max_entries

        if recursive:
            for p in sorted(target.rglob("*")):
                if not include_hidden and any(
                    part.startswith(".") for part in p.relative_to(root).parts
                ):
                    continue
                if any(part in self.skip_dirs for part in p.relative_to(root).parts):
                    continue
                if not _emit(p):
                    truncated = True
                    break
        else:
            for p in sorted(target.iterdir()):
                name = p.name
                if not include_hidden and name.startswith("."):
                    continue
                if name in self.skip_dirs:
                    continue
                if not _emit(p):
                    truncated = True
                    break

        if not lines:
            return f"(empty directory: {path})"
        header = f"{path} ({'recursive' if recursive else 'top-level'}):"
        body = "\n".join(lines)
        if truncated:
            body += f"\n… [truncated at {self.max_entries} entries]"
        return f"{header}\n{body}"
