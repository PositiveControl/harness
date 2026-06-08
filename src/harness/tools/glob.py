from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from harness.tools.base import ToolSpec

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
        # harness driver artifacts (traces, workspace tarballs, logs) —
        # never source the model should match. See grep.py for the
        # self-referential-trace overflow this guards against.
        ".harness",
    }
)


@dataclass
class GlobTool:
    """Find files by path pattern inside the workspace. Read-tier.
    Pairs with grep — glob for 'where does the module live?', grep for
    'who references it?'."""

    root: Path
    skip_dirs: frozenset[str] = field(default_factory=lambda: _DEFAULT_SKIP)
    max_results: int = 500

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="glob",
            description=(
                "List files in the workspace matching a glob pattern, "
                "one path per line. Examples: 'src/**/*.py', "
                "'docs/*.md'. Skips noise dirs (.git, .venv, "
                "node_modules, __pycache__, etc.). Capped at 500 "
                "results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py'.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Base directory (relative to workspace root). "
                            "Default is the workspace root."
                        ),
                    },
                },
                "required": ["pattern"],
            },
            tier="read",
            display_name="Glob",
        )

    def call(self, *, pattern: str, path: str = ".") -> str:
        if not pattern:
            raise ValueError("pattern must not be empty")
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

        results: list[str] = []
        truncated = False
        for p in sorted(target.glob(pattern)):
            if any(part in self.skip_dirs for part in p.relative_to(root).parts):
                continue
            results.append(p.relative_to(root).as_posix())
            if len(results) >= self.max_results:
                truncated = True
                break

        if not results:
            return f"(no files matching {pattern!r})"
        body = "\n".join(results)
        if truncated:
            body += f"\n… [truncated at {self.max_results} results]"
        return body
