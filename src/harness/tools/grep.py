from __future__ import annotations

import re
from collections.abc import Iterable
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
        # harness driver artifacts. .harness/loop_runs/*.vllm_trace.jsonl
        # embed entire prior prompts (spec + full source) one-per-line; a
        # workspace-root grep matched them and slurped a 162k-token line
        # into the next prompt, overflowing the window. The driver's own
        # scratch is never something the model should search.
        ".harness",
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
    # Byte ceilings on the RESULT (not the file). The match cap is by
    # line count; a single matched line of minified JS or a JSONL trace
    # record can itself be hundreds of KB, so a 100-line cap is no
    # protection against output that overflows the model's context
    # window. A workspace-root grep over a .harness trace did exactly
    # this — one matched line was a whole serialized prior prompt.
    # Clamp each emitted line, and stop once total output crosses the
    # ceiling, regardless of how many matches remain.
    max_line_chars: int = 1_000
    max_total_chars: int = 20_000

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="grep",
            description=(
                "Search file contents in the workspace for a regex "
                "pattern. Returns matching lines formatted as "
                "`PATH:LINE:TEXT`, capped at 100 hits by default. "
                "`path` may be a directory (walked recursively, noise "
                "dirs skipped) OR a single file. Restrict which files "
                "are searched with `glob` (e.g. 'src/**/*.py') when "
                "walking a directory. Pattern is a Python regex; "
                "escape literals as needed."
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
                            "Directory OR single file (relative to workspace "
                            "root) to search. Default is the workspace root."
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

        cap = max_results if max_results is not None else self.default_max_results
        hits: list[str] = []
        truncated = False
        total_chars = 0

        # harness-373e: when `path` resolves to a single file, grep just
        # that file. Matches the Unix `grep` and `ripgrep` convention
        # the model naturally reaches for. Previously raised
        # NotADirectoryError, which dropped the model into a dedup loop
        # because the same call kept failing the same way.
        single_file_mode = target.is_file()
        if single_file_mode:
            iterator: Iterable[Path] = (target,)
        else:
            iterator = target.rglob(glob) if glob else target.rglob("*")
        for p in iterator:
            if not p.is_file():
                continue
            # skip_dirs filters the recursive walk; an explicit
            # single-file pick from the model is honored even if it
            # lives under a normally-skipped path (the model intent is
            # clear).
            if not single_file_mode:
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
                    stripped = line.rstrip()
                    if len(stripped) > self.max_line_chars:
                        # Clamp a single runaway line (minified JS, a JSONL
                        # record) so one match can't dominate the result.
                        stripped = stripped[: self.max_line_chars] + " … [line truncated]"
                    hit = f"{rel}:{lineno}:{stripped}"
                    hits.append(hit)
                    total_chars += len(hit) + 1  # +1 for the join newline
                    if len(hits) >= cap or total_chars >= self.max_total_chars:
                        truncated = True
                        break
            if truncated:
                break

        if not hits:
            return f"(no matches for {pattern!r})"
        body = "\n".join(hits)
        if truncated:
            reason = (
                f"{cap} matches"
                if len(hits) >= cap
                else f"{self.max_total_chars}-char output ceiling"
            )
            body += f"\n… [truncated at {reason}]"
        return body
