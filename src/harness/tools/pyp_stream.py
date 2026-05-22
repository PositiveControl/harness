"""Candidate B of harness-bw27 — pyp (the Pyed Piper) wrapper.

Wraps the ``pyp`` CLI: a Python-expression pipe that gives the model
``x`` / ``line`` (per-line), ``lines`` (full input), and ``stdin`` as
bindings. The harness pipes input on stdin and captures stdout via
argv-only subprocess (``shell=False``). Sandboxing is *not* the pyp
binary's own — pyp does whatever Python it's asked to do — so the
guard is workspace-pinned cwd plus a separate write step in ``in_place``
mode (we never let pyp open files itself; the harness handles I/O).

Compared to Candidate C (``python_stream``), pyp leans on its automatic
``for`` loop over lines (very terse for one-shot tasks) at the cost of
a real external dep (``pypyp`` package). Phase 3's bench tells us
whether terseness translates to first-try model wins.

The tool is available iff the ``stream`` extra is installed:
``uv sync --extra stream``. Construction fails with a clear error
otherwise — the catalog never holds a half-loaded tool.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness.tools.base import ToolSpec

# Same shape as Candidate A/C — kept identical so the bench can compare
# apples-to-apples on token cost and validation behavior.
_MAX_INPUT_BYTES = 8 * 1024 * 1024
_MAX_STDIN_BYTES = 256 * 1024
_DEFAULT_MAX_OUTPUT_BYTES = 512 * 1024
_MAX_EXPR_BYTES = 8 * 1024


def _locate_pyp() -> str:
    """Find ``pyp`` at construction time. Missing binary means the
    ``stream`` extra wasn't installed — surface that here rather than
    silently producing a non-functional tool."""
    found = shutil.which("pyp")
    if found is None:
        raise FileNotFoundError(
            "pyp_stream: `pyp` not found on PATH — "
            "install the `stream` extra: uv sync --extra stream"
        )
    return found


def _validate_expr(expr: str) -> None:
    if not isinstance(expr, str):
        raise TypeError(f"pyp_stream: expr must be a string, got {type(expr).__name__}")
    if not expr.strip():
        raise ValueError("pyp_stream: expr must be non-empty")
    if len(expr.encode("utf-8")) > _MAX_EXPR_BYTES:
        raise ValueError(
            f"pyp_stream: expr is {len(expr.encode('utf-8'))} bytes — max is {_MAX_EXPR_BYTES}"
        )


def _resolve_path(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise ValueError("pyp_stream: path entries must be non-empty strings")
    root_abs = root.resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError(f"pyp_stream: path {rel!r} escapes workspace root") from exc
    if not target.exists():
        raise FileNotFoundError(f"pyp_stream: {rel!r} not found")
    if not target.is_file():
        raise IsADirectoryError(f"pyp_stream: {rel!r} is not a regular file")
    return target


def _load_text(root: Path, paths: list[str], stdin: str) -> str:
    if paths and stdin:
        raise ValueError("pyp_stream: pass either `paths` or `stdin`, not both")
    if not paths and not stdin:
        raise ValueError("pyp_stream: provide one of `paths` or `stdin`")
    if stdin:
        if len(stdin.encode("utf-8")) > _MAX_STDIN_BYTES:
            raise ValueError(
                f"pyp_stream: stdin is {len(stdin.encode('utf-8'))} bytes — max {_MAX_STDIN_BYTES}"
            )
        return stdin
    chunks: list[str] = []
    total = 0
    for rel in paths:
        abs_path = _resolve_path(root, rel)
        try:
            content = abs_path.read_text()
        except UnicodeDecodeError as exc:
            raise ValueError(f"pyp_stream: {rel!r} is not UTF-8 ({exc})") from exc
        total += len(content.encode("utf-8"))
        if total > _MAX_INPUT_BYTES:
            raise ValueError(f"pyp_stream: combined paths exceed {_MAX_INPUT_BYTES} bytes")
        chunks.append(content)
    return "".join(chunks)


def _truncate(text: str, cap: int) -> str:
    if len(text.encode("utf-8")) <= cap:
        return text
    encoded = text.encode("utf-8")[:cap]
    return encoded.decode("utf-8", errors="ignore") + f"\n[truncated at {cap} bytes]\n"


@dataclass
class PypStreamTool:
    """pyp wrapper. See module docstring for the threat model and the
    write-via-harness (never via pyp) design choice for ``in_place``."""

    root: Path
    timeout_seconds: float = 15.0
    binary: str = field(default="")
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        if not self.binary:
            self.binary = _locate_pyp()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="pyp_stream",
            description=(
                "Run a pyp (Pyed Piper) expression over a file (or "
                "inline text). pyp auto-iterates over input lines, "
                "binding `x` / `line` / `l` to the current line, "
                "`idx` / `index` / `i` to the line index, `lines` to "
                "the full list of rstripped lines, and `stdin` to the "
                "raw input stream. The expression's value is printed "
                "per-line (when it references `x`) or once (otherwise).\n\n"
                "Examples:\n"
                "  expr='x.split()[2]'              — extract col 3\n"
                "  expr='x.upper()'                 — uppercase each line\n"
                "  expr='len(lines)'                — count lines (once)\n"
                "  expr=\"json.loads(x)['user']\"     — parse JSONL\n\n"
                "Read input from `paths` (workspace-relative) or "
                "inline `stdin`. Without `in_place`, returns captured "
                "stdout (read-tier). With `in_place=true` and exactly "
                "one path, the output is written back to that path "
                "(write-tier).\n\n"
                "Requires the `stream` extra: uv sync --extra stream."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": (
                            "pyp expression. References `x` to auto-"
                            "iterate over input lines; references "
                            "`lines` to operate on the full input."
                        ),
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Files to read, concatenated in order. Mutually exclusive with `stdin`."
                        ),
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Inline input. Mutually exclusive with `paths`.",
                    },
                    "in_place": {
                        "type": "boolean",
                        "description": (
                            "When true and `paths` has exactly one "
                            "entry, overwrite that path with the "
                            "output; write-tier."
                        ),
                    },
                },
                "required": ["expr"],
            },
            tier="read",
            display_name="Pyp stream",
            high_noise=True,
        )

    def call(
        self,
        *,
        expr: str,
        paths: list[str] | None = None,
        stdin: str = "",
        in_place: bool = False,
        timeout: float | None = None,
    ) -> str:
        _validate_expr(expr)
        paths = list(paths or ())
        text = _load_text(self.root, paths, stdin)

        if in_place and len(paths) != 1:
            raise ValueError("pyp_stream: in_place requires exactly one entry in `paths`")

        effective_timeout = min(max(timeout or self.timeout_seconds, 0.1), 30.0)

        start = time.monotonic()
        try:
            # S603: pyp is the contract — argv-only invocation, no shell,
            # cwd pinned to the workspace. The expression is pyp's
            # responsibility; the harness owns the file I/O boundary.
            proc = subprocess.run(  # noqa: S603
                [self.binary, expr],
                input=text,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
                cwd=str(self.root.resolve()),
            )
        except subprocess.TimeoutExpired:
            return f"[pyp_stream] timed out after {effective_timeout:.1f}s"
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").rstrip()[-400:] or "(empty)"
            return f"[pyp_stream] exit={proc.returncode} ({elapsed_ms} ms)\nstderr: {stderr_tail}"

        output = proc.stdout

        if in_place:
            abs_path = _resolve_path(self.root, paths[0])
            abs_path.write_text(output)
            delta = len(output) - len(text)
            sign = "+" if delta >= 0 else ""
            return f"[pyp_stream] rewrote {paths[0]}: {sign}{delta} bytes ({elapsed_ms} ms)"

        return _truncate(output, self.max_output_bytes)


__all__ = ["PypStreamTool"]
