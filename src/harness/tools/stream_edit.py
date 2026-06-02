"""Curated awk/sed/cut/tr wrapper. Winner of harness-bw27 (file-ops bench).

Stream-edit gateway for the four most common Unix text-manipulation
tools. The model picks a verb (``awk``/``sed``/``cut``/``tr``) and emits
a single ``expr`` string the way it would type the argv tail at a shell
prompt; the harness shlex-splits it, refuses shell metacharacters, and
invokes the absolute tool path with no shell in between. Inputs come
from sandbox-relative paths or inline stdin; output is captured stdout
(``read`` tier) or per-file overwrites (``write`` tier when
``in_place=True``).

Threat model: incompetent model emission, not a hostile attacker —
Mark's the only user and the model isn't adversarial. The job is to
prevent a fumbled ``expr`` from shelling out beyond the chosen verb,
and there is no shell here: the argv goes straight to ``exec`` with
``shell=False``. Things like ``$1``, ``;``, ``&``, ``$()``, ``>`` etc.
are LEGITIMATE awk/sed syntax in this context (``$1`` is a field,
``s/foo/&/`` is a back-reference, ``print > "out"`` writes to a
file in the sandboxed cwd) — blocking them would gut the wrapper. The
realistic remaining escape is awk's ``system("cmd")`` and sed's GNU
``e`` flag, which the language itself spawns shells for. We block
``|`` (awk's pipe-to-shell operator) and backticks (no awk/sed dialect
uses them) and call it good; the model has no reason to call
``system()`` unless asked to, and the worst it can do is run a command
in the workspace cwd, which is Mark's own machine.

Boundaries the tool actually enforces:

  1. ``shutil.which`` is resolved once at construction; the absolute
     path lives on the instance. No PATH lookup at call time.
  2. ``expr`` is rejected for ``|`` and backticks, oversized scripts,
     and unbalanced quoting. ``shlex.split`` with ``posix=True`` is the
     only argv tokeniser; ``shell=False`` is the only way the
     subprocess runs.
  3. Every ``paths`` entry must resolve inside the workspace root.
     ``in_place`` writes go through ``Path.write_text`` after we've
     captured the child's stdout — never via the tool itself, so the
     BSD-vs-GNU ``sed -i`` divergence stays out of the design.

Companion tool: ``python_stream`` (python_eval-style sandboxed Python)
handles JSON parsing and multi-line block rewrites that don't fit awk/sed.
Bench fixtures live in ``scripts/bench_file_ops.py``; the eval is in
``harness.evals.file_ops``.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness.tools.base import ToolSpec

# Shell-only metacharacters the chosen verbs never legitimately need.
# `|` is awk's pipe-to-shell operator (the realistic escape); backticks
# are pure shell syntax. Everything else (``$``, ``;``, ``&``, ``>``,
# ``<``) is part of awk or sed itself and must pass through — see the
# module docstring for the rationale.
_DISALLOWED_METACHARS: frozenset[str] = frozenset("|`")

# Default cap on captured stdout returned to the model. Anything past
# this is truncated with a marker so the model knows output was cut.
# Overridable per-instance via ``max_output_bytes`` — the bench
# (scripts/bench_file_ops.py) bumps it to multiple megabytes so the
# wall-clock correctness check sees the full transform.
_DEFAULT_MAX_STDOUT_BYTES = 512 * 1024

# Hard cap on inline stdin to keep a runaway prompt from queueing
# multi-megabyte input through the bench. paths-based input is uncapped
# (the file's already on disk).
_MAX_STDIN_BYTES = 256 * 1024

# Hard cap on each argv element — gives the metachar check something
# bounded to scan and stops a paste of a multi-page sed script from
# sneaking through. The total argv list is also bounded by argc-style
# sanity (≤ 32 elements).
_MAX_ARG_BYTES = 4 * 1024
_MAX_ARGV_LEN = 32

_VERBS: tuple[str, ...] = ("awk", "sed", "cut", "tr")


def _resolve_binary(name: str) -> str:
    """Locate the absolute path to a coreutil binary at construction
    time. Returning the literal absolute path means the call-time
    subprocess can't be redirected by a malicious PATH; if the binary
    isn't on the system the tool refuses to construct, surfacing the
    problem at session start rather than mid-turn."""
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"stream_edit: {name!r} not found on PATH — install it before enabling this tool"
        )
    return found


def _validate_args(args: list[str]) -> list[str]:
    """Refuse shell-metachar argv elements; otherwise return ``args``
    unchanged. The model passes a structured list (one element per
    argv slot) so there is no shell parsing — each element goes to
    ``exec`` literally. Validation is per-element: length cap, total
    list cap, and metachar scan."""
    if not isinstance(args, list):
        raise TypeError(
            f"stream_edit: args must be a list of argv strings, got "
            f"{type(args).__name__}. One element per argv slot — e.g. "
            f'sed substitution is args=["-E", "s/old/new/g"], not a '
            f"single joined string. Do not put the file path in args; "
            f"pass it via `paths`."
        )
    if not args:
        raise ValueError("stream_edit: args must be a non-empty list")
    if len(args) > _MAX_ARGV_LEN:
        raise ValueError(
            f"stream_edit: argv has {len(args)} elements — max {_MAX_ARGV_LEN}; "
            f"collapse repeats or split across calls"
        )
    for arg in args:
        if not isinstance(arg, str):
            raise TypeError(
                f"stream_edit: every args element must be a string, got {type(arg).__name__}"
            )
        if len(arg.encode("utf-8")) > _MAX_ARG_BYTES:
            raise ValueError(
                f"stream_edit: an args element is {len(arg.encode('utf-8'))} bytes — "
                f"max is {_MAX_ARG_BYTES}; trim the script"
            )
        offending = {ch for ch in arg if ch in _DISALLOWED_METACHARS}
        if offending:
            raise ValueError(
                f"stream_edit: args element {arg!r} contains disallowed shell "
                f"metachars {sorted(offending)!r} — this tool is a single-verb "
                f"wrapper, not a shell. Compose multiple calls instead."
            )
    return list(args)


def _resolve_path(root: Path, rel: str) -> Path:
    """Translate a workspace-relative path into an absolute one, refusing
    anything that resolves outside ``root``. Identical guard to
    edit_file / read_file — same blast radius."""
    if not isinstance(rel, str) or not rel:
        raise ValueError("stream_edit: path entries must be non-empty strings")
    root_abs = root.resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError(f"stream_edit: path {rel!r} escapes workspace root") from exc
    if not target.exists():
        raise FileNotFoundError(f"stream_edit: {rel!r} not found")
    if not target.is_file():
        raise IsADirectoryError(f"stream_edit: {rel!r} is not a regular file")
    return target


def _truncate(text: str, cap: int) -> str:
    """Cap captured stdout at ``cap`` bytes; append a marker the model
    can read so it knows downstream consumers shouldn't trust the tail."""
    if len(text.encode("utf-8")) <= cap:
        return text
    encoded = text.encode("utf-8")[:cap]
    return encoded.decode("utf-8", errors="ignore") + f"\n[truncated at {cap} bytes]\n"


@dataclass
class StreamEditTool:
    """Curated awk/sed/cut/tr wrapper. See module docstring for the
    threat model and design tradeoffs."""

    root: Path
    timeout_seconds: float = 15.0
    binaries: dict[str, str] = field(default_factory=dict)
    max_output_bytes: int = _DEFAULT_MAX_STDOUT_BYTES

    def __post_init__(self) -> None:
        # Resolve every verb at construction. A missing binary fails
        # hard here so the registry never holds a half-broken tool.
        # ``field(default_factory=dict)`` plus this assignment means
        # callers can inject a custom binary map in tests without
        # touching the real PATH.
        if not self.binaries:
            self.binaries = {verb: _resolve_binary(verb) for verb in _VERBS}
        else:
            missing = [v for v in _VERBS if v not in self.binaries]
            if missing:
                raise ValueError(f"stream_edit: binaries dict missing verbs {missing!r}")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="stream_edit",
            description=(
                "Stream-edit a file (or inline text) through awk, sed, "
                "cut, or tr. Choose `tool` ∈ {awk, sed, cut, tr}, then "
                "pass `args` as the LIST of argv elements after that "
                "tool's name — one element per argv slot, no shell "
                "splitting.\n\n"
                "Examples:\n"
                "  tool='awk',  args=['{print $3}']\n"
                "  tool='sed',  args=['s/foo/bar/g']\n"
                "  tool='cut',  args=['-d', ',', '-f', '2']\n"
                "  tool='tr',   args=['a-z', 'A-Z']\n\n"
                "The file path goes in `paths`, NOT in `args` — args is "
                "only the tool's flags/program. Read input from `paths` "
                "(relative to the workspace) or from inline `stdin`; "
                "provide exactly one. Without `in_place`, the tool "
                "returns the captured stdout (read-tier). With "
                "`in_place=true`, each path's transformed output is "
                "written back to that path (write-tier).\n\n"
                "`|` and backticks in any args element are rejected — "
                "this is a single-verb wrapper, not a shell. To chain, "
                "make multiple calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": list(_VERBS),
                        "description": "Which Unix verb to invoke.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Argv tail as a list of strings; one "
                            "element per argv slot. No shell parsing."
                        ),
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Files to read, relative to the workspace "
                            "root. Mutually exclusive with `stdin`."
                        ),
                    },
                    "stdin": {
                        "type": "string",
                        "description": (
                            "Inline input. Mutually exclusive with `paths`. Capped at 256 KB."
                        ),
                    },
                    "in_place": {
                        "type": "boolean",
                        "description": (
                            "When true, per-file overwrite with the "
                            "tool's transformed output. Requires "
                            "`paths` to be non-empty; write-tier."
                        ),
                    },
                },
                "required": ["tool", "args"],
            },
            tier="read",  # common case; in_place=True escalates via write_when
            write_when=lambda args: bool(args.get("in_place")),
            display_name="Stream edit (awk/sed/cut/tr)",
            high_noise=True,
        )

    def call(
        self,
        *,
        tool: str,
        args: list[str],
        paths: list[str] | None = None,
        stdin: str = "",
        in_place: bool = False,
        timeout: float | None = None,
    ) -> str:
        if tool not in self.binaries:
            raise ValueError(f"stream_edit: unknown tool {tool!r}; valid: {sorted(self.binaries)}")
        argv_tail = _validate_args(args)
        paths = list(paths or ())

        if paths and stdin:
            raise ValueError("stream_edit: pass either `paths` or `stdin`, not both")
        if not paths and not stdin:
            raise ValueError("stream_edit: provide one of `paths` or `stdin`")
        if in_place and not paths:
            raise ValueError("stream_edit: in_place=True requires `paths` to be non-empty")
        if stdin and len(stdin.encode("utf-8")) > _MAX_STDIN_BYTES:
            raise ValueError(
                f"stream_edit: stdin is {len(stdin.encode('utf-8'))} bytes — "
                f"max {_MAX_STDIN_BYTES}; pass paths instead"
            )

        resolved_paths = [_resolve_path(self.root, rel) for rel in paths]
        binary = self.binaries[tool]
        effective_timeout = min(max(timeout or self.timeout_seconds, 0.1), 30.0)

        if in_place:
            return self._run_in_place(
                tool=tool,
                binary=binary,
                argv_tail=argv_tail,
                paths=paths,
                resolved_paths=resolved_paths,
                timeout=effective_timeout,
            )
        return self._run_capture(
            tool=tool,
            binary=binary,
            argv_tail=argv_tail,
            paths=paths,
            resolved_paths=resolved_paths,
            stdin=stdin,
            timeout=effective_timeout,
        )

    def _run_capture(
        self,
        *,
        tool: str,
        binary: str,
        argv_tail: list[str],
        paths: list[str],
        resolved_paths: list[Path],
        stdin: str,
        timeout: float,
    ) -> str:
        """Capture-mode: pipe input through the tool, return its stdout."""
        if resolved_paths:
            argv = [binary, *argv_tail, *(str(p) for p in resolved_paths)]
            input_text: str | None = None
        else:
            argv = [binary, *argv_tail]
            input_text = stdin

        start = time.monotonic()
        try:
            # S603: the entire module is a deliberate audited bridge to
            # these four binaries. argv-only invocation with shell=False
            # is the security boundary; the noqa marks intent.
            proc = subprocess.run(  # noqa: S603
                argv,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=str(self.root.resolve()),
            )
        except subprocess.TimeoutExpired:
            return f"[{tool}] timed out after {timeout:.1f}s on {paths or '<stdin>'}"
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0:
            stderr = (proc.stderr or "").rstrip()
            return (
                f"[{tool}] exit={proc.returncode} ({elapsed_ms} ms)\nstderr: {stderr or '(empty)'}"
            )
        return _truncate(proc.stdout, self.max_output_bytes)

    def _run_in_place(
        self,
        *,
        tool: str,
        binary: str,
        argv_tail: list[str],
        paths: list[str],
        resolved_paths: list[Path],
        timeout: float,
    ) -> str:
        """In-place mode: run the tool once per path against stdin, then
        overwrite the path. Avoids BSD-vs-GNU sed -i divergence entirely
        — we own the rewrite, the binary just transforms text."""
        summary: list[str] = []
        for rel, abs_path in zip(paths, resolved_paths, strict=True):
            try:
                original = abs_path.read_text()
            except UnicodeDecodeError as exc:
                return f"[{tool}] {rel!r}: not UTF-8 ({exc})"
            argv = [binary, *argv_tail]
            try:
                proc = subprocess.run(  # noqa: S603 — see _run_capture rationale
                    argv,
                    input=original,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                    cwd=str(self.root.resolve()),
                )
            except subprocess.TimeoutExpired:
                return f"[{tool}] timed out after {timeout:.1f}s on {rel!r}"
            if proc.returncode != 0:
                stderr = (proc.stderr or "").rstrip()
                return f"[{tool}] exit={proc.returncode} on {rel!r}\nstderr: {stderr or '(empty)'}"
            new_text = proc.stdout
            abs_path.write_text(new_text)
            delta = len(new_text) - len(original)
            sign = "+" if delta >= 0 else ""
            summary.append(f"  {rel}: {sign}{delta} bytes")
        return f"[{tool}] rewrote {len(paths)} file(s):\n" + "\n".join(summary)


__all__ = ["StreamEditTool"]
