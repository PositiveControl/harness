"""Sandboxed Python stream tool. Winner alongside stream_edit (harness-bw27).

Lets the model express stream transforms in Python — the right shape
for JSON parsing, multi-line block rewrites, and any task whose oracle
is easier to write in Python than awk. The user passes ``expr`` (Python
source) and either ``paths`` or ``stdin``; the child interpreter starts
with ``text`` / ``lines`` / ``paths`` pre-bound and a small allowlist of
pre-imported modules. If the final statement is an expression, its
repr() is the output; otherwise captured stdout is.

Sandboxing mirrors ``python_eval``: subprocess with ``-I -B -S``, an
import allowlist, ``sys.modules`` strip, restricted builtins, CPU +
memory rlimits (POSIX best-effort), and a wall-clock timeout. Same
threat model — incompetent emission, not adversarial. The difference
from ``python_eval`` is the namespace setup, not the boundary: the
model gets file content already loaded so it doesn't burn rounds on
``open()`` (which the sandbox blocks anyway).

Bench fixtures live in ``scripts/bench_file_ops.py``; the model-in-loop
eval is ``harness.evals.file_ops``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.tools.base import ToolSpec

# Hard caps. ``_MAX_INPUT_BYTES`` covers the union of file content +
# inline stdin so a giant fixture doesn't blow through the bench's
# memory budget. The output cap defaults to 512 KB (small enough for
# the model) but is overridable per-instance via ``max_output_bytes``
# — the bench (scripts/bench_file_ops.py) bumps it so wall-clock
# correctness checks see the full transform.
_MAX_INPUT_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_OUTPUT_BYTES = 512 * 1024
_MAX_EXPR_BYTES = 8 * 1024

_PRE_IMPORTS: tuple[str, ...] = (
    "re",
    "json",
    "math",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "textwrap",
    "string",
    "operator",
    "decimal",
    "fractions",
    "hashlib",
    "base64",
    "datetime",
)


_BOOTSTRAP = r"""
import ast, builtins, contextlib, io, json, sys, traceback

_ALLOWED = __ALLOWED__

# Resource limits (POSIX best-effort).
try:
    import resource
    cpu_limit = int(__CPU__)
    mem_limit = int(__MEM__)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except (ValueError, OSError):
        pass
except ImportError:
    pass

_BLOCKED_MODS = {
    "os", "subprocess", "socket", "urllib", "urllib.request",
    "urllib.parse", "urllib.error", "urllib.response",
    "ctypes", "ctypes.util", "shutil", "pathlib",
    "tempfile", "threading", "multiprocessing", "asyncio",
    "http", "http.client", "ftplib", "smtplib",
    "pickle", "marshal", "shelve", "dbm", "sqlite3",
    "pty", "fcntl", "termios", "select", "signal",
    "platform", "getpass", "pwd", "grp", "syslog",
    "_thread", "_ctypes",
}
for mod in list(sys.modules):
    if mod in _BLOCKED_MODS or mod.split(".")[0] in _BLOCKED_MODS:
        sys.modules.pop(mod, None)

_BAD_BUILTINS = {
    "open", "exec", "eval", "compile", "input",
    "exit", "quit", "help", "license", "copyright", "credits",
    "breakpoint", "__build_class__",
}
_safe_builtins = {k: v for k, v in builtins.__dict__.items() if k not in _BAD_BUILTINS}

_real_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED:
        raise ImportError(
            f"import {name!r} blocked by python_stream sandbox — "
            f"allowlist: {sorted(_ALLOWED)}"
        )
    return _real_import(name, globals, locals, fromlist, level)


_safe_builtins["__import__"] = _guarded_import

_preloaded = {}
for mod_name in _ALLOWED:
    try:
        _preloaded[mod_name] = _real_import(mod_name)
    except ImportError:
        pass

# Read the envelope: {expr: str, text: str, paths: [str]}.
envelope_in = json.loads(sys.stdin.read())
user_expr = envelope_in["expr"]
text = envelope_in["text"]
paths = tuple(envelope_in["paths"])
lines = text.splitlines()

namespace = dict(_preloaded)
namespace.update({
    "__builtins__": _safe_builtins,
    "text": text,
    "lines": lines,
    "paths": paths,
})

captured_stdout = io.StringIO()
captured_stderr = io.StringIO()
value_repr = None
ok = True
err = None

try:
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        tree = ast.parse(user_expr)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body[-1].value
            body_rest = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(body_rest, "<python_stream>", "exec"), namespace)
            result = eval(
                compile(ast.Expression(body=last_expr), "<python_stream>", "eval"),
                namespace,
            )
            value_repr = repr(result) if not isinstance(result, str) else result
        else:
            exec(compile(tree, "<python_stream>", "exec"), namespace)
except BaseException as exc:
    ok = False
    err = f"{type(exc).__name__}: {exc}"
    captured_stderr.write(traceback.format_exc())

envelope_out = {
    "value": value_repr,
    "stdout": captured_stdout.getvalue(),
    "stderr": captured_stderr.getvalue(),
    "ok": ok,
    "error": err,
}
sys.__stdout__.write(json.dumps(envelope_out))
sys.__stdout__.flush()
"""


# Corrective nudges for the blocked builtins the model reaches for most
# (harness-vszp). The sandbox already strips these (see _BAD_BUILTINS),
# but a bare `NameError: name 'open' is not defined` thrown from deep in
# a traceback doesn't tell the model what to do instead — so it retries
# the same call across turns. Catching the call at validation time lets
# us return the contract-specific fix as the tool result, turning N
# silent retries into one correction.
_BLOCKED_CALL_HINTS: dict[str, str] = {
    "open": (
        "python_stream has no `open()` — the sandbox has no filesystem "
        "access. Your input is ALREADY loaded: read it from the "
        "pre-bound `text` (full input as one string) or `lines` "
        "(text.splitlines()). To process a file, pass it via "
        '`paths=["file"]` and operate on `text`; do not call open().'
    ),
    "exec": "python_stream blocks `exec()` — write the code directly as the expr.",
    "eval": "python_stream blocks `eval()` — write the expression directly as the expr.",
    "compile": "python_stream blocks `compile()`.",
    "input": "python_stream blocks `input()` — there is no stdin prompt; use the `text` binding.",
    "breakpoint": "python_stream blocks `breakpoint()`.",
}


def _detect_blocked_call(expr: str) -> str | None:
    """Return a corrective hint if `expr` calls a blocked builtin as a
    bare name (e.g. `open(...)`), else None. AST-based so it doesn't
    false-positive on method calls (`io.open(...)`, `x.eval(...)`) or
    substrings (`reopen(...)`). Unparseable expr → None: let the normal
    path surface the syntax error rather than masking it."""
    try:
        # Suppress SyntaxWarning (e.g. the model's invalid escape
        # sequences like '\.') — this parse only inspects structure for
        # blocked builtins; unlike the child interpreter's parse it runs
        # in the PARENT process, where the warning would otherwise leak
        # to the operator's terminal (`<unknown>:N: SyntaxWarning`).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(expr)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _BLOCKED_CALL_HINTS
        ):
            return _BLOCKED_CALL_HINTS[node.func.id]
    return None


def _validate_expr(expr: str) -> None:
    if not isinstance(expr, str):
        raise TypeError(f"python_stream: expr must be a string, got {type(expr).__name__}")
    if not expr.strip():
        raise ValueError("python_stream: expr must be non-empty")
    if len(expr.encode("utf-8")) > _MAX_EXPR_BYTES:
        raise ValueError(
            f"python_stream: expr is {len(expr.encode('utf-8'))} bytes — max is {_MAX_EXPR_BYTES}"
        )
    hint = _detect_blocked_call(expr)
    if hint is not None:
        raise ValueError(f"python_stream: {hint}")


def _resolve_path(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise ValueError("python_stream: path entries must be non-empty strings")
    root_abs = root.resolve()
    target = (root / rel).resolve()
    try:
        target.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError(f"python_stream: path {rel!r} escapes workspace root") from exc
    if not target.exists():
        raise FileNotFoundError(f"python_stream: {rel!r} not found")
    if not target.is_file():
        raise IsADirectoryError(f"python_stream: {rel!r} is not a regular file")
    return target


def _load_text(root: Path, paths: list[str], stdin: str) -> str:
    """Materialize the child's ``text`` binding from paths-or-stdin.
    Files are concatenated in caller order, mirroring how the Unix tools
    in Candidate A consume multiple paths."""
    if paths and stdin:
        raise ValueError("python_stream: pass either `paths` or `stdin`, not both")
    if not paths and not stdin:
        raise ValueError("python_stream: provide one of `paths` or `stdin`")
    if stdin:
        if len(stdin.encode("utf-8")) > _MAX_INPUT_BYTES:
            raise ValueError(
                f"python_stream: stdin is {len(stdin.encode('utf-8'))} bytes — "
                f"max {_MAX_INPUT_BYTES}"
            )
        return stdin
    chunks: list[str] = []
    total = 0
    for rel in paths:
        abs_path = _resolve_path(root, rel)
        try:
            content = abs_path.read_text()
        except UnicodeDecodeError as exc:
            raise ValueError(f"python_stream: {rel!r} is not UTF-8 ({exc})") from exc
        total += len(content.encode("utf-8"))
        if total > _MAX_INPUT_BYTES:
            raise ValueError(f"python_stream: combined paths exceed {_MAX_INPUT_BYTES} bytes")
        chunks.append(content)
    return "".join(chunks)


def _truncate(text: str, cap: int) -> str:
    if len(text.encode("utf-8")) <= cap:
        return text
    encoded = text.encode("utf-8")[:cap]
    return encoded.decode("utf-8", errors="ignore") + f"\n[truncated at {cap} bytes]\n"


@dataclass
class PythonStreamTool:
    """Sandboxed Python stream-transform. See module docstring for the
    threat model and design tradeoffs."""

    root: Path
    timeout_seconds: float = 15.0
    cpu_limit_seconds: int = 30
    mem_limit_bytes: int = 512 * 1024 * 1024
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="python_stream",
            description=(
                "Run a Python expression or snippet over a file (or "
                "inline text) in a sandboxed child interpreter.\n\n"
                "DO NOT call open() — the sandbox has no filesystem "
                "access. Your input is pre-loaded under these names:\n"
                "  text   — the full input as a single string\n"
                "  lines  — text.splitlines()\n"
                "  paths  — tuple of input paths (empty when using stdin)\n"
                "Read from `text`/`lines`; never open() a file.\n"
                "Pre-imported: re, json, math, statistics, itertools, "
                "functools, collections, textwrap, string, operator, "
                "decimal, fractions, hashlib, base64, datetime. (No "
                "exec/eval/compile/input either.)\n\n"
                "If the final statement is an expression, its repr() (or "
                "value, if it's already a string) is returned. Otherwise "
                "the captured stdout is returned.\n\n"
                "Choose the input source: `paths` (workspace-relative "
                "list) OR inline `stdin` — provide exactly one. Without "
                "`in_place`, returns the result string (read-tier). With "
                "`in_place=true` (and exactly one path), the result is "
                "written back to that path (write-tier). No file I/O, "
                "no network, no subprocess inside the child."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": (
                            "Python source. Final expression's value "
                            "(or stdout, when no final expression) is "
                            "the output."
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
                        "description": ("Inline input. Mutually exclusive with `paths`."),
                    },
                    "in_place": {
                        "type": "boolean",
                        "description": (
                            "When true and `paths` has exactly one entry, "
                            "overwrite that path with the result; write-tier."
                        ),
                    },
                },
                "required": ["expr"],
            },
            tier="read",
            display_name="Python stream",
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
            raise ValueError("python_stream: in_place requires exactly one entry in `paths`")

        effective_timeout = min(max(timeout or self.timeout_seconds, 0.1), 30.0)

        bootstrap = (
            _BOOTSTRAP.replace("__ALLOWED__", json.dumps(list(_PRE_IMPORTS)))
            .replace("__CPU__", str(self.cpu_limit_seconds))
            .replace("__MEM__", str(self.mem_limit_bytes))
        )
        envelope = json.dumps({"expr": expr, "text": text, "paths": paths})

        start = time.monotonic()
        try:
            # S603: the bootstrap above IS the security boundary —
            # sys.modules strip + import guard + safe builtins. argv-only,
            # shell=False. See python_eval for the same noqa rationale.
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-I", "-B", "-S", "-c", bootstrap],
                input=envelope,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
                cwd=str(self.root.resolve()),
            )
        except subprocess.TimeoutExpired:
            return f"[python_stream] timed out after {effective_timeout:.1f}s"
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0 and not proc.stdout:
            return (
                f"[python_stream] child exited {proc.returncode} ({elapsed_ms} ms)\n"
                f"stderr: {proc.stderr[:400] if proc.stderr else '(empty)'}"
            )

        try:
            result: dict[str, Any] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return f"[python_stream] non-JSON envelope: {proc.stdout[:200]!r}"

        if not result.get("ok"):
            stderr = result.get("stderr") or ""
            return f"[python_stream] ERROR: {result.get('error')}\n{stderr.rstrip()[-400:]}"

        # Prefer the final-expression value when it carries content.
        # Fall back to captured stdout when the final value is None or
        # empty — this is the "the model called print() instead of
        # making the last line an expression" case.
        value = result.get("value")
        stdout = result.get("stdout") or ""
        output = stdout if value is None or value in ("None", "") else value
        if not isinstance(output, str):
            output = repr(output)

        if in_place:
            abs_path = _resolve_path(self.root, paths[0])
            abs_path.write_text(output)
            delta = len(output) - len(text)
            sign = "+" if delta >= 0 else ""
            return f"[python_stream] rewrote {paths[0]}: {sign}{delta} bytes ({elapsed_ms} ms)"

        return _truncate(output, self.max_output_bytes)


__all__ = ["PythonStreamTool"]
