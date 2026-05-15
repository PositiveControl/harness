"""python_eval — sandboxed Python expression / snippet evaluator.

The fourth reckon-profile tool. For computations beyond `calc`'s
arithmetic + unit-conversion grammar — JSON parsing, statistics,
decimal arithmetic, regex, datetime math — the model emits Python
into a child interpreter that has been stripped of:

  - filesystem access (no `open`)
  - network access (no socket / urllib / requests)
  - process control (no subprocess / os)
  - dynamic compile / exec / eval from user code
  - imports outside a small allowlist

The child runs in a fresh subprocess via `sys.executable -I -B -S`,
under a CPU and memory rlimit (POSIX best-effort), with a wall-clock
timeout enforced by `subprocess.run`. Output is a JSON envelope with
{value, stdout, stderr, ok, error, duration_ms}.

Threat model: the model emits incompetent code. Not "adversarial
attacker with physical access." Defense in depth, not bulletproof —
Mark is the only user and his model isn't malicious. The sandbox's
job is to keep mistakes contained.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from harness.tools.base import ToolSpec

# Bootstrap script. Runs inside the child interpreter. Reads user code
# from stdin, exec()s under a closed namespace, writes a JSON envelope
# to stdout. Resource limits are best-effort (macOS RLIMIT_AS often
# rejects low values).
_BOOTSTRAP = r"""
import ast, builtins, contextlib, io, json, sys, traceback

_ALLOWED = {
    "math", "statistics", "datetime", "json", "re",
    "decimal", "fractions", "itertools", "functools",
    "collections", "textwrap", "hashlib", "base64",
    "string", "operator", "heapq", "bisect", "array",
}

# Resource limits (POSIX best-effort).
try:
    import resource
    cpu_limit = int(__cpu_limit_seconds__)
    mem_limit = int(__mem_limit_bytes__)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except (ValueError, OSError):
        pass
except ImportError:
    pass  # resource module is POSIX-only

# Strip dangerous modules from sys.modules before user code runs.
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
    if mod in _BLOCKED_MODS or mod.split('.')[0] in _BLOCKED_MODS:
        sys.modules.pop(mod, None)

# Build a safe builtins dict. Remove things the model shouldn't need
# in compute snippets: open/exec/eval/compile/input/exit/quit/help.
_BAD_BUILTINS = {
    "open", "exec", "eval", "compile", "input",
    "exit", "quit", "help", "license", "copyright", "credits",
    "breakpoint", "__build_class__",
}
_safe_builtins = {k: v for k, v in builtins.__dict__.items() if k not in _BAD_BUILTINS}

# Replace __import__ with a guarded version that rejects non-allowlist
# roots. Attribute-based escapes (e.g. `math.__loader__`) remain
# theoretically possible but require effort the model isn't paying.
_real_import = builtins.__import__

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split('.')[0]
    if root not in _ALLOWED:
        raise ImportError(
            f"import {name!r} blocked by python_eval sandbox — "
            f"allowlist: {sorted(_ALLOWED)}"
        )
    return _real_import(name, globals, locals, fromlist, level)

_safe_builtins["__import__"] = _guarded_import

# Pre-import the allowlist so the model can use them without `import`.
_preloaded = {}
for mod_name in _ALLOWED:
    try:
        _preloaded[mod_name] = _real_import(mod_name)
    except ImportError:
        pass

namespace = {"__builtins__": _safe_builtins, **_preloaded}

# Read user code from stdin.
user_code = sys.stdin.read()

captured_stdout = io.StringIO()
captured_stderr = io.StringIO()
value_repr = None
ok = True
err = None

try:
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        tree = ast.parse(user_code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body[-1].value
            body_rest = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(body_rest, "<python_eval>", "exec"), namespace)
            result = eval(
                compile(ast.Expression(body=last_expr), "<python_eval>", "eval"),
                namespace,
            )
            value_repr = repr(result)
        else:
            exec(compile(tree, "<python_eval>", "exec"), namespace)
except BaseException as exc:  # noqa: BLE001 — surface every failure to the parent
    ok = False
    err = f"{type(exc).__name__}: {exc}"
    # Include a one-line traceback hint in stderr for the user.
    captured_stderr.write(traceback.format_exc())

envelope = {
    "value": value_repr,
    "stdout": captured_stdout.getvalue(),
    "stderr": captured_stderr.getvalue(),
    "ok": ok,
    "error": err,
}
sys.__stdout__.write(json.dumps(envelope))
sys.__stdout__.flush()
"""


@dataclass
class PythonEvalTool:
    """Sandboxed Python evaluator. Spawns a fresh child interpreter
    per call; no state leaks between calls."""

    timeout_seconds: float = 5.0
    cpu_limit_seconds: int = 10  # rlimit_cpu; subprocess timeout is the real guard
    mem_limit_bytes: int = 512 * 1024 * 1024  # 512 MB; macOS may ignore

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="python_eval",
            description=(
                "Evaluate a Python snippet in a sandboxed child "
                "interpreter. Pre-imported modules: math, statistics, "
                "datetime, json, re, decimal, fractions, itertools, "
                "functools, collections, textwrap, hashlib, base64, "
                "string, operator, heapq, bisect, array. No file I/O, "
                "no network, no subprocess. Returns the last "
                "expression's repr() + any captured stdout/stderr."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python source to run. If the last "
                            "statement is an expression, its repr() "
                            "is returned as `value`."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": ("Wall-clock timeout in seconds. Default 5.0, max 30.0."),
                    },
                },
                "required": ["code"],
            },
            tier="read",
            display_name="Python evaluator",
        )

    def call(self, *, code: str, timeout: float | None = None) -> str:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("python_eval: code must be a non-empty string")
        effective_timeout = min(max(timeout or self.timeout_seconds, 0.1), 30.0)

        # Inject the configured rlimit values into the bootstrap by
        # substitution. The bootstrap references __cpu_limit_seconds__
        # and __mem_limit_bytes__ as literal numbers.
        bootstrap = _BOOTSTRAP.replace(
            "__cpu_limit_seconds__", str(self.cpu_limit_seconds)
        ).replace("__mem_limit_bytes__", str(self.mem_limit_bytes))

        start = time.monotonic()
        try:
            # S603: executing the user's snippet IS the contract. The
            # entire bootstrap above is the answer to "is this safe?" —
            # see the import allowlist + sys.modules strip + builtins
            # filter. The sandbox is the security boundary, not a noqa.
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-I", "-B", "-S", "-c", bootstrap],
                input=code,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            envelope: dict[str, Any] = {
                "value": None,
                "stdout": "",
                "stderr": "",
                "ok": False,
                "error": f"timeout after {effective_timeout:.1f}s",
                "duration_ms": elapsed_ms,
            }
            return _render(envelope, code)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0 and not proc.stdout:
            # Child died before writing the envelope — surface stderr.
            return _render(
                {
                    "value": None,
                    "stdout": "",
                    "stderr": proc.stderr,
                    "ok": False,
                    "error": f"child exited {proc.returncode} before writing envelope",
                    "duration_ms": elapsed_ms,
                },
                code,
            )

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return _render(
                {
                    "value": None,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "ok": False,
                    "error": "child wrote non-JSON to stdout — bootstrap aborted",
                    "duration_ms": elapsed_ms,
                },
                code,
            )

        envelope["duration_ms"] = elapsed_ms
        return _render(envelope, code)


def _render(envelope: dict[str, Any], code: str) -> str:
    """Format the child envelope as a copy-pasteable provenance line."""
    snippet = code if len(code) < 60 else code[:57] + "..."
    snippet = snippet.replace("\n", " ¶ ")
    if envelope["ok"]:
        lines = [f"python_eval({snippet!r}) -> {envelope['value']}"]
    else:
        lines = [f"python_eval({snippet!r}) -> ERROR: {envelope['error']}"]
    if envelope.get("stdout"):
        lines.append(f"[stdout]\n{envelope['stdout'].rstrip()}")
    if envelope.get("stderr") and not envelope["ok"]:
        # Keep stderr terse on failure; full traceback is too much.
        stderr = envelope["stderr"]
        if len(stderr) > 400:
            stderr = stderr[:200] + "\n... [truncated]\n" + stderr[-200:]
        lines.append(f"[stderr]\n{stderr.rstrip()}")
    lines.append(f"[duration: {envelope['duration_ms']}ms]")
    return "\n".join(lines)


__all__ = ["PythonEvalTool"]
