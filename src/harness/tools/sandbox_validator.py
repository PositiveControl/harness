"""Tool sandbox validator — harness-l2ak.

The validator the synthesize_tool meta-tool (harness-yari) calls
before promoting a candidate Python source string into the catalog.
Same sandbox shape as `python_eval`: subprocess + `-I -B` + module
strip + import allowlist + RLIMIT + wall-clock timeout. The job is
*not* to defeat an attacker — Mark is the only user and the model
isn't malicious. The validator's job is to keep a fumbled synthesis
attempt from corrupting the catalog or hitting the filesystem on
import. (Unlike python_eval we don't pass `-S` because the candidate
tool must be able to import `harness.tools.base`, which lives in the
venv's site-packages.)

Contract:

    validate_tool_module(
        source: str,
        smoke_args: dict[str, Any] | None = None,
        allowed_imports: tuple[str, ...] = (),
        timeout_seconds: float = 5.0,
    ) -> ValidationResult

Validation steps:

    1. Reject source > MAX_SOURCE_BYTES (8 KB).
    2. Write source to a tmp file.
    3. Spawn a sandboxed subprocess that loads the module, locates a
       single top-level `Tool` (duck-typed: has `spec` property + `call`
       method), reads its spec, and — if `smoke_args` is supplied —
       invokes `call(**smoke_args)` and captures the output.
    4. Parse the envelope back into `ValidationResult`.

A successful result carries a `ToolSpec` reconstructed in the parent
process from the JSON the child emitted, so the spec is a normal
in-process object the catalog can serialize via the existing
`ToolCatalog` path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.tools.base import ToolSpec

# Hard cap on source size. Synthesised tools should be tiny — most of
# the catalog's tools are 100-300 LOC. 8 KB is generous and bounds the
# bootstrap's parse+exec time predictably.
MAX_SOURCE_BYTES = 8 * 1024

# Conservative default: only ToolSpec from harness.tools.base, which the
# child must import to declare a spec. Caller (synthesize_tool) extends
# this list per-tool from a small, curated allowlist.
DEFAULT_ALLOWED_IMPORTS: tuple[str, ...] = (
    "harness.tools.base",
    "math",
    "statistics",
    "json",
    "re",
    "datetime",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "collections",
    "textwrap",
    "string",
    "operator",
    "dataclasses",
    "typing",
)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a candidate tool module.

    `ok` is the single source of truth: True means the module compiled,
    exposed a valid Tool, and (if `smoke_args` was supplied) the smoke
    call succeeded. False means one of those gates failed — `error`
    carries the human-readable reason.

    `spec` is None on failure. On success it is a ToolSpec object the
    catalog can register directly. `smoke_output` carries the call's
    return string when a smoke call ran (None otherwise).
    """

    ok: bool
    spec: ToolSpec | None = None
    smoke_output: str | None = None
    error: str | None = None
    duration_ms: int = 0


# Bootstrap script that runs inside the child interpreter. Reads the
# module source path + smoke_args envelope from stdin, validates, and
# writes a JSON envelope to stdout. The bootstrap text below contains
# placeholder tokens (__ALLOWED__) substituted by the parent before
# `subprocess.run` so the allowlist is fixed at process launch.
_BOOTSTRAP = r"""
import builtins, inspect, json, sys, traceback

_ALLOWED = set(__ALLOWED__)

# Strip dangerous modules before user code runs. Same set as python_eval
# minus harness internals — synthesized tools may legitimately want
# pathlib / dataclasses, so we don't blanket-block stdlib.
_BLOCKED_MODS = {
    "os", "subprocess", "socket", "urllib", "urllib.request",
    "urllib.parse", "urllib.error", "urllib.response",
    "ctypes", "ctypes.util", "shutil",
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

# Best-effort rlimit.
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

# Pre-import the entire allowlist so the user code can use them without
# tripping the guard's hot path. `harness.tools.base` is hot too — the
# guard fires on user `from ... import` statements, not internal Python
# machinery that triggers imports during loader execution.
_real_import = builtins.__import__
_preloaded = {}
for mod_name in sorted(_ALLOWED):
    try:
        _preloaded[mod_name] = _real_import(mod_name)
    except ImportError:
        pass


_modules_ref = sys.modules  # capture by reference; avoid `import sys` in guard
_blocked_ref = _BLOCKED_MODS  # same


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Order matters:
    #   1. Blocklist always wins. Even if harness's transitive imports
    #      have re-pulled `os` into sys.modules, the model still
    #      mustn't be able to `import os` itself.
    #   2. Already-loaded modules pass straight through (loader
    #      machinery + harness-internal transitives).
    #   3. Allowlist roots are admitted on first user import.
    #   4. Everything else: refused.
    root = name.split(".")[0]
    if root in _blocked_ref or name in _blocked_ref:
        raise ImportError(
            f"import {name!r} blocked by tool sandbox — module is on the blocklist"
        )
    if name in _modules_ref:
        return _real_import(name, globals, locals, fromlist, level)
    if root not in _ALLOWED and name not in _ALLOWED:
        raise ImportError(
            f"import {name!r} blocked by tool sandbox — "
            f"allowlist roots: {sorted(_ALLOWED)}"
        )
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import

# Stdin carries a JSON envelope: {source_path: str, smoke_args: dict|None}.
inbound = json.loads(sys.stdin.read())
source_path = inbound["source_path"]
smoke_args = inbound.get("smoke_args")

def _emit(envelope):
    sys.__stdout__.write(json.dumps(envelope))
    sys.__stdout__.flush()


def _fail(reason):
    _emit({"ok": False, "error": reason, "spec": None, "smoke_output": None})
    sys.exit(0)


# Read the candidate source and exec it into a private namespace.
# Using exec() (rather than importlib) avoids the loader machinery
# pulling internal modules through the guarded __import__ during the
# load phase.
with open(source_path, "r", encoding="utf-8") as _src_fh:
    user_source = _src_fh.read()

namespace = {"__name__": "_candidate_tool", "__builtins__": builtins.__dict__}
try:
    exec(compile(user_source, "<candidate_tool>", "exec"), namespace)
except BaseException as exc:
    _fail(f"module load failed: {type(exc).__name__}: {exc}")

# Locate a single top-level class that quacks like Tool: has a `spec`
# property/attr that returns a ToolSpec-shaped object, and a `call`
# method. Reject 0 or >1 candidates so synthesis can't sneak a
# multi-tool module past the catalog.
candidates = []
for name, obj in list(namespace.items()):
    if name.startswith("_"):
        continue
    if not inspect.isclass(obj):
        continue
    # Ignore re-exported imports (anything that didn't get its
    # __module__ set to our namespace's __name__).
    if getattr(obj, "__module__", None) != "_candidate_tool":
        continue
    if not hasattr(obj, "spec") or not hasattr(obj, "call"):
        continue
    candidates.append((name, obj))

if not candidates:
    _fail("module exposes no Tool class (need a class with `spec` + `call`)")
if len(candidates) > 1:
    cand_names = ", ".join(n for n, _ in candidates)
    _fail(f"module exposes multiple Tool candidates: {cand_names} — pick one")

cls_name, cls = candidates[0]

# Instantiate with no args. Synthesised tools must have a default ctor —
# any required configuration goes through the spec's parameters, not the
# constructor.
try:
    instance = cls()
except BaseException as exc:
    _fail(f"could not instantiate {cls_name}(): {type(exc).__name__}: {exc}")

# Read the spec. ToolSpec is a frozen dataclass; we serialise its public
# fields for the parent. (Avoid pickle — we don't trust the child's
# class identity across the process boundary.)
try:
    spec = instance.spec
except BaseException as exc:
    _fail(f"reading .spec raised {type(exc).__name__}: {exc}")

required_attrs = ("name", "description", "parameters", "tier")
for attr in required_attrs:
    if not hasattr(spec, attr):
        _fail(f"ToolSpec missing attribute: {attr}")
    if getattr(spec, attr) in (None, ""):
        _fail(f"ToolSpec field empty: {attr}")

if not isinstance(spec.parameters, dict):
    _fail(f"ToolSpec.parameters must be a dict, got {type(spec.parameters).__name__}")
if spec.tier not in ("read", "write"):
    _fail(f"ToolSpec.tier must be 'read' or 'write', got {spec.tier!r}")

spec_dict = {
    "name": spec.name,
    "description": spec.description,
    "parameters": spec.parameters,
    "tier": spec.tier,
    "display_name": getattr(spec, "display_name", None),
    "high_noise": bool(getattr(spec, "high_noise", False)),
}

smoke_output = None
if smoke_args is not None:
    try:
        smoke_output = instance.call(**smoke_args)
    except BaseException as exc:
        _emit({
            "ok": False,
            "error": f"smoke call raised {type(exc).__name__}: {exc}",
            "spec": spec_dict,
            "smoke_output": None,
            "smoke_traceback": traceback.format_exc(),
        })
        sys.exit(0)
    # call() may return ToolResult; coerce to a string for the envelope.
    smoke_output = smoke_output if isinstance(smoke_output, str) else repr(smoke_output)

_emit({
    "ok": True,
    "error": None,
    "spec": spec_dict,
    "smoke_output": smoke_output,
})
"""


def validate_tool_module(
    *,
    source: str,
    smoke_args: dict[str, Any] | None = None,
    allowed_imports: tuple[str, ...] | None = None,
    timeout_seconds: float = 5.0,
    cpu_limit_seconds: int = 10,
    mem_limit_bytes: int = 256 * 1024 * 1024,
) -> ValidationResult:
    """Validate a candidate tool module in a sandboxed subprocess.

    `allowed_imports` extends `DEFAULT_ALLOWED_IMPORTS` — caller adds
    the per-tool extras (e.g. `("base64",)`). Pass `()` to use defaults
    only; pass `None` for the same effect.
    """

    if not isinstance(source, str) or not source.strip():
        return ValidationResult(ok=False, error="source must be a non-empty string")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return ValidationResult(
            ok=False,
            error=f"source exceeds {MAX_SOURCE_BYTES} bytes — synthesised tools must stay small",
        )

    effective_allowed = tuple(sorted({*DEFAULT_ALLOWED_IMPORTS, *(allowed_imports or ())}))
    effective_timeout = min(max(timeout_seconds, 0.1), 30.0)

    bootstrap = (
        _BOOTSTRAP.replace("__ALLOWED__", json.dumps(list(effective_allowed)))
        .replace("__CPU__", str(cpu_limit_seconds))
        .replace("__MEM__", str(mem_limit_bytes))
    )

    with tempfile.TemporaryDirectory(prefix="harness_tool_validate_") as tmp:
        source_path = Path(tmp) / "candidate.py"
        source_path.write_text(source, encoding="utf-8")
        envelope_in = json.dumps({"source_path": str(source_path), "smoke_args": smoke_args})

        start = time.monotonic()
        try:
            # Note on flags vs python_eval: we drop `-S` because the
            # synthesised tool *must* be able to import
            # `harness.tools.base` to declare its ToolSpec, and `-S`
            # skips site.py which is what adds the venv's
            # site-packages to sys.path. `-I` (isolated, ignore
            # PYTHONPATH, no user site) + the import allowlist is the
            # sandbox boundary here, not site.py.
            proc = subprocess.run(  # noqa: S603 — see python_eval rationale; bootstrap is the boundary
                [sys.executable, "-I", "-B", "-c", bootstrap],
                input=envelope_in,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                ok=False,
                error=f"validation timed out after {effective_timeout:.1f}s",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        duration_ms = int((time.monotonic() - start) * 1000)

        if proc.returncode != 0 and not proc.stdout:
            return ValidationResult(
                ok=False,
                error=(
                    f"child exited {proc.returncode} before writing envelope. "
                    f"stderr: {proc.stderr[:400] if proc.stderr else '(empty)'}"
                ),
                duration_ms=duration_ms,
            )

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return ValidationResult(
                ok=False,
                error=f"child wrote non-JSON envelope. stdout head: {proc.stdout[:200]!r}",
                duration_ms=duration_ms,
            )

    return _from_envelope(envelope, duration_ms=duration_ms)


def _from_envelope(envelope: dict[str, Any], *, duration_ms: int) -> ValidationResult:
    spec_dict = envelope.get("spec")
    spec_obj: ToolSpec | None
    if spec_dict is None:
        spec_obj = None
    else:
        try:
            spec_obj = ToolSpec(
                name=spec_dict["name"],
                description=spec_dict["description"],
                parameters=spec_dict["parameters"],
                tier=spec_dict["tier"],
                display_name=spec_dict.get("display_name"),
                high_noise=bool(spec_dict.get("high_noise", False)),
            )
        except (KeyError, TypeError) as exc:
            return ValidationResult(
                ok=False,
                error=f"could not reconstruct ToolSpec from child envelope: {exc}",
                duration_ms=duration_ms,
            )

    return ValidationResult(
        ok=bool(envelope.get("ok")),
        spec=spec_obj,
        smoke_output=envelope.get("smoke_output"),
        error=envelope.get("error"),
        duration_ms=duration_ms,
    )


__all__ = [
    "DEFAULT_ALLOWED_IMPORTS",
    "MAX_SOURCE_BYTES",
    "ValidationResult",
    "validate_tool_module",
]
