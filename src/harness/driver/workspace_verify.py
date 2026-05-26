"""Workspace-typed default VerifySteps for the executor loop — harness-oxj7.

Surfaced by loop run 26c39558 root-cause analysis. With the parse-gate
on write (harness-h6wa) catching SyntaxErrors at edit time, there's
still a residual class of "closed ≠ runnable" failures the VERIFY
phase should backstop: a model could compound several individually-
parseable edits into an artifact that fails to load (missing module,
malformed JSON config consumed by another tool, etc.). The VERIFY
phase already runs operator-supplied `VerifyStep` entries from
`--plan-draft`; this module adds workspace-typed BASELINE steps that
run regardless of whether the operator authored verify steps.

Self-gating policy mirrors `tools.parse_check`:
- No matching files in the workspace → no step (silent).
- Required parser binary missing from PATH → no step (silent).
The gate adds signal where it can and stays silent where it can't,
so a workspace this module can't verify doesn't get spurious failures.

Excluded paths: `node_modules`, `.venv`, `venv`, `.git`, `__pycache__`,
`.harness`, `.mypy_cache`, `.ruff_cache`, `.pytest_cache`, `dist`,
`build` — common heavy / vendored / generated directories where a
parse failure isn't actionable for the model.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
import shutil
import sys
from pathlib import Path

from harness.driver.planner import VerifyStep

# Directory names we never want to walk into when looking for project
# source files OR when handing the path to a recursive parser. Kept
# as a set for O(1) name lookups during the existence scan and used
# verbatim in the shell `find` exclusions below.
_EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".harness",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

# `find` exclusion clauses used in the JS / Python verify commands.
# Built once at module import so the strings stay stable + reviewable.
# `*/<name>/*` matches any depth — the loop runs `find .` from the
# workspace root and we want to skip these dirs wherever they appear.
_FIND_PRUNES = " ".join(f"-not -path '*/{name}/*'" for name in sorted(_EXCLUDED_DIR_NAMES))


def _workspace_has_file(workspace: Path, suffixes: tuple[str, ...]) -> bool:
    """True iff at least one regular file with any of `suffixes` exists
    in `workspace`, ignoring excluded directories at any depth. Short-
    circuits on first match — cheap on workspaces of any reasonable
    size. Tolerates permission errors silently (a directory we can't
    descend simply doesn't contribute matches)."""
    try:
        stack: list[Path] = [workspace]
        while stack:
            current = stack.pop()
            try:
                entries = list(current.iterdir())
            except (OSError, PermissionError):
                continue
            for entry in entries:
                if entry.is_dir():
                    if entry.name in _EXCLUDED_DIR_NAMES:
                        continue
                    stack.append(entry)
                    continue
                if entry.is_file() and entry.suffix in suffixes:
                    return True
    except (OSError, PermissionError):
        return False
    return False


def _js_verify_step() -> VerifyStep:
    """`node --check` over every workspace .js / .mjs / .cjs file.
    Catches the loop-run 26c39558 failure mode at the phase boundary
    even if the per-write parse-gate (harness-h6wa) was somehow
    bypassed — defense in depth. `xargs -0 -n1` propagates non-zero
    exit on any single failing file."""
    cmd = (
        f"find . -type f \\( -name '*.js' -o -name '*.mjs' -o -name '*.cjs' \\) "
        f"{_FIND_PRUNES} -print0 "
        f"| xargs -0 -n1 node --check"
    )
    # S604: `shell=True` here is a VerifyStep dataclass field, NOT a
    # subprocess kwarg. The actual subprocess call is in
    # `_exec_test_cmd` (already noqa'd as cmd-from-trusted-local-source).
    # The dataclass field tells the executor to run via /bin/sh -c so
    # the find | xargs pipe works; the cmd string is built from
    # constants above, no untrusted interpolation.
    return VerifyStep(cmd=cmd, shell=True)  # noqa: S604


def _py_verify_step() -> VerifyStep:
    """`ast.parse` over every workspace .py file. Uses ast (not
    py_compile) to avoid writing .pyc bytecode into the workspace —
    a verify step is a read-only contract over the project state.
    `xargs -0 -n1` propagates non-zero exit on any syntax error."""
    cmd = (
        f"find . -type f -name '*.py' {_FIND_PRUNES} -print0 "
        f"| xargs -0 -n1 python -c "
        f"'import ast,sys; ast.parse(open(sys.argv[1]).read(), filename=sys.argv[1])'"
    )
    # S604: `shell=True` here is a VerifyStep dataclass field, NOT a
    # subprocess kwarg. The actual subprocess call is in
    # `_exec_test_cmd` (already noqa'd as cmd-from-trusted-local-source).
    # The dataclass field tells the executor to run via /bin/sh -c so
    # the find | xargs pipe works; the cmd string is built from
    # constants above, no untrusted interpolation.
    return VerifyStep(cmd=cmd, shell=True)  # noqa: S604


# --- harness-4b8v: browser-app smoke-execute --------------------

# Matches ``<script ... src="..."...>``. Captures the src attribute so
# we can classify it (local .js/.mjs → workspace-authored runtime,
# CDN → third-party we don't gate). Case-insensitive because HTML is.
# ``[^>]*`` covers any other attributes between ``<script`` and
# ``src=``; newlines inside the tag are matched (the class excludes
# only ``>``).
_SCRIPT_SRC_RE = re.compile(
    r'<script\b[^>]*\bsrc\s*=\s*[\'"]([^\'"]+)[\'"]',
    re.IGNORECASE,
)

# Prefixes that classify a script src as remote. The protocol-relative
# ``//`` form resolves to the page's scheme — for ``file://`` pages it
# resolves to ``file://`` too, which would fail to load anyway, but the
# common case is operators copy-pasting CDN snippets from HTTPS docs.
# Treating ``//`` as remote keeps the gate from chasing CDN scripts on
# every drive turn.
_REMOTE_SRC_PREFIXES = ("http://", "https://", "//")

# Suffixes the smoke gate considers "workspace JS." `.cjs` is omitted
# because browsers can't load CommonJS via ``<script>`` directly; a
# workspace using .cjs is a Node app, not a browser app, and the
# node-parse-check above already covers it.
_LOCAL_JS_SUFFIXES = (".js", ".mjs")

# Entry HTML filenames the gate looks for at the workspace root.
# Nested entry HTML (e.g. ``docs/index.html``) is intentionally not
# scanned — the driver tells the model to load the root artifact, so
# that's what we smoke-test.
_INDEX_FILENAMES = ("index.html", "index.htm")


def _browser_app_index(workspace: Path) -> Path | None:
    """Return the entry HTML iff ``workspace`` looks like a browser app,
    else None.

    A "browser app" here means: workspace root contains
    ``index.html`` (or ``index.htm``) that references at least one
    local ``.js`` / ``.mjs`` file via a ``<script src="...">`` tag.
    CDN-hosted scripts (``http://``, ``https://``, ``//``) don't
    count — the smoke step is for code the workspace authors, not
    third-party libs.

    Returns the first matching index path so the caller can pass it
    verbatim to the runner; None when no candidate qualifies.
    """
    for name in _INDEX_FILENAMES:
        index = workspace / name
        if not index.is_file():
            continue
        try:
            html = index.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue
        for match in _SCRIPT_SRC_RE.finditer(html):
            src = match.group(1).strip()
            if not src or src.startswith(_REMOTE_SRC_PREFIXES):
                continue
            if src.lower().endswith(_LOCAL_JS_SUFFIXES):
                return index
    return None


def _playwright_available() -> bool:
    """True iff the ``playwright`` Python package is importable in the
    current interpreter. A missing Chromium binary still surfaces at
    smoke-runner exec time (the runner treats "Executable doesn't
    exist" as a skip, not a failure); the gate doesn't try to
    second-guess Playwright's installation state beyond the import."""
    try:
        return importlib.util.find_spec("playwright") is not None
    except (ImportError, ValueError):
        return False


def _smoke_execute_step(index: Path) -> VerifyStep:
    """Smoke-execute verify step (harness-4b8v).

    Invokes ``harness.driver.smoke_runner`` under the same interpreter
    the driver is running (``sys.executable``), passing the absolute
    path to the entry HTML. The runner launches headless Chromium,
    loads ``file://<index>``, waits for ``load`` plus a settle window,
    and exits non-zero on any console.error / unhandled page error.

    Catches the class of runtime-on-load JS bugs that ``node --check``
    cannot see: ``canvas.getContext('d')`` returning null,
    ``Math.flor`` silently returning undefined, ``addEventListner``
    typos, etc. The parse-gate accepts these as valid JS; only running
    the artifact surfaces the throw.
    """
    cmd = f"{shlex.quote(sys.executable)} -m harness.driver.smoke_runner {shlex.quote(str(index))}"
    # S604: ``shell=True`` here is a VerifyStep dataclass field, NOT a
    # subprocess kwarg (mirrors _js_verify_step / _py_verify_step
    # above). The cmd is built from sys.executable + a workspace-
    # resolved path; no untrusted interpolation.
    return VerifyStep(cmd=cmd, shell=True)  # noqa: S604


def browser_smoke_skip_reason(workspace: Path) -> str | None:
    """Return a human-readable reason iff ``workspace`` looks like a
    browser app (index.html referencing local JS) but the smoke-execute
    gate cannot contribute a step — else None.

    This exists so a *degraded* verify is never silent. The original
    b85f4008 false-close (two GTAII issues auto-closed while the game
    crashed on load) happened precisely because the smoke gate skipped
    invisibly: Playwright wasn't importable, so the strongest gate
    produced no step and no signal. The loop logs this reason loudly at
    startup so the operator knows runtime verification is OFF and can
    install the ``browser`` extra (``uv sync --extra browser`` +
    ``playwright install chromium``) before trusting auto-closes.

    Returns None when there's no browser app (nothing to warn about) or
    when the smoke gate WILL run (Playwright importable) — the
    chromium-binary-missing case still surfaces at smoke-runner exec
    time as a one-line skip hint, so we don't duplicate it here."""
    if _playwright_available():
        return None
    if _browser_app_index(workspace) is None:
        return None
    return (
        "browser app detected (index.html + local JS) but the smoke-execute "
        "verify gate is OFF: Playwright is not installed. Runtime-on-load JS "
        "bugs will NOT fail verify, so auto-closes are syntax-only. Install "
        "with `uv sync --extra browser && uv run playwright install chromium`."
    )


def default_workspace_verify_steps(workspace: Path) -> tuple[VerifyStep, ...]:
    """Synthesize the baseline VerifyStep tuple for a workspace.

    Inspects the workspace for known file types and returns one
    VerifyStep per type whose parser binary is on PATH. The returned
    tuple is appended to per-item operator-authored verify steps so
    both gates must pass for VERIFY to clear.

    Empty tuple when no recognized file types are present or all
    required parsers are missing — the loop's existing trust-the-close
    path is preserved when the gate has nothing to say.

    harness-4b8v: when the workspace looks like a browser app AND
    Playwright is importable, also appends a smoke-execute step that
    runtime-loads ``index.html`` to catch runtime-on-load JS bugs that
    parse gates can't see. Skips silently when either condition fails.
    """
    steps: list[VerifyStep] = []
    if _workspace_has_file(workspace, (".js", ".mjs", ".cjs")) and shutil.which("node"):
        steps.append(_js_verify_step())
    if _workspace_has_file(workspace, (".py",)) and shutil.which("python"):
        steps.append(_py_verify_step())
    if _playwright_available():
        index = _browser_app_index(workspace)
        if index is not None:
            steps.append(_smoke_execute_step(index))
    return tuple(steps)


__all__ = [
    "browser_smoke_skip_reason",
    "default_workspace_verify_steps",
]
