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


def _smoke_execute_step(
    index: Path,
    *,
    workspace: Path,
    enforce_blank_canvas: bool = True,
    capture_shot: Path | None = None,
) -> VerifyStep:
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

    The index is resolved to an ABSOLUTE path here (harness-53tp). The
    verify step executes with cwd set to the workspace, and the runner
    joins any relative path to its cwd — so a relative ``index`` (which
    ``_browser_app_index`` returns when the driver passes a relative
    ``--workspace``) would double-join to
    ``<workspace>/<workspace>/index.html`` and fail "index not found"
    on every turn. Absolute path → the runner uses it verbatim.

    ``enforce_blank_canvas`` (harness-6dsn): when False, append
    ``--no-blank-canvas`` so the runner skips the all-one-color check.
    The driver passes False during the early phase of an incremental
    from-scratch build, where a skeleton legitimately renders nothing
    until a later render milestone closes. Console / page-error checks
    stay active regardless — those are unconditional bugs.

    harness-5vn6t: when ``<workspace>/.harness/smoke_setup.js``
    exists, pass it via ``--setup=`` so the runner exercises it after
    load. This closes the "sprite never on screen at load" verify blind
    spot — a render path for a conditionally-spawned entity (cops,
    bullets, peds) that only runs once state is set up. The scenario
    spawns those entities so their render/init throws surface; a missing
    spawn function fails the smoke loudly. ``.harness`` is excluded from
    the file census / regression guard, so the scenario never counts as
    a deliverable edit.
    """
    flag = "" if enforce_blank_canvas else " --no-blank-canvas"
    setup = workspace / ".harness" / "smoke_setup.js"
    setup_flag = f" --setup={shlex.quote(str(setup.resolve()))}" if setup.is_file() else ""
    # harness-u1il5: optional behavioral assertions. When
    # ``.harness/smoke_assert.js`` exists it runs after the settle window
    # and fails the smoke on a non-empty failure array — closing the
    # render-only blind spot where input handlers (fire/walk/weapon) that
    # no draw check exercises could close blind. ``.harness`` is excluded
    # from the file census / regression guard, so the probe never counts
    # as a deliverable edit (mirrors smoke_setup.js).
    assert_js = workspace / ".harness" / "smoke_assert.js"
    assert_flag = (
        f" --assert={shlex.quote(str(assert_js.resolve()))}" if assert_js.is_file() else ""
    )
    # harness-ke4hx.4: when advisory vision-QA is on, ask the runner to
    # write a post-settle screenshot. The flag is harmless to the gate
    # itself — the smoke still passes/fails on errors + blank-canvas; the
    # screenshot is consumed out-of-band by the loop's advisory QA pass.
    shot_flag = f" --screenshot={shlex.quote(str(capture_shot.resolve()))}" if capture_shot else ""
    cmd = (
        f"{shlex.quote(sys.executable)} -m harness.driver.smoke_runner"
        f"{flag}{setup_flag}{assert_flag}{shot_flag} {shlex.quote(str(index.resolve()))}"
    )
    # S604: ``shell=True`` here is a VerifyStep dataclass field, NOT a
    # subprocess kwarg (mirrors _js_verify_step / _py_verify_step
    # above). The cmd is built from sys.executable + a workspace-
    # resolved path; no untrusted interpolation.
    return VerifyStep(cmd=cmd, shell=True)  # noqa: S604


# Browser globals that mark a .js/.mjs file as browser-authored (vs a
# Node script). Their presence + the ABSENCE of an entry index.html is
# the "lost / never-had the entry point" signal (harness-9ugc) — the
# runtime smoke gate needs index.html, so without it a drive verifies
# syntax-only and false-closes.
_BROWSER_GLOBAL_RE = re.compile(
    r"\b(?:document|window|requestAnimationFrame|getElementById|"
    r"getContext|addEventListener|canvas)\b"
)


def _workspace_has_browser_js(workspace: Path) -> bool:
    """True iff some workspace .js / .mjs references a browser global —
    i.e. it's browser-authored code that needs an index.html to run.
    Bounded read; tolerates unreadable files silently."""
    for path in workspace.rglob("*"):
        if path.suffix not in _LOCAL_JS_SUFFIXES or not path.is_file():
            continue
        if any(part in _EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            continue
        if _BROWSER_GLOBAL_RE.search(text):
            return True
    return False


def missing_entry_html_reason(workspace: Path) -> str | None:
    """Loud reason when the workspace has browser-authored JS but NO
    entry ``index.html`` at the root (harness-9ugc). The runtime
    smoke-execute gate requires index.html; without it the gate
    silently produces no step and the drive verifies syntax-only — the
    b85f4008 false-close class for the *absent*-index case that
    ``browser_smoke_skip_reason`` (which only fires when index.html
    EXISTS but Playwright is missing) does not cover. Run 3c7c9da2
    closed 8 beads this way against a wiped workspace.

    Returns None when an index.html exists (gate can run) OR there's no
    browser JS (nothing runtime to verify — e.g. a fresh §1 build that
    hasn't created any source yet)."""
    for name in _INDEX_FILENAMES:
        if (workspace / name).is_file():
            return None
    if not _workspace_has_browser_js(workspace):
        return None
    return (
        "workspace has browser-authored JS (uses document/canvas/window) but "
        "no index.html at the root — the runtime smoke gate is OFF, so closes "
        "would be SYNTAX-ONLY (the false-success class that closed 8 beads in "
        "run 3c7c9da2). Restore/create the entry index.html, or pass "
        "--allow-missing-smoke to proceed without runtime verification."
    )


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


# Markers the smoke runner emits to stderr (all printed by
# smoke_runner.py). Used to pull the actionable symptom out of a verify
# failure string so the handoff can surface it cleanly instead of the
# raw "<python -m harness.driver.smoke_runner …> exit=1: …" command echo.
_SMOKE_MARKERS: tuple[str, ...] = (
    "smoke-execute:",
    "pageerror:",
    "blank-canvas",
)


def extract_smoke_symptom(reason: str | None) -> str | None:
    """Pull the runtime-smoke symptom out of a verify/close failure
    ``reason`` — everything from the first smoke marker onward, with the
    leading command echo (`… python3 -m harness.driver.smoke_runner …
    exit=1:`) and failure-prefix noise stripped.

    Returns None when ``reason`` carries no smoke output (a parse-gate
    failure, a bd error, a plain "issue still open" reason, …) so callers
    can fall back to rendering the reason verbatim. harness-estby follow-up
    (handoff feedback): a clean symptom — "canvas rendered nothing …",
    "pageerror: X is not defined" — is far more actionable to the model
    than the buried command line, and the symptom is what the model has to
    fix to pass the gate and close."""
    if not reason:
        return None
    hits = [reason.find(m) for m in _SMOKE_MARKERS]
    hits = [i for i in hits if i >= 0]
    if not hits:
        return None
    symptom = reason[min(hits) :].strip()
    return symptom or None


def default_workspace_verify_steps(
    workspace: Path,
    *,
    enforce_blank_canvas: bool = True,
    capture_shot: Path | None = None,
) -> tuple[VerifyStep, ...]:
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

    harness-6dsn: ``enforce_blank_canvas`` is threaded to the smoke
    step. The driver sets it False during the early phase of an
    incremental from-scratch build (before the render milestone closes)
    so a legitimately-blank skeleton isn't failed. Console / page-error
    checks always run.
    """
    steps: list[VerifyStep] = []
    if _workspace_has_file(workspace, (".js", ".mjs", ".cjs")) and shutil.which("node"):
        steps.append(_js_verify_step())
    if _workspace_has_file(workspace, (".py",)) and shutil.which("python"):
        steps.append(_py_verify_step())
    if _playwright_available():
        index = _browser_app_index(workspace)
        if index is not None:
            steps.append(
                _smoke_execute_step(
                    index,
                    workspace=workspace,
                    enforce_blank_canvas=enforce_blank_canvas,
                    capture_shot=capture_shot,
                )
            )
    return tuple(steps)


__all__ = [
    "browser_smoke_skip_reason",
    "default_workspace_verify_steps",
    "extract_smoke_symptom",
    "missing_entry_html_reason",
]
