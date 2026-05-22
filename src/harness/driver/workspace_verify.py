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

import shutil
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


def default_workspace_verify_steps(workspace: Path) -> tuple[VerifyStep, ...]:
    """Synthesize the baseline VerifyStep tuple for a workspace.

    Inspects the workspace for known file types and returns one
    VerifyStep per type whose parser binary is on PATH. The returned
    tuple is appended to per-item operator-authored verify steps so
    both gates must pass for VERIFY to clear.

    Empty tuple when no recognized file types are present or all
    required parsers are missing — the loop's existing trust-the-close
    path is preserved when the gate has nothing to say."""
    steps: list[VerifyStep] = []
    if _workspace_has_file(workspace, (".js", ".mjs", ".cjs")) and shutil.which("node"):
        steps.append(_js_verify_step())
    if _workspace_has_file(workspace, (".py",)) and shutil.which("python"):
        steps.append(_py_verify_step())
    return tuple(steps)


__all__ = [
    "default_workspace_verify_steps",
]
