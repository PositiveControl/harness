"""Workspace integrity + hygiene helpers for the executor drive loop.

Two concerns, both surfaced by the GTA2 drive runs:

1. **Regression guard (harness-16w6).** A parked issue — or one that
   rewrites the deliverable to a stub — can leave the shared workspace
   broken or degraded, poisoning every issue the drive runs afterward
   (dec8c1ae left dangling refs that broke load for all later issues;
   993ae19c left a temporal-dead-zone crash AND a stub-rewrite that
   deleted §1-§6/§9-§11). The load/render smoke gate catches the *broken*
   case but not the *still-loads-but-gutted* case. These helpers snapshot
   the workspace per issue, restore the last-green snapshot as a rollback
   safety net, and detect catastrophic regressions (deleted source files,
   vanished top-level symbols, dramatic shrink) the smoke gate can't see.

2. **Scratch hygiene (harness-ul5z).** The executor creates planning /
   temp / validate / backup files in the workspace and re-reads them on
   later turns, burning the per-turn round budget and confusing itself.
   These helpers diff the workspace file list across an issue and archive
   agent-created scratch (matching configurable patterns) so it stops
   accumulating.

Everything here is pure + filesystem-only — no bd, no model, no loop
state — so it tests in isolation against a ``tmp_path`` workspace.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path

# Directories we never snapshot, scan, or sweep — vendored / generated /
# our own state. Mirrors loop.py's snapshot excludes (loop re-exports
# these for back-compat) plus the dirs the verify module prunes.
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
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

DEFAULT_EXCLUDE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")

# Glob patterns (matched against the file basename) that mark a file as
# agent-created scratch the drive should archive once the issue that
# made it completes. Deliberately conservative + name-shaped: real
# deliverables (game.js, index.html, app.py) never match these.
DEFAULT_SCRATCH_PATTERNS: tuple[str, ...] = (
    "*_plan.md",
    "*-plan.md",
    "*_plan.txt",
    "*_notes.md",
    "temp_*",
    "tmp_*",
    "*.tmp",
    "*_backup.*",
    "*.bak",
    "*.orig",
    "validate_*",
    "scratch*",
)

# Suffixes whose content we analyze for regression (symbol loss / shrink).
# A workspace's deliverable source lives in one of these; binaries and
# data files are out of scope for the symbol diff.
_SOURCE_SUFFIXES: tuple[str, ...] = (
    ".js",
    ".mjs",
    ".cjs",
    ".jsx",
    ".ts",
    ".tsx",
    ".py",
    # harness-9ugc: entry HTML + CSS count as deliverable source, so the
    # regression guard catches a deleted/gutted index.html — the exact
    # loss (no smoke gate, false success) that motivated this.
    ".html",
    ".htm",
    ".css",
)

# Top-level callable definitions we track across a snapshot boundary.
# Covers JS `function NAME(` / `NAME = function` / `NAME = (...)=>` and
# Python `def NAME` / `class NAME`. Losing one of these between a
# last-green snapshot and the current tree is the signal that an edit
# deleted working code (the stub-rewrite pattern).
_SYMBOL_RE = re.compile(
    r"(?:function\s+([A-Za-z_$][\w$]*)"
    r"|(?:^|\n)\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>)"
    r"|(?:^|\n)\s*def\s+([A-Za-z_][\w]*)"
    r"|(?:^|\n)\s*class\s+([A-Za-z_][\w]*))"
)


class WorkspaceTooBigError(RuntimeError):
    """Raised by :func:`archive_workspace` when the pre-tar uncompressed
    total exceeds the cap — the caller decides whether to abort or skip
    the snapshot for this issue."""


def iter_workspace_files(workspace: Path) -> Iterator[Path]:
    """Yield every regular file under ``workspace``, pruning
    :data:`DEFAULT_EXCLUDE_DIRS` and skipping
    :data:`DEFAULT_EXCLUDE_SUFFIXES`. followlinks=False guards against
    self-referential symlink loops."""
    for root, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDE_DIRS]
        for fname in files:
            if fname.endswith(DEFAULT_EXCLUDE_SUFFIXES):
                continue
            yield Path(root) / fname


def list_workspace_files(workspace: Path) -> set[str]:
    """Set of workspace-relative POSIX paths (the issue-boundary file
    census the scratch sweep diffs against)."""
    out: set[str] = set()
    for path in iter_workspace_files(workspace):
        try:
            out.add(path.relative_to(workspace).as_posix())
        except ValueError:
            continue
    return out


def archive_workspace(workspace: Path, dest: Path, *, size_cap_bytes: int) -> Path:
    """tar+gzip the (non-excluded) workspace tree to ``dest``. Raises
    :class:`WorkspaceTooBigError` before opening the tar if the
    uncompressed total exceeds ``size_cap_bytes`` (so an over-cap
    workspace leaves no partial tar behind)."""
    total = 0
    include: list[Path] = []
    for path in iter_workspace_files(workspace):
        try:
            total += path.stat().st_size
        except OSError:
            continue
        if total > size_cap_bytes:
            mb = size_cap_bytes // (1024 * 1024)
            raise WorkspaceTooBigError(f"workspace exceeds {mb}MB cap at {workspace}")
        include.append(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        for path in include:
            tf.add(path, arcname=path.relative_to(workspace).as_posix(), recursive=False)
    return dest


def _safe_members(tar: tarfile.TarFile, workspace: Path) -> list[tarfile.TarInfo]:
    """Members whose resolved destination stays inside ``workspace`` —
    drops any path-traversal entry. Our own archives are always safe;
    this guards a tampered/foreign tar from escaping the sandbox."""
    root = workspace.resolve()
    safe: list[tarfile.TarInfo] = []
    for m in tar.getmembers():
        dest = (workspace / m.name).resolve()
        if dest == root or root in dest.parents:
            safe.append(m)
    return safe


def restore_workspace(workspace: Path, snapshot: Path) -> tuple[int, list[str]]:
    """Roll ``workspace`` back to the state captured in ``snapshot``.

    Extracts every (safe) member over the workspace, then removes any
    current non-excluded file that is NOT in the snapshot — i.e. files
    the failing issue added after the snapshot was taken. Returns
    ``(restored_count, removed_relpaths)`` for logging. Excluded dirs
    (.harness etc.) are never touched, so run logs/snapshots survive."""
    with tarfile.open(snapshot, "r:gz") as tf:
        members = _safe_members(tf, workspace)
        member_names = {m.name for m in members}
        for m in members:
            # filter="data" (3.12+) rejects abs/.. paths + strips unsafe
            # metadata; _safe_members already vetted destinations.
            tf.extract(m, path=workspace, filter="data")
    removed: list[str] = []
    for path in iter_workspace_files(workspace):
        rel = path.relative_to(workspace).as_posix()
        if rel not in member_names:
            try:
                path.unlink()
                removed.append(rel)
            except OSError:
                continue
    return len(member_names), sorted(removed)


def extract_symbols(text: str) -> set[str]:
    """Top-level function/class/def names defined in ``text`` (see
    :data:`_SYMBOL_RE`). Used to detect an edit that deleted working
    code rather than adding to it."""
    names: set[str] = set()
    for groups in _SYMBOL_RE.findall(text):
        for g in groups:
            if g:
                names.add(g)
    return names


def _read_source_members(snapshot: Path) -> dict[str, str]:
    """Map of relative-path -> text for every source-suffixed file in the
    snapshot tar (best-effort UTF-8)."""
    out: dict[str, str] = {}
    with tarfile.open(snapshot, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(_SOURCE_SUFFIXES):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            out[m.name] = f.read().decode("utf-8", errors="replace")
    return out


def detect_regression(
    workspace: Path,
    snapshot: Path,
    *,
    shrink_ratio: float = 0.5,
    min_lines: int = 40,
) -> str | None:
    """Compare the current workspace source files against a (last-green)
    ``snapshot`` and return a human-readable reason if a catastrophic
    regression is present, else None.

    Fires on, per source file that existed in the snapshot:
      - the file is now **missing** entirely, or
      - it lost top-level **symbols** (functions/classes) that were
        defined before — names the message lists, or
      - it **shrank** below ``shrink_ratio`` of its snapshot line count
        (only for files with >= ``min_lines`` lines, so trimming a small
        file doesn't trip it).

    Conservative by design: adding code, renaming within a file (net
    symbol set unchanged), or reformatting never fires. This is the
    guard the load/render smoke gate cannot provide — a stub that still
    loads passes smoke but trips here."""
    before = _read_source_members(snapshot)
    if not before:
        return None
    for rel, old_text in before.items():
        cur = workspace / rel
        if not cur.is_file():
            return f"{rel} was deleted (existed in the last-green baseline)"
        try:
            new_text = cur.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        old_syms = extract_symbols(old_text)
        new_syms = extract_symbols(new_text)
        lost = old_syms - new_syms
        if lost:
            shown = ", ".join(sorted(lost)[:8])
            more = f" (+{len(lost) - 8} more)" if len(lost) > 8 else ""
            return (
                f"{rel} lost {len(lost)} previously-defined symbol(s): {shown}{more}. "
                f"Restore the deleted code — do not rewrite the file from scratch."
            )
        old_lines = old_text.count("\n") + 1
        new_lines = new_text.count("\n") + 1
        if old_lines >= min_lines and new_lines < old_lines * shrink_ratio:
            return (
                f"{rel} shrank from {old_lines} to {new_lines} lines "
                f"(<{int(shrink_ratio * 100)}% of the last-green baseline). "
                f"This looks like a rewrite that dropped working code; restore it."
            )
    return None


def workspace_changed(workspace: Path, snapshot: Path) -> bool:
    """True iff any source file under ``workspace`` differs from the
    last-green ``snapshot`` — content changed, a source file was added,
    or one was removed.

    Confirms an attempt actually produced edits before the driver
    auto-closes a still-open issue on the model's behalf (harness-82r1v).
    A green verify with no edits means no work was done this turn, so
    closing would be a false close. Scoped to ``_SOURCE_SUFFIXES`` to
    mirror :func:`detect_regression` — scratch/data files don't count as
    issue work."""
    before = _read_source_members(snapshot)
    current: dict[str, str] = {}
    for path in iter_workspace_files(workspace):
        if not path.name.endswith(_SOURCE_SUFFIXES):
            continue
        try:
            rel = path.relative_to(workspace).as_posix()
        except ValueError:
            continue
        try:
            current[rel] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    if set(before) != set(current):
        return True
    return any(before[rel] != current[rel] for rel in before)


def is_scratch(rel_path: str, patterns: Sequence[str]) -> bool:
    """True iff the basename of ``rel_path`` matches any scratch glob."""
    name = Path(rel_path).name
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def sweep_scratch(
    workspace: Path,
    baseline_files: set[str],
    *,
    patterns: Sequence[str],
    archive_dir: Path,
) -> list[str]:
    """Archive agent-created scratch produced during an issue.

    A file is swept iff it (a) is NOT in ``baseline_files`` (the census
    taken when the issue started — so it was created during the issue)
    and (b) matches a scratch pattern. Swept files are MOVED (not
    deleted) into ``archive_dir``, preserving their relative path, so a
    pattern misfire is recoverable. Returns the relative paths moved."""
    moved: list[str] = []
    for path in iter_workspace_files(workspace):
        rel = path.relative_to(workspace).as_posix()
        if rel in baseline_files or not is_scratch(rel, patterns):
            continue
        target = archive_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(target))
            moved.append(rel)
        except OSError:
            continue
    return sorted(moved)


__all__ = [
    "DEFAULT_EXCLUDE_DIRS",
    "DEFAULT_EXCLUDE_SUFFIXES",
    "DEFAULT_SCRATCH_PATTERNS",
    "WorkspaceTooBigError",
    "archive_workspace",
    "detect_regression",
    "extract_symbols",
    "is_scratch",
    "iter_workspace_files",
    "list_workspace_files",
    "restore_workspace",
    "sweep_scratch",
    "workspace_changed",
]
