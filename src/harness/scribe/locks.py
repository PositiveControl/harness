"""OS-level advisory lock for scribe runs.

Two scribe processes on the same session (e.g. interactive
`harness memory scribe` while a launchd job runs the same session) would
both read the same watermark, extract the same transcript window, and
double-write candidates. Filesystem-level `flock` serializes them
cheaply without requiring cross-process SQLite coordination."""

from __future__ import annotations

import contextlib
import fcntl
from collections.abc import Iterator
from pathlib import Path


class ScribeLockBusy(RuntimeError):  # noqa: N818 — short, explicit; not an Error subclass by convention
    """Raised when a non-blocking lock acquire fails because another
    process holds the lock for the same session."""


def _safe_filename(session_id: str) -> str:
    # Keep lock filenames human-readable while refusing path-traversal.
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id) or "unnamed"


@contextlib.contextmanager
def session_lock(
    session_id: str,
    lock_dir: Path,
    *,
    blocking: bool = True,
) -> Iterator[None]:
    """Acquire an advisory lock keyed by session_id. Blocking by default;
    pass blocking=False to fail fast with ScribeLockBusy.

    The lock file lives at lock_dir/scribe.<session>.lock and is reused
    across runs. We don't delete it on release — that would race with
    another process opening it before we unlink."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"scribe.{_safe_filename(session_id)}.lock"
    fd = path.open("w")
    try:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fd.fileno(), flags)
        except BlockingIOError as exc:
            raise ScribeLockBusy(f"another scribe is running for session {session_id!r}") from exc
        try:
            yield
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    finally:
        fd.close()
