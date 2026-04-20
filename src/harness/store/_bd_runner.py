"""Shared subprocess-runner base for BeadsAdapter (harness-vhoj).

Owns the state + infrastructure every domain mixin relies on: the
bd executable path, the ab bd_dir, the custom-types bootstrap, the
ab-assignee caps, and the `_run()` wrapper around `subprocess.run`.

Domain mixins (crud, focus, graph, memory) inherit this and
dispatch through `self._run(...)` without re-implementing the
subprocess plumbing.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

AB_CUSTOM_TYPES = ("project", "event", "habit")


class BeadsAdapterError(RuntimeError):
    """Raised when the bd CLI is missing, the bd_dir isn't initialized,
    or a bd subprocess call fails in a way the caller needs to know
    about."""


class TurnCapExceededError(BeadsAdapterError):
    """Raised when an ab-assignee bead create would exceed the
    per-turn budget. CaptureTool catches this and surfaces the hint
    to the model; the adapter does not swallow it, so any other
    caller sees it and can choose to ignore or report."""


class InflightCapExceededError(BeadsAdapterError):
    """Raised when an ab-assignee bead create would push the open
    ab-owned bead count past the in-flight cap. The message lists
    candidate beads suitable for closing first (low-priority and
    oldest-updated), so the caller can surface them to the model."""


class BeadsRunner:
    """Infrastructure layer — owns state + `_run`. Domain mixins
    inherit this and call `self._run(...)` directly."""

    def __init__(
        self,
        bd_dir: Path,
        *,
        bd_executable: str = "bd",
        default_exclude_assignee: str | None = None,
        ab_assignee: str | None = None,
        default_scope_allowlist: tuple[str, ...] | None = None,
        turn_cap: int = 3,
        inflight_cap: int = 10,
    ) -> None:
        """`default_exclude_assignee` (e.g. 'airton_b') hides beads owned
        by that assignee from the adapter's read methods — list_issues,
        ready, search, stale — unless a caller explicitly passes a
        positive `assignee=X` filter, which is respected as an opt-in.
        Write methods are never filtered.

        `ab_assignee` enables the ab-bead budget enforcement. When set,
        create() with that assignee checks both the in-flight cap
        (`inflight_cap`, default 10 open beads) and the per-turn cap
        (`turn_cap`, default 3 creates per turn). Cap hits raise
        InflightCapExceededError or TurnCapExceededError respectively.
        Callers call `reset_turn_counter()` at each user-turn boundary
        to refresh the turn budget. Leaving ab_assignee None disables
        both caps — tests and non-ab callers aren't affected.

        `default_scope_allowlist` (harness-j7y) narrows reads to
        items carrying any `scope:<value>` label in the allowlist,
        OR items whose assignee is `ab_assignee` (so ab's internal
        thought-graph beads aren't collateral-filtered — they flow
        through to `default_exclude_assignee` which decides whether
        to show them). Unset (None) = no scope filter; every read is
        unconstrained, same as before. Enables airton_b to ignore
        pure-project dev beads when browsing shared-dir state."""
        self._bd_dir = bd_dir
        self._bd = bd_executable
        self._types_ensured = False
        self._default_exclude_assignee = default_exclude_assignee
        self._ab_assignee = ab_assignee
        self._default_scope_allowlist = default_scope_allowlist
        self._turn_cap = turn_cap
        self._inflight_cap = inflight_cap
        self._ab_creates_this_turn = 0

    def reset_turn_counter(self) -> None:
        """Drop the per-turn ab-bead create counter to zero. Call from
        the chat loop right after a user-input boundary so the budget
        refreshes. Cheap — just an int write."""
        self._ab_creates_this_turn = 0

    @property
    def bd_dir(self) -> Path:
        return self._bd_dir

    def _run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self._bd, *args]
        try:
            result = subprocess.run(  # noqa: S603 — fixed bd executable, no shell
                cmd,
                cwd=self._bd_dir,
                capture_output=capture,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise BeadsAdapterError(
                f"bd executable not found at {self._bd!r}. Install beads and ensure it's on PATH."
            ) from exc
        if check and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise BeadsAdapterError(f"bd command failed ({' '.join(args)}): {stderr}")
        return result

    def verify(self, *, check_server: bool = True) -> None:
        """Confirm bd is runnable, bd_dir has a beads DB, and (by
        default) the project's Dolt server is reachable. Raises
        BeadsAdapterError with a repair hint on failure."""
        if shutil.which(self._bd) is None:
            raise BeadsAdapterError(
                f"bd executable not found at {self._bd!r}. Install beads and ensure it's on PATH."
            )
        if not self._bd_dir.exists():
            raise BeadsAdapterError(
                f"ab bd_dir {self._bd_dir} does not exist. "
                f"Create it and run `cd {self._bd_dir} && bd init` "
                "to initialize a fresh beads database."
            )
        beads_subdir = self._bd_dir / ".beads"
        if not beads_subdir.exists():
            raise BeadsAdapterError(
                f"ab bd_dir {self._bd_dir} has no .beads/ subdirectory. "
                f"Run `cd {self._bd_dir} && bd init` to initialize."
            )
        if check_server and not self._server_reachable():
            raise BeadsAdapterError(
                f"ab bd_dir {self._bd_dir} has its beads DB but the Dolt "
                "server isn't reachable. bd normally auto-starts on demand; "
                f"when that fails, run `cd {self._bd_dir} && bd dolt start` "
                "explicitly. Verify with `bd dolt test`."
            )

    def _server_reachable(self) -> bool:
        """Probe the project's Dolt server via `bd dolt test`. Returns
        False on subprocess failure, missing dolt subcommand, or
        explicit non-zero exit — caller decides whether to raise."""
        try:
            result = self._run(["dolt", "test"], check=False)
        except BeadsAdapterError:
            return False
        return result.returncode == 0

    def ensure_custom_types(self) -> None:
        """Idempotent: make sure bd's types.custom list includes ab's
        custom types (project, event, habit)."""
        if self._types_ensured:
            return
        existing = self._run(
            ["config", "get", "types.custom"],
            check=False,
        )
        current_raw = (existing.stdout or "").strip()
        current = {t.strip() for t in current_raw.split(",") if t.strip()} if current_raw else set()
        missing = [t for t in AB_CUSTOM_TYPES if t not in current]
        if missing:
            merged = sorted(current | set(AB_CUSTOM_TYPES))
            self._run(
                ["config", "set", "types.custom", ",".join(merged)],
            )
        self._types_ensured = True
