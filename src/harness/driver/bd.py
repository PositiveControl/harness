"""DriverBd — bd CLI client for the multi-turn driver (harness-b4wx).

The existing `BeadsAdapter` (src/harness/store/bd_adapter.py) is ab-scoped:
it targets the airton_b isolated database, enforces scope:professional /
scope:personal labels on every create, runs an in-flight cap, and bootstraps
ab-specific custom types. The driver targets the MAIN project bd (the same
database `bd ready` reads when Mark runs it from the repo root) — that's a
different data plane.

This module is the driver's bd client. It shells to `bd` with a configurable
cwd (the workspace root by default) and exposes only the operations the
loop driver needs:

  * `show(issue_id)`              read one issue
  * `ready()`                     list bd ready --json
  * `ready_under_epic(epic_id)`   ready ∩ descendants(epic) — v0 = direct children only
  * `create_with_labels(...)`     bd create + --label flags
  * `close(issue_id, reason=...)` bd close
  * `dep_add(blocker, blocked)`   bd dep add
  * `flag_human(issue_id, reason)` bd label add <id> human (+ note)
  * `write_thought(...)`          create-then-close a thought:* bead in one call
  * `write_session_state(...)`    convenience wrapper over write_thought
  * `thoughts_in_loop_run(id)`    bd search by loop-run-<id> label across thought:* types

Loop-run scoping is by bd label: every thought-bead the driver writes carries
a `loop-run-<id>` label, so `thoughts_in_loop_run` is a label query, not a
timestamp range. Cleaner than scoping by created_at (which has edge cases
when the operator writes their own beads mid-run).

Reuses `BeadsIssue` + `_issue_from_json` from `src/harness/store/_bd_types.py`
— that record is non-ab-coupled (no scope assumptions baked in).

`subprocess` and `shutil` stay imported at module level so tests can
monkeypatch `harness.driver.bd.subprocess.run` / `harness.driver.bd.shutil.which`
at the boundary, same pattern as `bd_adapter`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from harness.store._bd_types import BeadsIssue, _issue_from_json


class DriverBdError(RuntimeError):
    """Raised when the bd CLI is missing, unreachable, or returns a
    non-zero exit with stderr the caller needs to surface."""


# Default thought-bead label set the driver queries when assembling a
# handoff. Public so callers can override (e.g. include thought:hypothesis
# if a future handoff variant wants it).
DEFAULT_THOUGHT_TYPES: tuple[str, ...] = (
    "thought:decision",
    "thought:observation",
    "thought:question",
)


class DriverBd:
    """Subprocess wrapper around `bd` for the main project database.

    Construct once per loop run with the workspace root as `bd_dir`.
    All methods dispatch through `_run` with `cwd=bd_dir` so behavior
    is identical to what an operator would see invoking `bd` from the
    same directory.
    """

    def __init__(self, bd_dir: Path, *, bd_executable: str = "bd") -> None:
        self._bd_dir = bd_dir
        self._bd = bd_executable

    @property
    def bd_dir(self) -> Path:
        return self._bd_dir

    # --- subprocess plumbing -----------------------------------------

    def _run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self._bd, *args]
        try:
            result = subprocess.run(  # noqa: S603 — fixed bd executable, no shell
                cmd,
                cwd=self._bd_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise DriverBdError(
                f"bd executable not found at {self._bd!r}. Install beads and ensure it's on PATH."
            ) from exc
        if check and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise DriverBdError(f"bd command failed ({' '.join(args)}): {stderr}")
        return result

    def verify(self) -> None:
        """Confirm bd is on PATH and the workspace has a `.beads/`
        directory. Raises DriverBdError with a repair hint on failure.
        Does NOT probe the Dolt server — bd auto-starts it on demand
        in the main project workflow and an explicit probe here would
        slow every loop start by ~1 second."""
        if shutil.which(self._bd) is None:
            raise DriverBdError(
                f"bd executable not found at {self._bd!r}. Install beads and ensure it's on PATH."
            )
        if not (self._bd_dir / ".beads").exists():
            raise DriverBdError(
                f"workspace {self._bd_dir} has no .beads/ directory. "
                f"Run `cd {self._bd_dir} && bd init` to initialize a project bd database."
            )

    # --- reads --------------------------------------------------------

    def show(self, issue_id: str) -> BeadsIssue:
        """`bd show <id> --json`. Returns the issue. Raises on
        not-found (bd exits non-zero with a clear message)."""
        result = self._run(["show", issue_id, "--json"])
        data = json.loads(result.stdout)
        # bd show returns either a single object or a list of one;
        # normalize to a single record.
        if isinstance(data, list):
            if not data:
                raise DriverBdError(f"bd show {issue_id} returned an empty list")
            data = data[0]
        return _issue_from_json(data)

    def ready(self) -> list[BeadsIssue]:
        """All globally-ready issues. v0 doesn't filter by epic; use
        `ready_under_epic` for that.

        Passes `-n 9999` to bypass `bd ready`'s default cap of 10
        results (harness-c6yu). Without the override, `ready_under_epic`
        silently returns empty when none of an epic's children sit in
        the global top 10 — the loop then exits 'success' on turn 0
        without doing any work. 9999 is a pragmatic ceiling well above
        any realistic ready-queue depth; even a 10x project growth
        (~1000 ready beads) fits comfortably."""
        result = self._run(["ready", "-n", "9999", "--json"])
        return _parse_issue_list(result.stdout)

    def ready_under_epic(self, epic_id: str) -> list[BeadsIssue]:
        """Ready issues that are direct dependencies of `epic_id` (v0 —
        one-level-deep grouping). Returns the intersection in `bd ready`
        priority order so the executor consumes the queue in the same
        order an operator sees with `bd ready`.

        v0 limitation: walks only the direct `dependencies` array on
        `bd show <epic>`. Grandchildren (deps of deps) are not included.
        Future revisions can recurse if multi-level epics become a
        pattern — for now keeps the query simple and the dep traversal
        bounded."""
        epic = self.show(epic_id)
        # bd show emits the children inline under `dependencies` —
        # each entry has its own id and dependency_type. Pull those ids.
        raw_deps = epic.raw.get("dependencies") or []
        child_ids = {str(d.get("id")) for d in raw_deps if d.get("id")}
        if not child_ids:
            return []
        return [issue for issue in self.ready() if issue.id in child_ids]

    def thoughts_in_loop_run(
        self,
        loop_run_id: str,
        *,
        types: Sequence[str] = DEFAULT_THOUGHT_TYPES,
    ) -> list[BeadsIssue]:
        """All beads carrying both a `loop-run-<id>` label AND one of
        the `types` labels (thought:decision / :observation / :question
        by default). Returns newest-first by created_at, so the handoff
        builder can slice with recency-weighted truncation.

        Implementation: one `bd search` call per type, then merge +
        dedupe + sort. bd's search semantics don't support a single
        --label-and-label query, so we fan out client-side."""
        seen: dict[str, BeadsIssue] = {}
        loop_label = f"loop-run-{loop_run_id}"
        for thought_type in types:
            result = self._run(
                [
                    "search",
                    "--label",
                    loop_label,
                    "--label",
                    thought_type,
                    "--json",
                ],
                check=False,
            )
            # `bd search` exits non-zero when nothing matches in some
            # versions — tolerate empty results without raising.
            if result.returncode != 0:
                continue
            for issue in _parse_issue_list(result.stdout):
                seen.setdefault(issue.id, issue)
        return sorted(seen.values(), key=_created_at_key, reverse=True)

    # --- writes -------------------------------------------------------

    def create_with_labels(
        self,
        *,
        title: str,
        description: str,
        issue_type: str = "task",
        priority: int = 2,
        labels: Sequence[str] = (),
        acceptance: str | None = None,
    ) -> str:
        """`bd create` with one --label per entry in `labels`. Returns
        the new bead's id. Raises if bd doesn't emit a parseable
        confirmation line."""
        args: list[str] = [
            "create",
            f"--title={title}",
            f"--description={description}",
            f"--type={issue_type}",
            f"--priority={priority}",
        ]
        if acceptance:
            args.append(f"--acceptance={acceptance}")
        for label in labels:
            args.append(f"--label={label}")
        result = self._run(args)
        issue_id = _extract_created_id(result.stdout)
        if issue_id is None:
            raise DriverBdError(
                f"bd create did not return a parseable id; stdout was: {result.stdout!r}"
            )
        return issue_id

    def close(self, issue_id: str, *, reason: str | None = None) -> None:
        args = ["close", issue_id]
        if reason:
            args.append(f"--reason={reason}")
        self._run(args)

    def reopen(self, issue_id: str) -> None:
        """`bd update <id> --status=open`. Used by the loop driver when a
        verify gate fails after the model closed the bd issue
        (harness-xfh2): flipping the issue back to open re-admits it to
        `ready_under_epic` so the next iteration retries with the
        verify failure surfaced via `prior_attempt_failure` in the
        handoff."""
        self._run(["update", issue_id, "--status=open"])

    def dep_add(self, blocked: str, blocker: str) -> None:
        """`bd dep add <blocked> <blocker>` — `blocked` depends on
        `blocker`. Argument order matches the bd CLI."""
        self._run(["dep", "add", blocked, blocker])

    def flag_human(self, issue_id: str, *, reason: str) -> None:
        """Surface the issue to the operator's `bd human list` queue.

        bd's human-needed queue is keyed on the `human` label (added via
        `bd label add <id> human`); the older `bd human <id> --reason=`
        create form was removed when `bd human` became a parent command
        (`list`/`respond`/`dismiss`/`stats`). The label is the
        load-bearing flag — `reason` rides as a note so the operator sees
        the park context inline on the bead. The label add runs first so a
        note hiccup can't strip the flag."""
        self._run(["label", "add", issue_id, "human"])
        self._run(["note", issue_id, reason])

    def add_label(self, issue_id: str, label: str) -> None:
        """`bd label add <id> <label>`. Used to mark an issue with a
        process flag (e.g. `auto-decomposed` so the park path never
        re-decomposes the same umbrella)."""
        self._run(["label", "add", issue_id, label])

    def write_thought(
        self,
        *,
        thought_type: str,
        title: str,
        body: str,
        loop_run_id: str,
        extra_labels: Sequence[str] = (),
    ) -> str:
        """Create a thought-graph bead, label it with both `thought_type`
        and `loop-run-<id>`, then close it immediately so the existing
        harness-vu3 harvest path mirrors it into procedural memory on
        the next session bootstrap. Returns the new bead id.

        `title` should be short (one-line summary); `body` is the
        full content that lands in the harvested episodic row."""
        labels = [thought_type, f"loop-run-{loop_run_id}", *extra_labels]
        issue_id = self.create_with_labels(
            title=title,
            description=body,
            issue_type="task",
            priority=2,
            labels=labels,
        )
        # Close immediately — harvest only picks up CLOSED thought:*
        # beads (mirrors the harness-vu3 contract).
        self.close(issue_id, reason=f"auto-closed by loop driver ({loop_run_id})")
        return issue_id

    def write_session_state(
        self,
        *,
        loop_run_id: str,
        current_issue_id: str,
        status: str,
        body: str,
    ) -> str:
        """Convenience wrapper over `write_thought` for the per-turn
        session-state bead. `status` is one of: success | failure |
        halted | interrupted. The body is prepended with a structured
        header so the harvested episodic row carries machine-readable
        context the next session can read back."""
        header = f"[loop_run={loop_run_id} issue={current_issue_id} status={status}]\n\n"
        return self.write_thought(
            thought_type="thought:session-state",
            title=f"session-state {current_issue_id} ({status})",
            body=header + body,
            loop_run_id=loop_run_id,
        )


# --- module helpers --------------------------------------------------


# Identical shape to `_bd_crud._parse_issue_list`, duplicated here so
# the driver module doesn't reach into the ab-scoped crud module's
# internals. Cheap copy — five lines.
def _parse_issue_list(stdout: str) -> list[BeadsIssue]:
    stdout = (stdout or "").strip()
    if not stdout:
        return []
    data = json.loads(stdout)
    if isinstance(data, dict):
        data = [data]
    return [_issue_from_json(d) for d in data]


def _extract_created_id(stdout: str) -> str | None:
    """Pull a `Created issue: <id>` confirmation line out of bd's
    create stdout. Tolerates leading whitespace and unicode prefixes
    (✓). Returns None if the line is missing — caller decides whether
    to raise."""
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if "Created issue:" not in line:
            continue
        _, _, rest = line.partition("Created issue:")
        rest = rest.strip()
        token = rest.split()[0] if rest else ""
        if token:
            return token
    return None


def _created_at_key(issue: BeadsIssue) -> str:
    """Sort key for newest-first ordering. Falls back to empty string
    so issues missing `created_at` sink to the bottom rather than
    raising."""
    raw_value = issue.raw.get("created_at") or ""
    return str(raw_value)


__all__ = [
    "DEFAULT_THOUGHT_TYPES",
    "DriverBd",
    "DriverBdError",
]
