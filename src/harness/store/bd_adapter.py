"""Thin wrapper around the `bd` (beads) CLI — the single chokepoint
for ab's data-plane operations.

bd is treated as a black-box subprocess. Every call runs with
`cwd=<ab_bd_dir>` so the adapter targets ab's isolated beads database
and never touches the dev database co-located with the harness repo.
One-time setup (bd init / bd bootstrap in the ab_bd_dir) is the user's
responsibility; the adapter verifies on first use and raises with a
clear message if the dir hasn't been initialized.

Scope is enforced on every create. ab supports exactly two scopes —
`professional` and `personal` — applied as a `scope:<value>` label on
the bead. The adapter refuses to create an item without a scope.

Custom types (`project`, `event`, `habit`) are registered once on first
use via `bd config set types.custom`, merging with any pre-existing
custom types the user has configured.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_SCOPES = ("professional", "personal")
AB_CUSTOM_TYPES = ("project", "event", "habit")
# Built-in bd types plus the ab-specific additions. `bug`, `feature`,
# `chore`, `epic` are intentionally excluded from ab's vocabulary —
# keeps the ops surface clean and nudges the ops loop away from
# engineering-flavored types.
ALLOWED_TYPES = ("task", "decision", *AB_CUSTOM_TYPES)


class BeadsAdapterError(RuntimeError):
    """Raised when the bd CLI is missing, the bd_dir isn't initialized,
    or a bd subprocess call fails in a way the caller needs to know
    about."""


@dataclass(frozen=True)
class BeadsIssue:
    """Lightweight view of a bd issue. Only the fields ab cares about;
    the full JSON is available on `raw` for callers that need more."""

    id: str
    title: str
    status: str
    priority: int
    issue_type: str
    labels: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def scope(self) -> str | None:
        for label in self.labels:
            if label.startswith("scope:"):
                return label.split(":", 1)[1]
        return None


def _issue_from_json(data: dict[str, Any]) -> BeadsIssue:
    return BeadsIssue(
        id=str(data["id"]),
        title=str(data.get("title", "")),
        status=str(data.get("status", "")),
        priority=int(data.get("priority", 0)),
        issue_type=str(data.get("issue_type", "")),
        labels=tuple(data.get("labels", []) or []),
        raw=data,
    )


class BeadsAdapter:
    """Subprocess wrapper around the bd CLI. Construct once per session
    and pass to ab's tools; every method dispatches to bd with
    `cwd=bd_dir` so the isolation invariant holds without the caller
    having to think about it."""

    def __init__(self, bd_dir: Path, *, bd_executable: str = "bd") -> None:
        self._bd_dir = bd_dir
        self._bd = bd_executable
        self._types_ensured = False

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
        BeadsAdapterError with a repair hint on failure.

        `check_server=False` skips the Dolt reachability probe — useful
        in tests that have already stubbed subprocess and don't need
        to run a real `bd dolt test` round-trip."""
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
        custom types (project, event, habit). Skips silently if they
        are already registered."""
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

    def create(
        self,
        *,
        title: str,
        scope: str,
        issue_type: str = "task",
        description: str = "",
        priority: int = 2,
        parent: str | None = None,
        deps: Sequence[str] = (),
        extra_labels: Sequence[str] = (),
    ) -> str:
        """Create a bead with a mandatory scope label. Returns the new
        bead's id. Type defaults to `task` — override with `project`,
        `event`, `habit`, or `decision`."""
        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"scope must be one of {ALLOWED_SCOPES!r}, got {scope!r}")
        if issue_type not in ALLOWED_TYPES:
            raise ValueError(f"issue_type must be one of {ALLOWED_TYPES!r}, got {issue_type!r}")
        self.ensure_custom_types()
        args: list[str] = [
            "create",
            "--title",
            title,
            "--type",
            issue_type,
            "--priority",
            str(priority),
            "--description",
            description,
            "--add-label",
            f"scope:{scope}",
        ]
        for label in extra_labels:
            args.extend(["--add-label", label])
        if parent is not None:
            args.extend(["--parent", parent])
        result = self._run(args)
        issue_id = _extract_created_id(result.stdout)
        if issue_id is None:
            raise BeadsAdapterError(f"could not parse created id from bd output: {result.stdout!r}")
        for dep in deps:
            self._run(["dep", "add", issue_id, dep])
        return issue_id

    def close(self, issue_id: str, *, reason: str | None = None) -> None:
        args = ["close", issue_id]
        if reason is not None:
            args.extend(["--reason", reason])
        self._run(args)

    def update(self, issue_id: str, **fields: str) -> None:
        """Pass bd update fields as kwargs — e.g.
        `update("harness-x", priority="1", status="in_progress")`."""
        args = ["update", issue_id]
        for key, value in fields.items():
            flag = f"--{key.replace('_', '-')}"
            args.extend([flag, str(value)])
        self._run(args)

    def dep_add(self, issue: str, depends_on: str) -> None:
        self._run(["dep", "add", issue, depends_on])

    def list_issues(
        self,
        *,
        scope: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[BeadsIssue]:
        args = ["list", "--json"]
        if status is not None:
            args.extend(["--status", status])
        if limit is not None:
            args.extend(["--limit", str(limit)])
        result = self._run(args)
        issues = _parse_issue_list(result.stdout)
        if scope is not None:
            label = f"scope:{scope}"
            issues = [i for i in issues if label in i.labels]
        return issues

    def ready(self, *, scope: str | None = None, limit: int | None = None) -> list[BeadsIssue]:
        args = ["ready", "--json"]
        if limit is not None:
            args.extend(["--limit", str(limit)])
        result = self._run(args)
        issues = _parse_issue_list(result.stdout)
        if scope is not None:
            label = f"scope:{scope}"
            issues = [i for i in issues if label in i.labels]
        return issues

    def show(self, issue_id: str) -> BeadsIssue:
        result = self._run(["show", issue_id, "--json"])
        issues = _parse_issue_list(result.stdout)
        if not issues:
            raise BeadsAdapterError(f"no issue returned for id {issue_id!r}")
        return issues[0]

    def stale(self) -> list[BeadsIssue]:
        result = self._run(["stale", "--json"])
        return _parse_issue_list(result.stdout)

    def remember(self, insight: str) -> None:
        self._run(["remember", insight])

    def memories(self, query: str = "") -> str:
        args = ["memories"]
        if query:
            args.append(query)
        result = self._run(args)
        return result.stdout


def _parse_issue_list(stdout: str) -> list[BeadsIssue]:
    stdout = (stdout or "").strip()
    if not stdout:
        return []
    data = json.loads(stdout)
    if isinstance(data, dict):
        data = [data]
    return [_issue_from_json(d) for d in data]


def _extract_created_id(stdout: str) -> str | None:
    """bd's `create` prints a human-readable confirmation like
    `✓ Created issue: harness-xyz — <title>`. Pull the id out; tolerate
    whitespace and unicode variants."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if "Created issue:" not in line:
            continue
        _, _, rest = line.partition("Created issue:")
        rest = rest.strip()
        # rest looks like "harness-xyz — <title>" or "harness-xyz".
        token = rest.split()[0] if rest else ""
        if token:
            return token
    return None
