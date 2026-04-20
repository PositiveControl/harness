"""Core CRUD + list / search / ready / show / stale ops for BeadsAdapter.

Extracted from bd_adapter.py (harness-vhoj). Mixin over BeadsRunner —
methods use `self._run`, `self._ab_assignee`, `self._turn_cap`, etc.,
provided by the runner base.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from harness.store._bd_runner import (
    BeadsAdapterError,
    BeadsRunner,
    InflightCapExceededError,
    TurnCapExceededError,
)
from harness.store._bd_types import ALLOWED_SCOPES, ALLOWED_TYPES, BeadsIssue, _issue_from_json

if TYPE_CHECKING:
    pass


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
        token = rest.split()[0] if rest else ""
        if token:
            return token
    return None


class BeadsCrudMixin(BeadsRunner):
    """CRUD + list/search/ready/show/stale. Inherits state + `_run`
    from BeadsRunner."""

    def _check_inflight_cap(self) -> None:
        """Raise InflightCapExceededError if the count of open ab-owned
        beads is already at or above the configured cap. Runs before a
        new ab-owned create — so the cap is counted against the state
        at check time, not including the one about to be created.
        Candidate list sorted low-priority-first then oldest-updated-
        first so the model has concrete ids to close."""
        if self._ab_assignee is None:
            return
        open_ab = self.list_issues(status="open", assignee=self._ab_assignee)
        if len(open_ab) < self._inflight_cap:
            return
        candidates = sorted(
            open_ab,
            key=lambda i: (-i.priority, str(i.raw.get("updated_at") or "")),
        )[:5]
        candidate_fragment = ", ".join(f"{i.id} (P{i.priority})" for i in candidates)
        raise InflightCapExceededError(
            f"in-flight cap reached ({self._inflight_cap} open ab-owned beads). "
            "Close or defer one before capturing another. Candidates: "
            f"{candidate_fragment}."
        )

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
        assignee: str | None = None,
    ) -> str:
        """Create a bead with a mandatory scope label. Returns the new
        bead's id. Type defaults to `task` — override with `project`,
        `event`, `habit`, or `decision`. `assignee` marks provenance —
        pass `airton_b` for ab-captured beads, a user id for user-owned."""
        if scope not in ALLOWED_SCOPES:
            raise ValueError(f"scope must be one of {ALLOWED_SCOPES!r}, got {scope!r}")
        if issue_type not in ALLOWED_TYPES:
            raise ValueError(f"issue_type must be one of {ALLOWED_TYPES!r}, got {issue_type!r}")
        self.ensure_custom_types()
        labels = [f"scope:{scope}", *extra_labels]
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
            "--labels",
            ",".join(labels),
        ]
        if parent is not None:
            args.extend(["--parent", parent])
        if assignee is not None:
            args.extend(["--assignee", assignee])
        if self._ab_assignee is not None and assignee == self._ab_assignee:
            self._check_inflight_cap()
            if self._ab_creates_this_turn >= self._turn_cap:
                raise TurnCapExceededError(
                    f"turn-cap reached ({self._turn_cap} ab-owned beads this turn). "
                    "Close or defer an existing open ab-bead, or wait for the "
                    "next user turn to refresh the budget."
                )
        result = self._run(args)
        issue_id = _extract_created_id(result.stdout)
        if issue_id is None:
            raise BeadsAdapterError(f"could not parse created id from bd output: {result.stdout!r}")
        if self._ab_assignee is not None and assignee == self._ab_assignee:
            self._ab_creates_this_turn += 1
        for dep in deps:
            self._run(["dep", "add", issue_id, dep])
        return issue_id

    def close(self, issue_id: str, *, reason: str | None = None) -> None:
        args = ["close", issue_id]
        if reason is not None:
            args.extend(["--reason", reason])
        self._run(args)

    def reopen(self, issue_id: str, *, reason: str | None = None) -> None:
        """Wrap `bd reopen`. Re-opens a closed issue and clears its
        closed_at timestamp; emits a Reopened event in bd's audit log."""
        args = ["reopen", issue_id]
        if reason is not None:
            args.extend(["--reason", reason])
        self._run(args)

    def delete(
        self,
        issue_id: str,
        *,
        cascade: bool = False,
    ) -> None:
        """Wrap `bd delete`. Destructive — the bead and its reference
        links are removed from the database. Always passes --force
        because the agent-driven path already gates on a write-tier
        confirmation; requiring bd's own prompt would double-confirm.

        `cascade=True` recursively deletes every dependent. Default
        False so ab has to opt in explicitly; bd will orphan dependents
        (keep them, rewriting refs to `[deleted:ID]`) with --force
        alone."""
        args = ["delete", issue_id, "--force"]
        if cascade:
            args.append("--cascade")
        self._run(args)

    def update(self, issue_id: str, **fields: str) -> None:
        """Pass bd update fields as kwargs — e.g.
        `update("harness-x", priority="1", status="in_progress")`."""
        args = ["update", issue_id]
        for key, value in fields.items():
            flag = f"--{key.replace('_', '-')}"
            args.extend([flag, str(value)])
        self._run(args)

    def list_issues(
        self,
        *,
        scope: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        issue_type: str | None = None,
        limit: int | None = None,
        assignee: str | None = None,
    ) -> list[BeadsIssue]:
        """Wrap `bd list`. status/priority/type/assignee pass through to
        bd's native flags; scope is filtered client-side since bd
        doesn't recognize ab's scope: label as a first-class filter.

        `--flat` is required: bd 0.59 made `--tree` the default and it
        silently overrides `--json` so `bd list --json` alone returns
        the human tree view, which trips the JSON decoder in
        `_parse_issue_list` (harness-crh).

        `limit` handling (harness-z3f): when `scope` is set, `--limit`
        MUST NOT go to bd. bd applies the limit server-side before
        we see any rows, so any scope-matching items past the
        server-side cutoff get silently dropped and the caller sees
        an empty list even though qualifying rows exist. Defer
        slicing to client-side after the label filter in that case."""
        args = ["list", "--flat", "--json"]
        if status is not None:
            args.extend(["--status", status])
        if priority is not None:
            args.extend(["--priority", priority])
        if issue_type is not None:
            args.extend(["--type", issue_type])
        if assignee is not None:
            args.extend(["--assignee", assignee])
        # Only pass --limit to bd when no client-side filter (scope)
        # follows. Otherwise we'd prune rows the scope filter was
        # about to keep.
        defer_limit = limit is not None and scope is not None
        if limit is not None and not defer_limit:
            args.extend(["--limit", str(limit)])
        result = self._run(args)
        issues = _parse_issue_list(result.stdout)
        if scope is not None:
            label = f"scope:{scope}"
            issues = [i for i in issues if label in i.labels]
        if defer_limit:
            issues = issues[: limit or 0]
        return self._apply_default_exclude(issues, explicit_assignee=assignee)

    def search(
        self,
        query: str,
        *,
        status: str | None = None,
        limit: int | None = None,
        assignee: str | None = None,
    ) -> list[BeadsIssue]:
        """Wrap `bd search`. Defaults exclude closed issues; pass
        status='all' to include them. Empty query raises ValueError —
        bd rejects it and the error is clearer at the adapter."""
        if not query.strip():
            raise ValueError("search query must be non-empty")
        args = ["search", query, "--json"]
        if status is not None:
            args.extend(["--status", status])
        if assignee is not None:
            args.extend(["--assignee", assignee])
        if limit is not None:
            args.extend(["--limit", str(limit)])
        result = self._run(args)
        issues = _parse_issue_list(result.stdout)
        return self._apply_default_exclude(issues, explicit_assignee=assignee)

    def ready(
        self,
        *,
        scope: str | None = None,
        limit: int | None = None,
        assignee: str | None = None,
    ) -> list[BeadsIssue]:
        args = ["ready", "--json"]
        if assignee is not None:
            args.extend(["--assignee", assignee])
        if limit is not None:
            args.extend(["--limit", str(limit)])
        result = self._run(args)
        issues = _parse_issue_list(result.stdout)
        if scope is not None:
            label = f"scope:{scope}"
            issues = [i for i in issues if label in i.labels]
        return self._apply_default_exclude(issues, explicit_assignee=assignee)

    def show(self, issue_id: str) -> BeadsIssue:
        result = self._run(["show", issue_id, "--json"])
        issues = _parse_issue_list(result.stdout)
        if not issues:
            raise BeadsAdapterError(f"no issue returned for id {issue_id!r}")
        return issues[0]

    def stale(self) -> list[BeadsIssue]:
        result = self._run(["stale", "--json"])
        issues = _parse_issue_list(result.stdout)
        return self._apply_default_exclude(issues, explicit_assignee=None)

    def _apply_default_exclude(
        self,
        issues: list[BeadsIssue],
        *,
        explicit_assignee: str | None,
    ) -> list[BeadsIssue]:
        """Drop rows whose assignee matches the adapter's configured
        default exclude — but only if the caller didn't pass a positive
        `assignee=` filter. A positive filter is an opt-in and beats
        the default exclude."""
        if explicit_assignee is not None:
            return issues
        exclude = self._default_exclude_assignee
        if not exclude:
            return issues
        return [i for i in issues if i.assignee != exclude]


__all__ = [
    "BeadsCrudMixin",
    "_extract_created_id",
    "_parse_issue_list",
]


# keep unused re-export silent for mypy
_ = Any
