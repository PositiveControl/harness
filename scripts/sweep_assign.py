"""Bulk-assign unassigned bd issues to a target user (dry-run by default)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

from harness.config import settings
from harness.store.bd_adapter import BeadsAdapter, BeadsIssue


class _Adapter(Protocol):
    def list_issues(self, *, status: str | None = None) -> list[BeadsIssue]: ...

    def update(self, issue_id: str, **fields: str) -> None: ...


@dataclass(frozen=True)
class SweepPlan:
    """Result of classifying one bead for the sweep."""

    issue_id: str
    current: str | None
    action: str  # "assign" | "skip"


def classify(issues: list[BeadsIssue], target: str) -> list[SweepPlan]:
    """Return per-issue plans. Unassigned -> assign; anything else ->
    skip, regardless of whether it already matches target. The sweep is
    meant to fill holes, not re-home already-assigned work."""
    plans: list[SweepPlan] = []
    for issue in issues:
        if issue.assignee is None or not str(issue.assignee).strip():
            plans.append(SweepPlan(issue_id=issue.id, current=None, action="assign"))
        else:
            plans.append(SweepPlan(issue_id=issue.id, current=issue.assignee, action="skip"))
    return plans


def run_sweep(
    adapter: _Adapter,
    *,
    target: str,
    apply: bool,
    status: str | None = "open",
    out: IO[str] | None = None,
) -> list[SweepPlan]:
    """Core sweep: list issues, classify, print audit lines, optionally
    apply updates. Returns the plans so callers (tests) can assert on
    them without re-parsing stdout. `out=None` late-binds to sys.stdout
    so capsys in tests actually sees the audit lines."""
    issues = adapter.list_issues(status=status)
    plans = classify(issues, target)
    sink = out if out is not None else sys.stdout

    def _say(line: str) -> None:
        print(line, file=sink)

    mode = "apply" if apply else "dry-run"
    _say(f"sweep-assign: mode={mode} target={target} status={status or 'all'}")
    _say(f"sweep-assign: scanned {len(plans)} bead(s)")

    assign_count = 0
    skip_count = 0
    for plan in plans:
        if plan.action == "assign":
            assign_count += 1
            if apply:
                adapter.update(plan.issue_id, assignee=target)
                _say(f"assign: {plan.issue_id} -> {target}")
            else:
                _say(f"would-assign: {plan.issue_id} -> {target}")
        else:
            skip_count += 1
            _say(f"skip: {plan.issue_id} already assigned to {plan.current}")

    _say(f"sweep-assign: done ({assign_count} assign, {skip_count} skip)")
    return plans


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-assign unassigned bd issues to a target user. "
            "Dry-run by default; pass --apply to actually write."
        ),
    )
    parser.add_argument(
        "--assignee",
        default="mark",
        help="Target user id to assign unclaimed beads to (default: mark).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually call `bd update` per bead. Without this flag, the "
        "script prints what it would do and exits.",
    )
    parser.add_argument(
        "--status",
        default="open",
        help="Filter issues by bd status (default: open). Pass '' to scan all.",
    )
    parser.add_argument(
        "--bd-dir",
        type=Path,
        default=None,
        help="Override bd_dir (defaults to Settings.bd_dir).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    bd_dir = args.bd_dir if args.bd_dir is not None else settings.ab_bd_dir_resolved
    adapter = BeadsAdapter(bd_dir=bd_dir)
    status: str | None = args.status if args.status else None
    run_sweep(adapter, target=args.assignee, apply=args.apply, status=status)


if __name__ == "__main__":
    main()
