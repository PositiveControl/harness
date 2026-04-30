#!/usr/bin/env python3
"""Merge a `bd export --all` JSONL snapshot into the local Dolt store.

Use case: cross-machine pickup. One machine commits ``beads-snapshot.jsonl``
(produced via ``bd export --all -o beads-snapshot.jsonl``); another machine
runs this script to fold the snapshot's beads into local Dolt without losing
any local-only beads.

The two databases can diverge in three ways:
  * Snapshot-only beads (need to be created locally, IDs preserved)
  * Local-only beads (must NOT be touched)
  * Status drift on overlapping beads (snapshot wins by default — it's the
    pickup target, the newer write)

This is *not* what ``bd backup restore`` does (which is full-replace) and not
what ``bd dolt pull`` does (which needs a configured Dolt remote). It's a
mechanical merge that preserves bead IDs, dependencies, labels, notes, and
acceptance criteria.

Usage:
    python scripts/bd_merge_snapshot.py beads-snapshot.jsonl
    python scripts/bd_merge_snapshot.py beads-snapshot.jsonl --dry-run
    python scripts/bd_merge_snapshot.py beads-snapshot.jsonl --no-drift

Always export local state before running:
    bd export --all -o /tmp/local-pre-merge-$(date +%s).jsonl
    cp -a .beads/dolt /tmp/dolt-snapshot-$(date +%s)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BeadRecord = dict[str, Any]


def load_jsonl(path: Path) -> dict[str, BeadRecord]:
    out: dict[str, BeadRecord] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def export_local() -> dict[str, BeadRecord]:
    """Run ``bd export --all`` to a temp file, then load it."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(  # noqa: S603  # bd is on PATH by convention
            ["bd", "export", "--all", "-o", str(tmp_path)],  # noqa: S607
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"bd export failed: {proc.stderr}")
        return load_jsonl(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def run_bd(args: list[str], stdin: str | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(  # noqa: S603  # bd is on PATH by convention
        ["bd", *args],  # noqa: S607
        input=stdin,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


class Merger:
    def __init__(self, log_path: Path, dry_run: bool = False) -> None:
        self.log_path = log_path
        self.dry_run = dry_run
        self.log_path.write_text("")

    def log(self, line: str) -> None:
        print(line)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def _run(self, args: list[str], stdin: str | None = None) -> tuple[int, str, str]:
        if self.dry_run:
            self.log(f"  DRY-RUN bd {' '.join(args)}")
            return 0, "", ""
        return run_bd(args, stdin=stdin)

    def create_bead(self, rec: BeadRecord) -> bool:
        bid = rec["id"]
        args = [
            "create",
            "--id",
            bid,
            "--title",
            rec.get("title") or "",
            "--type",
            rec["issue_type"],
            "--priority",
            str(rec["priority"]),
        ]
        if rec.get("assignee"):
            args.extend(["--assignee", rec["assignee"]])
        if rec.get("acceptance_criteria"):
            args.extend(["--acceptance", rec["acceptance_criteria"]])
        if rec.get("notes"):
            args.extend(["--notes", rec["notes"]])
        # Dotted-id children: `bd create --id` rejects `--parent` (mutually
        # exclusive). The parent-child relationship is materialised in the
        # dependency phase via `bd dep add ... --type parent-child` from the
        # snapshot's `dependencies[]` array.

        desc = rec.get("description") or ""
        if desc:
            args.extend(["--body-file", "-"])
            rc, _, err = self._run(args, stdin=desc)
        else:
            rc, _, err = self._run(args)

        if rc != 0:
            self.log(f"  ERR create {bid}: rc={rc} stderr={err.strip()[:300]}")
            return False
        self.log(f"  ok create {bid} ({rec['issue_type']}, P{rec['priority']}, {rec['status']})")
        return True

    def close_bead(self, bid: str, reason: str) -> None:
        args = ["close", bid]
        if reason and reason != "Closed":
            args.extend(["--reason", reason])
        rc, _, err = self._run(args)
        if rc != 0:
            self.log(f"  ERR close {bid}: rc={rc} stderr={err.strip()[:200]}")
        else:
            self.log(f"  ok close {bid}")

    def set_in_progress(self, bid: str) -> None:
        rc, _, err = self._run(["update", bid, "--status", "in_progress"])
        if rc != 0:
            self.log(f"  ERR in_progress {bid}: rc={rc} stderr={err.strip()[:200]}")
        else:
            self.log(f"  ok in_progress {bid}")

    def reopen_bead(self, bid: str) -> None:
        rc, _, err = self._run(["reopen", bid])
        if rc != 0:
            self.log(f"  ERR reopen {bid}: stderr={err.strip()[:200]}")
        else:
            self.log(f"  ok reopen {bid}")

    def add_labels(self, bid: str, labels: list[str]) -> None:
        for label in labels:
            rc, _, err = self._run(["label", "add", bid, label])
            if rc != 0:
                self.log(f"  WARN label add {bid} {label}: {err.strip()[:120]}")

    def add_dep(self, issue: str, depends_on: str, dep_type: str = "blocks") -> None:
        args = ["dep", "add", issue, depends_on]
        if dep_type and dep_type != "blocks":
            args.extend(["--type", dep_type])
        rc, out, err = self._run(args)
        if rc != 0:
            msg = (err or out).strip()
            if "already exists" in msg.lower() or "duplicate" in msg.lower():
                return
            self.log(f"  WARN dep {issue} -> {depends_on}: {msg[:200]}")


def merge_snapshot_into_local(
    snapshot_path: Path,
    *,
    log_path: Path,
    dry_run: bool = False,
    apply_drift: bool = True,
) -> int:
    snapshot = load_jsonl(snapshot_path)
    local = export_local()

    merger = Merger(log_path, dry_run=dry_run)
    merger.log(f"=== merge start (snapshot={snapshot_path.name}) ===")
    merger.log(f"snapshot: {len(snapshot)} issues; local: {len(local)} issues")

    snapshot_only = [bid for bid in snapshot if bid not in local]
    merger.log(f"snapshot-only to import: {len(snapshot_only)}")

    # Sort so parents come before dotted children. Epics first as a
    # convention — they often anchor downstream deps even when --parent
    # isn't used.
    def sort_key(bid: str) -> tuple[int, str]:
        rec = snapshot[bid]
        is_epic = rec.get("issue_type") == "epic"
        is_child = "." in bid
        if is_epic:
            return (0, bid)
        if is_child:
            return (2, bid)
        return (1, bid)

    snapshot_only.sort(key=sort_key)

    merger.log("\n--- phase 1: create beads ---")
    created: list[str] = []
    failed: list[str] = []
    for bid in snapshot_only:
        if merger.create_bead(snapshot[bid]):
            created.append(bid)
        else:
            failed.append(bid)
    merger.log(f"created: {len(created)} / {len(snapshot_only)}; failed: {len(failed)}")

    merger.log("\n--- phase 2: close / in_progress ---")
    for bid in created:
        rec = snapshot[bid]
        status = rec["status"]
        if status == "closed":
            merger.close_bead(bid, rec.get("close_reason") or "")
        elif status == "in_progress":
            merger.set_in_progress(bid)

    merger.log("\n--- phase 3: labels ---")
    for bid in created:
        labels = snapshot[bid].get("labels") or []
        if labels:
            merger.add_labels(bid, labels)

    merger.log("\n--- phase 4: dependencies ---")
    dep_count = 0
    for bid in created:
        for dep in snapshot[bid].get("dependencies") or []:
            merger.add_dep(dep["issue_id"], dep["depends_on_id"], dep.get("type", "blocks"))
            dep_count += 1
    merger.log(f"deps replayed: {dep_count}")

    if apply_drift:
        merger.log("\n--- phase 5: status drifts (snapshot wins) ---")
        drifts: list[str] = [
            bid
            for bid in snapshot
            if bid in local and snapshot[bid]["status"] != local[bid]["status"]
        ]
        for bid in drifts:
            snap_status = snapshot[bid]["status"]
            local_status = local[bid]["status"]
            merger.log(f"drift {bid}: local={local_status} → snapshot={snap_status}")
            if snap_status == "closed":
                merger.close_bead(bid, snapshot[bid].get("close_reason") or "")
            elif snap_status == "open" and local_status == "closed":
                merger.reopen_bead(bid)
            elif snap_status == "in_progress":
                merger.set_in_progress(bid)
        merger.log(f"drifts resolved: {len(drifts)}")
    else:
        merger.log("\n--- phase 5 skipped (--no-drift) ---")

    merger.log("\n=== merge done ===")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("snapshot", type=Path, help="Path to bd export --all JSONL")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/tmp/bd-merge-log.txt"),  # noqa: S108  # CLI default; users can override
        help="Where to write the merge log (default: /tmp/bd-merge-log.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every bd command without executing",
    )
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="Skip the status-drift phase (overlapping beads with mismatched status)",
    )
    args = parser.parse_args()

    if not args.snapshot.is_file():
        parser.error(f"snapshot not found: {args.snapshot}")

    return merge_snapshot_into_local(
        args.snapshot,
        log_path=args.log,
        dry_run=args.dry_run,
        apply_drift=not args.no_drift,
    )


if __name__ == "__main__":
    sys.exit(main())
