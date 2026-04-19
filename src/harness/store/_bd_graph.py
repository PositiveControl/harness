"""Dep, label, comment, find-duplicates ops for BeadsAdapter (harness-vhoj).

Anything that manipulates the link graph around beads (dependencies,
labels, comments) or surfaces candidate duplicates. Pure thin wrappers
over `bd dep` / `bd label` / `bd comments` / `bd find-duplicates`.
"""

from __future__ import annotations

import json
from typing import Any

from harness.store._bd_runner import BeadsRunner


class BeadsGraphMixin(BeadsRunner):
    def dep_add(self, issue: str, depends_on: str) -> None:
        self._run(["dep", "add", issue, depends_on])

    def dep_rm(self, issue: str, depends_on: str) -> None:
        """Remove a dependency. Uses `bd dep remove` (bd aliases to rm).
        Wraps the positional-arg form; bd also accepts `--blocks`, but
        the positional shape matches `dep_add` for symmetry."""
        self._run(["dep", "remove", issue, depends_on])

    def label_add(self, issue_id: str, label: str) -> None:
        """Wrap `bd label add <issue> <label>`. Multi-issue add and
        `--set-labels` replace-all are out of scope for the ab path;
        ab works one item at a time."""
        if not label.strip():
            raise ValueError("label must be non-empty")
        self._run(["label", "add", issue_id, label])

    def label_rm(self, issue_id: str, label: str) -> None:
        if not label.strip():
            raise ValueError("label must be non-empty")
        self._run(["label", "remove", issue_id, label])

    def label_list(self, issue_id: str) -> str:
        """Return bd's raw label-list output. JSON parsing is skipped;
        callers render verbatim. Empty stdout means the issue has no
        labels (or doesn't exist — bd surfaces its own error)."""
        result = self._run(["label", "list", issue_id])
        return result.stdout

    def comment_add(self, issue_id: str, text: str) -> None:
        if not text.strip():
            raise ValueError("comment text must be non-empty")
        self._run(["comments", "add", issue_id, text])

    def comments_list(self, issue_id: str) -> str:
        """Wrap `bd comments <issue>`. Returns bd's raw output; ab
        renders verbatim so bd's formatting decisions (timestamps,
        author line) stay authoritative."""
        result = self._run(["comments", issue_id])
        return result.stdout

    def find_duplicates(
        self,
        *,
        threshold: float | None = None,
        limit: int | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Wrap `bd find-duplicates --json` (mechanical method only —
        AI method would leak the query to a cloud endpoint without the
        user's explicit ask and isn't wired here). Returns the parsed
        list of candidate pairs; each element carries whatever keys bd
        emits (typically issue ids, titles, similarity)."""
        args = ["find-duplicates", "--json", "--method", "mechanical"]
        if threshold is not None:
            args.extend(["--threshold", str(threshold)])
        if limit is not None:
            args.extend(["--limit", str(limit)])
        if status is not None:
            args.extend(["--status", status])
        result = self._run(args)
        stdout = (result.stdout or "").strip()
        if not stdout:
            return []
        data = json.loads(stdout)
        if isinstance(data, dict):
            return [data]
        return list(data)
