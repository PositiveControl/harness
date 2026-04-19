"""Focus + resume state for BeadsAdapter (harness-vhoj).

Owns the single-focus-per-assignee invariant: at most one in_progress
bead per assignee, reconciled lazily on every get_focus() read.
`persist_to_focus` appends compaction breadcrumbs to the notes field
so context reloads land on a fresh anchor.
"""

from __future__ import annotations

import warnings

from harness.store._bd_crud import BeadsCrudMixin
from harness.store._bd_runner import BeadsAdapterError
from harness.store._bd_types import BeadsIssue


class BeadsFocusMixin(BeadsCrudMixin):
    """Focus methods. Inherits from BeadsCrudMixin so it can reach
    `list_issues` / `update`; the MRO puts all mixins over
    BeadsRunner once by linearization."""

    def get_focus(self, assignee: str) -> BeadsIssue | None:
        """Return the single in_progress bead for `assignee`, or None.

        Reconciles on read: if bd's state somehow contains more than one
        in_progress bead for the same assignee (crash mid-set_focus,
        concurrent writer, manual edit) the method keeps the
        most-recently-updated and demotes the rest to `open`, emitting a
        RuntimeWarning. Lazy reconciliation replaces an explicit startup
        hook — any caller that reads focus gets a consistent answer."""
        issues = self.list_issues(status="in_progress", assignee=assignee)
        if not issues:
            return None
        if len(issues) == 1:
            return issues[0]
        issues_sorted = sorted(
            issues,
            key=lambda i: str(i.raw.get("updated_at") or ""),
            reverse=True,
        )
        keep, *to_demote = issues_sorted
        demoted_ids = [i.id for i in to_demote]
        warnings.warn(
            f"{len(issues)} in_progress beads for assignee={assignee!r}; "
            f"keeping most-recent {keep.id}, demoting {demoted_ids}.",
            RuntimeWarning,
            stacklevel=2,
        )
        for issue in to_demote:
            self.update(issue.id, status="open")
        return keep

    def persist_to_focus(self, summary: str, *, assignee: str) -> str:
        """Append `summary` to the current focus bead's notes field.
        Returns the focus bead id. Raises BeadsAdapterError when no
        focus is set — persist_to_focus is opt-in and load-bearing, so
        a silent no-op would mask a missing focus.

        Notes accumulate with newline separators (bd's --append-notes
        semantics), so repeated calls stack chronologically rather
        than overwriting — useful for compaction-summary breadcrumbs."""
        if not summary.strip():
            raise BeadsAdapterError("persist_to_focus: summary must be non-empty")
        focus = self.get_focus(assignee)
        if focus is None:
            raise BeadsAdapterError(
                f"persist_to_focus: no in_progress focus bead for assignee={assignee!r}. "
                "Call set_focus first."
            )
        self.update(focus.id, append_notes=summary)
        return focus.id

    def set_focus(self, issue_id: str, *, assignee: str) -> str | None:
        """Promote `issue_id` to in_progress for `assignee`. If another
        bead is currently in_progress for the same assignee, demote it
        to open first. Returns the demoted prior focus's id, or None if
        no demotion happened (no prior, or issue_id was already focus).

        Not atomic across bd subprocess calls — if demote succeeds but
        promote fails, no bead is in_progress for this assignee (clean,
        recoverable state; retry set_focus). The invariant 'at most one
        in_progress per assignee' is preserved at every observable
        boundary."""
        prior = self.get_focus(assignee)
        if prior is None:
            self.update(issue_id, status="in_progress")
            return None
        if prior.id == issue_id:
            return None
        self.update(prior.id, status="open")
        self.update(issue_id, status="in_progress")
        return prior.id
