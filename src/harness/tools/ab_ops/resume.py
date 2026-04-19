"""Session-resume helpers + PersistFocusNoteTool (harness-u73g).

`build_resume_summary` is the chat-startup + post-compaction banner
that surfaces thought-graph state so ab + user resume from the bead
graph rather than reconstructing from a cold conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from harness.store.bd_adapter import BeadsAdapterError, BeadsIssue
from harness.tools.ab_ops._shared import (
    AB_ASSIGNEE,
    AB_DRIFT_DAYS,
    _Adapter,
    _parse_iso_utc,
)
from harness.tools.base import ToolSpec


def build_resume_summary(
    adapter: _Adapter,
    *,
    memory_limit: int = 5,
    ab_drift_days: int = AB_DRIFT_DAYS,
) -> str:
    """Render a multi-line session-resume summary: focus bead, active
    ab-owned in-progress beads, recent bd memories, drifting items.
    Shown at chat start + after a compaction so the user (and ab
    itself) have the thought-graph state visible without guessing.

    Degrades gracefully on per-step adapter errors — a failing drift
    query shouldn't blank the whole summary."""
    lines = ["── session resume ──"]

    focus = _safe_get_focus(adapter)
    lines.append("Focus: " + _render_focus_line(focus))

    in_progress = _safe_in_progress(adapter)
    focus_id = focus.id if focus else None
    others = [i for i in in_progress if i.id != focus_id]
    if others:
        lines.append("In-progress ab-beads:")
        for issue in others:
            lines.append(f"  - {issue.id}: {issue.title}")
    elif not in_progress:
        lines.append("In-progress ab-beads: (none)")

    memories = _safe_memories(adapter, memory_limit)
    if memories:
        lines.append(f"Recent memories:\n{memories}")
    else:
        lines.append("Recent memories: (none)")

    drift = _safe_drift(adapter, ab_drift_days)
    if drift:
        drift_ids = ", ".join(i.id for i in drift[:5])
        lines.append(f"Drifting ab-beads: {drift_ids}")

    return "\n".join(lines)


def _safe_get_focus(adapter: _Adapter) -> BeadsIssue | None:
    try:
        return adapter.get_focus(AB_ASSIGNEE)
    except BeadsAdapterError:
        return None


def _safe_in_progress(adapter: _Adapter) -> list[BeadsIssue]:
    try:
        return adapter.list_issues(status="in_progress", assignee=AB_ASSIGNEE)
    except BeadsAdapterError:
        return []


def _safe_memories(adapter: _Adapter, limit: int) -> str:
    try:
        raw = adapter.memories("")
    except BeadsAdapterError:
        return ""
    lines = [line for line in raw.splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def _safe_drift(adapter: _Adapter, ab_drift_days: int) -> list[BeadsIssue]:
    try:
        candidates = adapter.list_issues(status="open", assignee=AB_ASSIGNEE)
    except BeadsAdapterError:
        return []
    cutoff = datetime.now(UTC) - timedelta(days=ab_drift_days)
    out: list[BeadsIssue] = []
    for issue in candidates:
        raw_ts = issue.raw.get("updated_at")
        if not raw_ts:
            continue
        parsed = _parse_iso_utc(str(raw_ts))
        if parsed is None:
            continue
        if parsed < cutoff:
            out.append(issue)
    return out


def _render_focus_line(issue: BeadsIssue | None) -> str:
    if issue is None:
        return "(none)"
    scope_tag = issue.scope or "?"
    return f"[{scope_tag}/{issue.id}] {issue.title} (P{issue.priority})"


@dataclass
class PersistFocusNoteTool:
    """Append a single-line summary to the current focus bead's notes.
    Opt-in breadcrumb: ab calls this explicitly (typically right after
    a compaction pass) when an observation is load-bearing enough to
    outlive the conversation window. No auto-persist — the caller is
    responsible for deciding what's worth writing."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="persist_focus_note",
            description=(
                "Append a summary string to the current focus bead's "
                "notes field. Use after compaction or when capturing "
                "a load-bearing observation that should survive the "
                "conversation window. Requires an active focus bead "
                "(capture + set-focus flow); fails loudly when none "
                "is set. Multiple calls stack with newline separators."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "One-line observation to append. Keep "
                            "terse — notes read back as prose later."
                        ),
                    },
                },
                "required": ["summary"],
            },
            tier="write",
            display_name="Persist focus note",
        )

    def call(self, *, summary: str) -> str:
        if not summary.strip():
            return "persist_focus_note failed: summary must be non-empty"
        try:
            focus_id = self.adapter.persist_to_focus(summary, assignee=AB_ASSIGNEE)
        except BeadsAdapterError as exc:
            return f"persist_focus_note failed: {exc}"
        return f"Appended note to focus bead {focus_id}."
