"""Memory tools for ab_ops — retro / memories / remember / forget
(harness-u73g).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from harness.store.bd_adapter import BeadsAdapterError
from harness.tools.ab_ops._shared import _Adapter
from harness.tools.base import ToolSpec


@dataclass
class RetroTool:
    """End-of-day retrospective. Summarizes today's Shall/Should/
    closed items, optionally persists user-provided insights as bd
    memories so next /plan consults them."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="retro",
            description=(
                "Run an end-of-day retrospective. Two modes: "
                "'summary' reads current state and returns a structured "
                "recap for the user to fill in; 'record' takes an "
                "insight string and persists it via bd remember."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["summary", "record"],
                        "description": "summary (read) or record (write).",
                    },
                    "insight": {
                        "type": "string",
                        "description": "Required when mode='record'.",
                    },
                },
                "required": ["mode"],
            },
            tier="write",  # record writes; summary reads. Mark as write
            # for orchestrator confirmation on opt-in.
            display_name="Retro",
        )

    def call(self, *, mode: str, insight: str | None = None) -> str:
        if mode == "summary":
            try:
                ready = self.adapter.ready()
                all_issues = self.adapter.list_issues()
            except BeadsAdapterError as exc:
                return f"retro failed: {exc}"
            today_open = [i for i in ready if i.priority <= 2]
            lines = [f"Retro — {date.today().isoformat()}"]
            lines.append(f"Open Shall/Should today: {len(today_open)}")
            for issue in today_open[:10]:
                scope_tag = issue.scope or "?"
                lines.append(f"  - [{scope_tag}/{issue.id}] {issue.title}")
            closed_recently = [i for i in all_issues if i.status == "closed"]
            lines.append(f"Total closed in history: {len(closed_recently)}")
            lines.append(
                "Ready to record insights? Call `retro` again with "
                "mode='record' and insight='<text>'."
            )
            return "\n".join(lines)
        if mode == "record":
            if not insight:
                return "retro failed: mode='record' requires insight text"
            try:
                self.adapter.remember(f"retro: {insight}")
            except BeadsAdapterError as exc:
                return f"retro failed: {exc}"
            return f"Recorded retro insight: {insight}"
        return f"retro failed: unknown mode {mode!r}; use 'summary' or 'record'"


@dataclass
class MemoriesTool:
    """Read-tier view of bd's persistent memories — free-text insights
    the agent or user stored via `retro record` or `bd remember`."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memories",
            description=(
                "List persistent memories, or search them by keyword. "
                "Returns bd's raw output verbatim. Read-only; use "
                "`retro` with mode='record' to add, `forget` to remove."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional keyword filter.",
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Memories",
        )

    def call(self, *, query: str = "") -> str:
        try:
            output = self.adapter.memories(query)
        except BeadsAdapterError as exc:
            return f"memories failed: {exc}"
        text = output.strip()
        return text if text else "(no memories)"


@dataclass
class ForgetTool:
    """Remove a persistent memory by key. Destructive — write-tier."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="forget",
            description=(
                "Remove a persistent memory by key. Irreversible — use "
                "`memories` first to confirm the key exists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
            tier="write",
            display_name="Forget memory",
        )

    def call(self, *, key: str) -> str:
        try:
            self.adapter.forget(key)
        except (ValueError, BeadsAdapterError) as exc:
            return f"forget failed: {exc}"
        return f"Forgot memory {key!r}."


@dataclass
class RememberTool:
    """Persist a free-form insight via `bd remember`. This is the right
    target for 'note to self' / 'remember that' / 'write it down'
    phrasings — use `comments` only when attaching text to a specific
    existing issue, and `retro` mode=record only at end-of-day recap
    time."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="remember",
            description=(
                "Persist a note to self as a durable memory via bd "
                "remember. Accepts one free-form string. Call this "
                "BEFORE replying whenever the user states a durable "
                "preference or rule — phrases like 'from now on', "
                "'always', 'remember to', 'keep in mind', 'note to "
                "self', 'remember that', 'write it down'. Capture the "
                "rule verbatim so later sessions can apply it. Do NOT "
                "use `comments` for notes to self — comments attach to "
                "a specific existing issue id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "insight": {
                        "type": "string",
                        "description": "The note to persist. Stored verbatim.",
                    },
                },
                "required": ["insight"],
            },
            tier="write",
            display_name="Remember",
        )

    def call(self, *, insight: str) -> str:
        if not insight.strip():
            return "remember failed: insight must be non-empty"
        try:
            self.adapter.remember(insight)
        except BeadsAdapterError as exc:
            return f"remember failed: {exc}"
        return f"Remembered: {insight}"
