"""Mutation tools for ab_ops — capture / close / defer / reopen /
delete / update (harness-u73g).
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.store.bd_adapter import (
    ALLOWED_SCOPES,
    ALLOWED_TYPES,
    BeadsAdapterError,
    BeadsIssue,
)
from harness.tools.ab_ops._shared import (
    _DEFER_COUNT_PREFIX,
    _SCOPE_PARAM_DESCRIPTION,
    _UPDATE_FIELD_FLAGS,
    AB_ASSIGNEE,
    STALL_DEFERS,
    STALL_LABEL,
    _Adapter,
)
from harness.tools.base import ToolSpec


@dataclass
class CaptureTool:
    """Stateless validator + creator. When any required field is
    missing the tool returns the missing-field hint; otherwise it
    dispatches `bd create` with the scope label applied. The chat loop
    drives the until-resolved dialogue by calling this tool
    progressively as the user fills in fields."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="capture",
            description=(
                "Create an item with a mandatory scope label. Returns "
                "the new issue id on success. If any required field is "
                "missing (scope, outcome, next_action), returns a "
                "missing-field hint so the caller can ask the user. "
                "Supports project / event / habit / task types.\n\n"
                "WHEN TO CALL: the user has explicitly committed to a "
                "NEW item and wants it tracked. Trigger phrases: "
                "'add', 'track', 'remind me', 'capture', 'plan for', "
                "'new task', or a direct commitment like 'I'll do X "
                "by Friday'. One capture per commitment.\n\n"
                "DO NOT CALL for read-only or planning asks — these "
                "belong to other tools or no tool at all:\n"
                "  • 'summarize / list / show my tasks' → `plan` (or "
                "no tool if the answer fits from memory)\n"
                "  • 'what's blocked / what's drifting' → `drift`\n"
                "  • 'what is X / status of X' → `status`\n"
                "  • 'suggest products / how do I / explain' → no "
                "tool; answer directly\n"
                "  • questions, recaps, clarifications, follow-ups, "
                "and 'do you have any ideas' → no tool\n"
                "If uncertain whether the user is committing to a new "
                "item, ask one clarifying question instead of capturing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "raw": {
                        "type": "string",
                        "description": "The user's original phrasing — recorded as context.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": list(ALLOWED_SCOPES),
                        "description": _SCOPE_PARAM_DESCRIPTION,
                    },
                    "outcome": {
                        "type": "string",
                        "description": "One-sentence 'done when' definition.",
                    },
                    "next_action": {
                        "type": "string",
                        "description": "A verb phrase naming the next concrete step.",
                    },
                    "issue_type": {
                        "type": "string",
                        "enum": list(ALLOWED_TYPES),
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                        "description": "bd priority (0=critical … 4=backlog). Default 2.",
                    },
                    "parent": {
                        "type": "string",
                        "description": "Existing project id to attach under. Omit for loose.",
                    },
                    "deadline": {
                        "type": "string",
                        "description": "Optional free-form deadline note; stored in description.",
                    },
                    "estimate": {
                        "type": "string",
                        "enum": ["S", "M", "L"],
                    },
                    "ab_owned": {
                        "type": "boolean",
                        "description": (
                            "True when the bead is ab-internal "
                            "thought-graph work (hypothesis, question, "
                            "plan-step). Counts against the per-turn "
                            "budget (3 max). False (default) for "
                            "user-initiated captures."
                        ),
                    },
                },
                "required": ["raw"],
            },
            tier="write",
            display_name="Capture item",
        )

    def call(
        self,
        *,
        raw: str,
        scope: str | None = None,
        outcome: str | None = None,
        next_action: str | None = None,
        issue_type: str = "task",
        priority: int = 2,
        parent: str | None = None,
        deadline: str | None = None,
        estimate: str | None = None,
        ab_owned: bool = False,
    ) -> str:
        missing = [
            f
            for f, v in {
                "scope": scope,
                "outcome": outcome,
                "next_action": next_action,
            }.items()
            if not v
        ]
        if missing:
            return (
                "missing field(s): "
                + ", ".join(missing)
                + f". Ask the user for {missing[0]} first; one question per turn."
            )
        description_lines = [f"raw: {raw}", f"outcome: {outcome}", f"next: {next_action}"]
        if deadline:
            description_lines.append(f"deadline: {deadline}")
        if estimate:
            description_lines.append(f"estimate: {estimate}")
        extra_labels: list[str] = []
        if estimate:
            extra_labels.append(f"est:{estimate}")
        try:
            new_id = self.adapter.create(
                title=raw if len(raw) < 80 else raw[:77] + "…",
                scope=scope,  # type: ignore[arg-type]  # checked above
                issue_type=issue_type,
                description="\n".join(description_lines),
                priority=priority,
                parent=parent,
                extra_labels=extra_labels,
                assignee=AB_ASSIGNEE if ab_owned else None,
            )
        except (ValueError, BeadsAdapterError) as exc:
            return f"capture failed: {exc}"
        deadline_frag = f", deadline {deadline}" if deadline else ""
        parent_frag = f" under {parent}" if parent else " (loose)"
        return (
            f"Captured {new_id} — [{scope}] {issue_type}{parent_frag}. "
            f"Next: {next_action}{deadline_frag}."
        )


@dataclass
class CloseTool:
    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="close",
            description=(
                "Close an item. Records an optional reason on the bead.\n\n"
                "REASON POLICY: Pass `reason` only if the user stated one "
                "explicitly in this turn (e.g. 'close harness-x, it shipped'). "
                "Do NOT synthesize a reason from prior context, the issue "
                "title, or your own inference. If a reason seems worth "
                "recording but none was given, reply WITHOUT calling `close` "
                "and ask the user for one — one question, no batching."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id"],
            },
            tier="write",
            display_name="Close item",
        )

    def call(self, *, id: str, reason: str | None = None) -> str:
        try:
            self.adapter.close(id, reason=reason)
        except BeadsAdapterError as exc:
            return f"close failed: {exc}"
        return f"Closed {id}."


@dataclass
class DeferTool:
    """Lower an item's priority — 'defer' in ab's ops vocabulary, bd
    priority-bump under the hood. Records a one-line reason so the
    drift tool can surface stalled deferrals later."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="defer",
            description=(
                "Defer an item by lowering its bd priority. Accepts a "
                "numeric priority (0-4) or the relative string 'down' "
                "(bump priority by 1). Reason is recorded on the bead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "description": (
                            "Target priority, either 0-4 or the literal "
                            "'down' to shift one step toward backlog."
                        ),
                    },
                    "reason": {"type": "string"},
                },
                "required": ["id", "priority"],
            },
            tier="write",
            display_name="Defer item",
        )

    def call(self, *, id: str, priority: str, reason: str | None = None) -> str:
        if priority != "down":
            try:
                new_priority = int(priority)
            except ValueError:
                return f"defer failed: priority must be 0-4 or 'down', got {priority!r}"
            if not 0 <= new_priority <= 4:
                return f"defer failed: priority {new_priority} must be 0-4"
        try:
            current = self.adapter.show(id)
            if priority == "down":
                new_priority = min(current.priority + 1, 4)
            self.adapter.update(id, priority=str(new_priority))
            if reason:
                self.adapter.remember(f"deferred {id} → P{new_priority}: {reason}")
            new_count = _bump_defer_count(self.adapter, current)
        except BeadsAdapterError as exc:
            return f"defer failed: {exc}"

        escalation_note = ""
        if new_count >= STALL_DEFERS and STALL_LABEL not in current.labels:
            try:
                child_id = _create_stall_escalation(self.adapter, current, new_count)
                self.adapter.label_add(id, STALL_LABEL)
                escalation_note = f" Stall-escalated: {child_id} (still relevant?)."
            except BeadsAdapterError as exc:
                escalation_note = f" (stall-escalation failed: {exc})"

        return f"Deferred {id} to P{new_priority}.{escalation_note}"


def _extract_defer_count(labels: tuple[str, ...]) -> int:
    """Read the current defer count off the bead's labels. Absence
    means zero — the first-ever defer lands at count=1."""
    for label in labels:
        if label.startswith(_DEFER_COUNT_PREFIX):
            try:
                return int(label[len(_DEFER_COUNT_PREFIX) :])
            except ValueError:
                return 0
    return 0


def _bump_defer_count(adapter: _Adapter, issue: BeadsIssue) -> int:
    """Increment the defer counter stored as a `defer-count:N` label.
    Two subprocess calls (remove old + add new) when the bead already
    has a count; one add call on the first defer."""
    old_count = _extract_defer_count(issue.labels)
    new_count = old_count + 1
    if old_count > 0:
        adapter.label_rm(issue.id, f"{_DEFER_COUNT_PREFIX}{old_count}")
    adapter.label_add(issue.id, f"{_DEFER_COUNT_PREFIX}{new_count}")
    return new_count


def _create_stall_escalation(
    adapter: _Adapter,
    parent: BeadsIssue,
    defer_count: int,
) -> str:
    """Create a `thought:question` child bead asking whether the
    stalled parent is still relevant. Parent-linked via bd's --parent."""
    parent_scope = parent.scope or "personal"
    return adapter.create(
        title=f"Still relevant? {parent.id}",
        scope=parent_scope,
        issue_type="task",
        description=(
            f"Parent {parent.id} deferred {defer_count} times — auto-escalated "
            "by DeferTool. Resolve by either closing parent (no longer "
            "relevant), promoting it back to active priority, or closing "
            "this question after a reason is captured."
        ),
        priority=2,
        parent=parent.id,
        extra_labels=["thought:question"],
        assignee=AB_ASSIGNEE,
    )


@dataclass
class ReopenTool:
    """Reopen a previously closed item. Thin wrapper over `bd reopen`;
    `reason` is recorded in bd's audit log so drift/retro can surface
    the rationale later."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="reopen",
            description=(
                "Reopen a closed item. Sets status back to open and "
                "emits a Reopened event. Accepts an optional reason."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id"],
            },
            tier="write",
            display_name="Reopen item",
        )

    def call(self, *, id: str, reason: str | None = None) -> str:
        try:
            self.adapter.reopen(id, reason=reason)
        except BeadsAdapterError as exc:
            return f"reopen failed: {exc}"
        return f"Reopened {id}."


@dataclass
class DeleteTool:
    """Permanently delete an item. Destructive: removes the bead and
    its dependency links from the database. Orphans dependents by
    default (bd rewrites their references to `[deleted:ID]`); set
    `cascade=true` to recursively delete every dependent.

    Tier is write — the orchestrator gates the first call per session
    on an explicit user confirmation, so bd's own --force is always
    passed through (double-confirm would be noise)."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="delete",
            description=(
                "Permanently delete an item. Orphans dependents by "
                "default; pass cascade=true to recursively delete "
                "every dependent. Destructive and irreversible — use "
                "`close` instead when the work simply finished."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "cascade": {
                        "type": "boolean",
                        "description": ("Recursively delete every dependent. Default false."),
                    },
                },
                "required": ["id"],
            },
            tier="write",
            display_name="Delete item",
        )

    def call(self, *, id: str, cascade: bool = False) -> str:
        try:
            self.adapter.delete(id, cascade=cascade)
        except BeadsAdapterError as exc:
            return f"delete failed: {exc}"
        suffix = " (cascade)" if cascade else ""
        return f"Deleted {id}{suffix}."


@dataclass
class UpdateTool:
    """Generic bd update over common fields: title, description, notes,
    assignee, priority, status. DeferTool stays as sugar for the
    priority-down flow; this tool covers everything else and also
    accepts `priority` directly when the caller knows the target."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="update",
            description=(
                "Update one or more fields on an item: title, "
                "description, notes, assignee, priority (0-4), or "
                "status. At least one field is required. For "
                "priority-down-one-step, prefer `defer`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "notes": {"type": "string"},
                    "assignee": {"type": "string"},
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                    },
                    "status": {"type": "string"},
                },
                "required": ["id"],
            },
            tier="write",
            display_name="Update item",
        )

    def call(
        self,
        *,
        id: str,
        title: str | None = None,
        description: str | None = None,
        notes: str | None = None,
        assignee: str | None = None,
        priority: int | None = None,
        status: str | None = None,
    ) -> str:
        fields: dict[str, str] = {}
        if title is not None:
            fields["title"] = title
        if description is not None:
            fields["description"] = description
        if notes is not None:
            fields["notes"] = notes
        if assignee is not None:
            fields["assignee"] = assignee
        if priority is not None:
            if not 0 <= priority <= 4:
                return f"update failed: priority {priority} must be 0-4"
            fields["priority"] = str(priority)
        if status is not None:
            fields["status"] = status
        if not fields:
            return (
                "update failed: no fields provided. Pass at least one of "
                + ", ".join(sorted(_UPDATE_FIELD_FLAGS))
                + "."
            )
        try:
            self.adapter.update(id, **fields)
        except BeadsAdapterError as exc:
            return f"update failed: {exc}"
        changed = ", ".join(sorted(fields))
        return f"Updated {id}: {changed}."
