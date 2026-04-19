"""ab's operations tool set — harness-inj.5.

Eight tools wrapping ab's slash commands. Every data-plane operation
dispatches through a shared BeadsAdapter; no tool here reads or writes
beads directly. Tier computation (Shall / Should / Shmaybe / Watching)
lives in-module and is pure: priority and dependent-count in, labelled
item + reason string out. Never stored.

Tools shipped:

| name           | tier  | purpose                                   |
|----------------|-------|-------------------------------------------|
| plan           | read  | today's tiered path with reasons          |
| capture        | write | until-resolved create with scope label    |
| status         | read  | bd show + transitive dep graph rollup     |
| drift          | read  | stale items with deadline-aware filtering |
| retro          | write | end-of-day summary; persists to memory    |
| reprioritize   | read  | recompute tiers, emit re-rank notice      |
| close          | write | thin wrapper over bd close                |
| defer          | write | thin wrapper over bd update --priority    |

Deadline-aware tier promotion (date-locked items → Shall at T-1) is a
follow-up; v1 classifies from priority + dependent-count only. Issues
can still carry deadlines as free-form text in their description.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, runtime_checkable

from harness.store.bd_adapter import (
    ALLOWED_SCOPES,
    ALLOWED_TYPES,
    BeadsAdapterError,
    BeadsIssue,
)
from harness.tools.base import ToolSpec


@runtime_checkable
class _Adapter(Protocol):
    """Narrow view of BeadsAdapter that ab's tools actually call. Kept
    behind a Protocol so FakeAdapters in tests don't have to inherit
    the concrete class — structural typing handles duck-shaped stubs.
    """

    def ready(self, *, scope: str | None = ..., limit: int | None = ...) -> list[BeadsIssue]: ...

    def list_issues(
        self,
        *,
        scope: str | None = ...,
        status: str | None = ...,
        limit: int | None = ...,
    ) -> list[BeadsIssue]: ...

    def show(self, issue_id: str) -> BeadsIssue: ...

    def stale(self) -> list[BeadsIssue]: ...

    def create(
        self,
        *,
        title: str,
        scope: str,
        issue_type: str = ...,
        description: str = ...,
        priority: int = ...,
        parent: str | None = ...,
        deps: Sequence[str] = ...,
        extra_labels: Sequence[str] = ...,
    ) -> str: ...

    def close(self, issue_id: str, *, reason: str | None = ...) -> None: ...

    def update(self, issue_id: str, **fields: Any) -> None: ...

    def remember(self, insight: str) -> None: ...


CAPTURE_REQUIRED_FIELDS = ("scope", "outcome", "next_action")

# Tier labels are fixed; display labels come from register_map via the
# rewriter. Order matters — render follows this order.
TIERS = ("shall", "should", "shmaybe", "watching")


@dataclass(frozen=True)
class _TieredLine:
    issue: BeadsIssue
    tier: str
    reason: str


def classify_issue(issue: BeadsIssue) -> tuple[str, str]:
    """Classify a BeadsIssue into one of the four tiers with a short
    reason suitable for rendering on the line. Pure: priority and
    dependent-count are the only signals v1 uses. Reason is a terse
    string the caller splices into the rendered plan line.

    Priority convention (bd): 0 = critical, 4 = backlog. ab maps:
      0 / 1          → shall
      2 + blocks     → shall
      2              → should
      3              → shmaybe
      4              → watching
    """
    blocks_others = int(issue.raw.get("dependent_count") or 0) > 0
    if issue.priority <= 1:
        return "shall", f"P{issue.priority}"
    if issue.priority == 2 and blocks_others:
        n = int(issue.raw["dependent_count"])
        return "shall", f"blocks {n}"
    if issue.priority == 2:
        return "should", "default P2"
    if issue.priority == 3:
        return "shmaybe", "low-priority"
    return "watching", "backlog"


def _render_plan(lines: Iterable[_TieredLine], today: date) -> str:
    """Render a tiered plan string. Empty tiers are listed explicitly
    ("(none)") so absence reads as deliberate, not as rendering bug."""
    buckets: dict[str, list[_TieredLine]] = {t: [] for t in TIERS}
    for line in lines:
        buckets[line.tier].append(line)
    out = [f"Today — {today.isoformat()}"]
    labels = {
        "shall": "Shall",
        "should": "Should",
        "shmaybe": "Shmaybe",
        "watching": "Watching",
    }
    counter = 1
    for tier in TIERS:
        label = labels[tier]
        items = buckets[tier]
        out.append(f"  {label}:")
        if not items:
            out.append("    (none)")
            continue
        for line in items:
            scope_tag = line.issue.scope or "?"
            out.append(
                f"    {counter}. [{scope_tag}/{line.issue.id}] {line.issue.title} — {line.reason}"
            )
            counter += 1
    return "\n".join(out)


def _render_status(issue: BeadsIssue, blockers: list[BeadsIssue]) -> str:
    """Render /status output: state, next, blockers, blocks. Blockers
    are listed one per line for readability; the raw JSON stays
    accessible on the issue for downstream tools that want more."""
    scope_tag = issue.scope or "?"
    header = (
        f"[{scope_tag}/{issue.id}] {issue.title} — "
        f"state: {issue.status}, priority: P{issue.priority}"
    )
    out = [header]
    dependent_count = int(issue.raw.get("dependent_count") or 0)
    blocker_count = int(issue.raw.get("dependency_count") or 0)
    if blocker_count:
        out.append(f"  Blockers ({blocker_count}):")
        for b in blockers:
            out.append(f"    - [{b.status}] {b.id} — {b.title}")
    else:
        out.append("  Blockers: none")
    if dependent_count:
        out.append(f"  Blocks: {dependent_count} item(s)")
    else:
        out.append("  Blocks: none")
    return "\n".join(out)


@dataclass
class PlanTool:
    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="plan",
            description=(
                "Render today's tiered operations path — Shall, Should, "
                "Shmaybe, Watching — with a reason on every non-empty line. "
                "Reads bd ready + bd list; tier is computed, not stored."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": list(ALLOWED_SCOPES),
                        "description": "Optional filter to a single scope.",
                    }
                },
                "required": [],
            },
            tier="read",
            display_name="Plan today",
        )

    def call(self, *, scope: str | None = None) -> str:
        err = _validate_scope(scope)
        if err:
            return err
        ready = self.adapter.ready(scope=scope)
        lines = [_TieredLine(issue=i, **_classify_dict(i)) for i in ready]
        return _render_plan(lines, date.today())


def _classify_dict(issue: BeadsIssue) -> dict[str, str]:
    tier, reason = classify_issue(issue)
    return {"tier": tier, "reason": reason}


def _validate_scope(scope: str | None) -> str | None:
    """Reject invalid scope values with a clear error string for the
    model. Tool schemas declare `enum` but small-model tool emitters
    sometimes ignore it; returning an explicit error lets the model
    course-correct instead of silently filtering to an empty set and
    then hallucinating state."""
    if scope is None:
        return None
    if scope in ALLOWED_SCOPES:
        return None
    return (
        f"invalid scope {scope!r}. Valid scopes: {', '.join(ALLOWED_SCOPES)}. "
        "Retry the call with a valid scope, or omit `scope` to see all."
    )


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
class StatusTool:
    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="status",
            description=(
                "Show an item's state, priority, blockers, and what it "
                "blocks. Accepts a bd issue id. Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "bd issue id."},
                },
                "required": ["id"],
            },
            tier="read",
            display_name="Status",
        )

    def call(self, *, id: str) -> str:
        try:
            issue = self.adapter.show(id)
        except BeadsAdapterError as exc:
            return f"status failed: {exc}"
        blockers_raw = issue.raw.get("dependencies") or []
        blockers: list[BeadsIssue] = []
        if isinstance(blockers_raw, list):
            for dep in blockers_raw:
                if isinstance(dep, dict) and "id" in dep:
                    blockers.append(
                        BeadsIssue(
                            id=str(dep["id"]),
                            title=str(dep.get("title", "")),
                            status=str(dep.get("status", "")),
                            priority=int(dep.get("priority") or 0),
                            issue_type=str(dep.get("issue_type", "")),
                            labels=tuple(dep.get("labels") or []),
                            raw=dep,
                        )
                    )
        return _render_status(issue, blockers)


@dataclass
class DriftTool:
    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="drift",
            description=(
                "Surface items that haven't been updated recently. Uses "
                "bd stale as its backing signal. Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": list(ALLOWED_SCOPES),
                        "description": "Optional scope filter.",
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Drift check",
        )

    def call(self, *, scope: str | None = None) -> str:
        err = _validate_scope(scope)
        if err:
            return err
        try:
            items = self.adapter.stale()
        except BeadsAdapterError as exc:
            return f"drift check failed: {exc}"
        if scope is not None:
            label = f"scope:{scope}"
            items = [i for i in items if label in i.labels]
        if not items:
            return "(no drift — backlog is current)"
        lines = ["Drifting:"]
        for issue in items:
            scope_tag = issue.scope or "?"
            lines.append(f"  - [{scope_tag}/{issue.id}] {issue.title} — status: {issue.status}")
        return "\n".join(lines)


@dataclass
class ReprioritizeTool:
    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="reprioritize",
            description=(
                "Recompute tiers from current bd state and emit a "
                "re-rank notice. Does NOT mutate bd priorities; it "
                "surfaces changes so the caller can confirm and apply "
                "via `defer` or `update` tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": list(ALLOWED_SCOPES),
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Reprioritize",
        )

    def call(self, *, scope: str | None = None) -> str:
        err = _validate_scope(scope)
        if err:
            return err
        ready = self.adapter.ready(scope=scope)
        lines = [_TieredLine(issue=i, **_classify_dict(i)) for i in ready]
        return _render_plan(lines, date.today())


@dataclass
class CloseTool:
    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="close",
            description="Close an item. Records an optional reason on the bead.",
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
        try:
            if priority == "down":
                current = self.adapter.show(id)
                new_priority = min(current.priority + 1, 4)
            else:
                new_priority = int(priority)
                if not 0 <= new_priority <= 4:
                    return f"defer failed: priority {new_priority} must be 0-4"
            self.adapter.update(id, priority=str(new_priority))
            if reason:
                # Reason is stored via bd remember so drift/retro can
                # surface rationale for stalled items later. No way to
                # attach structured notes without a richer bd schema.
                self.adapter.remember(f"deferred {id} → P{new_priority}: {reason}")
        except BeadsAdapterError as exc:
            return f"defer failed: {exc}"
        return f"Deferred {id} to P{new_priority}."


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


def make_ops_tools(
    adapter: _Adapter,
) -> tuple[
    PlanTool,
    CaptureTool,
    StatusTool,
    DriftTool,
    ReprioritizeTool,
    CloseTool,
    DeferTool,
    RetroTool,
]:
    """Single-point constructor for the full ab ops tool set. The CLI
    calls this once per session and passes the tuple to the registry."""
    return (
        PlanTool(adapter),
        CaptureTool(adapter),
        StatusTool(adapter),
        DriftTool(adapter),
        ReprioritizeTool(adapter),
        CloseTool(adapter),
        DeferTool(adapter),
        RetroTool(adapter),
    )
