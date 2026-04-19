"""ab's operations tool set — harness-inj.5.

Wraps ab's slash commands over bd. Every data-plane operation
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
| reopen         | write | re-open a closed item (bd reopen)         |
| delete         | write | permanently remove an item (bd delete)    |
| update         | write | edit title / description / notes / …      |
| search         | read  | keyword search over items                 |
| list           | read  | filtered cross-section of items           |
| memories       | read  | list / search persistent memories         |
| remember       | write | persist a note-to-self / insight          |
| forget         | write | remove a persistent memory by key         |
| dep            | write | add / remove dependency links             |
| label          | write | add / remove / list labels                |
| comments       | write | add / list comments on an item            |
| find_duplicates| read  | surface candidate dup pairs (mechanical)  |

Deadline-aware tier promotion (date-locked items → Shall at T-1) is a
follow-up; v1 classifies from priority + dependent-count only. Issues
can still carry deadlines as free-form text in their description.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
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
        priority: str | None = ...,
        issue_type: str | None = ...,
        limit: int | None = ...,
        assignee: str | None = ...,
    ) -> list[BeadsIssue]: ...

    def search(
        self,
        query: str,
        *,
        status: str | None = ...,
        limit: int | None = ...,
    ) -> list[BeadsIssue]: ...

    def show(self, issue_id: str) -> BeadsIssue: ...

    def stale(self) -> list[BeadsIssue]: ...

    def get_focus(self, assignee: str) -> BeadsIssue | None: ...

    def persist_to_focus(self, summary: str, *, assignee: str) -> str: ...

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
        assignee: str | None = ...,
    ) -> str: ...

    def close(self, issue_id: str, *, reason: str | None = ...) -> None: ...

    def reopen(self, issue_id: str, *, reason: str | None = ...) -> None: ...

    def delete(self, issue_id: str, *, cascade: bool = ...) -> None: ...

    def update(self, issue_id: str, **fields: Any) -> None: ...

    def dep_add(self, issue: str, depends_on: str) -> None: ...

    def dep_rm(self, issue: str, depends_on: str) -> None: ...

    def remember(self, insight: str) -> None: ...

    def memories(self, query: str = ...) -> str: ...

    def forget(self, key: str) -> None: ...

    def label_add(self, issue_id: str, label: str) -> None: ...

    def label_rm(self, issue_id: str, label: str) -> None: ...

    def label_list(self, issue_id: str) -> str: ...

    def comment_add(self, issue_id: str, text: str) -> None: ...

    def comments_list(self, issue_id: str) -> str: ...

    def find_duplicates(
        self,
        *,
        threshold: float | None = ...,
        limit: int | None = ...,
        status: str | None = ...,
    ) -> list[dict[str, Any]]: ...


CAPTURE_REQUIRED_FIELDS = ("scope", "outcome", "next_action")

# Tier labels are fixed; display labels come from register_map via the
# rewriter. Order matters — render follows this order.
TIERS = ("shall", "should", "shmaybe", "watching")

# Assignee name for ab's own thought-graph beads. Plan / status use
# this to surface the current focus without polluting the broader tool
# interface with an extra 'who owns this' arg.
AB_ASSIGNEE = "airton_b"

# Defer-count threshold for stall escalation. When a bead has been
# deferred this many times, DeferTool auto-creates a thought:question
# child to surface the 'still relevant?' question. Settings knob lives
# at Settings.ab_stall_defers (harness-6y5) — DeferTool doesn't read
# it directly to keep tool deps minimal; wire-up is a follow-up.
STALL_DEFERS = 3
STALL_LABEL = "stall-escalated"
_DEFER_COUNT_PREFIX = "defer-count:"

# Ab-internal drift threshold in days. bd's own `stale` signal is
# tuned for user work (~14d default); ab's thought-graph rots faster,
# so DriftTool augments bd.stale() with a client-side filter that
# flags ab-owned beads untouched for this many days. Settings knob
# lives at Settings.ab_drift_days (harness-6y5).
AB_DRIFT_DAYS = 7


# Shared description for every scope parameter (harness-ilru). Small
# models ignore JSON-schema enums when the description is vague and
# will pass 'tomorrow' / 'this week' after a time-phrased user turn,
# conflating scope (categorical) with when (temporal). Spelling out
# the contract and naming concrete anti-examples is what stops it.
_SCOPE_PARAM_DESCRIPTION = (
    "Scope is CATEGORICAL, not temporal — one of {professional, personal}. "
    "OMIT this field on time phrasings like 'tomorrow', 'today', 'this week', "
    "'next month' — scope does not filter by time. Omit also when the user "
    "hasn't specified a category."
)


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
                        "description": _SCOPE_PARAM_DESCRIPTION,
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
        plan = _render_plan(lines, date.today())
        focus = self.adapter.get_focus(AB_ASSIGNEE)
        if focus is None:
            return plan
        return f"{_render_focus_banner(focus)}\n{plan}"


def _render_focus_banner(issue: BeadsIssue) -> str:
    """One-line banner above the tier buckets. Announces what ab is
    currently holding in working memory so the plan isn't read in a
    vacuum."""
    scope_tag = issue.scope or "?"
    return f"Focus: [{scope_tag}/{issue.id}] {issue.title} (P{issue.priority})"


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
class StatusTool:
    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="status",
            description=(
                "Show an item's state, priority, blockers, and what it "
                "blocks. With no id, shows ab's current focus bead "
                "(the single in_progress thought-graph item). Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": ("bd issue id. Omit to show the current focus."),
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Status",
        )

    def call(self, *, id: str | None = None) -> str:
        focus = self.adapter.get_focus(AB_ASSIGNEE)
        if id is None:
            if focus is None:
                return (
                    "(no focus) — ab has no in_progress thought-graph bead. "
                    "Capture one or promote an existing open bead to focus."
                )
            return _render_status_with_focus(focus, focus_id=focus.id)
        try:
            issue = self.adapter.show(id)
        except BeadsAdapterError as exc:
            return f"status failed: {exc}"
        focus_id = focus.id if focus is not None else None
        return _render_status_with_focus(issue, focus_id=focus_id)


def _render_status_with_focus(issue: BeadsIssue, *, focus_id: str | None) -> str:
    """Shared rendering for StatusTool. Extracts the blockers from
    issue.raw and appends a focus-switch hint when the subject bead
    isn't the current focus — advisory only, does not mutate."""
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
    rendered = _render_status(issue, blockers)
    if focus_id is not None and focus_id != issue.id:
        rendered += f"\n  (Not current focus — ab is focused on {focus_id}.)"
    return rendered


@dataclass
class DriftTool:
    adapter: _Adapter
    ab_drift_days: int = AB_DRIFT_DAYS

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="drift",
            description=(
                "Surface items that haven't been updated recently. Uses "
                "bd stale for user-owned work plus a shorter-window "
                "client-side check for ab-owned thought-graph beads "
                f"(default {AB_DRIFT_DAYS}d). Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": list(ALLOWED_SCOPES),
                        "description": _SCOPE_PARAM_DESCRIPTION,
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
            ab_stale = self._compute_ab_stale()
        except BeadsAdapterError as exc:
            return f"drift check failed: {exc}"
        seen = {i.id for i in items}
        items = list(items) + [i for i in ab_stale if i.id not in seen]
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

    def _compute_ab_stale(self) -> list[BeadsIssue]:
        """Flag ab-owned open beads last-updated more than
        ab_drift_days ago. bd's native `stale` typically runs on a
        longer horizon suited to user work; ab's thought-graph should
        re-evaluate idle thoughts sooner."""
        candidates = self.adapter.list_issues(status="open", assignee=AB_ASSIGNEE)
        cutoff = datetime.now(UTC) - timedelta(days=self.ab_drift_days)
        stale: list[BeadsIssue] = []
        for issue in candidates:
            raw_ts = issue.raw.get("updated_at")
            if not raw_ts:
                continue
            parsed = _parse_iso_utc(str(raw_ts))
            if parsed is None:
                continue
            if parsed < cutoff:
                stale.append(issue)
        return stale


def _parse_iso_utc(ts: str) -> datetime | None:
    """Parse an ISO-8601 'Z' suffix timestamp into a timezone-aware
    UTC datetime. Returns None on anything unparseable so callers can
    skip rather than crash on an odd bd output."""
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


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
                        "description": _SCOPE_PARAM_DESCRIPTION,
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
        # Validate the priority spec before touching bd so bad input
        # bails cheap without spawning a show/update subprocess.
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
                # Reason is stored via bd remember so drift/retro can
                # surface rationale for stalled items later. No way to
                # attach structured notes without a richer bd schema.
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
    has a count; one add call on the first defer. Returns the new
    count so the caller can decide whether to escalate."""
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
    stalled parent is still relevant. Parent-linked via bd's --parent
    so the relationship is explicit; assigned to airton_b so the
    thought-graph view picks it up."""
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


_UPDATE_FIELD_FLAGS: dict[str, str] = {
    "title": "title",
    "description": "description",
    "notes": "notes",
    "assignee": "assignee",
    "priority": "priority",
    "status": "status",
}


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


def _render_issue_list(issues: list[BeadsIssue], *, empty_label: str) -> str:
    """Shared compact renderer for search / list results. One line per
    issue with scope, id, priority, status, title. Empty label lets
    callers distinguish 'no matches' from 'no open items'.

    Exact-title duplicates are collapsed onto the highest-priority
    sibling with an '(also: <id>, …)' annotation so the user sees the
    duplication instead of reading two independent rows and assuming
    they're distinct work (harness-mmz1). Fuzzy dedup is
    `find_duplicates`' job — this helper stays mechanical."""
    if not issues:
        return empty_label

    # Group by exact title, preserve first-seen order for the primary.
    primary_by_title: dict[str, BeadsIssue] = {}
    sibling_ids_by_title: dict[str, list[str]] = {}
    order: list[str] = []
    for issue in issues:
        key = issue.title
        if key not in primary_by_title:
            primary_by_title[key] = issue
            sibling_ids_by_title[key] = []
            order.append(key)
            continue
        # Dupe: keep the higher-priority one as primary (lower int = higher).
        current = primary_by_title[key]
        if issue.priority < current.priority:
            sibling_ids_by_title[key].append(current.id)
            primary_by_title[key] = issue
        else:
            sibling_ids_by_title[key].append(issue.id)

    lines: list[str] = []
    for key in order:
        issue = primary_by_title[key]
        scope_tag = issue.scope or "?"
        suffix = ""
        siblings = sibling_ids_by_title[key]
        if siblings:
            suffix = f" (also: {', '.join(siblings)})"
        lines.append(
            f"  - [{scope_tag}/{issue.id}] P{issue.priority} {issue.status}: {issue.title}{suffix}"
        )
    return "\n".join(lines)


@dataclass
class SearchTool:
    """Keyword search over bd. Defaults exclude closed issues; pass
    status='all' to include them. Distinct from drift/status — this
    is a free-text lookup, not a curated view."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search",
            description=(
                "Search items by keyword (title + id prefix by default). "
                "Excludes closed items unless status='all'. Returns a "
                "compact list; call `status` or `show` for detail."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "status": {
                        "type": "string",
                        "description": (
                            "Optional status filter (open, in_progress, "
                            "blocked, deferred, closed, all)."
                        ),
                    },
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
            },
            tier="read",
            display_name="Search",
        )

    def call(
        self,
        *,
        query: str,
        status: str | None = None,
        limit: int | None = None,
    ) -> str:
        if not query.strip():
            return "search failed: query must be non-empty"
        try:
            hits = self.adapter.search(query, status=status, limit=limit)
        except (ValueError, BeadsAdapterError) as exc:
            return f"search failed: {exc}"
        header = f"Search {query!r}:"
        body = _render_issue_list(hits, empty_label="  (no matches)")
        return f"{header}\n{body}"


@dataclass
class ListTool:
    """Filtered list. Distinct from `plan`, which renders tier buckets
    over the ready-subset. This tool is a raw cross-section: filter
    by status / priority / type / scope, return every matching row."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="list",
            description=(
                "List items matching optional filters: status, priority, "
                "type, scope. Distinct from `plan` — this is a raw "
                "filtered cross-section, not a tiered planning view."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                    },
                    "issue_type": {
                        "type": "string",
                        "enum": list(ALLOWED_TYPES),
                    },
                    "scope": {
                        "type": "string",
                        "enum": list(ALLOWED_SCOPES),
                        "description": _SCOPE_PARAM_DESCRIPTION,
                    },
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": [],
            },
            tier="read",
            display_name="List items",
        )

    def call(
        self,
        *,
        status: str | None = None,
        priority: int | None = None,
        issue_type: str | None = None,
        scope: str | None = None,
        limit: int | None = None,
    ) -> str:
        scope_err = _validate_scope(scope)
        if scope_err:
            return scope_err
        if priority is not None and not 0 <= priority <= 4:
            return f"list failed: priority {priority} must be 0-4"
        try:
            items = self.adapter.list_issues(
                scope=scope,
                status=status,
                priority=str(priority) if priority is not None else None,
                issue_type=issue_type,
                limit=limit,
            )
        except BeadsAdapterError as exc:
            return f"list failed: {exc}"
        return _render_issue_list(items, empty_label="(no matching items)")


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


@dataclass
class DepTool:
    """Add or remove dependency links. `add` creates an
    issue-depends-on-blocker relation; `remove` deletes the link.
    Single tool with an op switch keeps the schema compact — the
    model usually says either 'make X depend on Y' or 'drop dep';
    one tool handles both shapes."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="dep",
            description=(
                "Manage dependency links between items. op='add' makes "
                "`issue` depend on `depends_on` (i.e. depends_on blocks "
                "issue). op='remove' drops the link."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["add", "remove"],
                    },
                    "issue": {
                        "type": "string",
                        "description": "The dependent item (the one that is blocked).",
                    },
                    "depends_on": {
                        "type": "string",
                        "description": "The blocker item.",
                    },
                },
                "required": ["op", "issue", "depends_on"],
            },
            tier="write",
            display_name="Dependency",
        )

    def call(self, *, op: str, issue: str, depends_on: str) -> str:
        if op not in {"add", "remove"}:
            return f"dep failed: unknown op {op!r}; use 'add' or 'remove'"
        try:
            if op == "add":
                self.adapter.dep_add(issue, depends_on)
                return f"Linked {issue} → depends on {depends_on}."
            self.adapter.dep_rm(issue, depends_on)
        except BeadsAdapterError as exc:
            return f"dep failed: {exc}"
        return f"Unlinked {issue} from {depends_on}."


@dataclass
class LabelTool:
    """Add, remove, or list labels on an item. Mixed-op tool: list is
    read-only, add/remove mutate. Tiered write so the session-level
    confirmation gate applies to every invocation — over-gating list
    is preferable to letting the model silently tag or untag items."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="label",
            description=(
                "Manage labels on an item. op='add' or op='remove' "
                "needs a label string; op='list' returns the item's "
                "current labels. The scope: label is special — don't "
                "edit it via this tool; rescope by capturing a new "
                "item or closing and recreating."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["add", "remove", "list"],
                    },
                    "id": {"type": "string"},
                    "label": {
                        "type": "string",
                        "description": "Required for add/remove; ignored for list.",
                    },
                },
                "required": ["op", "id"],
            },
            tier="write",
            display_name="Label",
        )

    def call(self, *, op: str, id: str, label: str | None = None) -> str:
        if op not in {"add", "remove", "list"}:
            return f"label failed: unknown op {op!r}; use 'add', 'remove', or 'list'"
        if op in {"add", "remove"} and not (label and label.strip()):
            return f"label failed: op={op!r} requires a label"
        try:
            if op == "list":
                out = self.adapter.label_list(id)
                text = out.strip()
                return text if text else f"{id}: (no labels)"
            if op == "add":
                self.adapter.label_add(id, label)  # type: ignore[arg-type]  # validated above
                return f"Added label {label!r} to {id}."
            self.adapter.label_rm(id, label)  # type: ignore[arg-type]  # validated above
        except (ValueError, BeadsAdapterError) as exc:
            return f"label failed: {exc}"
        return f"Removed label {label!r} from {id}."


@dataclass
class CommentsTool:
    """List or add comments on an item. Same mixed-tier rationale as
    LabelTool: list is cheap but add mutates, so tier=write gates the
    whole tool."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="comments",
            description=(
                "List or add comments on an item. op='list' returns "
                "bd's raw comment output; op='add' appends a new "
                "comment with the provided text.\n\n"
                "The `id` MUST be an existing bd issue id (e.g. "
                "harness-abc) that you already know exists — either "
                "from a prior tool call this turn or because the user "
                "named it. Never invent an id from the user's phrasing. "
                "For 'note to self', 'remember that', 'write it down' "
                "use the `remember` tool, which stores a durable "
                "memory without needing an issue id."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["list", "add"],
                    },
                    "id": {"type": "string"},
                    "text": {
                        "type": "string",
                        "description": "Required for op='add'; ignored for op='list'.",
                    },
                },
                "required": ["op", "id"],
            },
            tier="write",
            display_name="Comments",
        )

    def call(self, *, op: str, id: str, text: str | None = None) -> str:
        if op not in {"list", "add"}:
            return f"comments failed: unknown op {op!r}; use 'list' or 'add'"
        if op == "add" and not (text and text.strip()):
            return "comments failed: op='add' requires non-empty text"
        try:
            if op == "list":
                out = self.adapter.comments_list(id)
                body = out.strip()
                return body if body else f"{id}: (no comments)"
            self.adapter.comment_add(id, text)  # type: ignore[arg-type]  # validated above
        except (ValueError, BeadsAdapterError) as exc:
            return f"comments failed: {exc}"
        return f"Added comment to {id}."


@dataclass
class FindDuplicatesTool:
    """Surface candidate duplicate pairs using bd's mechanical
    (token-similarity) method. AI-backed detection is intentionally
    disabled — it would leak issue text to a cloud endpoint without
    the user's explicit ask. Read-tier; no auto-merge."""

    adapter: _Adapter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="find_duplicates",
            description=(
                "Find pairs of items that look like duplicates via "
                "token similarity (mechanical method only). Returns "
                "candidate pairs; merging is not automated — surface "
                "the pairs and let the user decide."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "Similarity cutoff 0.0-1.0; lower surfaces "
                            "more pairs. bd default is 0.5."
                        ),
                    },
                    "limit": {"type": "integer", "minimum": 1},
                    "status": {
                        "type": "string",
                        "description": "Optional status filter (defaults to non-closed).",
                    },
                },
                "required": [],
            },
            tier="read",
            display_name="Find duplicates",
        )

    def call(
        self,
        *,
        threshold: float | None = None,
        limit: int | None = None,
        status: str | None = None,
    ) -> str:
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            return f"find_duplicates failed: threshold {threshold} must be 0.0-1.0"
        try:
            pairs = self.adapter.find_duplicates(
                threshold=threshold,
                limit=limit,
                status=status,
            )
        except BeadsAdapterError as exc:
            return f"find_duplicates failed: {exc}"
        if not pairs:
            return "(no duplicate candidates)"
        lines = [f"Duplicate candidates ({len(pairs)}):"]
        for pair in pairs:
            a_id = pair.get("a_id") or pair.get("id_a") or pair.get("first")
            b_id = pair.get("b_id") or pair.get("id_b") or pair.get("second")
            sim = pair.get("similarity") or pair.get("score")
            a_title = pair.get("a_title") or pair.get("title_a") or ""
            b_title = pair.get("b_title") or pair.get("title_b") or ""
            sim_frag = f" ({sim:.2f})" if isinstance(sim, int | float) else ""
            lines.append(f"  - {a_id} ↔ {b_id}{sim_frag}: {a_title} | {b_title}")
        return "\n".join(lines)


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
    if in_progress:
        labels = ", ".join(i.id for i in in_progress)
        lines.append(f"In-progress ab-beads: {labels}")
    else:
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
    ReopenTool,
    DeleteTool,
    UpdateTool,
    SearchTool,
    ListTool,
    MemoriesTool,
    RememberTool,
    ForgetTool,
    DepTool,
    LabelTool,
    CommentsTool,
    FindDuplicatesTool,
    PersistFocusNoteTool,
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
        ReopenTool(adapter),
        DeleteTool(adapter),
        UpdateTool(adapter),
        SearchTool(adapter),
        ListTool(adapter),
        MemoriesTool(adapter),
        RememberTool(adapter),
        ForgetTool(adapter),
        DepTool(adapter),
        LabelTool(adapter),
        CommentsTool(adapter),
        FindDuplicatesTool(adapter),
        PersistFocusNoteTool(adapter),
    )
