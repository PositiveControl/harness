"""Plan / status / drift / reprioritize tools — read-tier planning
views over bd state (harness-u73g).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from harness.store.bd_adapter import ALLOWED_SCOPES, BeadsAdapterError, BeadsIssue
from harness.tools.ab_ops._shared import (
    _SCOPE_PARAM_DESCRIPTION,
    AB_ASSIGNEE,
    AB_DRIFT_DAYS,
    _Adapter,
    _classify_dict,
    _parse_iso_utc,
    _render_focus_banner,
    _render_plan,
    _render_status_with_focus,
    _TieredLine,
    _validate_scope,
)
from harness.tools.base import ToolSpec


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
        ab_drift_days ago. bd's native `stale` runs on a longer horizon
        suited to user work; ab's thought-graph should re-evaluate idle
        thoughts sooner."""
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
