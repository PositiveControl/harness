"""Search + filtered-list tools for ab_ops (harness-u73g)."""

from __future__ import annotations

from dataclasses import dataclass

from harness.store.bd_adapter import ALLOWED_SCOPES, ALLOWED_TYPES, BeadsAdapterError
from harness.tools.ab_ops._shared import (
    _SCOPE_PARAM_DESCRIPTION,
    _Adapter,
    _render_issue_list,
    _validate_scope,
)
from harness.tools.base import ToolSpec


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
