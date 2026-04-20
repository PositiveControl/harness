"""Graph tools for ab_ops — dependency links, labels, comments, and
duplicate-surface detection (harness-u73g).
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.store.bd_adapter import BeadsAdapterError
from harness.tools.ab_ops._shared import _Adapter
from harness.tools.base import ToolSpec


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
                "current labels.\n\n"
                "Scope is special: to set or change scope on an item "
                "('make harness-x professional', 'change scope to "
                "personal') use the `update` tool with scope=... — it "
                "swaps the underlying scope:* label atomically. Don't "
                "pass a bare 'professional'/'personal' to this tool; "
                "you'll create an unprefixed label that no read path "
                "filters on."
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
