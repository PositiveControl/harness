"""Render a Plan as a compact context block for the model — harness-uzan.

Plain text, ~200-token budget so the orchestrator can prepend it to a
system prompt without crowding out tool schemas. The renderer prefers
breadth (one line per active subgoal) over depth (deeply nested
precondition trees) — the model needs to know *what's in flight* and
*what's waiting*, not the full ontology.

Output shape::

    # Active plan: <plan title>
    - <subgoal title>
        waits on: <bead-id> (closed)
        waits until: <ISO timestamp>
    - <subgoal title>

    Upcoming (N pending):
    - <subgoal title>

Truncated with an ellipsis when over `char_budget`. Caller decides
whether to elide further; the renderer just guarantees it never
exceeds the budget.
"""

from __future__ import annotations

from harness.plan.model import Plan, Precondition

DEFAULT_CHAR_BUDGET = 800  # ~200 tokens on a Qwen tokenizer (4 chars/token).
DEFAULT_MAX_SUBGOALS = 8


def _format_precondition(p: Precondition) -> str | None:
    """Render one precondition as a compact one-liner. Returns None
    for kinds the renderer doesn't recognize so the system prompt
    doesn't carry uninterpretable noise."""
    if p.kind == "bd_closed":
        bead = p.payload.get("bead")
        if isinstance(bead, str) and bead:
            return f"waits on: {bead} (closed)"
        return None
    if p.kind == "bd_open":
        bead = p.payload.get("bead")
        if isinstance(bead, str) and bead:
            return f"waits on: {bead} (open)"
        return None
    if p.kind == "timestamp_past":
        when = p.payload.get("when")
        if isinstance(when, str) and when:
            return f"waits until: {when}"
        return None
    return None


def render_plan_block(
    plan: Plan,
    *,
    max_subgoals: int = DEFAULT_MAX_SUBGOALS,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> str:
    """Render `plan` as plain text suitable for a system-prompt block.

    Active subgoals come first with their preconditions; pending
    subgoals follow as a flat upcoming list with a count. Achieved /
    abandoned subgoals are deliberately omitted — they're not load-
    bearing for the model's current-turn reasoning, and including
    them inflates the budget for little gain.

    Args:
        plan: the Plan to render.
        max_subgoals: cap on the active + pending lists. Excess
            entries collapse into a `(+N more)` line so the model
            knows the block is incomplete.
        char_budget: hard upper bound on the returned string. Anything
            past this is truncated with an ellipsis.
    """
    lines: list[str] = [f"# Active plan: {plan.title}"]
    active = [sg for sg in plan.active() if sg.id != plan.root_subgoal_id]
    pending = [
        sg
        for sg in plan.subgoals.values()
        if sg.status == "pending" and sg.id != plan.root_subgoal_id
    ]

    if not active and not pending:
        lines.append("(no active or pending subgoals)")
    else:
        if active:
            visible_active = active[:max_subgoals]
            for sg in visible_active:
                lines.append(f"- {sg.title}")
                for p in sg.preconditions:
                    rendered = _format_precondition(p)
                    if rendered is not None:
                        lines.append(f"    {rendered}")
            remaining_active = len(active) - len(visible_active)
            if remaining_active > 0:
                lines.append(f"  (+{remaining_active} more active)")

        if pending:
            lines.append("")
            lines.append(f"Upcoming ({len(pending)} pending):")
            visible_pending = pending[:max_subgoals]
            for sg in visible_pending:
                lines.append(f"- {sg.title}")
            remaining_pending = len(pending) - len(visible_pending)
            if remaining_pending > 0:
                lines.append(f"  (+{remaining_pending} more pending)")

    text = "\n".join(lines)
    if len(text) <= char_budget:
        return text
    # Truncate with a clear ellipsis marker so the model knows the
    # block was cut.
    return text[: char_budget - 3] + "..."
