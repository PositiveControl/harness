"""Subagent primitive: a meta-tool that runs an isolated tool loop.

The parent agent calls `spawn_subagent(task=..., tools=[...])` when it
wants to investigate something without polluting its own context with
40 grep results. The tool opens a fresh `run_tool_loop` pass over the
same adapter + hooks, restricted to a caller-chosen read-tier subset
of the parent's tools, and returns the child's final reply verbatim
as a string.

Design invariants (locked 2026-04-22 — see harness-qxr):

- **Read-only subagent.** The child registry is filtered to read-tier
  tools only. Any write work reports back through the summary and the
  parent agent decides whether to act. This keeps subagent spawning
  safe to use autonomously without the write-tier confirmation UX.
- **Max depth 1.** Recursion is blocked by a ContextVar counter *and*
  by excluding `spawn_subagent` from the child registry, so a
  compromised / confused child can't burrow deeper.
- **Shared hooks.** The child loop runs the parent's HookPipeline, so
  fabrication catchers fire in the child too — no silent hallucination
  gap across the boundary.
- **Router on by default.** Small-model intent routing is cheap and
  independently useful inside the child; reuses the parent's Router
  instance (classification is stateless per call).
- **Isolated context.** The child `working` starts as
  `[system, user(task)]` — parent history is NOT leaked. The child's
  final reply is the summary (no second summarize pass — it's already
  a terminating reply by the tool loop's contract).

Observer events are not forwarded from the child to the parent in
v1. The parent sees a single tool_call_start/end pair around the
`spawn_subagent` invocation, the way every other tool looks. Inline
step-by-step visibility can be added later (harness-qxr follow-up).
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harness.model.adapter import ChatMessage
from harness.tools.base import Tool, ToolRegistry, ToolSpec

if TYPE_CHECKING:
    from harness.orchestrator.hooks import HookPipeline
    from harness.orchestrator.tool_loop import _ToolCapableAdapter
    from harness.router.intent import Router


DEFAULT_SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused subagent spawned by the main agent to "
    "investigate a specific question. You have been given a small set "
    "of read-only tools and a task. Complete the task in as few rounds "
    "as possible and return a concise summary. Preserve file paths, "
    "identifiers, and line numbers verbatim in your answer so the "
    "parent agent can act on them. You CANNOT modify files or state — "
    "if the task requires writes, describe precisely what should be "
    "done and let the parent agent act on your findings."
)


# ContextVar-scoped depth counter. Using ContextVar (vs threading.local)
# means an asyncio-driven future gateway inherits depth correctly
# across awaits. Default = 0 at the top level; incremented inside each
# call, restored on exit.
_SUBAGENT_DEPTH: ContextVar[int] = ContextVar("_SUBAGENT_DEPTH", default=0)


@dataclass
class SpawnSubagentTool:
    """Runs a child `run_tool_loop` with a caller-chosen read-tier
    subset of the parent's registry. Returns the child's final reply
    verbatim as the summary.

    `adapter`, `registry`, and `hooks` are references to the parent's
    objects — the same ones the parent tool loop uses. Construction
    must happen after the parent registry is populated so `registry`
    contains every tool the subagent could choose from.

    `max_depth` caps nesting; the default of 1 means the top-level
    parent can spawn a subagent but the subagent cannot spawn a
    sub-subagent. Self-recursion is additionally blocked by stripping
    `spawn_subagent` from the child registry."""

    adapter: _ToolCapableAdapter
    registry: ToolRegistry
    hooks: HookPipeline
    router: Router | None = None
    default_max_rounds: int = 6
    default_system_prompt: str = DEFAULT_SUBAGENT_SYSTEM_PROMPT
    max_depth: int = 1
    # Populated at construction time; kept as a tuple for introspection /
    # schema-rendering but the tool uses `registry.names()` at call time
    # since the parent registry is mutable.
    _profile_tools: tuple[str, ...] = field(default_factory=tuple)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="spawn_subagent",
            description=(
                "Run an isolated read-only subagent on a focused task. "
                "Use this when a question needs 5+ tool calls to answer "
                "but you don't want the intermediate results eating your "
                "own context. The subagent has its own fresh conversation, "
                "a caller-chosen subset of read-tier tools, and returns a "
                "summary string. The subagent CANNOT modify files, memory, "
                "or state — if the task needs a write, have the subagent "
                "investigate + report, then you perform the write yourself."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "The question or investigation for the "
                            "subagent to answer. Be concrete — the "
                            "subagent won't see your history."
                        ),
                    },
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Names of tools to make available to the "
                            "subagent. Must be a subset of your own "
                            "read-tier tools."
                        ),
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": ("Hard cap on the subagent's loop rounds. Defaults to 6."),
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": (
                            "Optional override for the subagent's system "
                            "prompt. Defaults to a generic 'investigate "
                            "and report' brief."
                        ),
                    },
                },
                "required": ["task", "tools"],
            },
            tier="read",
            display_name="Spawn subagent",
        )

    def call(
        self,
        *,
        task: str,
        tools: list[str],
        max_rounds: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        depth = _SUBAGENT_DEPTH.get()
        if depth >= self.max_depth:
            return (
                f"spawn_subagent error: max depth {self.max_depth} exceeded. "
                "Subagents cannot spawn further subagents. Do the "
                "investigation inline in this loop instead."
            )

        # Late import so the subagent module is safe to import at
        # CLI wiring time even while tool_loop is being evaluated.
        from harness.orchestrator.tool_loop import run_tool_loop

        child_registry_or_error = self._build_child_registry(tools)
        if isinstance(child_registry_or_error, str):
            return child_registry_or_error
        child_registry = child_registry_or_error

        rounds = max_rounds if max_rounds is not None else self.default_max_rounds
        if rounds < 1:
            return "spawn_subagent error: max_rounds must be >= 1"

        prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        child_messages: list[ChatMessage] = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=task),
        ]

        token = _SUBAGENT_DEPTH.set(depth + 1)
        try:
            result = run_tool_loop(
                self.adapter,
                child_messages,
                child_registry,
                max_rounds=rounds,
                hooks=self.hooks,
                router=self.router,
            )
        finally:
            _SUBAGENT_DEPTH.reset(token)

        summary = result.content.strip()
        # The tool loop's exhaustion branch emits this exact marker
        # when it runs out of rounds with pending tool calls instead
        # of a final reply. Translate it into a subagent-framed
        # message so the parent can distinguish budget exhaustion
        # from a genuine empty reply.
        if summary == "[tool loop exhausted without final reply]":
            return (
                f"[spawn_subagent: budget exhausted after {rounds} rounds "
                "without a final reply. Nothing to report.]"
            )
        if not summary:
            return (
                "[spawn_subagent: loop returned empty content. The subagent "
                "may have emitted only tool calls without a closing reply.]"
            )
        return summary

    def _build_child_registry(self, tools: list[str]) -> ToolRegistry | str:
        """Validate + filter the requested tool list, return a fresh
        ToolRegistry containing only those tools, OR a user-facing
        error string on validation failure. Rejected reasons:
        - `spawn_subagent` in the list (recursion guard)
        - tool not present in the parent registry
        - tool is write-tier
        """
        if not tools:
            return (
                "spawn_subagent error: tools list is empty. The subagent "
                "needs at least one read-tier tool to do any work."
            )

        rejected: list[str] = []
        unknown: list[str] = []
        write_tier: list[str] = []
        selected: list[Tool] = []
        seen: set[str] = set()
        for name in tools:
            if name == self.spec.name:
                rejected.append(name)
                continue
            if name in seen:
                continue
            seen.add(name)
            if name not in self.registry:
                unknown.append(name)
                continue
            tool = self.registry.get(name)
            if tool.spec.tier != "read":
                write_tier.append(name)
                continue
            selected.append(tool)

        if rejected:
            return (
                "spawn_subagent error: subagents cannot spawn further "
                "subagents. Remove 'spawn_subagent' from the tools list."
            )
        if unknown:
            return (
                f"spawn_subagent error: unknown tools {unknown!r}. "
                f"Available: {sorted(self.registry.names())}."
            )
        if write_tier:
            return (
                f"spawn_subagent error: tools {write_tier!r} are write-tier. "
                "Subagents are read-only; investigate with read-tier tools "
                "and let the parent agent perform any writes."
            )

        child = ToolRegistry()
        for tool in selected:
            child.register(tool)
        return child


def current_subagent_depth() -> int:
    """Inspection helper for tests + diagnostics. Returns the current
    ContextVar depth counter; 0 means we're at the top-level loop."""
    return _SUBAGENT_DEPTH.get()
