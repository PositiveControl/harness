"""Parse-gate escalation pair (harness-0v6d).

When the per-write parse-gate (harness-h6wa) trips repeatedly on the
same file in one turn, the model is stuck on the wrong edit primitive —
loop_run=3e295564 turn 17 ground two `edit_file` parse-gate failures
on game.js with `function gameLoop\\(now\\) \\{[^}]*\\}` regex variants
that can't match nested braces. Each retry compounded the broken edit.

This module installs two cooperating hooks that share a mutable
`ParseGateState`:

* `ParseGateFailureObserver` (post_tool) counts parse-gate failures
  per file path.
* `ParseGateEscalationHook` (pre_tool) intercepts the next
  edit_file / stream_edit / python_stream call on a path that's
  crossed the failure threshold and Skips it with a nudge instructing
  the model to use `read_file` + `write_file` (whole-file rewrite)
  instead.

Shared state is per-pipeline-instance — `_build_driver_hook_pipeline`
constructs a fresh pair per executor turn, so the counter resets
between turns (matches the bead's "within a turn" scope).

Backed by the file-ops-bench-2026-05 memory: `function_body_rewrite`
was 0% pass for stream_edit / pyp_stream / python_stream alike — the
escalation steers off a known-failing primitive onto the one that
works for that task shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.orchestrator.hooks import (
    Continue,
    PostToolContext,
    PostToolOutcome,
    PreToolContext,
    PreToolOutcome,
    Skip,
)
from harness.tools.base import ToolResult

# Tools that can land a broken edit and that we want to ESCALATE AWAY
# from once a file has tripped the parse-gate enough times this turn.
# `write_file` is intentionally absent — it IS the recommended primitive
# at escalation time (whole-file rewrite), so we must let it through.
_PATCH_TOOLS: frozenset[str] = frozenset({"edit_file", "stream_edit", "python_stream"})

# The substring every parse-gate failure carries (see
# `_enforce_parse_check` in tools/edit_file.py and tools/write_file.py).
_PARSE_GATE_MARKER: str = "no longer parses"

DEFAULT_THRESHOLD: int = 2


@dataclass
class ParseGateState:
    """Per-turn counter + blocked-set shared between the observer and
    escalation hook. Mutable by design — both hooks read+write it
    during a single pipeline lifetime."""

    failures_by_path: dict[str, int] = field(default_factory=dict)
    blocked_paths: set[str] = field(default_factory=set)


def _is_parse_gate_failure(result: ToolResult) -> bool:
    """True iff `result` is the parse-gate's signature failure shape:
    success=False with the 'no longer parses' marker in either output
    or error. Both edit_file and write_file raise the same wording."""
    if result.success:
        return False
    return _PARSE_GATE_MARKER in result.output or (
        result.error is not None and _PARSE_GATE_MARKER in result.error
    )


def _extract_path(args: dict[str, Any]) -> str | None:
    """Pull the file path from a tool call's arguments. Handles
    `path: str` (edit_file / write_file) and `paths: list[str]`
    (stream_edit / python_stream). Returns the first path for multi-
    path calls — failure-tracking is per-path, so multi-path edits
    accumulate against the first listed file. Returns None when the
    args carry no usable path."""
    path = args.get("path")
    if isinstance(path, str) and path:
        return path
    paths = args.get("paths")
    if isinstance(paths, list) and paths and isinstance(paths[0], str):
        return paths[0]
    if isinstance(paths, str) and paths.strip():
        # python_stream historically accepted a stringy `paths` (the
        # bench prompts in the trace did this); take the first non-
        # blank line.
        for line in paths.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return None


def _escalation_nudge(path: str, count: int) -> str:
    return (
        f"[PARSE_GATE_ESCALATION] {path} has tripped the per-write "
        f"parse-gate {count} times this turn — the patch primitive isn't "
        f"working on this file. STOP using edit_file / stream_edit / "
        f"python_stream on {path}. Instead: read_file({path!r}) to load "
        f"the current content, decide your changes against that text, "
        f"then write_file({path!r}, ...) with the FULL rewritten file. "
        f"Whole-file rewrite avoids the brace-matching / regex pitfalls "
        f"that broke the previous edits."
    )


@dataclass
class ParseGateFailureObserver:
    """post_tool hook: count parse-gate failures per file path. No
    return-value side effect — always Continue. The counter feeds the
    paired ParseGateEscalationHook."""

    name: str = "parse_gate_observer"
    state: ParseGateState = field(default_factory=ParseGateState)
    threshold: int = DEFAULT_THRESHOLD

    def check(self, ctx: PostToolContext) -> PostToolOutcome:
        if not _is_parse_gate_failure(ctx.result):
            return Continue()
        path = _extract_path(ctx.call.arguments)
        if path is None:
            return Continue()
        self.state.failures_by_path[path] = self.state.failures_by_path.get(path, 0) + 1
        if self.state.failures_by_path[path] >= self.threshold:
            self.state.blocked_paths.add(path)
        return Continue()


@dataclass
class ParseGateEscalationHook:
    """pre_tool hook: when a file has accumulated >= threshold parse-
    gate failures this turn, intercept the next edit_file /
    stream_edit / python_stream call against that path with a Skip
    carrying a whole-file-rewrite nudge."""

    name: str = "parse_gate_escalation"
    state: ParseGateState = field(default_factory=ParseGateState)

    def check(self, ctx: PreToolContext) -> PreToolOutcome:
        if ctx.call.name not in _PATCH_TOOLS:
            return Continue()
        path = _extract_path(ctx.call.arguments)
        if path is None or path not in self.state.blocked_paths:
            return Continue()
        count = self.state.failures_by_path.get(path, 0)
        return Skip(
            ToolResult(
                tool_name=ctx.call.name,
                output=_escalation_nudge(path, count),
                success=False,
                error="parse_gate_escalation",
            )
        )


def make_parse_gate_escalation_pair(
    *, threshold: int = DEFAULT_THRESHOLD
) -> tuple[ParseGateFailureObserver, ParseGateEscalationHook]:
    """Construct an observer + escalation pair that share one
    ParseGateState. Caller adds the observer to `post_tool` and the
    escalation hook to `pre_tool`. Threshold defaults to 2 (i.e.,
    intercept on the 2nd-and-later patch call after 2 prior failures
    on the same file)."""
    shared = ParseGateState()
    return (
        ParseGateFailureObserver(state=shared, threshold=threshold),
        ParseGateEscalationHook(state=shared),
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "ParseGateEscalationHook",
    "ParseGateFailureObserver",
    "ParseGateState",
    "make_parse_gate_escalation_pair",
]
