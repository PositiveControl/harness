"""Orchestrator — drives tool-calling loops on top of a base model
adapter. Sits between the CLI and the adapter. Does not know about
persona or retrieval (those are the caller's responsibility)."""

from harness.orchestrator.tool_loop import (
    _FABRICATED_SEARCH_RE,
    _FALSE_SUCCESS_RE,
    _META_CONFIRM_RE,
    _TOOL_INTENT_RE,
    ConfirmFn,
    ObserverFn,
    ToolLoopEvent,
    ToolLoopResult,
    run_tool_loop,
)

__all__ = [
    # Private regexes exposed for the CLI stream renderer so it can
    # suppress meta-confirm / false-success / fabricated-output / bare-
    # intent text before it lands on the user's terminal. Used by
    # stream-level filtering; not a stable API.
    "_FABRICATED_SEARCH_RE",
    "_FALSE_SUCCESS_RE",
    "_META_CONFIRM_RE",
    "_TOOL_INTENT_RE",
    "ConfirmFn",
    "ObserverFn",
    "ToolLoopEvent",
    "ToolLoopResult",
    "run_tool_loop",
]
