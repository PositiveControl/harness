"""Orchestrator — drives tool-calling loops on top of a base model
adapter. Sits between the CLI and the adapter. Does not know about
persona or retrieval (those are the caller's responsibility)."""

from harness.orchestrator.tool_loop import (
    ConfirmFn,
    ObserverFn,
    ToolLoopEvent,
    ToolLoopResult,
    run_tool_loop,
)

__all__ = [
    "ConfirmFn",
    "ObserverFn",
    "ToolLoopEvent",
    "ToolLoopResult",
    "run_tool_loop",
]
