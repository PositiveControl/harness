"""Small-model intent router — pre-pass that picks a tool + arguments
so a weak main model doesn't burn rounds fabricating tool output.

This package owns parsing + shape. Orchestrator integration (synthetic
tool-call injection) lives in `harness.orchestrator.tool_loop` and is
tracked under harness-ut3. CLI wiring is harness-t7t. Evals, harness-zgu."""

from harness.router.intent import Router, RouterIntent
from harness.router.model_router import ModelRouter

__all__ = ["ModelRouter", "Router", "RouterIntent"]
