"""Shared turn service (harness-fl313).

One grounded chat turn — retrieval, system-prompt assembly, transcript
history, optional tool loop, persona rewrite, persistence, audit — lifted
out of CLI-only assembly so CLI, TUI, web, and daemon callers run the
same path instead of each reimplementing (or, in web's case, skipping)
it. UI side effects (spinner, streaming, confirm prompts, console
notices) are injected via `TurnIO`; a headless caller passes the
defaults and gets a plain reply.
"""

from __future__ import annotations

from harness.turn.service import TurnContext, TurnIO, TurnResult, TurnService

__all__ = ["TurnContext", "TurnIO", "TurnResult", "TurnService"]
