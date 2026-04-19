"""Write-tier confirmation controller for the TUI chat app.

Extracted from chat_app.py::ChatApp (harness-iwco). Owns the
ConfirmStrip widget lifecycle and the blocking future the worker
thread awaits while the user decides approve / decline / always.

State flow (harness-drd):
  worker calls `confirm(call)` → UI thread shows strip + creates
  future → user presses y/n/a or escape → action posts verdict via
  `resolve()` → future fulfilled → worker resumes. If tool was
  marked 'always', subsequent calls short-circuit.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from textual.widgets import Input

if TYPE_CHECKING:
    from harness.tools import ToolCall
    from harness.tui.chat_app import ChatApp, _ChatAppState


APPROVE = "approve"
DECLINE = "decline"
ALWAYS = "always"


class ConfirmController:
    def __init__(self, app: ChatApp, state: _ChatAppState) -> None:
        self._app = app
        self._state = state
        self._approved_tools: set[str] = set()

    def request(self, call: ToolCall) -> bool:
        """Worker-thread. Block until the user resolves the inline
        confirm strip. Auto-approve when the tool is in the session's
        always-allow set."""
        if call.name in self._approved_tools:
            return True
        call_from_thread: Any = self._app.call_from_thread
        decision: str = call_from_thread(self._prompt, call)
        if decision == APPROVE:
            return True
        if decision == ALWAYS:
            self._approved_tools.add(call.name)
            return True
        return False

    async def _prompt(self, call: ToolCall) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._state.confirm_future = future
        self._show_strip(call)
        try:
            return await future
        except asyncio.CancelledError:
            return DECLINE
        finally:
            self._hide_strip()
            self._state.confirm_future = None

    def is_pending(self) -> bool:
        fut = self._state.confirm_future
        return fut is not None and not fut.done()

    def resolve(self, decision: str) -> None:
        fut = self._state.confirm_future
        if fut is None or fut.done():
            return
        fut.set_result(decision)

    def _show_strip(self, call: ToolCall) -> None:
        from harness.tui.chat_app import ConfirmStrip

        strip = self._app.query_one(ConfirmStrip)
        label = self._tool_label(call.name)
        args_preview = format_confirm_args(call.arguments)
        summary = f"[bold yellow]⚠ {label}[/bold yellow]"
        if args_preview:
            summary = f"{summary} [dim]{args_preview}[/dim]"
        strip.show_for(summary)

    def _hide_strip(self) -> None:
        from harness.tui.chat_app import ConfirmStrip

        try:
            strip = self._app.query_one(ConfirmStrip)
        except Exception:
            return
        strip.hide()
        try:
            self._app.query_one("#prompt", Input).focus()
        except Exception:
            return

    def _tool_label(self, name: str) -> str:
        registry = self._app._tool_registry
        if registry is None:
            return name
        if name in registry:
            return registry.get(name).spec.label
        return name


def format_confirm_args(arguments: object) -> str:
    """Compact single-line preview of tool arguments. JSON when
    serializable, str() otherwise; truncated so a long write_file
    payload can't blow out the row height."""
    try:
        text = json.dumps(arguments, ensure_ascii=False)
    except TypeError:
        text = str(arguments)
    text = text.replace("\n", " ")
    if len(text) > 60:
        text = f"{text[:57]}…"
    return text
