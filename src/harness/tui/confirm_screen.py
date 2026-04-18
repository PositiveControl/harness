"""Write-tier tool confirmation modal for the Textual chat app
(harness-mz2).

The classic REPL asked `console.input("   approve? [y/N/always]: ")`
synchronously before running any write-tier tool. The TUI can't
block on stdin the same way — the event loop owns the terminal —
so this `ModalScreen` takes the same role: it pops over the chat
view when the orchestrator is about to run a write-tier tool and
returns one of three verdicts the worker then maps to confirm=True
/ confirm=False / confirm=True + session-level always-approve.

Keybindings mirror the classic prompt:
    y  — approve this call
    n  — decline this call (also bound to escape)
    a  — approve and auto-approve future calls to the same tool
         for the rest of this session
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:
    from harness.tools import ToolCall


# Public constants so the calling code doesn't stringly-type the
# three possible return values. `None` from dismiss()` (e.g. the
# app closes mid-prompt) is treated as a decline by the worker.
APPROVE = "approve"
DECLINE = "decline"
ALWAYS = "always"


class ConfirmToolScreen(ModalScreen[str]):
    """Modal approval prompt for a write-tier tool call.

    Dismisses with one of APPROVE / DECLINE / ALWAYS (literal
    strings, exported as constants above so callers don't have to
    string-match). The calling ChatApp maps the result onto the
    confirm(call) -> bool contract that run_tool_loop expects.

    Rendered as a compact panel centered over the chat view so the
    scrolling log stays visible behind it — the user can double-
    check the last tool output before approving."""

    CSS = """
    ConfirmToolScreen {
        align: center middle;
    }

    #dialog {
        padding: 1 2;
        width: auto;
        max-width: 80;
        height: auto;
        background: $panel;
        border: thick $warning;
    }

    #title {
        text-style: bold;
        color: $warning;
        padding-bottom: 1;
    }

    #args {
        color: $text-muted;
        padding-bottom: 1;
    }

    #hint {
        color: $text-muted;
        text-style: italic;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("y", "approve", "approve"),
        ("n", "decline", "decline"),
        ("a", "always", "always for session"),
        ("escape", "decline", "cancel"),
    ]

    def __init__(self, call: ToolCall, *, label: str) -> None:
        super().__init__()
        self._call = call
        self._label = label

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(f"🔧 Airton wants to {self._label}", id="title")
            # Pretty-print args so multi-line / long-arg write calls
            # are legible. dumps falls back to str() for non-JSON-
            # serializable values — tool args are free-form dicts,
            # so hedge.
            try:
                args_text = json.dumps(self._call.arguments, indent=2)
            except TypeError:
                args_text = str(self._call.arguments)
            yield Static(args_text, id="args")
            yield Static("[y] approve   [n] decline   [a] always", id="hint")

    def action_approve(self) -> None:
        self.dismiss(APPROVE)

    def action_decline(self) -> None:
        self.dismiss(DECLINE)

    def action_always(self) -> None:
        self.dismiss(ALWAYS)
