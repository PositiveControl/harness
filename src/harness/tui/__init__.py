"""Textual-based chat TUI. Optional; only loaded when the CLI is
invoked with --tui (see harness-56b).

Public surface: `ChatApp` — the Textual `App` subclass the CLI
launches in place of the classic REPL. Everything else in this
package is implementation detail that may move between phases.
"""

from harness.tui.chat_app import ChatApp

__all__ = ["ChatApp"]
