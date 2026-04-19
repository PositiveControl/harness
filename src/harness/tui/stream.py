"""Per-turn stream buffer + live preview for the TUI chat app.

Extracted from chat_app.py::ChatApp (harness-iwco). The flow:

- `feed` appends a token delta to `_state.stream_buffer` and updates the
  `#stream_preview` Static below the main log.
- `flush` commits the buffered text to the main `#output` RichLog as a
  rich.Markdown block, then clears the preview. Called at round
  boundaries (tool call emitted, truncated_retry, turn end).
- `hide_preview` clears the preview without committing — used on
  truncated_retry / bail_retry where the partial is discarded.

The helper holds weak references by always calling `self._app.query_one`
at use-time, so a teardown-mid-render doesn't raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import RichLog, Static

if TYPE_CHECKING:
    from harness.tui.chat_app import ChatApp, _ChatAppState


def _make_assistant_badge(speaker: str) -> Text:
    line = Text()
    line.append(f" {speaker} › ", style="reverse bold green")
    line.append(" ")
    return line


class StreamView:
    """Owns the per-turn stream buffer state + preview/log rendering."""

    def __init__(self, app: ChatApp, state: _ChatAppState, character_name: str) -> None:
        self._app = app
        self._state = state
        self._character_name = character_name

    def feed(self, delta: str) -> None:
        if not delta:
            return
        self._state.stream_buffer += delta
        try:
            preview = self._app.query_one("#stream_preview", Static)
        except Exception:
            return
        preview.update(Text(self._state.stream_buffer))
        preview.add_class("-visible")

    def flush(self) -> None:
        text = self._state.stream_buffer
        self._state.stream_buffer = ""
        self.hide_preview()
        if not text.strip():
            return
        log = self._app.query_one("#output", RichLog)
        if self._state.stream_first_chunk:
            log.write(_make_assistant_badge(self._character_name))
            self._state.stream_first_chunk = False
        log.write(Markdown(text, code_theme="monokai"))

    def hide_preview(self) -> None:
        try:
            preview = self._app.query_one("#stream_preview", Static)
        except Exception:
            return
        preview.update("")
        preview.remove_class("-visible")

    def reset(self) -> None:
        """Drop any in-flight buffer + hide preview. Used on interrupt
        and on truncated_retry / bail_retry where the partial is
        superseded by the next round's output."""
        self._state.stream_buffer = ""
        self._state.stream_first_chunk = True
        self.hide_preview()
