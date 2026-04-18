"""Phase 1 scaffold for the Textual chat app (harness-1o8).

Shape is the end-state layout — scrolling output on top, a persistent
Input at the bottom, and a metrics Static between them — but the
wiring is deliberately minimal: input submissions echo back to the
log, there is no model, no memory, no tools, no persona. Later
phases (harness-29c onward) graft those layers on.

The point of shipping this scaffold alone is to lock in the layout,
CSS, and event plumbing so each subsequent phase can be reviewed in
isolation instead of as one monolithic Textual migration.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.widgets import Input, RichLog, Static


class ChatApp(App[None]):
    """Persistent-input chat TUI scaffold.

    Layout (bottom-docked for stability across terminal resizes):

        +----------------------------+
        |        RichLog             |  scrollable history
        |  (user ›, airton ›, …)     |
        +----------------------------+
        | ctx — · elapsed —          |  metrics strip
        +----------------------------+
        | > input prompt             |  persistent Input
        +----------------------------+

    Phase 1 echoes input back into the log so you can verify the
    event loop is alive. Later phases replace the echo in
    `on_input_submitted` with a worker that drives retrieval +
    persona + tool_loop."""

    CSS = """
    RichLog {
        height: 1fr;
        border: none;
        padding: 1 2 0 2;
        background: $background;
    }

    #metrics {
        dock: bottom;
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 2;
    }

    Input {
        dock: bottom;
        border: tall $accent;
        margin: 0;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "quit", "quit"),
        ("ctrl+d", "quit", "quit"),
    ]

    def __init__(self, *, character_name: str = "airton", speaker: str = "mark") -> None:
        super().__init__()
        self._character_name = character_name
        self._speaker = speaker

    def compose(self) -> ComposeResult:
        # Order matters here because Input + Static are dock:bottom;
        # Textual stacks docked widgets in reverse declaration order.
        # Declaring Input last means it lands above Static — we want
        # the opposite (metrics strip *above* the input prompt), so
        # yield Input first, Static second. RichLog fills whatever's
        # left.
        yield RichLog(id="output", wrap=True, markup=True, highlight=False)
        yield Input(id="prompt", placeholder="type a message… (ctrl+c to quit)")
        yield Static("ctx — · elapsed —", id="metrics")

    def on_mount(self) -> None:
        log = self.query_one("#output", RichLog)
        log.write(
            f"[dim]Chat with {self._character_name}. Phase 1 scaffold — "
            f"messages echo back. Model, tools, persona, and memory land "
            f"in later phases (see harness-29c).[/dim]"
        )
        self.query_one("#prompt", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        log = self.query_one("#output", RichLog)
        log.write(f"[bold cyan]{self._speaker} ›[/bold cyan] {text}")
        log.write(
            f"[bold green]{self._character_name} ›[/bold green] [dim](phase 1 echo)[/dim] {text}"
        )
        event.input.value = ""
