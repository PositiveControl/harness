"""Metrics strip for the TUI chat app.

Extracted from chat_app.py::ChatApp (harness-iwco). Owns the
`#metrics` one-row Static: context meter + live thinking timer +
last-elapsed readout + pending-queue depth.

The helper recomputes nothing on the 250ms tick — `_state.ctx_used`
is refreshed once per turn in `recompute_ctx`; the tick only
re-renders the string. Exception-swallow on `count_tokens` is kept
here so a tokenizer hiccup can't wedge the whole turn.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.text import Text
from textual.css.query import NoMatches
from textual.widgets import Static

from harness.cli import _format_ctx_meter
from harness.model.adapter import approx_token_count

if TYPE_CHECKING:
    from harness.model.adapter import ModelAdapter
    from harness.tui.chat_app import ChatApp, _ChatAppState


class MetricsView:
    def __init__(self, app: ChatApp, state: _ChatAppState, adapter: ModelAdapter) -> None:
        self._app = app
        self._state = state
        self._adapter = adapter

    def refresh(self) -> None:
        # harness-k1kx: the 250ms tick outlives the widget tree. On
        # shutdown Textual prunes the DOM while timers are still
        # scheduled, so a tick can land after `#metrics` is gone —
        # query_one then raises NoMatches, which Textual stores as the
        # app's exception and re-raises on exit (a traceback on quit
        # for a real user; a load-sensitive flake in the test suite).
        # No strip to draw on is a no-op, not an error.
        try:
            metrics = self._app.query_one("#metrics", Static)
        except NoMatches:
            return
        meter = _format_ctx_meter(self._state.ctx_used, self._adapter.context_window) or "ctx —"
        parts = [meter]
        if self._state.turn_started_at is not None:
            elapsed = time.monotonic() - self._state.turn_started_at
            parts.append(f"thinking {elapsed:.1f}s")
        elif self._state.last_elapsed is not None:
            parts.append(f"last {self._state.last_elapsed:.1f}s")
        else:
            parts.append("idle")
        if self._state.pending_prompts:
            parts.append(f"queued {len(self._state.pending_prompts)}")
        # layout=False: the #metrics strip is height:1 (fixed), so we
        # don't need a full layout pass 4×/sec — and a 250ms layout
        # cycle was measuring hidden height:auto siblings and letting
        # their computed height drift upward (harness-7xz0).
        metrics.update(Text.from_markup(" · ".join(parts)), layout=False)

    def recompute_ctx(self) -> None:
        count_fn = getattr(self._adapter, "count_tokens", None)
        try:
            if callable(count_fn):
                self._state.ctx_used = int(count_fn(self._state.history))
            else:
                self._state.ctx_used = approx_token_count(self._state.history)
        except Exception:
            self._state.ctx_used = approx_token_count(self._state.history)
