"""Slash-command operations for the TUI chat app.

Extracted from chat_app.py::ChatApp (harness-iwco). These are the
background tasks kicked off by `/compact`, `/scribe`, `/consolidate`,
plus the synchronous `/edit` (voice-capture) and `/retro` (read-only
bd query). The worker-thread methods hop back to the UI via
call_from_thread(_render_op_result) so they stay single-threaded at
the Textual reactive layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Input, RichLog

from harness.cli import _open_in_editor, _write_voice_capture
from harness.compaction import run_compaction
from harness.config import settings
from harness.consolidate import run_consolidation
from harness.scribe import run_scribe

if TYPE_CHECKING:
    from harness.tui.chat_app import ChatApp


class SlashOps:
    def __init__(self, app: ChatApp) -> None:
        self._app = app

    def run_edit_capture(self) -> None:
        """Inline /edit + /capture handler. Pulls the last exchange,
        opens Airton's reply in $EDITOR via app.suspend(), appends the
        edited text as a new captured voice sample. Runs on the UI
        thread because app.suspend() needs the main thread."""
        app = self._app
        log = app.query_one("#output", RichLog)
        tail = app._transcript.tail(app._session, limit=50)
        user_turns = [m for m in tail if m.role == "user"]
        assistant_turns = [m for m in tail if m.role == "assistant"]
        if not user_turns or not assistant_turns:
            log.write(
                Text(
                    "no exchange to capture yet — have a turn first, then /edit",
                    style="yellow",
                )
            )
            return
        prev_prompt = user_turns[-1].content
        prev_reply = assistant_turns[-1].content
        with app.suspend():
            edited = _open_in_editor(prev_reply)
        if edited is None:
            log.write(Text("(no changes — nothing captured)", style="dim"))
            return
        captured_path, sample_id, total = _write_voice_capture(
            prompt=prev_prompt,
            gold=edited,
            session=app._session,
            original=prev_reply,
        )
        line = Text()
        line.append("captured ", style="green")
        line.append(f"id={sample_id!r} → ", style="dim")
        line.append(str(captured_path.relative_to(settings.root)))
        line.append(f" (now {total} captured sample(s))", style="dim")
        log.write(line)

    def run_retro(self) -> None:
        """/retro — ab's thought-graph retrospective (read-only bd
        query, fast + safe on the UI thread)."""
        app = self._app
        log = app.query_one("#output", RichLog)
        if app._ab_adapter is None:
            log.write(
                Text(
                    "/retro: only available when character=airton_b",
                    style="yellow",
                )
            )
            return
        from harness.tools.ab_ops import RetroTool

        try:
            summary = RetroTool(app._ab_adapter).call(mode="summary")
        except Exception as exc:
            log.write(Text(f"/retro failed: {exc}", style="red"))
            return
        log.write(Text(summary, style="dim"))

    def run_compact_sync(self) -> None:
        """Worker-thread. /compact — fold older turns into a session
        summary. Hops back to the UI thread with call_from_thread."""
        app = self._app
        if app._compaction_store is None:
            app.call_from_thread(
                self._render,
                "compact: no compaction store wired for this session",
                True,
            )
            return
        try:
            outcome = run_compaction(
                app._adapter,
                app._transcript,
                app._compaction_store,
                session_id=app._session,
            )
        except Exception as exc:
            app.call_from_thread(
                self._render, f"compact failed: {type(exc).__name__}: {exc}", True
            )
            return
        msg = (
            f"✓ compacted {outcome.covered_turns} turn(s) up to id={outcome.new_up_to_turn_id}"
            if outcome.wrote
            else "compact: nothing new to fold (not enough turns past the watermark)"
        )
        app.call_from_thread(self._render, msg, False)

    def run_scribe_sync(self) -> None:
        """Worker-thread. /scribe — extract memory candidates from the
        session's unprocessed transcript window."""
        app = self._app
        missing = [
            name
            for name, obj in (
                ("memory_store", app._memory_store),
                ("semantic_store", app._semantic_store),
            )
            if obj is None
        ]
        if missing:
            app.call_from_thread(
                self._render,
                f"scribe: missing {', '.join(missing)} — "
                "launch with --memories > 0 and --facts > 0",
                True,
            )
            return
        try:
            summary = run_scribe(
                app._adapter,
                app._character,
                app._transcript,
                app._memory_store,  # type: ignore[arg-type]
                app._semantic_store,  # type: ignore[arg-type]
                session_id=app._session,
                user_id=app._scribe_user_id or app._speaker,
                lock_dir=app._scribe_lock_dir,
            )
        except Exception as exc:
            app.call_from_thread(
                self._render, f"scribe failed: {type(exc).__name__}: {exc}", True
            )
            return
        msg = (
            f"✓ scribed: {summary.episodic_written} episodic + "
            f"{summary.semantic_written} semantic candidates "
            f"across {summary.windows} window(s)"
        )
        app.call_from_thread(self._render, msg, False)

    def run_consolidate_sync(self) -> None:
        """Worker-thread. /consolidate — merge near-duplicate memories
        + facts by cosine-similarity clustering."""
        app = self._app
        if app._memory_store is None or app._semantic_store is None:
            app.call_from_thread(
                self._render, "consolidate: need both --memories and --facts stores", True
            )
            return
        try:
            summary = run_consolidation(app._memory_store, app._semantic_store)
        except Exception as exc:
            app.call_from_thread(
                self._render, f"consolidate failed: {type(exc).__name__}: {exc}", True
            )
            return
        msg = (
            f"✓ consolidated: episodic {summary.episodic_clusters_merged} cluster(s) merged, "
            f"{summary.episodic_superseded} superseded; semantic "
            f"{summary.semantic_groups_merged} group(s) merged, "
            f"{summary.semantic_superseded} superseded"
        )
        app.call_from_thread(self._render, msg, False)

    def kick(self, label: str, worker_name: str) -> None:
        """UI-thread. Shared helper for `/compact`, `/scribe`,
        `/consolidate`. Writes a dim '▸ running {label}…' line and
        kicks the named worker method on a thread so the UI stays
        responsive. Refuses to run concurrently with a model turn —
        MLX is not thread-safe across complete() calls."""
        app = self._app
        log = app.query_one("#output", RichLog)
        if app._state.is_busy:
            log.write(
                Text(
                    f"{label}: wait for the current turn to finish, then retry",
                    style="yellow",
                )
            )
            return
        log.write(Text(f"▸ running {label}…", style="dim magenta"))
        worker = getattr(self, worker_name)
        app.run_worker(worker, thread=True, exclusive=False, group="op")

    def _render(self, msg: str, error: bool) -> None:
        app = self._app
        log = app.query_one("#output", RichLog)
        style = "red" if error else "dim green"
        log.write(Text(msg, style=style))

    def focus_input(self) -> None:
        try:
            self._app.query_one("#prompt", Input).focus()
        except Exception:
            return
