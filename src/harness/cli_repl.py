"""Classic-REPL helpers for `harness chat`.

Extracted from cli.chat() (harness-0n1r). Covers:

- `ContextMeter` — per-turn context-token measurement + header print
  + automatic compaction when the meter crosses `compact_at`.
- `handle_retro_slash` — /retro summary + optional insight record.
- `handle_edit_slash` — /edit + /capture: open $EDITOR on the last
  reply, write the edited text as a new voice sample.

The chat() command still owns the REPL loop shape; these helpers let
it stop juggling closure state and make the individual flows
testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from harness.compaction import run_compaction, should_compact
from harness.config import settings
from harness.model.adapter import ChatMessage, count_tokens
from harness.tools.ab_ops import RetroTool, build_resume_summary

if TYPE_CHECKING:
    from rich.console import Console

    from harness.character import Character
    from harness.compaction import CompactionStore
    from harness.model.adapter import ModelAdapter
    from harness.store.bd_adapter import BeadsAdapter
    from harness.store.transcript import Transcript


@dataclass
class ContextMeter:
    """Per-turn context-token measurement + optional auto-compaction.

    Owns the closures that used to live inside cli.chat(): history
    load through the compaction pointer, token count across the
    pending system + history, header print, and the 'ctx crossed
    threshold → fold older turns' path.
    """

    adapter: ModelAdapter
    character: Character
    transcript: Transcript
    compaction_store: CompactionStore | None
    session: str
    console: Console

    def load_history(self) -> tuple[ChatMessage | None, list[ChatMessage]]:
        """Return (optional summary-system-message, turns-since-pointer).
        When a compaction summary exists, turns before the pointer are
        represented by the summary only; raw rows stay in the transcript
        for audit but never hit the model."""
        from harness.cli import _decode_transcript_message

        record = (
            self.compaction_store.latest_for_session(self.session)
            if self.compaction_store
            else None
        )
        if record is not None:
            summary_msg = ChatMessage(
                role="system",
                content=(
                    "Earlier conversation in this session (summarized; "
                    f"{record.covered_turns} turns folded in):\n\n{record.summary}"
                ),
            )
            rows = self.transcript.fetch_after(self.session, after_id=record.up_to_turn_id)
            return summary_msg, [_decode_transcript_message(m) for m in rows]
        rows = self.transcript.tail(self.session, limit=50)
        return None, [_decode_transcript_message(m) for m in rows]

    def measure(self) -> int:
        """Estimate tokens for what the NEXT turn will start with:
        system prompt + optional compaction summary + history since the
        pointer. Undercounts slightly (retrieved memories/facts add
        text per turn) but tracks transcript growth accurately."""
        baseline_system = ChatMessage(
            role="system", content=self.character.system_prompt(now=date.today())
        )
        summary_msg, history_msgs = self.load_history()
        msgs: list[ChatMessage] = [baseline_system]
        if summary_msg is not None:
            msgs.append(summary_msg)
        msgs.extend(history_msgs)
        return count_tokens(self.adapter, msgs)

    def print_ctx(self) -> None:
        from harness.cli import _format_ctx_meter

        used = self.measure()
        meter = _format_ctx_meter(used, self.adapter.context_window)
        if meter:
            self.console.print(meter)

    def maybe_compact(
        self,
        *,
        compact_at: float,
        compact_keep_recent: int,
        thinking: object,
        ab_adapter: BeadsAdapter | None,
    ) -> None:
        """Fire the compactor when the context meter crosses
        `compact_at × context_window`. No-op when the store is
        unconfigured or the threshold hasn't been reached."""
        if self.compaction_store is None:
            return
        used = self.measure()
        if not should_compact(
            used_tokens=used,
            context_window=self.adapter.context_window,
            threshold_pct=compact_at,
        ):
            return
        self.console.print(
            f"[dim]compacting history (ctx {used / 1000:.1f}k, threshold "
            f"{compact_at * 100:.0f}%)…[/dim]"
        )
        thinking.start()  # type: ignore[attr-defined]
        try:
            outcome = run_compaction(
                self.adapter,
                self.transcript,
                self.compaction_store,
                session_id=self.session,
                keep_recent=compact_keep_recent,
            )
        finally:
            thinking.stop()  # type: ignore[attr-defined]
        if outcome.wrote:
            self.console.print(
                f"[dim]compacted {outcome.covered_turns} turns "
                f"(pointer → #{outcome.new_up_to_turn_id})[/dim]"
            )
            if ab_adapter is not None:
                # Post-compaction resume — context window shrank,
                # reprint thought-graph state so the anchor is fresh.
                self.console.print(f"[dim]{build_resume_summary(ab_adapter)}[/dim]")
        else:
            self.console.print(
                "[yellow]compaction skipped — nothing qualified "
                "(fewer turns than keep-recent, or model returned empty).[/yellow]"
            )


def handle_retro_slash(ab_adapter: BeadsAdapter | None, console: Console) -> None:
    """/retro — summary first; then prompt for an optional insight to
    persist via RetroTool mode=record. Empty response skips recording."""
    if ab_adapter is None:
        console.print(
            "[yellow]/retro is only available when ab's bd adapter "
            "is configured (character=airton_b).[/yellow]"
        )
        return
    retro = RetroTool(ab_adapter)
    console.print(f"[dim]{retro.call(mode='summary')}[/dim]")
    insight = console.input(
        "[bold cyan]insight to record (blank to skip) › [/bold cyan]"
    ).strip()
    if insight:
        console.print(f"[dim]{retro.call(mode='record', insight=insight)}[/dim]")


def handle_edit_slash(
    *, transcript: Transcript, session: str, console: Console
) -> None:
    """/edit + /capture — open $EDITOR on the last assistant reply,
    write the edited text as a new captured voice sample paired with
    the preceding user prompt. Closes the loop between 'reply was
    off-register' and 'new training sample' without leaving chat."""
    from harness.cli import _open_in_editor, _write_voice_capture

    history_tail = transcript.tail(session, limit=50)
    user_turns = [m for m in history_tail if m.role == "user"]
    assistant_turns = [m for m in history_tail if m.role == "assistant"]
    if not user_turns or not assistant_turns:
        console.print(
            "[yellow]no exchange to capture yet — have a turn first, "
            "then run /edit.[/yellow]"
        )
        return
    prev_prompt = user_turns[-1].content
    prev_reply = assistant_turns[-1].content
    edited = _open_in_editor(prev_reply)
    if edited is None:
        console.print("[dim](no changes — nothing captured)[/dim]")
        return
    captured_path, sample_id, total = _write_voice_capture(
        prompt=prev_prompt,
        gold=edited,
        session=session,
        original=prev_reply,
    )
    console.print(
        f"[green]captured[/green] id={sample_id!r} "
        f"→ {captured_path.relative_to(settings.root)} "
        f"(now {total} captured sample(s))"
    )
