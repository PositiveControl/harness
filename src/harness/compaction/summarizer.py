from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from harness.model.adapter import ChatMessage

if TYPE_CHECKING:
    from harness.model.adapter import ModelAdapter
    from harness.store.transcript import TranscriptMessage


_SYSTEM_INSTRUCTIONS = """\
You are compacting a long chat transcript so the conversation can keep
running inside a shrinking context window. Write a single dense summary
of the turns below. Keep:

- Names, handles, file paths, commands, decisions, and refusals
- Open questions and unresolved threads
- Anything the next turn might need to reference

Drop chit-chat, filler, re-asked questions, and repeated tool output.

Use past tense. 10-25 short lines max. No headers, no bullets that
exceed one clause. Plain prose or tight dashes. Return only the summary
— no preamble, no explanation of what you changed."""


def _format_turn(turn: TranscriptMessage) -> str:
    speaker = turn.speaker or turn.role
    return f"[{turn.id}] {speaker} ({turn.role}): {turn.content.strip()}"


def build_summarizer_messages(
    turns: Sequence[TranscriptMessage],
    *,
    prior_summary: str | None = None,
) -> list[ChatMessage]:
    """Build the ChatMessage list the adapter will see. The system slot
    carries the summarization rubric; the user slot carries the prior
    summary (if any) and the turns to fold in. Kept as two messages so
    the base chat template formats correctly for every backend."""
    lines: list[str] = []
    if prior_summary:
        lines.append("Prior summary (extend this, don't discard it):")
        lines.append(prior_summary.strip())
        lines.append("")
    lines.append("Turns to add:")
    for turn in turns:
        lines.append(_format_turn(turn))
    return [
        ChatMessage(role="system", content=_SYSTEM_INSTRUCTIONS),
        ChatMessage(role="user", content="\n".join(lines)),
    ]


def summarize_turns(
    adapter: ModelAdapter,
    turns: Sequence[TranscriptMessage],
    *,
    prior_summary: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Ask the adapter to fold `turns` (and any prior summary) into a
    single new summary. Temperature is low — we want faithful
    compression, not creative reinterpretation."""
    if not turns:
        return prior_summary or ""
    messages = build_summarizer_messages(turns, prior_summary=prior_summary)
    return adapter.complete(messages, max_tokens=max_tokens, temperature=temperature).strip()
