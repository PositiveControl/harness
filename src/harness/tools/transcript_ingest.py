"""Transcript ingest — writes a recorded session (simulator export,
cockpit recording, manual notes) into episodic memory as working-tier
rows so later turns in the current (or a future) chat session can
`search_memory` for them.

Each turn becomes one episodic row, idempotent by
`external_id=transcript:<session_id>:<index>`. The `session_id`
parameter is the *transcribed* session (e.g. a MaxSim session ID),
distinct from the current chat session. The `user_id` is the trainee
the transcript belongs to — NOT inferred from the current speaker,
because an instructor typically ingests a trainee's session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from harness.tools.base import ToolSpec

if TYPE_CHECKING:
    from harness.store.episodic import EpisodicStore


_TITLE_SNIPPET_CHARS = 100


@dataclass
class TranscriptIngestTool:
    """Ingest a recorded pilot/controller (or any speaker-tagged)
    session transcript as JSON turns into episodic working memory.

    Substrate for the Phase-1 trainee-debrief flow (harness-cvi):
    after ingest, the debrief prompt can call `search_memory` scoped
    to the transcribed session to pull relevant turns by topic."""

    store: EpisodicStore

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="transcript_ingest",
            description=(
                "Ingest a recorded session transcript as JSON turns "
                "into episodic working memory. Each turn becomes one "
                "searchable row. Idempotent by session_id — re-ingest "
                "of the same transcript is a no-op. Use when the user "
                "provides a simulator export, cockpit recording, or "
                "manually transcribed session and wants later turns "
                "to reference it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {
                                    "type": "string",
                                    "description": ("Who spoke (e.g. 'TWR', 'N123AB', 'trainee')."),
                                },
                                "text": {
                                    "type": "string",
                                    "description": "What was said, verbatim.",
                                },
                                "timestamp": {
                                    "type": "string",
                                    "description": (
                                        "When — ISO-8601 or scenario-relative "
                                        "(e.g. '14:32:15' or 'T+00:03:22')."
                                    ),
                                },
                            },
                            "required": ["speaker", "text", "timestamp"],
                        },
                        "description": "Ordered list of transcript turns.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": (
                            "Identifier for the TRANSCRIBED session "
                            "(e.g. simulator session ID). Distinct from the "
                            "current chat session. Used for search scoping "
                            "and for idempotent re-ingest."
                        ),
                    },
                    "user_id": {
                        "type": "string",
                        "description": (
                            "Trainee ID — the user who owns these turns. "
                            "Required for per-user scoping; an instructor "
                            "ingesting a trainee's session supplies the "
                            "trainee's ID, not their own."
                        ),
                    },
                    "source_tag": {
                        "type": "string",
                        "description": (
                            "Source of the transcript: 'maxsim', "
                            "'beyondatc', 'manual', or a similar short "
                            "identifier. Recorded in the row's source "
                            "column for audit."
                        ),
                    },
                },
                "required": ["turns", "session_id", "user_id", "source_tag"],
            },
            tier="write",
            display_name="Ingest transcript",
        )

    def call(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        user_id: str,
        source_tag: str,
    ) -> str:
        if not turns:
            raise ValueError("turns must be non-empty")
        if not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not source_tag.strip():
            raise ValueError("source_tag must be non-empty")

        ingested = 0
        skipped = 0
        for i, turn in enumerate(turns):
            speaker = turn.get("speaker")
            text = turn.get("text")
            timestamp = turn.get("timestamp")
            if not speaker or not text or not timestamp:
                raise ValueError(f"turn {i} missing required field (speaker/text/timestamp)")

            ext_id = f"transcript:{session_id}:{i}"
            already_present = self.store.has(ext_id)
            snippet = str(text)[:_TITLE_SNIPPET_CHARS]
            ellipsis = "…" if len(str(text)) > _TITLE_SNIPPET_CHARS else ""
            self.store.ingest(
                external_id=ext_id,
                title=f"{speaker} @ {timestamp}: {snippet}{ellipsis}",
                body=str(text),
                tier="working",
                source=f"transcript:{source_tag}",
                session_id=session_id,
                user_id=user_id,
            )
            if already_present:
                skipped += 1
            else:
                ingested += 1

        return (
            f"ingested {ingested} turns into session {session_id!r} "
            f"(trainee={user_id}, source={source_tag}); "
            f"{skipped} already present"
        )
