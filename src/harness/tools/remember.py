"""Write-tier memory tools — let the model commit things to the
semantic / episodic stores from within a turn, rather than waiting
for the scribe's batch pass.

Both tools attribute writes to `tool:remember_*` in the store's
`source` column, and scope to the active speaker so they don't
contaminate shared memory or other users' relationship memory.
Confirmation is required (write-tier), so the user sees exactly
what's being recorded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.tools.base import ToolSpec

if TYPE_CHECKING:
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore


@dataclass
class RememberFactTool:
    """Record a (subject, predicate, object) triple the model learned
    this turn. Goes into the working tier — the consolidator will
    later de-dup against existing facts."""

    store: SemanticStore
    user_id: str | None = None
    session_id: str | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="remember_fact",
            description=(
                "Permanently record a fact the user just told you as a "
                "(subject, predicate, object) triple. Use when the user "
                "shares a durable preference, relationship, or project "
                "detail that should carry into future sessions. Do NOT "
                "use for things the user is telling you to do right now "
                "(those are actions, not facts). Triples are stored in "
                "working memory; the consolidator later merges duplicates."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": (
                            "Who/what the fact is about (e.g. 'mark', 'project:harness', 'mac')."
                        ),
                    },
                    "predicate": {
                        "type": "string",
                        "description": (
                            "Relation or attribute (e.g. 'prefers', 'uses', 'has_ram')."
                        ),
                    },
                    "object": {
                        "type": "string",
                        "description": ("The value. Keep it short and concrete."),
                    },
                    "confidence": {
                        "type": "number",
                        "description": (
                            "How certain you are, 0.0–1.0. Default 0.9. Use <0.7 for guesses."
                        ),
                    },
                },
                "required": ["subject", "predicate", "object"],
            },
            tier="write",
            display_name="Remember fact",
        )

    def call(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.9,
    ) -> str:
        if not subject or not predicate or not object:
            raise ValueError("subject, predicate, and object must all be non-empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        fact = self.store.add(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            source="tool:remember_fact",
            session_id=self.session_id,
            user_id=self.user_id,
            tier="working",
        )
        scope = f"user={self.user_id}" if self.user_id else "shared"
        return (
            f"recorded fact #{fact.id}: ({subject} {predicate} {object}) "
            f"conf={confidence:.2f} [{scope}]"
        )


@dataclass
class RememberEventTool:
    """Record an episodic memory — a narrative snippet with optional
    principle and tags. Goes into the working tier for later
    consolidation."""

    store: EpisodicStore
    user_id: str | None = None
    session_id: str | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="remember_event",
            description=(
                "Permanently record a narrative memory of what just "
                "happened or what the user shared. Use for stories, "
                "decisions, lessons, context that isn't cleanly a "
                "subject-predicate-object triple. Provide a short "
                "title and a body. Optional `principle` is the lesson "
                "in one line; `tags` are free-form strings. Goes into "
                "working memory; the consolidator later merges "
                "near-duplicates."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "One-line summary.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The narrative itself, 1–5 sentences.",
                    },
                    "principle": {
                        "type": "string",
                        "description": "Optional — the lesson in one line.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional free-form tags.",
                    },
                },
                "required": ["title", "body"],
            },
            tier="write",
            display_name="Remember event",
        )

    def call(
        self,
        *,
        title: str,
        body: str,
        principle: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        if not title.strip() or not body.strip():
            raise ValueError("title and body must both be non-empty")
        rec = self.store.ingest(
            external_id=None,
            title=title,
            body=body,
            principle=principle,
            tags=tags or (),
            tier="working",
            source="tool:remember_event",
            session_id=self.session_id,
            user_id=self.user_id,
        )
        scope = f"user={self.user_id}" if self.user_id else "shared"
        return f"recorded memory #{rec.id}: {title!r} [{scope}]"
