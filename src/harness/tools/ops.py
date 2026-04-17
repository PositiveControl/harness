"""Self-operation tools — let Airton run the scribe and consolidator
on its own harness without the user dropping out of chat.

Both are write-tier (they mutate memory), and both need the same
stores + adapter wiring the memory CLI subcommands use. Tools carry
those deps as attributes so the CLI can configure them once at session
start."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from harness.consolidate import run_consolidation
from harness.scribe import run_scribe
from harness.tools.base import ToolSpec

if TYPE_CHECKING:
    from harness.character import Character
    from harness.model.adapter import ModelAdapter
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore
    from harness.store.transcript import Transcript


@dataclass
class ScribeSessionTool:
    """Run the scribe on a named session inline. Uses the same fcntl
    lock as the CLI subcommand, so concurrent runs serialize rather
    than racing the watermark."""

    adapter: ModelAdapter
    character: Character
    transcript: Transcript
    episodic_store: EpisodicStore
    semantic_store: SemanticStore
    default_user_id: str | None = None
    lock_dir: Path | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="scribe_session",
            description=(
                "Walk unprocessed transcript turns for a session and "
                "extract episodic + semantic memory candidates. "
                "Watermark-tracked — reruns are incremental. Pass "
                "`session` to pick which session; default is the "
                "current one. Use when the user asks you to 'remember "
                "what we just talked about' or after a meaningful "
                "session ends."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": ("Session id to scribe. Default: the active session."),
                    },
                    "user_id": {
                        "type": "string",
                        "description": (
                            "Override the user scope for candidates. Default: the current speaker."
                        ),
                    },
                },
                "required": [],
            },
            tier="write",
            display_name="Scribe session",
        )

    def call(
        self,
        *,
        session: str | None = None,
        user_id: str | None = None,
    ) -> str:
        # Defaults stitched in call() rather than captured in a closure so
        # changes to the tool's attributes (e.g. a long-running session that
        # later renames itself) still pick up.
        resolved_session = session or "local"
        resolved_user = user_id if user_id is not None else self.default_user_id
        summary = run_scribe(
            self.adapter,
            self.character,
            self.transcript,
            self.episodic_store,
            self.semantic_store,
            session_id=resolved_session,
            user_id=resolved_user,
            lock_dir=self.lock_dir,
        )
        return (
            f"scribed session={resolved_session}: "
            f"{summary.episodic_written} episodic + "
            f"{summary.semantic_written} semantic candidates "
            f"across {summary.windows} windows; "
            f"parse_errors={len(summary.parse_errors)}"
        )


@dataclass
class ConsolidateMemoryTool:
    """Run the consolidator inline — cluster near-duplicate working-tier
    records into consolidated-tier survivors and mark the rest
    superseded."""

    episodic_store: EpisodicStore
    semantic_store: SemanticStore
    default_threshold: float = 0.80

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="consolidate_memory",
            description=(
                "Cluster near-duplicate episodic memories and merge "
                "fact triples in the working tier. The winner is "
                "promoted to consolidated tier; losers are marked "
                "superseded (still visible for audit, dropped from "
                "retrieval). Use after a batch of scribe runs, or "
                "when the working tier feels cluttered."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": (
                            "Cosine-similarity floor for clustering episodic records. Default 0.80."
                        ),
                    },
                },
                "required": [],
            },
            tier="write",
            display_name="Consolidate memory",
        )

    def call(self, *, threshold: float | None = None) -> str:
        summary = run_consolidation(
            self.episodic_store,
            self.semantic_store,
            episodic_threshold=threshold or self.default_threshold,
        )
        return (
            f"consolidated: episodic "
            f"{summary.episodic_considered} considered, "
            f"{summary.episodic_clusters_merged} clusters merged, "
            f"{summary.episodic_superseded} superseded; "
            f"semantic {summary.semantic_considered} considered, "
            f"{summary.semantic_groups_merged} groups merged, "
            f"{summary.semantic_superseded} superseded"
        )
