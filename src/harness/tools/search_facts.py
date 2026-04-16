from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.tools.base import ToolSpec

if TYPE_CHECKING:
    from harness.store.semantic import SemanticStore


@dataclass
class SearchFactsTool:
    """Let Airton query atomic facts on demand. Complements the
    automatic fact retrieval at turn start."""

    store: SemanticStore
    user_id: str | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_facts",
            description=(
                "Search atomic semantic facts about participants, the "
                "project, or recurring patterns. Returns (subject, "
                "predicate, object) triples with confidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Max results (default 5)",
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Filter out facts below this confidence",
                    },
                },
                "required": ["query"],
            },
            tier="read",
        )

    def call(
        self,
        *,
        query: str,
        k: int = 5,
        min_confidence: float = 0.0,
    ) -> str:
        hits = self.store.search(
            query,
            k=k,
            min_confidence=min_confidence,
            user_id=self.user_id,
        )
        if not hits:
            return "(no facts above threshold)"
        lines = []
        for fact, score in hits:
            lines.append(
                f"[{score:.3f}] {fact.subject} {fact.predicate} {fact.object} "
                f"(conf={fact.confidence:.2f}, tier={fact.tier})"
            )
        return "\n".join(lines)
