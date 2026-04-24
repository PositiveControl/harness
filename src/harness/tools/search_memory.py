from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harness.retrieval.query_expander import NullQueryExpander, QueryExpander
from harness.tools.base import ToolSpec

if TYPE_CHECKING:
    from harness.store.episodic import EpisodicStore


@dataclass
class SearchMemoryTool:
    """Let Airton query its own episodic memory on demand. Complements
    the automatic top-K retrieval at turn start — useful when Airton
    wants more context than was injected, or wants to look up something
    specific by lesson/topic.

    When a `QueryExpander` is supplied, the incoming query is augmented
    with section-anchored lay-term synonyms before it hits the store
    (harness-ajn). Identity-preserving on queries that don't trigger a
    synonym match, so non-corpus characters (that ship a
    `NullQueryExpander`) pay no cost."""

    store: EpisodicStore
    user_id: str | None = None
    expander: QueryExpander = field(default_factory=NullQueryExpander)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_memory",
            description=(
                "Search episodic memory for relevant past events and "
                "lessons. Use when the pre-retrieved memories didn't "
                "cover what you need, or when you want more detail on "
                "a specific situation."
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
                },
                "required": ["query"],
            },
            tier="read",
            display_name="Recall memory",
        )

    def call(self, *, query: str, k: int = 5) -> str:
        # Expand lay-term queries into the section's full synonym field
        # so dense cosine + BM25 both see the jargon-space version of
        # the user's question. NullQueryExpander (the default) is
        # identity, so this call stays free for non-corpus characters.
        expanded = self.expander.expand(query)
        hits = self.store.search(expanded, k=k, user_id=self.user_id)
        if not hits:
            return (
                "(no memories matched — if this is about external facts, "
                "people, places, or live information, try search_web next; "
                "otherwise answer from general knowledge or ask a clarifying "
                "question)"
            )
        lines: list[str] = []
        for rec, score in hits:
            lines.append(f"[{score:.3f}] {rec.title}")
            if rec.principle:
                lines.append(f"  lesson: {rec.principle}")
            snippet = rec.body[:400]
            ellipsis = "…" if len(rec.body) > 400 else ""
            lines.append(f"  {snippet}{ellipsis}")
            lines.append("")
        return "\n".join(lines).strip()
