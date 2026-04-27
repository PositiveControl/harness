from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harness.retrieval.query_expander import NullQueryExpander, QueryExpander
from harness.tools.base import ToolHit, ToolResult, ToolSpec
from harness.tools.citations import extract_citations

if TYPE_CHECKING:
    from harness.store.episodic import EpisodicStore

# Per-hit body cap. Sized to fit one post-chunker-fold (harness-1s4)
# section — typical airton_c1 JO 7110.65 chunk is ~1,000 chars; the old
# 400-char cap sliced TBL 4-1-2 in half, dropping the CL/MH rows and
# letting the model fabricate them from priors (harness-5uq repro).
# Budget: k=12 hits * 1,200 chars ~= 14.4k chars (~3.6k tokens) of tool
# output, tolerable on 7B/32B Qwen and still well under compaction
# thresholds.
_BODY_CAP_CHARS = 1200


# Default number of hits returned. Bumped 5 -> 8 after harness-0tb to
# give the sanitize_fts_query phrase-clause fix room to reach the
# model: §13-1-2 landed at hybrid rank 6 on the 2026-04-24
# aircraft-to-aircraft repro, which k=5 silently truncated. Bumped
# 8 -> 12 (harness-6aa1) for low-cosine corpora like JO 7110.65 where
# raw cosines cluster at 0.016-0.033 — pre-retrieval (memories_threshold
# 0.5) drops everything, so the tool call is the only path and needs
# enough hits that lesson summary + multiple rule-body chunks both
# land. Was: model saw a single lesson summary at top, fell through to
# search_web; with the wider window the §-body chunks ride alongside
# the summary and the wrap-up answers from corpus.
_DEFAULT_K = 12


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
                        "description": f"Max results (default {_DEFAULT_K})",
                    },
                },
                "required": ["query"],
            },
            tier="read",
            display_name="Recall memory",
        )

    def call(self, *, query: str, k: int = _DEFAULT_K) -> ToolResult:
        # Expand lay-term queries into the section's full synonym field
        # so dense cosine + BM25 both see the jargon-space version of
        # the user's question. NullQueryExpander (the default) is
        # identity, so this call stays free for non-corpus characters.
        expanded = self.expander.expand(query)
        hits = self.store.search(expanded, k=k, user_id=self.user_id)
        if not hits:
            return ToolResult.text(
                self.spec.name,
                "(no memories matched — if this is about external facts, "
                "people, places, or live information, try search_web next; "
                "otherwise answer from general knowledge or ask a clarifying "
                "question)",
            )
        lines: list[str] = []
        for rec, score in hits:
            lines.append(f"[{score:.3f}] {rec.title}")
            if rec.principle:
                lines.append(f"  lesson: {rec.principle}")
            snippet = rec.body[:_BODY_CAP_CHARS]
            ellipsis = "…" if len(rec.body) > _BODY_CAP_CHARS else ""
            lines.append(f"  {snippet}{ellipsis}")
            lines.append("")
        # Surface structured hits so the per-turn audit log (harness-ywp.2)
        # and the low-confidence fallback hook (harness-ywp.3) can read
        # retrieval scores without regex-parsing the tool output text.
        tool_hits = tuple(
            ToolHit(
                source="episodic",
                external_id=rec.external_id,
                title=rec.title,
                score=score,
                principle=rec.principle,
            )
            for rec, score in hits
        )
        # Declare the citations this tool actually grounded (harness-ywp.5).
        # Extracted from title + body + principle of every hit — not from
        # the model's reply. The audit log and low-confidence fallback
        # use this to ask 'did any tool ground this §-reference?' as a
        # structured check instead of string matching on the reply.
        grounded = frozenset().union(
            *(
                extract_citations(
                    " ".join(part for part in (rec.title, rec.body, rec.principle) if part)
                )
                for rec, _ in hits
            )
        )
        return ToolResult(
            tool_name=self.spec.name,
            output="\n".join(lines).strip(),
            hits=tool_hits,
            citations_grounded=grounded,
        )
