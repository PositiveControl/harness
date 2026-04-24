"""Tests for ToolLoopResult's per-turn summary properties
(harness-ywp.2 sub-commit B). These derive retrieval scores,
grounded citations, and tools_ran from the event log so cli.py can
write an audit row without walking the events itself."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.model.adapter import ChatMessage
from harness.orchestrator import run_tool_loop
from harness.tools import (
    ModelReply,
    ToolCall,
    ToolHit,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


@dataclass
class _ScriptedAdapter:
    replies: list[ModelReply]

    def complete_with_tools(
        self,
        messages: object,
        *,
        tools: object = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        return self.replies.pop(0)

    def complete(
        self, messages: object, *, max_tokens: int = 1024, temperature: float = 0.5
    ) -> str:
        return self.replies.pop(0).content


@dataclass
class _StructuredMemoryStub:
    """search_memory-shaped tool that returns a structured ToolResult
    with hits + grounded citations."""

    name: str = "search_memory"
    hits: tuple[ToolHit, ...] = ()
    citations: frozenset[str] = field(default_factory=frozenset)
    output: str = "result text"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="stub",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            tier="read",
        )

    def call(self, *, query: str) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            output=self.output,
            hits=self.hits,
            citations_grounded=self.citations,
        )


def test_tool_results_property_empty_on_no_tool_turn() -> None:
    adapter = _ScriptedAdapter(replies=[ModelReply(content="just text")])
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="hi")],
        ToolRegistry(),
    )
    assert result.tool_results == []
    assert result.retrieval_top_score is None
    assert result.citations_grounded == frozenset()
    assert result.tools_ran == frozenset()


def test_tool_results_captures_structured_result() -> None:
    stub = _StructuredMemoryStub(
        hits=(
            ToolHit(source="episodic", external_id="a", title="A", score=0.4),
            ToolHit(source="episodic", external_id="b", title="B", score=0.82),
        ),
        citations=frozenset({"§4-1-1", "TBL 4-1-2"}),
    )
    registry = ToolRegistry()
    registry.register(stub)

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="search_memory", arguments={"query": "clearance"}),),
            ),
            ModelReply(content="here is the answer"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="q")],
        registry,
    )
    assert len(result.tool_results) == 1
    assert result.retrieval_top_score == 0.82
    assert result.citations_grounded == frozenset({"§4-1-1", "TBL 4-1-2"})
    assert result.tools_ran == frozenset({"search_memory"})


def test_tool_results_unions_across_multiple_calls() -> None:
    """Two grounding calls in one turn — hits/citations merge via max/
    union so the audit row summarises the whole turn, not just the
    last tool call."""
    stub_a = _StructuredMemoryStub(
        name="search_memory",
        hits=(ToolHit(source="episodic", external_id="a", title="A", score=0.5),),
        citations=frozenset({"§4-1-1"}),
        output="a",
    )
    stub_b = _StructuredMemoryStub(
        name="search_facts",
        hits=(ToolHit(source="semantic", external_id="b", title="B", score=0.9),),
        citations=frozenset({"§5-5-4"}),
        output="b",
    )
    registry = ToolRegistry()
    registry.register(stub_a)
    registry.register(stub_b)

    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(
                    ToolCall(name="search_memory", arguments={"query": "q1"}),
                    ToolCall(name="search_facts", arguments={"query": "q2"}),
                ),
            ),
            ModelReply(content="final"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="q")],
        registry,
    )
    assert result.retrieval_top_score == 0.9
    assert result.citations_grounded == frozenset({"§4-1-1", "§5-5-4"})
    assert result.tools_ran == frozenset({"search_memory", "search_facts"})


def test_tool_results_ignore_str_returning_tool() -> None:
    """A tool that returns plain str (legacy contract) still appears in
    tools_ran but contributes no hits / citations. Confirms the
    backward-compat path from harness-ywp.4 reaches the summary."""

    @dataclass
    class _LegacyTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(
                name="plain",
                description="legacy",
                parameters={"type": "object", "properties": {}},
                tier="read",
            )

        def call(self) -> str:
            return "plain text output"

    registry = ToolRegistry()
    registry.register(_LegacyTool())
    adapter = _ScriptedAdapter(
        replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="plain", arguments={}),),
            ),
            ModelReply(content="done"),
        ]
    )
    result = run_tool_loop(
        adapter,
        [ChatMessage(role="user", content="q")],
        registry,
    )
    assert result.tools_ran == frozenset({"plain"})
    assert result.retrieval_top_score is None
    assert result.citations_grounded == frozenset()
