"""TurnService — the shared grounded-turn path (harness-fl313).

Three turns through the service: a plain no-tool turn, a retrieval-backed
turn (episodic + semantic blocks reach the system prompt), and a
tool-backed turn (the loop runs, the exchange persists). The service is
exercised headless — no console, no renderer — which is exactly how web
and daemon callers use it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from harness.character import load_character
from harness.cli import _RetrievalState
from harness.config import settings
from harness.model.adapter import ChatMessage
from harness.orchestrator import DEFAULT_ROUND_MAX_TOKENS
from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore
from harness.store.transcript import Transcript
from harness.tools import ModelReply, ReadFileTool, ToolCall, ToolRegistry, ToolSpec
from harness.turn import TurnContext, TurnService


@dataclass
class _FakeEmbedder:
    id: str = "fake"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else v)
        return np.stack(out)


@dataclass
class _CapturingAdapter:
    """Records the system message it sees and returns canned replies.
    No `stream` method, so the service's headless path calls complete()."""

    id: str = "cap"
    context_window: int = 8192
    reply: str = "FINAL REPLY"
    seen_system: list[str] = field(default_factory=list)
    seen_max_tokens: list[int] = field(default_factory=list)
    tool_replies: list[ModelReply] = field(default_factory=list)

    def _record(self, messages: Iterable[ChatMessage]) -> None:
        for m in messages:
            if m.role == "system":
                self.seen_system.append(m.content)

    def complete(
        self, messages: Iterable[ChatMessage], *, max_tokens: int = 512, temperature: float = 0.7
    ) -> str:
        self._record(messages)
        return self.reply

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        self._record(messages)
        self.seen_max_tokens.append(max_tokens)
        return self.tool_replies.pop(0) if self.tool_replies else ModelReply(content=self.reply)


def _ctx(tmp_path: Path, adapter: Any, **over: Any) -> tuple[TurnContext, Transcript]:
    character = load_character(settings.character_path)
    transcript = Transcript(tmp_path / "t.sqlite")
    base: dict[str, Any] = {
        "character": character,
        "adapter": adapter,
        "transcript": transcript,
        "load_history": lambda: (None, []),
        "speaker": "mark",
        "session": "s1",
        "channel": "cli",
        "retrieval_state": _RetrievalState(),
    }
    base.update(over)
    return TurnContext(**base), transcript


def test_no_tool_turn_persists_and_returns(tmp_path: Path) -> None:
    adapter = _CapturingAdapter(reply="hello back")
    ctx, transcript = _ctx(tmp_path, adapter)
    result = TurnService(ctx).run_turn("hi there")

    assert result.reply == "hello back"
    assert result.loop_result is None
    assert result.streamed is False

    rows = transcript.tail("s1", limit=10)
    roles = [(r.role, r.speaker) for r in rows]
    assert ("user", "mark") in roles
    assert ("assistant", ctx.character.name) in roles
    # No tool grounding block when there's no registry.
    assert adapter.seen_system
    assert "TOOL-USE RULES" not in adapter.seen_system[-1]


def test_retrieval_backed_turn_injects_memory_and_fact_blocks(tmp_path: Path) -> None:
    emb = _FakeEmbedder()
    epi = EpisodicStore(tmp_path / "e.sqlite", embedder=emb)
    sem = SemanticStore(tmp_path / "s.sqlite", embedder=emb)
    epi.ingest(
        external_id=None,
        title="Mark likes terse answers",
        body="Mark wants dense, decision-shaped replies, not hedged ones.",
        principle="voice",
        user_id="mark",
    )
    sem.add(
        subject="Mark",
        predicate="prefers",
        object="terse decision-shaped answers",
        source="test",
        user_id="mark",
    )
    adapter = _CapturingAdapter(reply="ok")
    ctx, _ = _ctx(
        tmp_path,
        adapter,
        memory_store=epi,
        semantic_store=sem,
        memories=3,
        memories_threshold=0.0,
        facts=5,
        facts_threshold=0.0,
    )
    try:
        TurnService(ctx).run_turn("what does mark like")
    finally:
        epi.close()
        sem.close()

    sysmsg = adapter.seen_system[-1]
    # Episodic memory block reached the prompt...
    assert "Relevant past experience" in sysmsg
    assert "terse" in sysmsg.lower()
    # ...and the semantic fact block too.
    assert "Relevant facts I know" in sysmsg


def test_tool_backed_turn_runs_loop_and_persists_exchange(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("file contents here")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    adapter = _CapturingAdapter(
        tool_replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "f.txt"}),),
            ),
            ModelReply(content="the file says: file contents here"),
        ]
    )
    ctx, transcript = _ctx(tmp_path, adapter, registry=registry, workspace_path=tmp_path)
    result = TurnService(ctx).run_turn("read f.txt")

    assert result.loop_result is not None
    assert result.reply == "the file says: file contents here"
    # The tool-call exchange (incl. the tool result row) was persisted.
    rows = transcript.tail("s1", limit=20)
    blob = "\n".join(r.content for r in rows)
    assert "file contents here" in blob
    # Tool grounding block was injected because a registry was present.
    assert "TOOL-USE RULES" in adapter.seen_system[-1]


def test_round_budget_defaults_to_the_orchestrator_default(tmp_path: Path) -> None:
    """harness-gebo5: an unset `round_max_tokens` keeps the tuned local-MLX
    operating point, so nothing about the default chat path moves."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    adapter = _CapturingAdapter(reply="done")
    ctx, _ = _ctx(tmp_path, adapter, registry=registry, workspace_path=tmp_path)
    TurnService(ctx).run_turn("hi")

    assert adapter.seen_max_tokens == [DEFAULT_ROUND_MAX_TOKENS]


def test_round_budget_reaches_the_tool_loop(tmp_path: Path) -> None:
    """harness-gebo5: `--max-tokens` has to land on the adapter call, not
    just on the dataclass. The wrap-up rounds get the same budget — the
    round that writes a file is often the wrap-up one, and a wrap-up cap
    below the file body reproduces harness-4s6fv."""
    (tmp_path / "f.txt").write_text("file contents here")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    adapter = _CapturingAdapter(
        tool_replies=[
            ModelReply(
                content="",
                tool_calls=(ToolCall(name="read_file", arguments={"path": "f.txt"}),),
            ),
            ModelReply(content="the file says: file contents here"),
        ]
    )
    ctx, _ = _ctx(
        tmp_path,
        adapter,
        registry=registry,
        workspace_path=tmp_path,
        round_max_tokens=8192,
    )
    TurnService(ctx).run_turn("read f.txt")

    # Round 0 (tool call) and the post-tool wrap-up round both honor it.
    assert adapter.seen_max_tokens == [8192, 8192]
