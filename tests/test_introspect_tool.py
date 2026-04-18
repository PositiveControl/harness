"""Tests for the IntrospectTool (harness-4jg).

Exercises each `scope` value against a hand-built IntrospectContext
with real stores (sqlite in tmp_path), the real Airton character, and
a stub adapter so we don't pay for MLX cold-load in the suite.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.character import load_character
from harness.cli_introspect import CommandInfo
from harness.config import settings
from harness.model.adapter import ChatMessage
from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore
from harness.tools import IntrospectContext, IntrospectTool, ReadFileTool, ToolRegistry


@dataclass
class _FakeEmbedder:
    id: str = "fake-introspect"
    dimension: int = 4

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            h = sum(ord(c) for c in text.lower())
            v = np.array([h % 7, h % 11, h % 13, h % 17], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            vectors.append(v / norm if norm > 0 else v)
        return np.stack(vectors)


class _StubAdapter:
    """Echo-shaped adapter exposing the introspect-visible attrs (id,
    context_window, repo, adapter_path). repo/adapter_path are only on
    concrete MLXAdapter in production; exposing them here verifies the
    tool's getattr-based fallback picks them up when present."""

    id = "test:stub"
    context_window = 12345
    repo = "mlx-community/Qwen-Test"
    adapter_path = None

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return "(stub reply)"


@pytest.fixture
def ctx(tmp_path: Path) -> IntrospectContext:
    repo = Path(__file__).resolve().parents[1]
    character = load_character(repo / "character" / "airton")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    episodic = EpisodicStore(tmp_path / "ep.sqlite", embedder=_FakeEmbedder())
    semantic = SemanticStore(tmp_path / "sem.sqlite", embedder=_FakeEmbedder())
    episodic.ingest(external_id="seed-1", title="a seed", body="body", tier="seed", source="yaml")
    episodic.ingest(
        external_id="mine",
        title="mine",
        body="body",
        tier="working",
        source="user",
        user_id="mark",
    )
    semantic.add(
        subject="mark",
        predicate="uses",
        object="mac",
        source="user",
        user_id="mark",
    )
    commands = (
        CommandInfo(path="chat", summary="Start a chat session."),
        CommandInfo(path="memory list", summary="List memory rows."),
    )
    ctx = IntrospectContext(
        registry=registry,
        adapter=_StubAdapter(),
        character=character,
        settings=settings,
        episodic=episodic,
        semantic=semantic,
        workspace=tmp_path,
        user_id="mark",
        commands=commands,
    )
    # Register introspect itself so scope=tools sees it reporting on itself.
    registry.register(IntrospectTool(context=ctx))
    return ctx


def test_scope_tools_lists_registered_tools(ctx: IntrospectContext) -> None:
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="tools")
    assert "read_file" in out
    assert "[read]" in out
    assert "introspect" in out


def test_scope_model_reports_adapter_surface(ctx: IntrospectContext) -> None:
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="model")
    assert "test:stub" in out
    assert "12345" in out  # context window
    assert "mlx-community/Qwen-Test" in out  # repo via getattr
    assert "LoRA" not in out  # adapter_path is None → section omitted


def test_scope_memory_reports_counts_and_user_scope(ctx: IntrospectContext) -> None:
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="memory")
    # Shared seed + mark's private row = 2 episodic visible to mark.
    assert "episodic: 2 active records" in out
    assert "(1 seed)" in out
    assert "semantic: 1 active facts" in out
    assert "shared + mark" in out


def test_scope_character_splits_voice_counts(ctx: IntrospectContext) -> None:
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="character")
    assert "name: airton" in out
    c = ctx.character
    expected = (
        f"voice samples: {c.canonical_voice_count} canonical + "
        f"{c.captured_voice_count} captured = {len(c.voice_samples)}"
    )
    assert expected in out


def test_scope_commands_renders_from_context(ctx: IntrospectContext) -> None:
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="commands")
    assert "harness chat — Start a chat session." in out
    assert "harness memory list — List memory rows." in out


def test_scope_all_concatenates_every_section(ctx: IntrospectContext) -> None:
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="all")
    for header in ("Tools", "Model", "Memory", "Character", "CLI commands"):
        assert header in out


def test_scope_unknown_returns_error_message(ctx: IntrospectContext) -> None:
    """Model hallucinating a bogus scope gets a self-correcting error
    line, not an exception (tool failures get fed back as tool-role
    messages)."""
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="universe")
    assert "unknown scope" in out.lower()
    assert "tools" in out  # lists valid scopes


def test_scope_memory_handles_disabled_stores(tmp_path: Path) -> None:
    """When episodic / semantic are None (no --memories / --facts flag),
    the scope=memory output stays informative instead of crashing."""
    repo = Path(__file__).resolve().parents[1]
    character = load_character(repo / "character" / "airton")
    registry = ToolRegistry()
    ctx = IntrospectContext(
        registry=registry,
        adapter=_StubAdapter(),
        character=character,
        settings=settings,
    )
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="memory")
    assert "episodic: (not enabled this session)" in out
    assert "semantic: (not enabled this session)" in out


def test_spec_shape(ctx: IntrospectContext) -> None:
    tool = IntrospectTool(context=ctx)
    spec = tool.spec
    assert spec.name == "introspect"
    assert spec.tier == "read"
    assert spec.parameters["required"] == ["scope"]
    enum = spec.parameters["properties"]["scope"]["enum"]
    assert set(enum) == {"tools", "model", "memory", "character", "commands", "all"}


def test_scope_tools_reports_capability_gaps(ctx: IntrospectContext) -> None:
    """harness-8is: when the registry lacks web / shell / git / memory-
    write / etc. tools, scope=tools spells out what the agent CANNOT
    do this session and hints how to enable it. The fixture registry
    only has read_file + introspect — so every gap should be listed."""
    tool = IntrospectTool(context=ctx)
    out = tool.call(scope="tools")
    assert "Capability gaps" in out
    assert "browse the web" in out
    assert "run shell commands" in out
    assert "inspect git history" in out
    assert "--tool-set" in out  # hint contains the remediation flag


def test_scope_tools_omits_gaps_section_when_everything_loaded(tmp_path: Path) -> None:
    """When every capability is covered, the gaps section is omitted
    rather than rendered empty — keeps the output terse on well-
    equipped sessions."""
    from harness.tools import (
        EditFileTool,
        GitStatusTool,
        RememberFactTool,
        ScribeSessionTool,
        SearchWebTool,
        ShellTool,
        WriteFileTool,
    )

    repo = Path(__file__).resolve().parents[1]
    character = load_character(repo / "character" / "airton")
    registry = ToolRegistry()
    semantic = SemanticStore(tmp_path / "sem.sqlite", embedder=_FakeEmbedder())
    episodic = EpisodicStore(tmp_path / "ep.sqlite", embedder=_FakeEmbedder())
    registry.register(SearchWebTool())
    registry.register(EditFileTool(root=tmp_path))
    registry.register(WriteFileTool(root=tmp_path))
    registry.register(ShellTool(cwd=tmp_path))
    registry.register(GitStatusTool(root=tmp_path))
    registry.register(RememberFactTool(store=semantic, user_id="mark", session_id="s"))

    class _FakeAdapter:
        id = "fake"
        context_window = 4096

    # Stub scribe to satisfy the curate-memory capability — don't
    # actually invoke it.
    from harness.character import load_character as _lc

    registry.register(
        ScribeSessionTool(
            adapter=_FakeAdapter(),  # type: ignore[arg-type]
            character=_lc(repo / "character" / "airton"),
            transcript=None,  # type: ignore[arg-type]
            episodic_store=episodic,
            semantic_store=semantic,
        )
    )
    ctx = IntrospectContext(
        registry=registry,
        adapter=_StubAdapter(),
        character=character,
        settings=settings,
    )
    out = IntrospectTool(context=ctx).call(scope="tools")
    assert "Capability gaps" not in out


def test_scope_model_reports_persona_and_router(tmp_path: Path) -> None:
    """harness-8is: the model scope reports persona on/off and the
    router label so the agent can answer 'are you using the rewriter'
    / 'what's fronting the tool loop' truthfully."""
    repo = Path(__file__).resolve().parents[1]
    character = load_character(repo / "character" / "airton")
    registry = ToolRegistry()
    ctx_on = IntrospectContext(
        registry=registry,
        adapter=_StubAdapter(),
        character=character,
        settings=settings,
        persona_active=True,
        router_id="grammar:Hermes-3-Llama-3.2-3B-4bit",
    )
    out_on = IntrospectTool(context=ctx_on).call(scope="model")
    assert "persona rewriter: on" in out_on
    assert "intent router: grammar:Hermes-3-Llama-3.2-3B-4bit" in out_on

    ctx_off = IntrospectContext(
        registry=registry,
        adapter=_StubAdapter(),
        character=character,
        settings=settings,
    )
    out_off = IntrospectTool(context=ctx_off).call(scope="model")
    assert "persona rewriter: off" in out_off
    assert "intent router: off" in out_off


def test_scope_memory_surfaces_retrieval_health(tmp_path: Path) -> None:
    """harness-8is: when a retrieval source has been disabled this
    session (it raised mid-turn and the CLI flipped the flag), the
    memory scope reports it so the agent can answer 'why didn't you
    recall X' honestly. Also asserts the all-ok path renders cleanly."""

    @dataclass
    class _FakeHealth:
        voice_ok: bool = True
        episodic_ok: bool = True
        semantic_ok: bool = True

    repo = Path(__file__).resolve().parents[1]
    character = load_character(repo / "character" / "airton")
    registry = ToolRegistry()

    ok = _FakeHealth()
    ctx_ok = IntrospectContext(
        registry=registry,
        adapter=_StubAdapter(),
        character=character,
        settings=settings,
        retrieval_health=ok,
    )
    out_ok = IntrospectTool(context=ctx_ok).call(scope="memory")
    assert "retrieval health: all sources ok" in out_ok

    degraded = _FakeHealth(episodic_ok=False, semantic_ok=False)
    ctx_bad = IntrospectContext(
        registry=registry,
        adapter=_StubAdapter(),
        character=character,
        settings=settings,
        retrieval_health=degraded,
    )
    out_bad = IntrospectTool(context=ctx_bad).call(scope="memory")
    assert "episodic" in out_bad
    assert "semantic" in out_bad
    assert "disabled this session" in out_bad


def test_retrieval_health_is_live_reference(tmp_path: Path) -> None:
    """Mutating the shared retrieval_health AFTER tool construction
    must show up in the next call — confirms we're not snapshotting."""

    @dataclass
    class _FakeHealth:
        voice_ok: bool = True
        episodic_ok: bool = True
        semantic_ok: bool = True

    repo = Path(__file__).resolve().parents[1]
    character = load_character(repo / "character" / "airton")
    registry = ToolRegistry()
    health = _FakeHealth()
    tool = IntrospectTool(
        context=IntrospectContext(
            registry=registry,
            adapter=_StubAdapter(),
            character=character,
            settings=settings,
            retrieval_health=health,
        )
    )
    assert "all sources ok" in tool.call(scope="memory")
    health.voice_ok = False
    assert "voice disabled this session" in tool.call(scope="memory")
