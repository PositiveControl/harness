"""Tests for the tool grounding-block system-prompt fragment.

Covers harness-u71: when the introspect tool is loaded, the block
appends a one-line nudge telling the model to call `introspect`
instead of guessing its capabilities. Gated on tool presence so
introspect-less profiles (minimal, research) don't carry the line.
"""

from __future__ import annotations

from pathlib import Path

from harness.character import load_character
from harness.cli import _build_tool_grounding_block
from harness.config import settings
from harness.model.echo import EchoAdapter
from harness.tools import (
    IntrospectContext,
    IntrospectTool,
    ReadFileTool,
    ToolRegistry,
)


def _registry_with(tools: list[object]) -> ToolRegistry:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)  # type: ignore[arg-type]
    return registry


def test_grounding_block_omits_nudge_without_introspect(tmp_path: Path) -> None:
    """Registry without introspect: no introspection directive lands
    in the grounding block. This is the existing behaviour; the nudge
    shouldn't fire if the tool isn't even loaded."""
    registry = _registry_with([ReadFileTool(root=tmp_path)])
    block = _build_tool_grounding_block(registry, tmp_path)
    assert "introspect" not in block.lower()


def test_grounding_block_adds_nudge_when_introspect_present(tmp_path: Path) -> None:
    """harness-u71: with introspect loaded, the block carries a
    directive pointing the model at it for self-capability questions."""
    repo = Path(__file__).resolve().parents[1]
    character = load_character(repo / "character" / "airton")
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    ctx = IntrospectContext(
        registry=registry,
        adapter=EchoAdapter(),
        character=character,
        settings=settings,
    )
    registry.register(IntrospectTool(context=ctx))

    block = _build_tool_grounding_block(registry, tmp_path)
    assert "call the `introspect` tool" in block
    assert "do not guess" in block.lower()
    assert "introspect" in block.lower()
