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


def test_grounding_block_adds_preference_capture_nudge_when_remember_present(
    tmp_path: Path,
) -> None:
    """harness-7jda: when the `remember` tool is loaded (ab ops profile),
    the grounding block must carry a directive telling the model to call
    remember on durable-preference phrasings ('from now on', 'always',
    'remember to', 'keep in mind'). Otherwise the model acknowledges the
    preference in-chat but never persists it."""
    from harness.store.bd_adapter import BeadsAdapter  # noqa: F401  (imported for clarity)
    from harness.tools.ab_ops import RememberTool

    class _StubAdapter:
        def remember(self, _insight: str) -> None:
            return None

    registry = ToolRegistry()
    registry.register(RememberTool(adapter=_StubAdapter()))  # type: ignore[arg-type]

    block = _build_tool_grounding_block(registry, tmp_path)
    assert "remember" in block.lower()
    # Must list at least two of the canonical trigger phrases so the
    # model has concrete pattern-matches, not a vague directive.
    triggers = ("from now on", "always", "remember to", "keep in mind")
    matched = [t for t in triggers if t in block.lower()]
    assert len(matched) >= 2, f"expected preference triggers, found only {matched}"


def test_grounding_block_omits_preference_nudge_without_remember(tmp_path: Path) -> None:
    """When remember isn't loaded (non-ops profiles), the preference
    nudge doesn't appear — the model can't act on it anyway."""
    registry = _registry_with([ReadFileTool(root=tmp_path)])
    block = _build_tool_grounding_block(registry, tmp_path)
    assert "from now on" not in block.lower()
    assert "remember to" not in block.lower()


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
