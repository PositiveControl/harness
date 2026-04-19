"""Tests for the ab bd-memory system-prompt injection (harness-hc9k).

The chat pipeline already auto-surfaces harness-local EpisodicStore /
SemanticStore memories (_render_memory_block). bd remember writes live
in a separate bd-owned store that the pipeline never injected, so
persisted preferences stayed dormant across sessions unless the model
proactively called the `memories` tool. These tests pin the new
`_render_ab_memories_block` helper and its failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.cli import _render_ab_memories_block
from harness.store.bd_adapter import BeadsAdapterError


@dataclass
class _FakeBd:
    """Minimal stand-in: exposes only the one attribute the helper reads."""

    output: str = ""
    raise_on_call: bool = False
    calls: int = 0

    def memories(self) -> str:
        self.calls += 1
        if self.raise_on_call:
            raise BeadsAdapterError("bd unavailable")
        return self.output


def test_ab_memories_block_renders_nonempty_output() -> None:
    """When bd memories returns stored insights, the block wraps them
    with a header explaining their role (durable preferences from
    earlier sessions) so the model knows to apply them automatically."""
    adapter = _FakeBd(
        output=(
            "Memories (1):\n\n"
            "  include-full-bead-id-postfix-when-displaying-tasks\n"
            "    Include full bead id postfix when displaying tasks."
        )
    )
    block = _render_ab_memories_block(adapter)  # type: ignore[arg-type]
    assert block is not None
    assert "Durable preferences" in block
    assert "include-full-bead-id-postfix" in block
    assert adapter.calls == 1


def test_ab_memories_block_returns_none_on_empty_store() -> None:
    """bd prints 'No memories stored.' when the store is empty. Helper
    must detect that and return None so the caller skips injection —
    the block header would be meaningless without content."""
    adapter = _FakeBd(output="No memories stored. Use 'bd remember \"insight\"' to add one.")
    assert _render_ab_memories_block(adapter) is None  # type: ignore[arg-type]


def test_ab_memories_block_returns_none_on_adapter_error() -> None:
    """If bd is transiently unavailable, the helper swallows the
    BeadsAdapterError and returns None — ab chat continues without
    memories rather than crashing the whole turn."""
    adapter = _FakeBd(raise_on_call=True)
    assert _render_ab_memories_block(adapter) is None  # type: ignore[arg-type]
