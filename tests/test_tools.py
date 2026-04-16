from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from harness.store.episodic import EpisodicStore
from harness.store.semantic import SemanticStore
from harness.tools import (
    ReadFileTool,
    SearchFactsTool,
    SearchMemoryTool,
    ShellTool,
    Tool,
    ToolRegistry,
    ToolSpec,
    WriteFileTool,
)


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


# ---------- ToolRegistry ----------


def test_registry_registers_and_dispatches(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool = ReadFileTool(root=tmp_path)
    registry.register(tool)

    assert "read_file" in registry
    assert registry.names() == ["read_file"]
    assert registry.get("read_file") is tool

    (tmp_path / "hi.txt").write_text("hello")
    result = registry.call("read_file", {"path": "hi.txt"})
    assert result.success
    assert result.output == "hello"


def test_registry_refuses_duplicate_registration(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ReadFileTool(root=tmp_path))


def test_registry_unknown_tool_returns_failed_result() -> None:
    registry = ToolRegistry()
    result = registry.call("nonexistent", {})
    assert not result.success
    assert result.error == "unknown_tool"


def test_registry_catches_tool_exceptions(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    result = registry.call("read_file", {"path": "does-not-exist.txt"})
    assert not result.success
    assert "FileNotFoundError" in (result.error or "")


def test_every_tool_satisfies_protocol(tmp_path: Path) -> None:
    """Structural check — every concrete tool implements the Tool protocol."""
    store_ep = EpisodicStore(tmp_path / "e.sqlite", embedder=_FakeEmbedder())
    store_sem = SemanticStore(tmp_path / "s.sqlite", embedder=_FakeEmbedder())
    try:
        assert isinstance(ReadFileTool(root=tmp_path), Tool)
        assert isinstance(WriteFileTool(root=tmp_path), Tool)
        assert isinstance(ShellTool(), Tool)
        assert isinstance(SearchMemoryTool(store=store_ep), Tool)
        assert isinstance(SearchFactsTool(store=store_sem), Tool)
    finally:
        store_ep.close()
        store_sem.close()


# ---------- ReadFileTool ----------


def test_read_file_returns_contents(tmp_path: Path) -> None:
    tool = ReadFileTool(root=tmp_path)
    (tmp_path / "file.txt").write_text("sample contents")
    assert tool.call(path="file.txt") == "sample contents"


def test_read_file_refuses_path_traversal(tmp_path: Path) -> None:
    tool = ReadFileTool(root=tmp_path)
    with pytest.raises(ValueError, match="escapes workspace root"):
        tool.call(path="../secret.txt")


def test_read_file_truncates_large_files(tmp_path: Path) -> None:
    tool = ReadFileTool(root=tmp_path, max_bytes=100)
    (tmp_path / "big.txt").write_text("x" * 500)
    result = tool.call(path="big.txt")
    assert "truncated" in result
    # 100 bytes of 'x' + truncation notice
    assert result.count("x") == 100


def test_read_file_raises_for_missing(tmp_path: Path) -> None:
    tool = ReadFileTool(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        tool.call(path="missing.txt")


# ---------- WriteFileTool ----------


def test_write_file_creates_file(tmp_path: Path) -> None:
    tool = WriteFileTool(root=tmp_path)
    result = tool.call(path="out.txt", content="hello world")
    assert "11 chars" in result
    assert (tmp_path / "out.txt").read_text() == "hello world"


def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    tool = WriteFileTool(root=tmp_path)
    tool.call(path="nested/dir/file.txt", content="data")
    assert (tmp_path / "nested" / "dir" / "file.txt").read_text() == "data"


def test_write_file_refuses_path_traversal(tmp_path: Path) -> None:
    tool = WriteFileTool(root=tmp_path)
    with pytest.raises(ValueError, match="escapes workspace root"):
        tool.call(path="../evil.txt", content="nope")


# ---------- ShellTool ----------


def test_shell_captures_stdout() -> None:
    tool = ShellTool()
    result = tool.call(cmd="echo hello")
    assert "hello" in result
    assert "exit=0" in result


def test_shell_reports_nonzero_exit() -> None:
    tool = ShellTool()
    result = tool.call(cmd="exit 2")
    assert "exit=2" in result


def test_shell_captures_stderr() -> None:
    tool = ShellTool()
    result = tool.call(cmd="echo oops 1>&2")
    assert "oops" in result


def test_shell_times_out() -> None:
    tool = ShellTool(timeout_seconds=1)
    result = tool.call(cmd="sleep 5")
    assert "timed out" in result


# ---------- SearchMemoryTool / SearchFactsTool ----------


def test_search_memory_returns_formatted_hits(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "e.sqlite", embedder=_FakeEmbedder())
    try:
        store.ingest(
            external_id="m1",
            title="Debug session",
            body="We chased a tricky bug.",
            principle="Root cause or nothing.",
            tier="working",
            source="t",
        )
        tool = SearchMemoryTool(store=store)
        result = tool.call(query="bug")
        assert "Debug session" in result
        assert "Root cause or nothing." in result
    finally:
        store.close()


def test_search_memory_empty_store_returns_note(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "e.sqlite", embedder=_FakeEmbedder())
    try:
        tool = SearchMemoryTool(store=store)
        assert tool.call(query="anything") == "(no memories above similarity threshold)"
    finally:
        store.close()


def test_search_facts_returns_triples(tmp_path: Path) -> None:
    store = SemanticStore(tmp_path / "s.sqlite", embedder=_FakeEmbedder())
    try:
        store.add(subject="mark", predicate="prefers", object="raw sqlite", source="t")
        tool = SearchFactsTool(store=store)
        result = tool.call(query="sql")
        assert "mark" in result
        assert "prefers" in result
    finally:
        store.close()


# ---------- ToolSpec shape ----------


def test_tool_specs_have_required_schema_fields(tmp_path: Path) -> None:
    tool = ReadFileTool(root=tmp_path)
    spec = tool.spec
    assert spec.name == "read_file"
    assert spec.tier == "read"
    assert "type" in spec.parameters
    assert spec.parameters["type"] == "object"
    assert "path" in spec.parameters["properties"]


def test_write_tier_tools_are_marked_write(tmp_path: Path) -> None:
    assert WriteFileTool(root=tmp_path).spec.tier == "write"
    assert ShellTool().spec.tier == "write"
    assert ReadFileTool(root=tmp_path).spec.tier == "read"


def test_tool_spec_type_annotation(tmp_path: Path) -> None:
    spec: ToolSpec = ReadFileTool(root=tmp_path).spec
    assert spec.name == "read_file"
