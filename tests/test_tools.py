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


def test_registry_unknown_kwarg_returns_actionable_error(tmp_path: Path) -> None:
    """harness-d7e: when the model passes an arg the tool's signature
    doesn't accept, the bare Python TypeError ('unexpected keyword
    argument deadline') doesn't tell the model what IS accepted, so
    it tends to repeat the same call. Rewrite that case into a
    'rejected unknown argument X. Accepts: a, b, c. Retry without
    the unknown field.' message keyed by the tool's JSON-schema
    properties."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))
    (tmp_path / "hi.txt").write_text("hello")

    result = registry.call("read_file", {"path": "hi.txt", "deadline": "today"})

    assert not result.success
    assert "rejected unknown argument 'deadline'" in result.output
    # The accepted-args list is sourced from the spec's properties so
    # additions to the schema flow into the hint automatically.
    assert "Accepts:" in result.output
    assert "path" in result.output
    assert (result.error or "").startswith("unknown_kwarg:")


def test_registry_typeerror_without_unknown_kwarg_falls_through(tmp_path: Path) -> None:
    """A TypeError that ISN'T about unknown kwargs (shape mismatch,
    missing required arg) flows through the generic error path so the
    rewrite doesn't claim to know more than it does."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=tmp_path))

    # path is required; omitting it raises a different TypeError shape.
    result = registry.call("read_file", {})

    assert not result.success
    # Generic TypeError prefix from the catch-all path, NOT the
    # unknown-kwarg rewrite.
    assert "rejected unknown argument" not in result.output
    assert (result.error or "").startswith("TypeError")


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


def test_every_tool_declares_a_human_readable_label(tmp_path: Path) -> None:
    """`ToolSpec.label` is what the CLI prints — every shipped tool must
    set a display_name so users never see the machine name like
    'search_facts' in confirm prompts or call logs."""
    store_ep = EpisodicStore(tmp_path / "e.sqlite", embedder=_FakeEmbedder())
    store_sem = SemanticStore(tmp_path / "s.sqlite", embedder=_FakeEmbedder())
    try:
        tools: list[Tool] = [
            ReadFileTool(root=tmp_path),
            WriteFileTool(root=tmp_path),
            ShellTool(),
            SearchMemoryTool(store=store_ep),
            SearchFactsTool(store=store_sem),
        ]
        for tool in tools:
            spec = tool.spec
            assert spec.display_name is not None, f"{spec.name} missing display_name"
            assert spec.label == spec.display_name
            assert spec.label != spec.name  # label must be prettier than machine name
    finally:
        store_ep.close()
        store_sem.close()


def test_tool_spec_label_falls_back_to_name_when_no_display_name() -> None:
    spec = ToolSpec(
        name="raw_name",
        description="d",
        parameters={"type": "object", "properties": {}},
        tier="read",
    )
    assert spec.label == "raw_name"


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


def test_write_file_refuses_existing_without_overwrite(tmp_path: Path) -> None:
    """Regression: write_file used to clobber existing files silently.
    The model misinterpreted 'add something' as a write_file call and
    destroyed the previous content. Now it must opt in explicitly."""
    tool = WriteFileTool(root=tmp_path)
    target = tmp_path / ".gitignore"
    target.write_text("existing line\n")
    with pytest.raises(ValueError, match="already exists"):
        tool.call(path=".gitignore", content="scratch\n")
    assert target.read_text() == "existing line\n"


def test_write_file_overwrite_opt_in(tmp_path: Path) -> None:
    tool = WriteFileTool(root=tmp_path)
    target = tmp_path / "config.toml"
    # Prior content similar in size to the new content — not a shrink.
    target.write_text("old_key = 1\nold_other = 2\n")
    result = tool.call(
        path="config.toml",
        content="new_key = 2\nnew_other = 3\n",
        overwrite=True,
    )
    assert "overwrote" in result
    assert target.read_text() == "new_key = 2\nnew_other = 3\n"


def test_write_file_refuses_drastic_shrink_even_with_overwrite(tmp_path: Path) -> None:
    """Regression (harness-2tq): 'add scratch to .gitignore' drove the
    model to write_file(content='scratch\\n', overwrite=True) — a catastrophic
    clobber of a 400-byte file with 8 bytes. Refuse the shrink and redirect
    to edit_file."""
    tool = WriteFileTool(root=tmp_path)
    target = tmp_path / ".gitignore"
    target.write_text(
        ".venv/\n__pycache__/\nnode_modules/\ndist/\n.mypy_cache/\n"
        ".ruff_cache/\n.pytest_cache/\ndata/\n.env\n.env.local\n.DS_Store\n"
    )
    with pytest.raises(ValueError, match="looks like you meant to append"):
        tool.call(path=".gitignore", content="scratch\n", overwrite=True)
    # Existing content untouched.
    assert "scratch\n" not in target.read_text()
    assert "__pycache__" in target.read_text()


def test_write_file_allows_overwrite_of_small_config(tmp_path: Path) -> None:
    """Legitimate use case: regenerating a small config file. The shrink
    guard has a 1KB floor so this isn't blocked."""
    tool = WriteFileTool(root=tmp_path)
    target = tmp_path / "config.toml"
    target.write_text("old = 1\n")
    # New content is smaller but both are under 1KB and under the 50% rule
    # threshold: existing=8, new=6 — existing//2 = 4, so new (6) >= 4. Pass.
    result = tool.call(path="config.toml", content="new=2\n", overwrite=True)
    assert "overwrote" in result


def test_write_file_spec_lists_overwrite_param(tmp_path: Path) -> None:
    spec = WriteFileTool(root=tmp_path).spec
    assert "overwrite" in spec.parameters["properties"]
    # Pointer to edit_file should live in the description so the model sees it.
    assert "edit_file" in spec.description


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


def test_shell_runs_in_cwd(tmp_path: Path) -> None:
    tool = ShellTool(cwd=tmp_path)
    result = tool.call(cmd="pwd")
    # macOS /tmp is a symlink to /private/tmp — resolve both sides for the compare.
    assert str(tmp_path.resolve()) in result or str(tmp_path) in result


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
        result = tool.call(query="anything")
        assert result.startswith("(no memories matched")
        assert "search_web" in result
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
