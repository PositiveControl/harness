from __future__ import annotations

import shutil
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
    ToolHit,
    ToolRegistry,
    ToolResult,
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


@dataclass
class _StructuredReturnStub:
    """Tool that returns ToolResult directly with structured metadata.
    Exercises the harness-ywp.4 Registry dispatch path."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="_stub_structured",
            description="stub",
            parameters={"type": "object", "properties": {}},
            tier="read",
        )

    def call(self) -> ToolResult:
        return ToolResult(
            tool_name="_stub_structured",
            output="hello",
            hits=(
                ToolHit(source="episodic", external_id="x1", title="T1", score=0.91),
                ToolHit(source="episodic", external_id="x2", title="T2", score=0.33),
            ),
        )


def test_registry_passes_through_structured_tool_result() -> None:
    """harness-ywp.4: tools may return ToolResult directly to carry
    retrieval hits / grounded citations. The Registry trusts that
    result and propagates it to the caller."""
    registry = ToolRegistry()
    registry.register(_StructuredReturnStub())
    result = registry.call("_stub_structured", {})
    assert result.success
    assert result.output == "hello"
    assert len(result.hits) == 2
    assert result.hits[0].external_id == "x1"
    assert result.top_score == 0.91


@dataclass
class _MismatchedToolNameStub:
    """Tool that returns a ToolResult whose tool_name disagrees with
    the registered name — Registry must re-stamp to the registry name
    so audit attribution stays honest."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="_stub_mismatched",
            description="stub",
            parameters={"type": "object", "properties": {}},
            tier="read",
        )

    def call(self) -> ToolResult:
        return ToolResult(tool_name="wrong_name", output="ok")


def test_registry_restamps_tool_name_on_mismatch() -> None:
    registry = ToolRegistry()
    registry.register(_MismatchedToolNameStub())
    result = registry.call("_stub_mismatched", {})
    assert result.tool_name == "_stub_mismatched"
    assert result.output == "ok"


def test_tool_result_text_factory_has_empty_metadata() -> None:
    r = ToolResult.text("x", "some output")
    assert r.tool_name == "x"
    assert r.output == "some output"
    assert r.hits == ()
    assert r.citations_grounded == frozenset()
    assert r.top_score is None


def test_tool_result_top_score_returns_max_hit_score() -> None:
    r = ToolResult(
        tool_name="x",
        output="",
        hits=(
            ToolHit(source="episodic", external_id=None, title="a", score=0.2),
            ToolHit(source="episodic", external_id=None, title="b", score=0.8),
            ToolHit(source="episodic", external_id=None, title="c", score=0.5),
        ),
    )
    assert r.top_score == 0.8


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


# --- harness-0tni: offset + limit ----------------------------------


def _multiline(tmp_path: Path, *, n: int = 100) -> Path:
    """Helper: create a `lines.txt` file with N numbered lines."""
    target = tmp_path / "lines.txt"
    target.write_text("\n".join(f"line {i}" for i in range(1, n + 1)) + "\n")
    return target


def test_read_file_with_offset_starts_from_line(tmp_path: Path) -> None:
    """offset=N returns the file starting at the N-th line (1-based).
    Marker reports the actual range + total."""
    _multiline(tmp_path, n=50)
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="lines.txt", offset=10)
    assert result.startswith("line 10\n")
    assert "line 9" not in result
    assert "line 50" in result
    assert "[showing lines 10-50 of 50]" in result


def test_read_file_with_limit_returns_first_n_lines(tmp_path: Path) -> None:
    """limit=N (without offset) returns the first N lines."""
    _multiline(tmp_path, n=50)
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="lines.txt", limit=5)
    assert "line 1" in result
    assert "line 5" in result
    assert "line 6" not in result
    assert "[showing lines 1-5 of 50]" in result


def test_read_file_with_offset_and_limit_returns_slice(tmp_path: Path) -> None:
    """Combined: offset=20, limit=10 returns lines 20-29."""
    _multiline(tmp_path, n=50)
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="lines.txt", offset=20, limit=10)
    assert "line 20" in result
    assert "line 29" in result
    assert "line 19" not in result
    assert "line 30" not in result
    assert "[showing lines 20-29 of 50]" in result


def test_read_file_offset_past_eof_returns_note(tmp_path: Path) -> None:
    """offset past last line returns a 'no lines in range' note
    instead of an opaque empty string. The model needs to know it
    asked for something out of range, not that the call failed."""
    _multiline(tmp_path, n=10)
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="lines.txt", offset=999)
    assert "no lines in range" in result
    assert "10 lines" in result
    assert "offset=999" in result


def test_read_file_offset_zero_rejected(tmp_path: Path) -> None:
    """offset is 1-based — 0 (or negative) is the most common
    off-by-one mistake; reject explicitly with a clear hint."""
    _multiline(tmp_path, n=10)
    tool = ReadFileTool(root=tmp_path)
    with pytest.raises(ValueError, match="offset must be >= 1"):
        tool.call(path="lines.txt", offset=0)


def test_read_file_limit_zero_rejected(tmp_path: Path) -> None:
    _multiline(tmp_path, n=10)
    tool = ReadFileTool(root=tmp_path)
    with pytest.raises(ValueError, match="limit must be >= 1"):
        tool.call(path="lines.txt", limit=0)


def test_read_file_without_offset_or_limit_returns_full_file(tmp_path: Path) -> None:
    """Backward compat: no slice args = legacy full-file behavior,
    no slicing marker appended. Existing callers see zero change."""
    _multiline(tmp_path, n=10)
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="lines.txt")
    assert "line 1" in result
    assert "line 10" in result
    assert "showing lines" not in result


# --- harness-2kob: symbol addressing ------------------------------

try:
    import tree_sitter_language_pack  # noqa: F401

    _HAS_CODE = True
except ImportError:  # pragma: no cover - exercised only on lean installs
    _HAS_CODE = False

_requires_code = pytest.mark.skipif(not _HAS_CODE, reason="requires the [code] extra")

_PY_MODULE = """import os


class Foo:
    def bar(self, x):
        return x + 1

    def baz(self):
        return 2


def top(a, b):
    return a + b
"""


@_requires_code
def test_read_file_symbol_returns_whole_span(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(_PY_MODULE)
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="m.py", symbol="Foo.bar")
    assert "def bar(self, x):" in result
    assert "return x + 1" in result
    # the *whole* span, not a guessed slice — and nothing from baz/top.
    assert "def baz" not in result
    assert "def top" not in result
    assert "[symbol Foo.bar, lines 5-6 of 13]" in result


@_requires_code
def test_read_file_symbol_bare_name_resolves(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(_PY_MODULE)
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="m.py", symbol="top")
    assert "def top(a, b):" in result
    assert "[symbol top, lines" in result


@_requires_code
def test_read_file_symbol_ambiguous_lists_candidates(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "class A:\n    def run(self):\n        pass\n\nclass B:\n    def run(self):\n        pass\n"
    )
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="m.py", symbol="run")
    assert "2 symbols match" in result
    assert "A.run" in result
    assert "B.run" in result
    # ambiguous => no body returned, just the candidate list.
    assert "pass" not in result


@_requires_code
def test_read_file_symbol_not_found_note(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(_PY_MODULE)
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="m.py", symbol="nonexistent")
    assert "no symbol named 'nonexistent'" in result


def test_read_file_symbol_with_offset_is_rejected(tmp_path: Path) -> None:
    """Mutual exclusion: symbol + offset/limit is a usage error, not a
    silent precedence. Needs no parser — the guard fires first."""
    (tmp_path / "m.py").write_text(_PY_MODULE)
    tool = ReadFileTool(root=tmp_path)
    with pytest.raises(ValueError, match="cannot be combined with offset/limit"):
        tool.call(path="m.py", symbol="top", offset=1)


def test_read_file_symbol_unsupported_extension_degrades(tmp_path: Path) -> None:
    """A symbol read of a file with no grammar degrades to a guidance
    note (never raises) — and needs no [code] extra, since the
    unsupported-extension check precedes the lazy tree-sitter import."""
    (tmp_path / "notes.txt").write_text("just prose, no code\n")
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="notes.txt", symbol="anything")
    assert "symbol read unavailable" in result
    assert "offset/limit" in result


def test_read_file_slice_preserves_trailing_newlines(tmp_path: Path) -> None:
    """splitlines(keepends=True) keeps newlines so the slice round-
    trips faithfully — edit_file's old_string matching depends on
    this when the model is iterating on a slice it just read."""
    target = tmp_path / "lines.txt"
    target.write_text("a\nb\nc\n")
    tool = ReadFileTool(root=tmp_path)
    result = tool.call(path="lines.txt", offset=1, limit=2)
    # Body should be "a\nb\n", marker appended after.
    assert result.startswith("a\nb\n")
    assert "[showing lines 1-2 of 3]" in result


def test_read_file_spec_lists_offset_and_limit(tmp_path: Path) -> None:
    """Schema must advertise the new params so model adapters' tool-
    spec serialization includes them. The model's prior is that
    read_file accepts offset/limit; if the schema doesn't list them,
    the adapter strips them client-side."""
    spec = ReadFileTool(root=tmp_path).spec
    props = spec.parameters["properties"]
    assert "offset" in props
    assert "limit" in props
    assert props["offset"]["type"] == "integer"
    assert props["limit"]["type"] == "integer"


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


# --- harness-h6wa: post-write parse gate ---------------------------


_requires_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="parse-check JS integration requires `node` on PATH",
)


@_requires_node
def test_write_file_rejects_broken_js(tmp_path: Path) -> None:
    """harness-h6wa: WriteFileTool runs the parse gate too. A new .js
    file containing a SyntaxError surfaces as a failure, with the
    parser's stderr in the message — matching EditFileTool's behavior
    so the model sees the same signal regardless of which write tool
    it reached for."""
    tool = WriteFileTool(root=tmp_path)
    with pytest.raises(ValueError, match="no longer parses"):
        tool.call(
            path="broken.js",
            content="function f() {\n  const x = 1;\n  const x = 2;\n}\n",
        )


@_requires_node
def test_write_file_passes_valid_js(tmp_path: Path) -> None:
    """Sanity: valid JS goes through the gate untouched."""
    tool = WriteFileTool(root=tmp_path)
    result = tool.call(path="ok.js", content="const x = 1;\n")
    assert "wrote" in result
    assert (tmp_path / "ok.js").read_text() == "const x = 1;\n"


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
        assert "Debug session" in result.output
        assert "Root cause or nothing." in result.output
        # Structured return (harness-ywp.4): hit surfaces retrieval
        # metadata so the audit log and confidence-fallback hook can
        # read scores without parsing the tool output string.
        assert len(result.hits) == 1
        assert result.hits[0].source == "episodic"
        assert result.hits[0].title == "Debug session"
        assert result.hits[0].external_id == "m1"
        assert result.top_score == result.hits[0].score
    finally:
        store.close()


def test_search_memory_empty_store_returns_note(tmp_path: Path) -> None:
    store = EpisodicStore(tmp_path / "e.sqlite", embedder=_FakeEmbedder())
    try:
        tool = SearchMemoryTool(store=store)
        result = tool.call(query="anything")
        assert result.output.startswith("(no memories matched")
        assert "search_web" in result.output
        assert result.hits == ()
        assert result.top_score is None
        assert result.citations_grounded == frozenset()
    finally:
        store.close()


def test_search_memory_declares_grounded_citations(tmp_path: Path) -> None:
    """harness-ywp.5: grounding-tier tools declare the §-citations
    present in the chunks they returned, so the audit log + confidence
    fallback don't have to regex-parse the output text to know what was
    actually grounded."""
    store = EpisodicStore(tmp_path / "e.sqlite", embedder=_FakeEmbedder())
    try:
        store.ingest(
            external_id="jo-4-1-1",
            title="§4-1-1 Air Traffic Clearances",
            body="Controllers issue clearances per §4-1-1 and TBL 4-1-2.",
            principle="See §4-1-1 for phraseology.",
            tier="seed",
            source="t",
        )
        tool = SearchMemoryTool(store=store)
        result = tool.call(query="clearance")
        # Citations are the CANONICAL set the search_memory chunk
        # actually contains — not whatever regex the hook later runs
        # on the model's reply.
        assert "§4-1-1" in result.citations_grounded
        assert "TBL 4-1-2" in result.citations_grounded
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
