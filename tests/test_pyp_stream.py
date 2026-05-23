"""Tests for pyp_stream (Candidate B of harness-bw27).

Pyp itself is exercised end-to-end by the bench; here we pin the
harness-level guarantees: binary resolution, validation, sandboxing,
and the in-place writeback semantics. Skips automatically when the
``stream`` extra isn't installed so a slim install can still run the
rest of the suite.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pyp_available = shutil.which("pyp") is not None

pytestmark = pytest.mark.skipif(
    not pyp_available,
    reason="pyp binary not on PATH — install the `stream` extra: uv sync --extra stream",
)

from harness.tools.pyp_stream import PypStreamTool  # noqa: E402 — gated by pytestmark above


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def tool(workspace: Path) -> PypStreamTool:
    return PypStreamTool(root=workspace)


# --- construction -----------------------------------------------------------


def test_construct_locates_pyp(tool: PypStreamTool) -> None:
    assert tool.binary.endswith("pyp"), f"pyp lookup looks wrong: {tool.binary}"
    assert Path(tool.binary).exists()


def test_construct_with_injected_binary(workspace: Path) -> None:
    real_pyp = shutil.which("pyp")
    assert real_pyp is not None
    tool = PypStreamTool(root=workspace, binary=real_pyp)
    assert tool.binary == real_pyp


# --- expr validation --------------------------------------------------------


def test_empty_expr_rejected(tool: PypStreamTool) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tool.call(expr="   ", stdin="x")


def test_oversized_expr_rejected(tool: PypStreamTool) -> None:
    with pytest.raises(ValueError, match="bytes"):
        tool.call(expr="x" * (8 * 1024 + 1), stdin="x")


# --- path / stdin plumbing --------------------------------------------------


def test_extract_column_from_file(tool: PypStreamTool, workspace: Path) -> None:
    (workspace / "log.txt").write_text("a b c\nd e f\ng h i\n")
    out = tool.call(expr="x.split()[2]", paths=["log.txt"])
    assert out == "c\nf\ni\n"


def test_extract_column_from_stdin(tool: PypStreamTool) -> None:
    out = tool.call(expr="x.split()[0]", stdin="alpha 1\nbeta 2\n")
    assert out == "alpha\nbeta\n"


def test_lines_binding(tool: PypStreamTool, workspace: Path) -> None:
    (workspace / "log.txt").write_text("one\ntwo\nthree\n")
    out = tool.call(expr="len(lines)", paths=["log.txt"])
    assert out.strip() == "3"


def test_multiple_paths_concatenate(tool: PypStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("alpha\n")
    (workspace / "b.txt").write_text("beta\n")
    out = tool.call(expr="x.upper()", paths=["a.txt", "b.txt"])
    assert out == "ALPHA\nBETA\n"


def test_both_paths_and_stdin_rejected(tool: PypStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("x")
    with pytest.raises(ValueError, match="either"):
        tool.call(expr="x", paths=["a.txt"], stdin="y")


def test_neither_paths_nor_stdin_rejected(tool: PypStreamTool) -> None:
    with pytest.raises(ValueError, match="one of"):
        tool.call(expr="x")


def test_oversized_stdin_rejected(tool: PypStreamTool) -> None:
    too_big = "x" * (256 * 1024 + 1)
    with pytest.raises(ValueError, match="max"):
        tool.call(expr="x", stdin=too_big)


# --- sandbox ----------------------------------------------------------------


def test_path_outside_workspace_rejected(tool: PypStreamTool, tmp_path: Path) -> None:
    (tmp_path.parent / "outside.txt").write_text("secret")
    with pytest.raises(ValueError, match="escapes workspace"):
        tool.call(expr="x", paths=["../outside.txt"])


def test_missing_path_raises(tool: PypStreamTool) -> None:
    with pytest.raises(FileNotFoundError):
        tool.call(expr="x", paths=["nope.txt"])


def test_directory_rejected(tool: PypStreamTool, workspace: Path) -> None:
    (workspace / "sub").mkdir()
    with pytest.raises(IsADirectoryError):
        tool.call(expr="x", paths=["sub"])


# --- in-place ---------------------------------------------------------------


def test_in_place_single_path(tool: PypStreamTool, workspace: Path) -> None:
    src = workspace / "a.py"
    src.write_text("foo and foo\n")
    summary = tool.call(expr="x.replace('foo', 'bar')", paths=["a.py"], in_place=True)
    assert "rewrote a.py" in summary
    assert src.read_text() == "bar and bar\n"


def test_in_place_rejects_multi_path(tool: PypStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("x")
    (workspace / "b.txt").write_text("y")
    with pytest.raises(ValueError, match="exactly one"):
        tool.call(expr="x", paths=["a.txt", "b.txt"], in_place=True)


# --- failure surface --------------------------------------------------------


def test_pyp_runtime_error_returned_as_text(tool: PypStreamTool) -> None:
    """pyp returns a nonzero exit when the user code raises. We surface
    the exit + stderr instead of crashing the tool."""
    out = tool.call(expr="x.no_such_method()", stdin="hello\n")
    assert "exit=" in out
    assert "stderr" in out


def test_timeout_returns_marker(workspace: Path) -> None:
    """Expression must reference input bindings (pyp rejects expressions
    that don't), so we sleep inside a tuple that still touches ``x``."""
    tool = PypStreamTool(root=workspace, timeout_seconds=0.3)
    out = tool.call(
        expr="(__import__('time').sleep(5), x)[1]",
        stdin="hi\n",
        timeout=0.3,
    )
    assert "timed out" in out
