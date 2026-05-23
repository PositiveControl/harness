"""Tests for python_stream (Candidate C of harness-bw27).

The sandboxed-Python control candidate. Three boundaries to pin:

  1. Sandbox properties — blocked imports/builtins stay blocked; the
     ``text`` / ``lines`` / ``paths`` bindings are populated; allowlisted
     imports work.
  2. Capture path — repr() vs raw-string-pass-through for the final
     expression, stdout fallback when no final expression, oversized
     output truncation, timeout handling.
  3. in-place — single-path overwrite works, multi-path is rejected,
     workspace-escape guards apply.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.python_stream import PythonStreamTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def tool(workspace: Path) -> PythonStreamTool:
    return PythonStreamTool(root=workspace)


# --- input plumbing --------------------------------------------------------


def test_text_binding_populated_from_paths(tool: PythonStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("hello\nworld\n")
    out = tool.call(expr="text", paths=["a.txt"])
    assert out == "hello\nworld\n"


def test_lines_binding_split(tool: PythonStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("alpha\nbeta\ngamma\n")
    out = tool.call(expr="lines[1]", paths=["a.txt"])
    assert out == "beta"


def test_paths_binding_is_input_paths(tool: PythonStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("x")
    out = tool.call(expr="list(paths)", paths=["a.txt"])
    assert out == "['a.txt']"


def test_stdin_path(tool: PythonStreamTool) -> None:
    out = tool.call(expr="text.upper()", stdin="hello")
    assert out == "HELLO"


def test_multiple_paths_concatenate(tool: PythonStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("one\n")
    (workspace / "b.txt").write_text("two\n")
    out = tool.call(expr="len(lines)", paths=["a.txt", "b.txt"])
    assert out == "2"


# --- output shape ---------------------------------------------------------


def test_string_result_passes_through_unquoted(tool: PythonStreamTool) -> None:
    """If the final expression is already a string, return it verbatim —
    no repr() with quotes. Makes the tool behave like sed/awk when you
    want the raw transform."""
    out = tool.call(expr="text.upper()", stdin="hi")
    assert out == "HI"
    assert not out.startswith("'")


def test_non_string_result_returns_repr(tool: PythonStreamTool) -> None:
    out = tool.call(expr="[1, 2, 3]", stdin="x")
    assert out == "[1, 2, 3]"


def test_no_final_expression_returns_stdout(tool: PythonStreamTool) -> None:
    out = tool.call(expr="print('via stdout')", stdin="x")
    assert "via stdout" in out


def test_extract_column_via_python(tool: PythonStreamTool, workspace: Path) -> None:
    """Practical bench task — col 3 extract, in pure Python."""
    (workspace / "log.txt").write_text("a b c\nd e f\ng h i\n")
    out = tool.call(
        expr="'\\n'.join(line.split()[2] for line in lines) + '\\n'",
        paths=["log.txt"],
    )
    assert out == "c\nf\ni\n"


def test_filter_jsonl_via_python(tool: PythonStreamTool, workspace: Path) -> None:
    """JSONL filter — the kind of task pyp markets itself for."""
    src = workspace / "events.jsonl"
    src.write_text(
        '{"id": 1, "user": "alice", "amount": 10}\n'
        '{"id": 2, "user": "bob", "amount": 5}\n'
        '{"id": 3, "user": "alice", "amount": 20}\n'
    )
    out = tool.call(
        expr=(
            "'\\n'.join("
            "json.dumps({'id': r['id'], 'amount': r['amount']}) "
            "for r in (json.loads(l) for l in lines) if r['user'] == 'alice'"
            ") + '\\n'"
        ),
        paths=["events.jsonl"],
    )
    assert out == '{"id": 1, "amount": 10}\n{"id": 3, "amount": 20}\n'


# --- sandbox ---------------------------------------------------------------


def test_open_is_blocked(tool: PythonStreamTool) -> None:
    out = tool.call(expr="open('/etc/passwd').read()", stdin="x")
    assert "ERROR" in out
    assert "open" in out or "NameError" in out


def test_os_import_blocked(tool: PythonStreamTool) -> None:
    out = tool.call(expr="__import__('os').listdir('/')", stdin="x")
    assert "ERROR" in out
    assert "blocked" in out or "ImportError" in out


def test_subprocess_import_blocked(tool: PythonStreamTool) -> None:
    out = tool.call(expr="import subprocess; subprocess.run(['ls'])", stdin="x")
    assert "ERROR" in out


def test_eval_builtin_blocked(tool: PythonStreamTool) -> None:
    out = tool.call(expr="eval('1+1')", stdin="x")
    assert "ERROR" in out


def test_re_is_pre_imported(tool: PythonStreamTool) -> None:
    out = tool.call(expr="re.findall(r'[a-z]+', text)", stdin="hello123world")
    assert out == "['hello', 'world']"


def test_json_is_pre_imported(tool: PythonStreamTool) -> None:
    out = tool.call(expr="json.dumps({'a': 1})", stdin="x")
    assert out == '{"a": 1}'


# --- validation ------------------------------------------------------------


def test_empty_expr_rejected(tool: PythonStreamTool) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tool.call(expr="   ", stdin="x")


def test_oversized_expr_rejected(tool: PythonStreamTool) -> None:
    with pytest.raises(ValueError, match="bytes"):
        tool.call(expr="x" * (8 * 1024 + 1), stdin="x")


def test_path_outside_workspace_rejected(tool: PythonStreamTool, tmp_path: Path) -> None:
    (tmp_path.parent / "outside.txt").write_text("secret")
    with pytest.raises(ValueError, match="escapes workspace"):
        tool.call(expr="text", paths=["../outside.txt"])


def test_neither_paths_nor_stdin_rejected(tool: PythonStreamTool) -> None:
    with pytest.raises(ValueError, match="one of"):
        tool.call(expr="text")


def test_both_paths_and_stdin_rejected(tool: PythonStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("hi")
    with pytest.raises(ValueError, match="either"):
        tool.call(expr="text", paths=["a.txt"], stdin="x")


# --- in-place --------------------------------------------------------------


def test_in_place_single_path(tool: PythonStreamTool, workspace: Path) -> None:
    src = workspace / "a.txt"
    src.write_text("foo and foo\n")
    summary = tool.call(
        expr="text.replace('foo', 'bar')",
        paths=["a.txt"],
        in_place=True,
    )
    assert "rewrote a.txt" in summary
    assert src.read_text() == "bar and bar\n"


def test_in_place_rejects_multi_path(tool: PythonStreamTool, workspace: Path) -> None:
    (workspace / "a.txt").write_text("x")
    (workspace / "b.txt").write_text("y")
    with pytest.raises(ValueError, match="exactly one"):
        tool.call(expr="text", paths=["a.txt", "b.txt"], in_place=True)


# --- failure surface -------------------------------------------------------


def test_runtime_error_returned_as_text(tool: PythonStreamTool) -> None:
    out = tool.call(expr="1 / 0", stdin="x")
    assert "ERROR" in out
    assert "ZeroDivisionError" in out


def test_syntax_error_returned_as_text(tool: PythonStreamTool) -> None:
    out = tool.call(expr="def (no_name", stdin="x")
    assert "ERROR" in out
    assert "Syntax" in out or "syntax" in out


def test_timeout_returns_marker(workspace: Path) -> None:
    tool = PythonStreamTool(root=workspace, timeout_seconds=0.3)
    out = tool.call(expr="while True: pass", stdin="x", timeout=0.3)
    assert "timed out" in out


def test_oversized_output_truncated(tool: PythonStreamTool) -> None:
    """A 600 KB result is capped at 512 KB with a marker."""
    out = tool.call(expr="'y' * (600 * 1024)", stdin="x")
    assert "[truncated at" in out
