"""Tests for the `python_eval` reckon-profile tool — harness-jojo.

Heavy on sandbox negatives: every blocked module + escape attempt
gets a named test so a future change to the bootstrap can't silently
re-open a hole. These tests spawn real child interpreters and are
slower than the other tool tests (~50-100ms each); kept under the
default suite because they're the security contract.
"""

from __future__ import annotations

import pytest

from harness.tools.python_eval import PythonEvalTool


@pytest.fixture
def tool() -> PythonEvalTool:
    return PythonEvalTool(timeout_seconds=5.0)


# ----------------- spec shape -----------------


def test_spec(tool: PythonEvalTool) -> None:
    spec = tool.spec
    assert spec.name == "python_eval"
    assert spec.tier == "read"
    assert "code" in spec.parameters["properties"]
    assert "timeout" in spec.parameters["properties"]


# ----------------- happy path -----------------


def test_simple_expression(tool: PythonEvalTool) -> None:
    out = tool.call(code="1 + 2")
    assert "-> 3" in out
    assert "[duration:" in out


def test_math_module_works(tool: PythonEvalTool) -> None:
    out = tool.call(code="math.sqrt(2)")
    assert "1.4142" in out


def test_statistics_median(tool: PythonEvalTool) -> None:
    out = tool.call(code="statistics.median([3, 1, 4, 1, 5, 9, 2, 6])")
    assert "3.5" in out


def test_json_roundtrip(tool: PythonEvalTool) -> None:
    out = tool.call(code='json.loads(\'{"a": 1, "b": [2, 3]}\')')
    assert "{'a': 1, 'b': [2, 3]}" in out


def test_datetime_arithmetic(tool: PythonEvalTool) -> None:
    out = tool.call(code="(datetime.date(2026, 5, 15) - datetime.date(2026, 4, 1)).days")
    assert "-> 44" in out


def test_regex_match(tool: PythonEvalTool) -> None:
    out = tool.call(code='re.findall(r"\\d+", "a1 b22 c333")')
    assert "['1', '22', '333']" in out


def test_decimal_arithmetic(tool: PythonEvalTool) -> None:
    out = tool.call(code='decimal.Decimal("0.1") + decimal.Decimal("0.2")')
    assert "0.3" in out


def test_explicit_import_in_allowlist(tool: PythonEvalTool) -> None:
    out = tool.call(code="import fractions; fractions.Fraction(1, 3) + fractions.Fraction(1, 6)")
    assert "Fraction(1, 2)" in out


def test_multi_statement_captures_last_expr(tool: PythonEvalTool) -> None:
    out = tool.call(code="x = 5\ny = x * 3\ny + 1")
    assert "-> 16" in out


def test_print_captures_stdout(tool: PythonEvalTool) -> None:
    out = tool.call(code='print("hello"); 42')
    assert "-> 42" in out
    assert "hello" in out
    assert "[stdout]" in out


# ----------------- sandbox negatives -----------------


def test_import_os_blocked(tool: PythonEvalTool) -> None:
    out = tool.call(code="import os")
    assert "ERROR" in out
    assert "blocked" in out or "ImportError" in out


def test_import_subprocess_blocked(tool: PythonEvalTool) -> None:
    out = tool.call(code="import subprocess")
    assert "ERROR" in out
    assert "blocked" in out or "ImportError" in out


def test_import_socket_blocked(tool: PythonEvalTool) -> None:
    out = tool.call(code="import socket")
    assert "ERROR" in out


def test_import_urllib_blocked(tool: PythonEvalTool) -> None:
    out = tool.call(code="import urllib.request")
    assert "ERROR" in out


def test_import_ctypes_blocked(tool: PythonEvalTool) -> None:
    out = tool.call(code="import ctypes")
    assert "ERROR" in out


def test_open_builtin_removed(tool: PythonEvalTool) -> None:
    out = tool.call(code="open('/etc/passwd').read()")
    assert "ERROR" in out
    assert "open" in out  # NameError mentions 'open'


def test_eval_builtin_removed(tool: PythonEvalTool) -> None:
    out = tool.call(code="eval('1+1')")
    assert "ERROR" in out


def test_exec_builtin_removed(tool: PythonEvalTool) -> None:
    out = tool.call(code="exec('x=1')")
    assert "ERROR" in out


def test_compile_builtin_removed(tool: PythonEvalTool) -> None:
    out = tool.call(code="compile('1+1', '<x>', 'eval')")
    assert "ERROR" in out


# ----------------- timeout -----------------


def test_timeout_triggers() -> None:
    # 0.3s timeout, infinite loop. Should kill cleanly.
    fast = PythonEvalTool(timeout_seconds=0.3)
    out = fast.call(code="while True: pass")
    assert "ERROR" in out
    assert "timeout" in out


def test_call_level_timeout_overrides_default() -> None:
    slow = PythonEvalTool(timeout_seconds=10.0)
    out = slow.call(code="while True: pass", timeout=0.3)
    assert "timeout" in out


# ----------------- error surfaces -----------------


def test_syntax_error_surfaces(tool: PythonEvalTool) -> None:
    out = tool.call(code="1 +")
    assert "ERROR" in out
    assert "Syntax" in out


def test_runtime_error_surfaces(tool: PythonEvalTool) -> None:
    out = tool.call(code="1 / 0")
    assert "ERROR" in out
    assert "ZeroDivisionError" in out


def test_name_error_surfaces(tool: PythonEvalTool) -> None:
    out = tool.call(code="undefined_var + 1")
    assert "ERROR" in out
    assert "NameError" in out


def test_empty_code_rejected(tool: PythonEvalTool) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tool.call(code="")


def test_whitespace_only_rejected(tool: PythonEvalTool) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tool.call(code="   \n\t  ")
