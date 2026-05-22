"""Unit tests for driver/workspace_verify.py — harness-oxj7.

The helper scans a workspace at run_loop start and synthesizes a tuple
of `VerifyStep`s gating every issue close on baseline file-type parses.
These tests pin the file-discovery walk (including excluded-dir
pruning), the skip-on-missing-binary policy, and the integration with
the FSM-path `_resolve_verify_outcome` (the commands actually run +
flag real syntax errors)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.driver.fsm_turn import _exec_test_cmd
from harness.driver.workspace_verify import (
    _workspace_has_file,
    default_workspace_verify_steps,
)

_HAS_NODE = shutil.which("node") is not None
_HAS_PYTHON = shutil.which("python") is not None

_requires_node = pytest.mark.skipif(
    not _HAS_NODE, reason="workspace JS verify step requires `node` on PATH"
)
_requires_python = pytest.mark.skipif(
    not _HAS_PYTHON, reason="workspace Python verify step requires `python` on PATH"
)


# --- file discovery -----------------------------------------------


def test_workspace_has_file_finds_js_at_root(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("const x = 1;\n")
    assert _workspace_has_file(tmp_path, (".js",)) is True


def test_workspace_has_file_finds_nested_match(tmp_path: Path) -> None:
    """Walk descends arbitrary depth — a buried .py file in
    `src/lib/util/helper.py` must be discovered so a project with
    typical structure picks up the gate."""
    nested = tmp_path / "src" / "lib" / "util"
    nested.mkdir(parents=True)
    (nested / "helper.py").write_text("x = 1\n")
    assert _workspace_has_file(tmp_path, (".py",)) is True


def test_workspace_has_file_excludes_node_modules(tmp_path: Path) -> None:
    """JS files under `node_modules/` must NOT count — vendored
    dependencies aren't workspace source, and their parse errors
    aren't the model's problem to fix."""
    nm = tmp_path / "node_modules" / "some-pkg"
    nm.mkdir(parents=True)
    (nm / "broken.js").write_text("function f() { const x = 1; const x = 2; }\n")
    assert _workspace_has_file(tmp_path, (".js",)) is False


def test_workspace_has_file_excludes_venv(tmp_path: Path) -> None:
    """Python files under `.venv/` must NOT count — same vendoring
    argument as node_modules."""
    venv = tmp_path / ".venv" / "lib" / "site-packages"
    venv.mkdir(parents=True)
    (venv / "broken.py").write_text("def f(:\n")
    assert _workspace_has_file(tmp_path, (".py",)) is False


def test_workspace_has_file_excludes_pycache(tmp_path: Path) -> None:
    cache = tmp_path / "src" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "broken.py").write_text("def f(:\n")
    assert _workspace_has_file(tmp_path, (".py",)) is False


def test_workspace_has_file_empty_workspace_returns_false(tmp_path: Path) -> None:
    assert _workspace_has_file(tmp_path, (".js",)) is False
    assert _workspace_has_file(tmp_path, (".py",)) is False


# --- default_workspace_verify_steps -------------------------------


def test_empty_workspace_returns_no_steps(tmp_path: Path) -> None:
    """No recognized file types → empty tuple. The gate stays silent
    when it has nothing to say."""
    assert default_workspace_verify_steps(tmp_path) == ()


def test_workspace_with_only_js_adds_js_step(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("const x = 1;\n")
    steps = default_workspace_verify_steps(tmp_path)
    if not _HAS_NODE:
        # Node missing → no step even though JS present.
        assert steps == ()
        return
    assert len(steps) == 1
    assert "node --check" in steps[0].cmd


def test_workspace_with_only_python_adds_python_step(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = 1\n")
    steps = default_workspace_verify_steps(tmp_path)
    if not _HAS_PYTHON:
        assert steps == ()
        return
    assert len(steps) == 1
    assert "ast.parse" in steps[0].cmd


def test_workspace_with_both_adds_both_steps(tmp_path: Path) -> None:
    """JS + Python in the same workspace → two steps, JS first then
    Python. Order matters: parse-checks run in sequence and the
    first non-zero exit short-circuits, so a stable order is part
    of the contract."""
    (tmp_path / "game.js").write_text("const x = 1;\n")
    (tmp_path / "main.py").write_text("x = 1\n")
    steps = default_workspace_verify_steps(tmp_path)
    if _HAS_NODE and _HAS_PYTHON:
        assert len(steps) == 2
        assert "node --check" in steps[0].cmd
        assert "ast.parse" in steps[1].cmd


def test_missing_node_drops_js_step(tmp_path: Path) -> None:
    """JS present but `node` not on PATH → no JS step. Mirrors
    parse_check's missing-binary policy: silent skip, no failure."""
    (tmp_path / "game.js").write_text("const x = 1;\n")

    def _missing_node(name: str) -> str | None:
        return None if name == "node" else f"/usr/bin/{name}"

    with patch("harness.driver.workspace_verify.shutil.which", side_effect=_missing_node):
        steps = default_workspace_verify_steps(tmp_path)
    js_steps = [s for s in steps if "node --check" in s.cmd]
    assert js_steps == []


# --- end-to-end: synthesized steps actually run -------------------


@_requires_node
def test_js_step_passes_on_clean_workspace(tmp_path: Path) -> None:
    """Synthesized JS step exits 0 against a workspace of valid JS."""
    (tmp_path / "ok.js").write_text("const x = 1;\n")
    steps = default_workspace_verify_steps(tmp_path)
    js_step = next(s for s in steps if "node --check" in s.cmd)
    exit_code, _tail = _exec_test_cmd(js_step.cmd, tmp_path, shell_mode=js_step.shell)
    assert exit_code == 0


@_requires_node
def test_js_step_fails_on_broken_workspace(tmp_path: Path) -> None:
    """Synthesized JS step exits non-zero against a workspace
    containing a SyntaxError — exactly the loop run 26c39558
    failure mode that this gate is here to catch at the phase
    boundary."""
    (tmp_path / "ok.js").write_text("const x = 1;\n")
    (tmp_path / "broken.js").write_text(
        "function f() {\n  const tileX = 1;\n  const tileX = 2;\n}\n"
    )
    steps = default_workspace_verify_steps(tmp_path)
    js_step = next(s for s in steps if "node --check" in s.cmd)
    exit_code, _tail = _exec_test_cmd(js_step.cmd, tmp_path, shell_mode=js_step.shell)
    assert exit_code != 0


@_requires_python
def test_python_step_passes_on_clean_workspace(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def f():\n    return 42\n")
    steps = default_workspace_verify_steps(tmp_path)
    py_step = next(s for s in steps if "ast.parse" in s.cmd)
    exit_code, _tail = _exec_test_cmd(py_step.cmd, tmp_path, shell_mode=py_step.shell)
    assert exit_code == 0


@_requires_python
def test_python_step_fails_on_broken_workspace(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def f():\n    return 42\n")
    (tmp_path / "broken.py").write_text("def f(:\n    return 1\n")
    steps = default_workspace_verify_steps(tmp_path)
    py_step = next(s for s in steps if "ast.parse" in s.cmd)
    exit_code, _tail = _exec_test_cmd(py_step.cmd, tmp_path, shell_mode=py_step.shell)
    assert exit_code != 0


@_requires_python
def test_python_step_excludes_pycache_paths(tmp_path: Path) -> None:
    """Even if a `__pycache__` dir contains malformed .py files (very
    unusual but possible from a botched compile), the verify step
    must exclude them so we don't false-positive on artifacts the
    model never wrote."""
    (tmp_path / "main.py").write_text("x = 1\n")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "garbage.py").write_text("def f(:\n")
    steps = default_workspace_verify_steps(tmp_path)
    py_step = next(s for s in steps if "ast.parse" in s.cmd)
    exit_code, _tail = _exec_test_cmd(py_step.cmd, tmp_path, shell_mode=py_step.shell)
    assert exit_code == 0
