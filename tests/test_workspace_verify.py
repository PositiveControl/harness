"""Unit tests for driver/workspace_verify.py — harness-oxj7.

The helper scans a workspace at run_loop start and synthesizes a tuple
of `VerifyStep`s gating every issue close on baseline file-type parses.
These tests pin the file-discovery walk (including excluded-dir
pruning), the skip-on-missing-binary policy, and the integration with
the FSM-path `_resolve_verify_outcome` (the commands actually run +
flag real syntax errors)."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from harness.driver.fsm_turn import _exec_test_cmd
from harness.driver.workspace_verify import (
    _browser_app_index,
    _workspace_has_file,
    browser_smoke_skip_reason,
    default_workspace_verify_steps,
    extract_smoke_symptom,
    missing_entry_html_reason,
)

_HAS_NODE = shutil.which("node") is not None
_HAS_PYTHON = shutil.which("python") is not None
_HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None

_requires_node = pytest.mark.skipif(
    not _HAS_NODE, reason="workspace JS verify step requires `node` on PATH"
)
_requires_python = pytest.mark.skipif(
    not _HAS_PYTHON, reason="workspace Python verify step requires `python` on PATH"
)
_requires_playwright = pytest.mark.skipif(
    not _HAS_PLAYWRIGHT,
    reason="smoke-execute step requires `playwright` + chromium installed",
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


# --- harness-4b8v: browser-app smoke-execute detection ------------


def test_browser_app_index_finds_index_html_with_local_script(tmp_path: Path) -> None:
    """Root-level index.html with a local script tag → that's a
    browser app the smoke gate can verify."""
    (tmp_path / "index.html").write_text(
        '<!DOCTYPE html><html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    assert _browser_app_index(tmp_path) == tmp_path / "index.html"


def test_browser_app_index_accepts_index_htm(tmp_path: Path) -> None:
    """index.htm (legacy 8.3 extension) is treated the same as
    index.html — some scaffold tools still emit it."""
    (tmp_path / "index.htm").write_text(
        '<html><body><script src="app.js"></script></body></html>',
    )
    assert _browser_app_index(tmp_path) == tmp_path / "index.htm"


def test_browser_app_index_accepts_module_mjs(tmp_path: Path) -> None:
    """ES module entry: `<script type="module" src="game.mjs">`.
    Other attributes between `<script` and `src=` must not break
    detection."""
    (tmp_path / "index.html").write_text(
        '<html><body><script type="module" src="./game.mjs"></script></body></html>',
    )
    assert _browser_app_index(tmp_path) == tmp_path / "index.html"


def test_browser_app_index_skips_when_no_index(tmp_path: Path) -> None:
    """JS in the workspace but no index.html → not a browser app
    we know how to load; smoke gate stays silent (per the spec's
    skip-on-non-browser-workspace policy)."""
    (tmp_path / "game.js").write_text("const x = 1;\n")
    assert _browser_app_index(tmp_path) is None


def test_browser_app_index_skips_cdn_only_scripts(tmp_path: Path) -> None:
    """index.html whose only script tags reference CDN-hosted libs
    isn't an artifact the workspace authors — the smoke step is for
    code the model is editing, not jQuery."""
    (tmp_path / "index.html").write_text(
        "<html><body>"
        '<script src="https://cdn.example.com/lib.js"></script>'
        '<script src="http://other.example/foo.js"></script>'
        "</body></html>",
    )
    assert _browser_app_index(tmp_path) is None


def test_browser_app_index_treats_protocol_relative_as_remote(tmp_path: Path) -> None:
    """`//cdn.example/lib.js` resolves to the page's protocol; for
    HTTPS pages that's a CDN. Match the heuristic the operator
    intuitively expects — gate stays silent."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="//cdn.example/lib.js"></script></body></html>',
    )
    assert _browser_app_index(tmp_path) is None


def test_browser_app_index_skips_html_without_scripts(tmp_path: Path) -> None:
    """A static HTML page with no <script> tags has no JS we can
    smoke-test — skip silently."""
    (tmp_path / "index.html").write_text("<html><body><h1>Hi</h1></body></html>")
    assert _browser_app_index(tmp_path) is None


def test_browser_app_index_mixed_remote_and_local_triggers(tmp_path: Path) -> None:
    """When some scripts are CDN and some are local, the presence of
    even one local script means the workspace authors JS that should
    be smoke-tested."""
    (tmp_path / "index.html").write_text(
        "<html><body>"
        '<script src="https://cdn.example.com/lib.js"></script>'
        '<script src="game.js"></script>'
        "</body></html>",
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    assert _browser_app_index(tmp_path) == tmp_path / "index.html"


def test_smoke_step_added_for_browser_app_when_playwright_present(tmp_path: Path) -> None:
    """Integration: browser-app workspace + Playwright importable →
    default_workspace_verify_steps includes the smoke-execute step."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(tmp_path)
    smoke_steps = [s for s in steps if "smoke_runner" in s.cmd]
    assert len(smoke_steps) == 1
    # The step must reference the absolute, resolved index path so
    # subprocess cwd doesn't change the resolution surface (harness-53tp).
    # Use .resolve() on both sides — on macOS tmp_path is under a
    # /var -> /private/var symlink that resolve() canonicalizes.
    assert str((tmp_path / "index.html").resolve()) in smoke_steps[0].cmd


def test_smoke_step_enforces_blank_canvas_by_default(tmp_path: Path) -> None:
    """Default: the smoke cmd does NOT carry --no-blank-canvas, so the
    runner's blank-canvas check stays on (verifying an already-built
    game)."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(tmp_path)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    assert "--no-blank-canvas" not in smoke.cmd


def test_smoke_step_suppresses_blank_canvas_when_disabled(tmp_path: Path) -> None:
    """harness-6dsn: enforce_blank_canvas=False threads --no-blank-canvas
    into the smoke cmd so an incremental from-scratch skeleton isn't
    failed for a legitimately-blank canvas."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(tmp_path, enforce_blank_canvas=False)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    assert "--no-blank-canvas" in smoke.cmd


def test_smoke_step_adds_assert_flag_when_probe_present(tmp_path: Path) -> None:
    """harness-u1il5: a .harness/smoke_assert.js probe → the smoke cmd
    threads --assert=<resolved path> so the runner enforces the
    behavioral check."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    harness_dir = tmp_path / ".harness"
    harness_dir.mkdir()
    probe = harness_dir / "smoke_assert.js"
    probe.write_text("return [];\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(tmp_path)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    assert f"--assert={probe.resolve()}" in smoke.cmd


def test_smoke_step_no_assert_flag_when_probe_absent(tmp_path: Path) -> None:
    """No .harness/smoke_assert.js → no --assert flag (the behavioral
    check is opt-in per workspace)."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(tmp_path)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    assert "--assert=" not in smoke.cmd


def test_smoke_step_adds_screenshot_flag_when_capture_shot_set(tmp_path: Path) -> None:
    """harness-ke4hx.4: capture_shot threads --screenshot=<resolved path>
    into the smoke cmd so the runner writes a shot for advisory vision-QA."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    shot = tmp_path / ".harness" / "vision_shot.png"
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(tmp_path, capture_shot=shot)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    assert f"--screenshot={shot.resolve()}" in smoke.cmd


def test_smoke_step_no_screenshot_flag_by_default(tmp_path: Path) -> None:
    """No capture_shot → no --screenshot flag (vision-QA is opt-in; the
    smoke gate is byte-identical to its pre-vision form)."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(tmp_path)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    assert "--screenshot=" not in smoke.cmd


def test_smoke_step_index_absolute_with_relative_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """harness-53tp regression: when the driver passes a RELATIVE
    workspace path, the smoke step must still embed an ABSOLUTE index
    path — otherwise the runner double-joins it to its cwd
    (<workspace>/<workspace>/index.html) and fails 'index not found'
    on every turn, exhausting the run (loop_run 104e40b5)."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    # Reproduce the driver's calling convention: cwd at the parent,
    # workspace passed as a relative path.
    monkeypatch.chdir(tmp_path.parent)
    relative_ws = Path(tmp_path.name)
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(relative_ws)
    smoke_step = next(s for s in steps if "smoke_runner" in s.cmd)
    expected = str((tmp_path / "index.html").resolve())
    assert expected in smoke_step.cmd
    # And crucially: the workspace path segment must not be doubled.
    assert f"{tmp_path.name}/{tmp_path.name}" not in smoke_step.cmd


def test_smoke_step_skipped_when_playwright_unavailable(tmp_path: Path) -> None:
    """Skip-on-missing-deps acceptance criterion (c): Playwright
    not importable → no smoke step even though the workspace looks
    like a browser app."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=False):
        steps = default_workspace_verify_steps(tmp_path)
    smoke_steps = [s for s in steps if "smoke_runner" in s.cmd]
    assert smoke_steps == []


def test_smoke_step_skipped_when_not_a_browser_app(tmp_path: Path) -> None:
    """Skip-on-no-index acceptance: Playwright available but the
    workspace has no index.html → smoke step is not added."""
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        steps = default_workspace_verify_steps(tmp_path)
    smoke_steps = [s for s in steps if "smoke_runner" in s.cmd]
    assert smoke_steps == []


# --- harness-7bxm: loud skip-warning when the gate is degraded ----


def test_skip_reason_set_when_browser_app_but_no_playwright(tmp_path: Path) -> None:
    """The b85f4008 root cause: a browser app whose runtime-verify gate
    is OFF because Playwright isn't importable. browser_smoke_skip_reason
    must return a non-empty message so the loop can log it loudly."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=False):
        reason = browser_smoke_skip_reason(tmp_path)
    assert reason is not None
    assert "Playwright" in reason


def test_skip_reason_none_when_playwright_present(tmp_path: Path) -> None:
    """Gate WILL run → no warning."""
    (tmp_path / "index.html").write_text(
        '<html><body><script src="game.js"></script></body></html>',
    )
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=True):
        assert browser_smoke_skip_reason(tmp_path) is None


def test_skip_reason_none_when_not_a_browser_app(tmp_path: Path) -> None:
    """No browser app → nothing to warn about, even without Playwright."""
    (tmp_path / "game.js").write_text("const x = 1;\n")
    with patch("harness.driver.workspace_verify._playwright_available", return_value=False):
        assert browser_smoke_skip_reason(tmp_path) is None


# --- end-to-end: smoke-execute against real Playwright -----------


@_requires_playwright
def test_smoke_step_fails_on_runtime_canvas_error(tmp_path: Path) -> None:
    """harness-4b8v canonical failure: getContext('d') returns null
    → TypeError on first frame. Parse-gate passes; smoke step must
    fail. This is the bug the gate exists to catch."""
    (tmp_path / "index.html").write_text(
        "<!DOCTYPE html><html><body>"
        '<canvas id="game" width="100" height="100"></canvas>'
        '<script src="game.js"></script>'
        "</body></html>",
    )
    (tmp_path / "game.js").write_text(
        "const canvas = document.getElementById('game');\n"
        "const ctx = canvas.getContext('d');\n"
        "ctx.clearRect(0, 0, 100, 100);\n",
    )
    steps = default_workspace_verify_steps(tmp_path)
    smoke_step = next(s for s in steps if "smoke_runner" in s.cmd)
    exit_code, tail = _exec_test_cmd(smoke_step.cmd, tmp_path, shell_mode=smoke_step.shell)
    assert exit_code != 0
    # The runner emits either a console.error or a pageerror line —
    # either way the tail must mention runtime/error context so the
    # handoff back to the model is actionable.
    assert "error" in tail.lower() or "TypeError" in tail


@_requires_playwright
def test_smoke_step_passes_on_clean_canvas_workspace(tmp_path: Path) -> None:
    """A clean canvas init that actually renders content (a colored
    fill plus a contrasting rect) loads without console errors AND
    isn't flagged blank — smoke step exits 0."""
    (tmp_path / "index.html").write_text(
        "<!DOCTYPE html><html><body>"
        '<canvas id="game" width="100" height="100"></canvas>'
        '<script src="game.js"></script>'
        "</body></html>",
    )
    # Draw two distinct colors so the canvas is non-uniform — this is
    # what a working render loop produces and what the blank-canvas
    # check (harness-7bxm) expects to see.
    (tmp_path / "game.js").write_text(
        "const canvas = document.getElementById('game');\n"
        "const ctx = canvas.getContext('2d');\n"
        "ctx.fillStyle = '#3a3a3a';\n"
        "ctx.fillRect(0, 0, 100, 100);\n"
        "ctx.fillStyle = '#1e6fd9';\n"
        "ctx.fillRect(40, 40, 20, 20);\n",
    )
    steps = default_workspace_verify_steps(tmp_path)
    smoke_step = next(s for s in steps if "smoke_runner" in s.cmd)
    exit_code, _tail = _exec_test_cmd(smoke_step.cmd, tmp_path, shell_mode=smoke_step.shell)
    assert exit_code == 0


@_requires_playwright
def test_smoke_step_fails_on_blank_canvas(tmp_path: Path) -> None:
    """harness-7bxm: a workspace that loads clean (no console/page
    error) but never draws to its canvas — the all-uniform "blank
    canvas" failure mode (e.g. §2 tile render never wired into the
    rAF loop) — must fail the smoke step."""
    (tmp_path / "index.html").write_text(
        "<!DOCTYPE html><html><body>"
        '<canvas id="game" width="100" height="100"></canvas>'
        '<script src="game.js"></script>'
        "</body></html>",
    )
    (tmp_path / "game.js").write_text(
        "const canvas = document.getElementById('game');\n"
        "const ctx = canvas.getContext('2d');\n"
        "// draw loop never wired up — canvas stays uniform\n"
        "function draw() { /* TODO */ }\n",
    )
    steps = default_workspace_verify_steps(tmp_path)
    smoke_step = next(s for s in steps if "smoke_runner" in s.cmd)
    exit_code, tail = _exec_test_cmd(smoke_step.cmd, tmp_path, shell_mode=smoke_step.shell)
    assert exit_code != 0
    assert "blank" in tail.lower() or "one color" in tail.lower()


_BEHAVIORAL_INDEX = (
    "<!DOCTYPE html><html><body>"
    '<canvas id="game" width="100" height="100"></canvas>'
    '<script src="game.js"></script>'
    "</body></html>"
)

# A game that renders two colors (so the blank-canvas check passes) and
# — in the working version — wires a keydown listener that flips a flag
# on KeyJ (the §9b-i "firing trigger" shape). The render proves the page
# is otherwise clean so the behavioral assert is the only variable.
_GAME_WITH_FIRE = (
    "const canvas = document.getElementById('game');\n"
    "const ctx = canvas.getContext('2d');\n"
    "ctx.fillStyle = '#3a3a3a'; ctx.fillRect(0, 0, 100, 100);\n"
    "ctx.fillStyle = '#1e6fd9'; ctx.fillRect(40, 40, 20, 20);\n"
    "document.addEventListener('keydown', (e) => {\n"
    "  if (e.code === 'KeyJ') { window.fired = true; }\n"
    "});\n"
)
# Same render, but the fire handler was never wired (the §9b-i blind
# spot — bullets/projectile exist, nothing triggers a shot).
_GAME_WITHOUT_FIRE = (
    "const canvas = document.getElementById('game');\n"
    "const ctx = canvas.getContext('2d');\n"
    "ctx.fillStyle = '#3a3a3a'; ctx.fillRect(0, 0, 100, 100);\n"
    "ctx.fillStyle = '#1e6fd9'; ctx.fillRect(40, 40, 20, 20);\n"
)
# Setup drives the input; assert checks the effect post-settle.
_SETUP_PRESS_FIRE = "document.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyJ'}));\n"
_ASSERT_FIRE_WORKED = "return window.fired ? [] : ['fire (KeyJ) did not set window.fired'];\n"


@_requires_playwright
def test_smoke_step_fails_when_behavioral_assert_fails(tmp_path: Path) -> None:
    """harness-u1il5 core: a game that loads clean and renders (passes
    the render + blank-canvas gates) but whose fire handler is missing
    must FAIL the smoke once a behavioral probe drives KeyJ and asserts
    the effect. This is the input-handler-closes-blind class the
    render-only gate let through."""
    (tmp_path / "index.html").write_text(_BEHAVIORAL_INDEX)
    (tmp_path / "game.js").write_text(_GAME_WITHOUT_FIRE)
    harness_dir = tmp_path / ".harness"
    harness_dir.mkdir()
    (harness_dir / "smoke_setup.js").write_text(_SETUP_PRESS_FIRE)
    (harness_dir / "smoke_assert.js").write_text(_ASSERT_FIRE_WORKED)
    steps = default_workspace_verify_steps(tmp_path)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    exit_code, tail = _exec_test_cmd(smoke.cmd, tmp_path, shell_mode=smoke.shell)
    assert exit_code != 0
    assert "assert failure" in tail.lower()
    assert "fire" in tail.lower()


@_requires_playwright
def test_smoke_step_passes_when_behavioral_assert_passes(tmp_path: Path) -> None:
    """The working version (fire handler wired) — same setup + assert —
    passes: the probe drives KeyJ, the listener flips the flag, the
    assert returns an empty failure array, smoke exits 0."""
    (tmp_path / "index.html").write_text(_BEHAVIORAL_INDEX)
    (tmp_path / "game.js").write_text(_GAME_WITH_FIRE)
    harness_dir = tmp_path / ".harness"
    harness_dir.mkdir()
    (harness_dir / "smoke_setup.js").write_text(_SETUP_PRESS_FIRE)
    (harness_dir / "smoke_assert.js").write_text(_ASSERT_FIRE_WORKED)
    steps = default_workspace_verify_steps(tmp_path)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    exit_code, _tail = _exec_test_cmd(smoke.cmd, tmp_path, shell_mode=smoke.shell)
    assert exit_code == 0


@_requires_playwright
def test_smoke_step_fails_when_assert_throws(tmp_path: Path) -> None:
    """A probe that throws (references an undefined global, bad assertion
    code) fails loud rather than silently passing — a broken probe must
    not green-light a close."""
    (tmp_path / "index.html").write_text(_BEHAVIORAL_INDEX)
    (tmp_path / "game.js").write_text(_GAME_WITH_FIRE)
    harness_dir = tmp_path / ".harness"
    harness_dir.mkdir()
    (harness_dir / "smoke_assert.js").write_text("return nonexistentGlobal.value;\n")
    steps = default_workspace_verify_steps(tmp_path)
    smoke = next(s for s in steps if "smoke_runner" in s.cmd)
    exit_code, tail = _exec_test_cmd(smoke.cmd, tmp_path, shell_mode=smoke.shell)
    # A throwing probe fails loud (non-zero) and the JS error surfaces in
    # the tail — the prefix line may be pushed out of the tail window by
    # the stack trace, so assert on the stable error content.
    assert exit_code != 0
    assert "not defined" in tail.lower()


@_requires_playwright
def test_blank_canvas_check_disabled_by_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blank-canvas check has an off-switch
    (HARNESS_SMOKE_BLANK_CANVAS=0) for the rare canvas app that
    intentionally renders nothing on load. With it disabled, an
    otherwise-clean blank canvas passes."""
    (tmp_path / "index.html").write_text(
        "<!DOCTYPE html><html><body>"
        '<canvas id="game" width="100" height="100"></canvas>'
        '<script src="game.js"></script>'
        "</body></html>",
    )
    (tmp_path / "game.js").write_text(
        "const canvas = document.getElementById('game');\nconst ctx = canvas.getContext('2d');\n",
    )
    monkeypatch.setenv("HARNESS_SMOKE_BLANK_CANVAS", "0")
    steps = default_workspace_verify_steps(tmp_path)
    smoke_step = next(s for s in steps if "smoke_runner" in s.cmd)
    exit_code, _tail = _exec_test_cmd(smoke_step.cmd, tmp_path, shell_mode=smoke_step.shell)
    assert exit_code == 0


@_requires_playwright
def test_blank_canvas_suppressed_via_enforce_flag(tmp_path: Path) -> None:
    """harness-6dsn end-to-end: enforce_blank_canvas=False (the driver's
    early-phase setting) lets a legitimately-blank skeleton pass the
    smoke step, while console/page-error checks still apply. This is
    what lets an incremental from-scratch build clear §1 instead of
    failing blank-canvas on every turn (loop_run 8e73902f)."""
    (tmp_path / "index.html").write_text(
        "<!DOCTYPE html><html><body>"
        '<canvas id="game" width="100" height="100"></canvas>'
        '<script src="game.js"></script>'
        "</body></html>",
    )
    (tmp_path / "game.js").write_text(
        "const canvas = document.getElementById('game');\nconst ctx = canvas.getContext('2d');\n",
    )
    steps = default_workspace_verify_steps(tmp_path, enforce_blank_canvas=False)
    smoke_step = next(s for s in steps if "smoke_runner" in s.cmd)
    exit_code, _tail = _exec_test_cmd(smoke_step.cmd, tmp_path, shell_mode=smoke_step.shell)
    assert exit_code == 0


# --- harness-9ugc: missing entry-html guard --------------------------


def test_missing_entry_html_fires_on_browser_js_without_index(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("const ctx = canvas.getContext('2d');\n")
    reason = missing_entry_html_reason(tmp_path)
    assert reason is not None
    assert "index.html" in reason
    assert "SYNTAX-ONLY" in reason


def test_missing_entry_html_none_when_index_present(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("document.getElementById('x')\n")
    (tmp_path / "index.html").write_text("<script src='game.js'></script>")
    assert missing_entry_html_reason(tmp_path) is None


def test_missing_entry_html_none_when_no_browser_js(tmp_path: Path) -> None:
    # A Node script (no browser globals) + no index.html is legit — nothing
    # runtime to smoke-test, so no false-success risk.
    (tmp_path / "tool.js").write_text("const fs = require('fs'); fs.readFileSync('x');\n")
    assert missing_entry_html_reason(tmp_path) is None


def test_missing_entry_html_none_on_empty_workspace(tmp_path: Path) -> None:
    # A fresh §1 build with no source yet must not be refused.
    assert missing_entry_html_reason(tmp_path) is None


# --- extract_smoke_symptom (harness-estby handoff feedback) ----------


def test_extract_smoke_symptom_strips_command_echo_blank_canvas() -> None:
    reason = (
        "claim_without_close: /Users/x/.venv/bin/python3 -m harness.driver.smoke_run... "
        "exit=1: smoke-execute: canvas rendered nothing (entirely one color) after "
        "1500ms — draw loop likely not wired up: blank-canvas #0 (1280x960) uniform "
        "rgba=[0, 0, 0, 0]"
    )
    out = extract_smoke_symptom(reason)
    assert out is not None
    assert out.startswith("smoke-execute: canvas rendered nothing")
    # Command echo + exit prefix stripped.
    assert "smoke_runner" not in out
    assert "exit=1" not in out
    assert "claim_without_close" not in out


def test_extract_smoke_symptom_pageerror() -> None:
    reason = "verify_failed: node --check ... exit=1: pageerror: foo is not defined"
    out = extract_smoke_symptom(reason)
    assert out == "pageerror: foo is not defined"


def test_extract_smoke_symptom_none_for_non_smoke_failure() -> None:
    assert extract_smoke_symptom("fabrication_fallback fired") is None
    assert extract_smoke_symptom("verify_failed: pytest exit=1: 2 failed") is None
    assert extract_smoke_symptom("issue still open after turn") is None


def test_extract_smoke_symptom_none_for_empty() -> None:
    assert extract_smoke_symptom("") is None
    assert extract_smoke_symptom(None) is None


# --- smoke-setup scenario (harness-5vn6t verify-observability) --------


def test_smoke_step_adds_setup_flag_when_scenario_present(tmp_path: Path) -> None:
    from harness.driver.workspace_verify import _smoke_execute_step

    (tmp_path / "index.html").write_text("<canvas></canvas>")
    harness_dir = tmp_path / ".harness"
    harness_dir.mkdir()
    (harness_dir / "smoke_setup.js").write_text("player.wanted = 3; spawnPoliceCar();")
    step = _smoke_execute_step(tmp_path / "index.html", workspace=tmp_path)
    assert "--setup=" in step.cmd
    assert "smoke_setup.js" in step.cmd


def test_smoke_step_no_setup_flag_when_scenario_absent(tmp_path: Path) -> None:
    from harness.driver.workspace_verify import _smoke_execute_step

    (tmp_path / "index.html").write_text("<canvas></canvas>")
    step = _smoke_execute_step(tmp_path / "index.html", workspace=tmp_path)
    assert "--setup=" not in step.cmd


def test_read_setup_returns_source_when_present(tmp_path: Path) -> None:
    from harness.driver.smoke_runner import _read_setup

    f = tmp_path / "scene.js"
    f.write_text("spawnPoliceCar();")
    source, error = _read_setup(["prog", f"--setup={f}", "index.html"])
    assert source == "spawnPoliceCar();"
    assert error is None


def test_read_setup_none_when_no_flag(tmp_path: Path) -> None:
    from harness.driver.smoke_runner import _read_setup

    source, error = _read_setup(["prog", "index.html"])
    assert source is None
    assert error is None


def test_read_setup_errors_when_file_missing(tmp_path: Path) -> None:
    from harness.driver.smoke_runner import _read_setup

    source, error = _read_setup(["prog", f"--setup={tmp_path / 'nope.js'}", "index.html"])
    assert source is None
    assert error is not None
    assert "unreadable" in error


@_requires_playwright
def test_smoke_setup_scenario_throw_fails_smoke(tmp_path: Path) -> None:
    """End-to-end: a setup scenario that calls a missing function (the
    render/spawn fn the issue was supposed to provide) surfaces as a
    smoke failure — the verify-observability payoff."""
    from harness.driver import smoke_runner

    (tmp_path / "index.html").write_text(
        "<!doctype html><canvas id='c' width='8' height='8'></canvas>"
        "<script src='game.js'></script>"
    )
    (tmp_path / "game.js").write_text("// loads clean, defines no spawn fn\n")
    setup = tmp_path / ".harness" / "smoke_setup.js"
    setup.parent.mkdir()
    setup.write_text("spawnPoliceCarThatDoesNotExist();")
    rc = smoke_runner.main(
        ["prog", "--no-blank-canvas", f"--setup={setup}", str(tmp_path / "index.html")]
    )
    assert rc == 1


@_requires_playwright
def test_smoke_setup_scenario_benign_passes(tmp_path: Path) -> None:
    """A setup that runs cleanly doesn't fail the smoke."""
    from harness.driver import smoke_runner

    (tmp_path / "index.html").write_text(
        "<!doctype html><canvas id='c' width='8' height='8'></canvas>"
        "<script src='game.js'></script>"
    )
    (tmp_path / "game.js").write_text("window.__ok = true;\n")
    setup = tmp_path / ".harness" / "smoke_setup.js"
    setup.parent.mkdir()
    setup.write_text("window.__ok = true;")
    rc = smoke_runner.main(
        ["prog", "--no-blank-canvas", f"--setup={setup}", str(tmp_path / "index.html")]
    )
    assert rc == 0
