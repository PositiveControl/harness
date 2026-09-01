"""Eval-blind gate tell — harness-815wm (loop_run=069d6172).

Run 069d6172 parked harness-xdgtb after every authored gate followed the
same shape: stub the browser globals, ``eval(fs.readFileSync('game.js'))``,
assert on a top-level binding. Direct-eval ``let``/``const`` bindings never
escape the eval scope, so the gate threw ``ReferenceError: traffic is not
defined`` while ``game.js`` declared ``let traffic = []`` — red forever,
3 verify retries + a re-authored gate burned against it.

Unit tests cover ``eval_blind_reference`` (the pure tell); the turn-level
tests pin the first-failure halt and the submit-time lint rejection.

The runnable fake gates are Python scripts saved with a ``.js`` name: the
tell reads the script resolved out of ``test_cmd`` as TEXT (looking for
the eval/readFileSync markers), so a Python body carrying the markers in
comments exercises the real path without a Node dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.driver.fsm_executor import _lint_submitted_gate, run_fsm_turn
from harness.driver.gate_blind import (
    eval_blind_reference,
    eval_blind_typeof_guard,
    first_blind_tell,
    phantom_member_assertion,
    regex_body_truncation,
)
from harness.driver.turn_fsm import TurnPhase
from tests.test_fsm_gate_suspect import (
    _ClosedBd,
    _FakeCharacter,
    _handoff,
    _OpenBd,
    _scripted_tool_loop,
)

_REFERENCE_ERROR_OUTPUT = (
    "=== Testing Traffic Array Initialization ===\n"
    "ReferenceError: traffic is not defined\n"
    "    at testTrafficArrayInitialization (test_gate.js:22:40)\n"
    "    at Object.<anonymous> (test_gate.js:188:19)\n"
    "Node.js v23.11.0"
)


def _blind_test_file(workspace: Path, *, name: str = "test_gate.js") -> str:
    """The eval-the-source idiom: reads game.js, direct-evals it, asserts
    on a top-level binding from outside the eval'd string."""
    (workspace / name).write_text(
        "const fs = require('fs');\n"
        "eval(fs.readFileSync('game.js', 'utf8'));\n"
        "if (!Array.isArray(traffic)) { process.exit(1); }\n"
    )
    return name


def _typeof_blind_file(
    workspace: Path, *, name: str = "test_typeof.js", ident: str = "foot"
) -> str:
    """The typeof-guarded trap: eval the source, then `typeof <ident>` —
    returns 'undefined' for a trapped let/const, no ReferenceError thrown."""
    (workspace / name).write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "eval(src);\n"
        f"if (typeof {ident} !== 'object') {{ process.exit(1); }}\n"
    )
    return name


# --- eval_blind_typeof_guard unit tests -----------------------------


def test_typeof_guard_flags_trapped_let_binding(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("let foot = { x: 0 };\nfunction update(dt) {}\n")
    test_path = _typeof_blind_file(tmp_path)

    diag = eval_blind_typeof_guard(test_path, tmp_path)

    assert diag is not None
    assert "`foot`" in diag
    assert "typeof" in diag
    assert "INTO the eval'd string" in diag


def test_typeof_guard_flags_inline_eval_readfilesync(tmp_path: Path) -> None:
    """The inline `eval(fs.readFileSync(...))` form (no intermediate var) is
    the same trap and must also be flagged."""
    (tmp_path / "game.js").write_text("const foot = { x: 0 };\n")
    (tmp_path / "test_inline.js").write_text(
        "const fs = require('fs');\n"
        "eval(fs.readFileSync('game.js', 'utf8'));\n"
        "if (typeof foot !== 'object') { process.exit(1); }\n"
    )

    assert eval_blind_typeof_guard("test_inline.js", tmp_path) is not None


def test_typeof_guard_ignores_append_into_eval_idiom(tmp_path: Path) -> None:
    """Working idiom: assertions concatenated INTO the eval string share its
    scope, so the binding is visible — not blind, never flagged."""
    (tmp_path / "game.js").write_text("let foot = { x: 0 };\n")
    (tmp_path / "test_ok.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "eval(src + '\\n;if (typeof foot !== \"object\") process.exit(1);');\n"
    )

    assert eval_blind_typeof_guard("test_ok.js", tmp_path) is None


def test_typeof_guard_ignores_undeclared_deliverable(tmp_path: Path) -> None:
    """`typeof` on a name the source does NOT declare top-level let/const is
    the genuine gap (the bead must build it), never the trap."""
    (tmp_path / "game.js").write_text("function update(dt) {}\n")
    test_path = _typeof_blind_file(tmp_path, ident="spawnFoot")

    assert eval_blind_typeof_guard(test_path, tmp_path) is None


def test_typeof_guard_none_for_text_regex_gate(tmp_path: Path) -> None:
    """The recommended source-text gate (readFileSync + regex, no eval) is
    exactly what we steer toward — must never be flagged."""
    (tmp_path / "game.js").write_text("let foot = {};\n")
    (tmp_path / "test_text.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "if (!/let foot/.test(src)) process.exit(1);\n"
    )

    assert eval_blind_typeof_guard("test_text.js", tmp_path) is None


# --- regex_body_truncation unit tests -------------------------------


def _body_truncation_file(workspace: Path, *, name: str = "test_body.js") -> str:
    """The k18er trap: read the source, slice a function body with a
    non-nesting `{[^}]*}` regex, assert on the (truncated) slice."""
    (workspace / name).write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "const body = src.match(/function\\s+update\\([^)]*\\)\\s*{[^}]*}/s)[0];\n"
        "if (!/player\\.mode/.test(body)) process.exit(1);\n"
    )
    return name


def test_body_truncation_flags_nonnesting_slice(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("function update(dt) { if (x) { foo(); } }\n")
    test_path = _body_truncation_file(tmp_path)

    diag = regex_body_truncation(test_path, tmp_path)

    assert diag is not None
    assert "FIRST inner" in diag
    assert "src.includes" in diag


def test_body_truncation_matches_plus_quantifier(tmp_path: Path) -> None:
    """`{[^}]+}` is the same non-nesting trap as `{[^}]*}`."""
    (tmp_path / "game.js").write_text("function update() { if (x) {} }\n")
    (tmp_path / "test_plus.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "if (!/update\\(\\)\\s*{[^}]+}/.test(src)) process.exit(1);\n"
    )

    assert regex_body_truncation("test_plus.js", tmp_path) is not None


def test_body_truncation_none_for_whole_source_regex(tmp_path: Path) -> None:
    """A regex on the FULL source (no body slice) is the correct idiom —
    must never be flagged."""
    (tmp_path / "game.js").write_text("function update() {}\n")
    (tmp_path / "test_ok.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "if (!/player\\.mode\\s*===\\s*'foot'/.test(src)) process.exit(1);\n"
    )

    assert regex_body_truncation("test_ok.js", tmp_path) is None


def test_body_truncation_none_when_source_not_read(tmp_path: Path) -> None:
    """Conservative: the `{[^}]*}` token in a test that never reads source
    (asserts on its own scope) isn't the source-blind trap — not flagged."""
    (tmp_path / "test_noread.js").write_text(
        "const body = someString.match(/f\\(\\)\\s*{[^}]*}/)[0];\n"
        "if (!/x/.test(body)) process.exit(1);\n"
    )

    assert regex_body_truncation("test_noread.js", tmp_path) is None


def test_body_truncation_none_for_missing_file(tmp_path: Path) -> None:
    assert regex_body_truncation("nonexistent.js", tmp_path) is None
    assert regex_body_truncation(None, tmp_path) is None


# --- phantom_member_assertion unit tests ----------------------------


def _phantom_member_file(workspace: Path, *, name: str = "test_phantom.js") -> str:
    """The o4cbj shape: reads source, then asserts (required-present) on
    `player.car.speed` — a path the source never produces because speed
    lives flat on `player.speed`."""
    (workspace / name).write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "const patterns = [\n"
        "  /Math\\.abs\\(player\\.car\\.speed\\)\\s*<\\s*20/,\n"
        "  /player\\.car\\.speed\\s*=\\s*0/,\n"
        "];\n"
        "for (const p of patterns) { if (!p.test(src)) process.exit(1); }\n"
    )
    return name


def test_phantom_member_flags_invented_intermediate_segment(tmp_path: Path) -> None:
    """`player.car.speed` is absent from source while `player.speed` IS
    present — the gate names a member path the source can't produce."""
    (tmp_path / "game.js").write_text(
        "let player = { x: 0, speed: 0 };\nplayer.speed = 0;\nplayer.car = null;\n"
    )
    test_path = _phantom_member_file(tmp_path)

    diag = phantom_member_assertion(test_path, tmp_path)

    assert diag is not None
    assert "`player.car.speed`" in diag
    assert "`player.speed`" in diag
    assert "game.js" in diag


def test_phantom_member_none_when_full_path_present(tmp_path: Path) -> None:
    """`player.car.driver = null` genuinely lives on `player.car` — the full
    path IS in the source, so it is not a phantom and must be accepted."""
    (tmp_path / "game.js").write_text("player.car = c;\nplayer.car.driver = null;\n")
    (tmp_path / "test_driver.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "if (!/player\\.car\\.driver\\s*=\\s*null/.test(src)) process.exit(1);\n"
    )

    assert phantom_member_assertion("test_driver.js", tmp_path) is None


def test_phantom_member_none_for_greenfield_add(tmp_path: Path) -> None:
    """Conservative: when NEITHER `player.car.speed` nor `player.speed` is in
    the source yet, the gate may legitimately require the impl to add it —
    no contradiction, so never flag."""
    (tmp_path / "game.js").write_text("let player = { x: 0 };\n")
    _phantom_member_file(tmp_path, name="test_add.js")

    assert phantom_member_assertion("test_add.js", tmp_path) is None


def test_phantom_member_none_when_only_negated(tmp_path: Path) -> None:
    """A should-be-ABSENT assertion (`!src.includes(path)`) goes green when
    the path is absent — that is the gap, not a phantom. A path whose every
    occurrence is `!`-negated on its line is never flagged."""
    (tmp_path / "game.js").write_text("let player = { speed: 0 };\nplayer.speed = 0;\n")
    (tmp_path / "test_neg.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "if (!src.includes('player.car.speed')) { /* expected absent */ }\n"
    )

    assert phantom_member_assertion("test_neg.js", tmp_path) is None


def test_phantom_member_none_when_source_not_read(tmp_path: Path) -> None:
    """A `player.car.speed` mention in a test that never reads source isn't
    the source-blind trap."""
    (tmp_path / "test_noread.js").write_text(
        "const player = { car: { speed: 5 } };\nif (player.car.speed < 20) process.exit(1);\n"
    )

    assert phantom_member_assertion("test_noread.js", tmp_path) is None


def test_phantom_member_none_for_missing_file(tmp_path: Path) -> None:
    assert phantom_member_assertion("nonexistent.js", tmp_path) is None
    assert phantom_member_assertion(None, tmp_path) is None


# --- eval_blind_reference unit tests --------------------------------


def test_flags_let_binding_the_eval_d_source_declares(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("let traffic = [];\nfunction update(dt) {}\n")
    test_path = _blind_test_file(tmp_path)

    diag = eval_blind_reference(_REFERENCE_ERROR_OUTPUT, test_path, tmp_path)

    assert diag is not None
    assert "`traffic`" in diag
    assert "game.js" in diag
    # The diagnostic must teach a working idiom, not just name the problem.
    assert "INTO the eval'd string" in diag


def test_flags_const_binding_too(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("const traffic = [];\n")
    test_path = _blind_test_file(tmp_path)

    assert eval_blind_reference(_REFERENCE_ERROR_OUTPUT, test_path, tmp_path) is not None


def test_undeclared_name_is_the_genuine_gap_not_flagged(tmp_path: Path) -> None:
    """``ReferenceError: traffic is not defined`` where game.js never
    declares traffic IS the gap the bead exists to close — never flag."""
    (tmp_path / "game.js").write_text("let pedestrians = [];\nfunction update(dt) {}\n")
    test_path = _blind_test_file(tmp_path)

    assert eval_blind_reference(_REFERENCE_ERROR_OUTPUT, test_path, tmp_path) is None


def test_var_binding_not_flagged(tmp_path: Path) -> None:
    """``var`` leaks out of a sloppy direct eval, so the gate CAN see it —
    a ReferenceError on a var name means something else went wrong."""
    (tmp_path / "game.js").write_text("var traffic = [];\n")
    test_path = _blind_test_file(tmp_path)

    assert eval_blind_reference(_REFERENCE_ERROR_OUTPUT, test_path, tmp_path) is None


def test_test_without_eval_not_flagged(tmp_path: Path) -> None:
    """A source-text gate (readFileSync + regex, no eval) that throws a
    ReferenceError is broken for its own reasons, not eval-blind."""
    (tmp_path / "game.js").write_text("let traffic = [];\n")
    (tmp_path / "test_gate.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "if (!/let traffic/.test(src)) { process.exit(1); }\n"
    )

    assert eval_blind_reference(_REFERENCE_ERROR_OUTPUT, "test_gate.js", tmp_path) is None


def test_no_reference_error_in_output_not_flagged(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("let traffic = [];\n")
    test_path = _blind_test_file(tmp_path)

    assert (
        eval_blind_reference("AssertionError: expected 6 cars, got 0", test_path, tmp_path) is None
    )


def test_unresolvable_test_path_not_flagged(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("let traffic = [];\n")

    assert eval_blind_reference(_REFERENCE_ERROR_OUTPUT, None, tmp_path) is None
    assert eval_blind_reference(_REFERENCE_ERROR_OUTPUT, "missing.js", tmp_path) is None


def test_reference_error_buried_in_long_output_is_found(tmp_path: Path) -> None:
    """The ReferenceError line sits at the HEAD of a Node stack dump — the
    tell must work on full output where a 200-char tail would miss it."""
    (tmp_path / "game.js").write_text("let traffic = [];\n")
    test_path = _blind_test_file(tmp_path)
    output = _REFERENCE_ERROR_OUTPUT + "\n" + ("    at frame (loader:1:1)\n" * 50)

    assert eval_blind_reference(output, test_path, tmp_path) is not None


# --- submit-time lint (third shape) ---------------------------------


def test_submit_lint_rejects_eval_blind_gate(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("let traffic = [];\n")
    test_path = _blind_test_file(tmp_path)

    rejection = _lint_submitted_gate(
        tmp_path, test_path, "node test_gate.js", 1, _REFERENCE_ERROR_OUTPUT
    )

    assert rejection is not None
    assert "structural" in rejection
    assert "INTO the eval'd string" in rejection


def test_submit_lint_accepts_gate_red_on_the_genuine_gap(tmp_path: Path) -> None:
    """Same failure output, but the source does NOT declare the name —
    the red is the gap; the gate must be accepted."""
    (tmp_path / "game.js").write_text("let pedestrians = [];\n")
    test_path = _blind_test_file(tmp_path)

    assert (
        _lint_submitted_gate(tmp_path, test_path, "node test_gate.js", 1, _REFERENCE_ERROR_OUTPUT)
        is None
    )


def test_submit_lint_rejects_typeof_guarded_trap(tmp_path: Path) -> None:
    """loop_run=55568949: the typeof-guarded trap throws no ReferenceError,
    so the runtime tell misses it — its custom FAIL tail carries no error to
    parse. The submit lint must still reject it via the static read."""
    (tmp_path / "game.js").write_text("let foot = { x: 0 };\nfunction update(dt) {}\n")
    test_path = _typeof_blind_file(tmp_path, name="test_foot.js")
    custom_tail = "FAIL: foot state not found or not an object"

    rejection = _lint_submitted_gate(tmp_path, test_path, "node test_foot.js", 1, custom_tail)

    assert rejection is not None
    assert "structural" in rejection
    assert "`foot`" in rejection
    assert "INTO the eval'd string" in rejection


def test_submit_lint_rejects_body_truncation_gate(tmp_path: Path) -> None:
    """harness-k18er: a `{[^}]*}` body-slice gate reads the source and throws
    nothing, so both eval tells miss it — the static body-truncation read must
    reject it at submit, before the byte-identical detector burns the budget."""
    (tmp_path / "game.js").write_text("function update(dt) { if (x) { foo(); } }\n")
    test_path = _body_truncation_file(tmp_path, name="test_body.js")
    custom_tail = "FAIL: No foot mode detection found in update function"

    rejection = _lint_submitted_gate(tmp_path, test_path, "node test_body.js", 1, custom_tail)

    assert rejection is not None
    assert "structural" in rejection
    assert "FIRST inner" in rejection
    assert "src.includes" in rejection


def test_submit_lint_rejects_phantom_member_gate(tmp_path: Path) -> None:
    """harness-o4cbj: a gate asserting `player.car.speed` against a source
    that models speed flat as `player.speed` reds out byte-identically across
    every IMPLEMENT pass. The static phantom-member read must reject it at
    submit, before the verify detector burns the whole attempt budget."""
    (tmp_path / "game.js").write_text(
        "let player = { x: 0, speed: 0 };\nplayer.speed = 0;\nplayer.car = null;\n"
    )
    test_path = _phantom_member_file(tmp_path, name="test_phantom.js")
    custom_tail = "FAIL: Missing required patterns (0, 1)"

    rejection = _lint_submitted_gate(tmp_path, test_path, "node test_phantom.js", 1, custom_tail)

    assert rejection is not None
    assert "structural" in rejection
    assert "`player.car.speed`" in rejection
    assert "`player.speed`" in rejection


def test_submit_lint_accepts_member_path_the_source_uses(tmp_path: Path) -> None:
    """A gate whose required path genuinely lives on the source object
    (`player.car.driver`) is not a phantom — it must be accepted."""
    (tmp_path / "game.js").write_text("player.car = c;\nplayer.car.driver = null;\n")
    (tmp_path / "test_driver.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "if (!/player\\.car\\.driver\\s*=\\s*null/.test(src)) process.exit(1);\n"
    )

    assert (
        _lint_submitted_gate(tmp_path, "test_driver.js", "node test_driver.js", 1, "FAIL") is None
    )


# --- turn-level: first-failure halt ----------------------------------


def _blind_runnable_gate(workspace: Path) -> str:
    """A gate the FSM can actually execute: Python body (constant red,
    emitting the xdgtb-shaped ReferenceError) under a .js name whose TEXT
    carries the eval/readFileSync markers the tell looks for."""
    (workspace / "test_gate.js").write_text(
        "# eval(fs.readFileSync('game.js', 'utf8'))  # idiom markers\n"
        "import sys\n"
        "sys.stderr.write(\n"
        "    'ReferenceError: traffic is not defined\\n'\n"
        "    '    at testTraffic (test_gate.js:22:40)\\n'\n"
        "    'Node.js v23.11.0'\n"
        ")\n"
        "sys.exit(1)\n"
    )
    return "python3 test_gate.js"


def test_gate_blind_halts_on_first_verify_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-815wm acceptance: the xdgtb shape halts on the FIRST verify
    failure — one IMPLEMENT pass, no verify-retry ceiling, no byte-identical
    second lap — and drops the carried test for re-authoring."""
    (tmp_path / "game.js").write_text("let traffic = [];\nfunction update(dt) {}\n")
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]  # never reached; run_tool_loop is stubbed
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd=_blind_runnable_gate(tmp_path),
    )

    assert not result.succeeded
    assert result.final_phase == TurnPhase.HALTED
    assert "gate-blind verify gate" in result.reason
    assert "INTO the eval'd string" in result.reason
    assert result.gate_suspect
    assert result.last_test_cmd is None
    assert phases_seen.count("implement") == 1


def test_genuine_gap_reference_error_does_not_trip_gate_blind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same gate, but the source never declares `traffic` — the red is the
    gap. The first failure must NOT halt gate-blind; the constant tail then
    reaches the ordinary byte-identical suspect path on the second lap."""
    (tmp_path / "game.js").write_text("let pedestrians = [];\n")
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd=_blind_runnable_gate(tmp_path),
    )

    assert not result.succeeded
    assert "gate-blind" not in result.reason
    assert "suspect verify gate" in result.reason
    assert phases_seen.count("implement") == 2


def test_suspect_gate_halt_names_pitfall_on_browser_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-815wm: the byte-identical suspect halt used to recommend a
    gate that 'LOADS the source' — steering the model back into the blind
    eval idiom. On a browser-JS workspace it must name the let/const
    pitfall and a working idiom instead."""
    (tmp_path / "game.js").write_text(
        "const canvas = document.getElementById('gameCanvas');\nlet traffic = [];\n"
    )
    # Constant red WITHOUT the eval/readFileSync markers and WITHOUT a
    # declared-name ReferenceError, so only the byte-identical path fires.
    (tmp_path / "fail_const.py").write_text(
        "import sys\nsys.stderr.write('gap: traffic spawn missing')\nsys.exit(1)\n"
    )
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd="python3 fail_const.py",
    )

    assert "suspect verify gate" in result.reason
    assert "LOADS the source" not in result.reason
    assert "browser-JS workspace" in result.reason
    assert "INTO the eval'd string" in result.reason


def test_write_test_hint_names_scoping_pitfall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The browser-JS WRITE_TEST hint must teach the let/const-eval
    pitfall and the assertions-inside-the-eval idiom."""
    (tmp_path / "game.js").write_text("const canvas = document.getElementById('gameCanvas');\n")
    phases_seen: list[str] = []
    prompts: dict[str, str] = {}
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, prompts, write_test_action="skip"),
    )

    run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_ClosedBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
    )

    hint = prompts["write_test"]
    assert "SCOPING PITFALL" in hint
    assert "eval(src + " in hint


# --- first_blind_tell: the shared roster ----------------------------


def test_first_blind_tell_includes_phantom_member(tmp_path: Path) -> None:
    """Regression (loop_run=135f0d99): the shared entry point must run the
    FULL roster — including phantom_member_assertion, the 4th trap. The
    carried-gate VERIFY check drifted to only the first three, so an o4cbj
    phantom-member gate got no first-failure tell."""
    (tmp_path / "game.js").write_text(
        "let player = { x: 0, speed: 0 };\nplayer.speed = 0;\nplayer.car = null;\n"
    )
    test_path = _phantom_member_file(tmp_path)

    # Output carries no ReferenceError, so only the static phantom-member
    # read can catch it — the trap the VERIFY site used to omit.
    diag = first_blind_tell("FAIL: Missing required patterns (0, 1)", test_path, tmp_path)

    assert diag is not None
    assert "`player.car.speed`" in diag
    assert "`player.speed`" in diag


def test_first_blind_tell_none_for_clean_gate(tmp_path: Path) -> None:
    """A gate whose required path genuinely lives on the source object is no
    trap on any roster entry — the shared tell returns None."""
    (tmp_path / "game.js").write_text("player.car = c;\nplayer.car.driver = null;\n")
    (tmp_path / "test_clean.js").write_text(
        "const fs = require('fs');\n"
        "const src = fs.readFileSync('game.js', 'utf8');\n"
        "if (!/player\\.car\\.driver\\s*=\\s*null/.test(src)) process.exit(1);\n"
    )

    assert first_blind_tell("FAIL", "test_clean.js", tmp_path) is None


def _phantom_runnable_gate(workspace: Path) -> str:
    """A carried phantom-member gate the FSM can execute: Python body under a
    `.js` name, exiting red WITHOUT a ReferenceError (so only the static
    phantom-member read can catch it) while its TEXT carries the readFileSync
    marker and a required-present `player.car.speed` path."""
    (workspace / "test_phantom_gate.js").write_text(
        "# const src = fs.readFileSync('game.js', 'utf8')  # idiom marker\n"
        "# required-present assertion: player.car.speed must appear in src\n"
        "import sys\n"
        "sys.stderr.write('FAIL: missing required pattern player.car.speed\\n')\n"
        "sys.exit(1)\n"
    )
    return "python3 test_phantom_gate.js"


def test_carried_phantom_member_gate_halts_on_first_verify_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-o4cbj / loop_run=135f0d99 acceptance: a phantom-member gate
    CARRIED from a prior attempt (reused without re-submitting, so it never
    sees the submit-time lint) must now halt gate-blind on the FIRST verify
    failure — not burn the verify-retry ceiling + a byte-identical second lap
    before parking. Proves first_blind_tell wired the 4th trap into the
    carried-gate VERIFY path."""
    (tmp_path / "game.js").write_text(
        "let player = { x: 0, speed: 0 };\nplayer.speed = 0;\nplayer.car = null;\n"
    )
    phases_seen: list[str] = []
    monkeypatch.setattr(
        "harness.driver.fsm_executor.run_tool_loop",
        _scripted_tool_loop(phases_seen, {}),
    )

    result = run_fsm_turn(
        adapter=None,  # type: ignore[arg-type]  # never reached; run_tool_loop is stubbed
        character=_FakeCharacter(),  # type: ignore[arg-type]
        bd=_OpenBd(),  # type: ignore[arg-type]
        handoff_builder=_handoff,
        workspace=tmp_path,
        current_issue_id="harness-x",
        prior_test_cmd=_phantom_runnable_gate(tmp_path),
    )

    assert not result.succeeded
    assert result.final_phase == TurnPhase.HALTED
    assert "gate-blind verify gate" in result.reason
    assert "`player.speed`" in result.reason
    assert result.gate_suspect
    assert result.last_test_cmd is None
    assert phases_seen.count("implement") == 1
