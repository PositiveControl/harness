"""Tests for the runtime behavioral-gate synthesizer (pure) and the
fsm-side red-now adoption predicate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from harness.driver.fsm_turn import _adopt_synthesized_runtime_gate, _runtime_gate_red_now
from harness.driver.runtime_gate_synth import build_runtime_gates

_HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None
_requires_playwright = pytest.mark.skipif(
    not _HAS_PLAYWRIGHT,
    reason="runtime-gate adoption requires `playwright` + chromium installed",
)


def test_declines_when_not_browser_js() -> None:
    a = {"gap": "NPC cars in traffic never move; no position integration"}
    assert build_runtime_gates(a, is_browser_js=False) == []


def test_declines_on_empty_assessment() -> None:
    assert build_runtime_gates(None, is_browser_js=True) == []
    assert build_runtime_gates({}, is_browser_js=True) == []


def test_motion_candidate_from_position_gap() -> None:
    a = {
        "gap": "No position integration — there is no car.x += Math.cos(...). "
        "NPC cars in traffic never move.",
        "approach": "add x/y integration to the traffic update loop",
        "current_state": "traffic[] positions frozen",
    }
    cands = build_runtime_gates(a, is_browser_js=True)
    kinds = {c.kind for c in cands}
    assert "motion" in kinds
    # The real array name is surfaced as a candidate.
    descs = " ".join(c.description for c in cands)
    assert "traffic" in descs
    motion = next(c for c in cands if c.kind == "motion" and "traffic" in c.description)
    # Probe shape: setup stashes baseline, assert returns RGFAIL on no movement.
    assert "window.__rg0" in motion.setup_js
    assert "RGFAIL:" in motion.assert_js
    assert "return" in motion.assert_js


def test_no_motion_candidate_without_motion_signal() -> None:
    # Names a collection but no position/motion gap → no motion probe.
    a = {"gap": "the traffic array should be colored differently per car"}
    cands = build_runtime_gates(a, is_browser_js=True)
    assert [c for c in cands if c.kind == "motion"] == []


def test_reached_candidate_from_state_and_key() -> None:
    a = {
        "gap": "game.js never assigns mode='foot'. Exit is unreachable.",
        "approach": "on Enter or KeyF keydown set player.mode='foot'",
        "current_state": "player.mode='driving'",
    }
    cands = build_runtime_gates(a, is_browser_js=True)
    reached = [c for c in cands if c.kind == "reached"]
    assert reached
    # The unreached literal ('foot') is among the candidates; the dispatcher
    # fires the named keys.
    foot = next(c for c in reached if "foot" in c.description)
    assert "dispatchEvent" in foot.setup_js
    assert "KeyboardEvent" in foot.setup_js
    assert "'Enter'" in foot.setup_js or "'KeyF'" in foot.setup_js
    assert "RGFAIL:" in foot.assert_js


def test_no_reached_candidate_without_a_key() -> None:
    # State literal named but no trigger key → no reached probe.
    a = {"gap": "mode='foot' is never set anywhere in the file"}
    cands = build_runtime_gates(a, is_browser_js=True)
    assert [c for c in cands if c.kind == "reached"] == []


def test_candidates_are_bounded() -> None:
    # A noisy assessment can't fan out unbounded smoke launches.
    a = {
        "gap": "traffic cars npcs peds bullets enemies entities never move; "
        "no position integration anywhere. mode='a' mode='b' mode='c' "
        "mode='d' on Enter KeyF KeyQ KeyR keys",
        "approach": "x += stuff",
        "current_state": "frozen",
    }
    cands = build_runtime_gates(a, is_browser_js=True)
    motion = [c for c in cands if c.kind == "motion"]
    reached = [c for c in cands if c.kind == "reached"]
    assert len(motion) <= 3
    assert len(reached) <= 3


# ---- _runtime_gate_red_now adoption predicate ----------------------------


def test_red_now_true_on_gap_failure() -> None:
    tail = "assert failure: RGFAIL: no element of `traffic` changed position"
    assert _runtime_gate_red_now(1, tail) is True


def test_red_now_false_when_green() -> None:
    assert _runtime_gate_red_now(0, "OK: all required shapes present") is False


def test_red_now_false_on_probe_error() -> None:
    # A mis-extracted symbol throws → scenario error, never a false red.
    assert _runtime_gate_red_now(1, "assert-scenario error: traffic is not defined") is False
    assert _runtime_gate_red_now(1, "setup-scenario error: foo is not defined") is False


def test_red_now_false_without_sentinel() -> None:
    # A page that errors on load fails without our RGFAIL marker → not ours.
    assert _runtime_gate_red_now(1, "runtime errors detected on load: TypeError x") is False


# ---- end-to-end adoption against a real headless page --------------------

_INDEX_HTML = "<!doctype html><html><body><canvas id='g' width='64' height='64'></canvas>"
_INDEX_HTML += "<script src='game.js'></script></body></html>\n"

# A game whose NPC array is declared and spawned but whose loop NEVER
# integrates position — the exact frozen-traffic gap. The canvas is painted
# (non-blank) so the load-only smoke gate would pass; only stepping the loop
# and reading positions reveals the gap.
_FROZEN_GAME = """
let traffic = [{x: 10, y: 10, angle: 0, speed: 50}, {x: 20, y: 20, angle: 0, speed: 50}];
const ctx = document.getElementById('g').getContext('2d');
function update(dt) { for (const c of traffic) { c.speed += 1; } }  // accelerate, never move
function draw() { ctx.fillStyle = '#123'; ctx.fillRect(0, 0, 64, 64); }
let last = 0;
function loop(t) {
  const dt = (t - last) / 1000; last = t;
  update(dt); draw(); requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
"""

# Same game with position integration added — cars move; the gate must NOT
# adopt (green = nothing to gate).
_MOVING_GAME = _FROZEN_GAME.replace(
    "for (const c of traffic) { c.speed += 1; }",
    "for (const c of traffic) { c.speed += 1; c.x += Math.cos(c.angle) * c.speed * dt; }",
)

_TRAFFIC_ASSESSMENT = {
    "gap": "No position integration — no car.x += in the traffic loop. NPC cars never move.",
    "approach": "add x/y integration to the traffic update loop",
    "current_state": "traffic[] positions frozen",
}


def _write_workspace(tmp_path: Path, game_js: str) -> Path:
    (tmp_path / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    (tmp_path / "game.js").write_text(game_js, encoding="utf-8")
    return tmp_path


@_requires_playwright
def test_adopts_runtime_gate_when_traffic_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_SETTLE_MS", "500")
    ws = _write_workspace(tmp_path, _FROZEN_GAME)
    outcome = _adopt_synthesized_runtime_gate(
        ws, _TRAFFIC_ASSESSMENT, is_browser_js=True, issue_id="harness-test"
    )
    assert outcome is not None, "frozen traffic must produce a red-now runtime gate"
    assert outcome.kind == "failing_test_submitted"
    # The adopted gate's probe files survive (the carried gate references them).
    assert any((ws / ".harness").glob("rg_harness-test_*_assert.js"))


@_requires_playwright
def test_declines_runtime_gate_when_traffic_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARNESS_SMOKE_SETTLE_MS", "500")
    ws = _write_workspace(tmp_path, _MOVING_GAME)
    outcome = _adopt_synthesized_runtime_gate(
        ws, _TRAFFIC_ASSESSMENT, is_browser_js=True, issue_id="harness-test"
    )
    assert outcome is None, "moving traffic is green — nothing to gate, must decline"
    # Declined candidates clean up their probe files.
    assert not list((ws / ".harness").glob("rg_harness-test_*_assert.js"))
