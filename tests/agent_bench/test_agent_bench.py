"""Unit coverage for the agent-bench rig: pure scorers + the results store.

These never touch gx10, node, or a browser — they pin the deterministic parts
(regex milestone probe, token/diff accounting, SQLite round-trip). The
external-tool scorers (`builds` via node, `runs_headless` via Playwright) are
exercised end-to-end by the runner, not here; `builds` gets a smoke test only
when node is present.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from adapters.base import RunArtifacts
from scorers.builds import BuildsScorer
from scorers.cost import CostScorer
from scorers.feature_checklist import FeatureChecklistScorer
from scorers.process import ProcessScorer
from store import ResultsStore, RunRecord

# A workspace that hits a known subset of milestones (m1, m2, m5, m7) and
# deliberately misses others, so the count is an exact assertion.
_GAME_JS = """
const ctx = canvas.getContext('2d');
let player = { angle: 0 };
addEventListener('keydown', e => {});
function loop() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.translate(-camera.x, -camera.y);
  drawTile();
  ctx.fillText('HUD', 10, 10);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
"""
_INDEX_HTML = '<canvas id="c" width="640" height="480"></canvas><script src="game.js"></script>'


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "game.js").write_text(_GAME_JS)
    (tmp_path / "index.html").write_text(_INDEX_HTML)
    return tmp_path


def _artifacts(**kw: object) -> RunArtifacts:
    base: dict[str, object] = {"exit_ok": True, "transcript": "", "diff": "", "duration_s": 1.0}
    base.update(kw)
    return RunArtifacts(**base)  # type: ignore[arg-type]


def test_feature_checklist_counts_exact_milestones(tmp_path: Path) -> None:
    scores = FeatureChecklistScorer().score(_workspace(tmp_path), _artifacts())
    assert scores["m1_loop"] is True  # requestAnimationFrame + keydown
    assert scores["m7_hud"] is True  # fillText + setTransform(1,0,0,1
    assert scores["m8_police"] is False  # no police/wanted signals
    assert scores["milestones_total"] == 8
    # reached == number of true milestone flags
    reached = sum(1 for k, v in scores.items() if k.startswith("m") and v is True)
    assert scores["milestones_reached"] == reached


def test_cost_trusts_artifact_tokens() -> None:
    scores = CostScorer().score(
        Path("."), _artifacts(tokens_prompt=100, tokens_completion=50, duration_s=10.0)
    )
    assert scores["tokens_total"] == 150
    assert scores["tokens_per_s"] == pytest.approx(5.0)  # completion / duration
    assert scores["cost_source"] == "artifacts"


def test_cost_falls_back_to_transcript_scrape() -> None:
    transcript = 'usage {"prompt_tokens": 30, "completion_tokens": 20}'
    scores = CostScorer().score(Path("."), _artifacts(transcript=transcript))
    assert (scores["tokens_prompt"], scores["tokens_completion"]) == (30, 20)
    assert scores["cost_source"] == "transcript"


def test_process_counts_diff_and_transcript_signals() -> None:
    diff = (
        "diff --git a/game.js b/game.js\n"
        "new file mode 100644\n"
        "--- /dev/null\n+++ b/game.js\n"
        "+line one\n+line two\n"
    )
    transcript = 'Traceback (most recent call last):\n{"type":"tool_use"}'
    scores = ProcessScorer().score(Path("."), _artifacts(diff=diff, transcript=transcript, turns=7))
    assert scores["files_created"] == 1
    assert scores["lines_added"] == 2
    assert scores["lines_removed"] == 0
    assert scores["turns"] == 7  # surfaced from artifacts, not guessed
    assert scores["tool_calls"] == 1
    assert scores["tracebacks"] == 1


def test_process_reports_zero_turns_when_unknown() -> None:
    scores = ProcessScorer().score(Path("."), _artifacts(turns=None))
    assert scores["turns"] == 0  # never guessed


def test_store_round_trips_scores_and_mirrors_jsonl(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "bench.sqlite3")
    rec = RunRecord(
        framework="aider",
        version="0.86.2",
        run_idx=0,
        spec_sha="abc123",
        model_id="qwen",
        exit_ok=True,
        duration_s=12.5,
        scores={"builds": True, "milestones_reached": 6},
        manifest={"temperature": 0.0},
    )
    run_id = store.append(rec)
    assert run_id == 1

    rows = store.all_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["framework"] == "aider"
    assert row["exit_ok"] == 1  # SQLite stores bool as int
    assert row["scores"] == {"builds": True, "milestones_reached": 6}
    assert row["manifest"] == {"temperature": 0.0}

    jsonl = (tmp_path / "bench.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_builds_passes_on_valid_game(tmp_path: Path) -> None:
    scores = BuildsScorer().score(_workspace(tmp_path), _artifacts())
    assert scores["builds"] is True
    assert scores["js_syntax_failures"] == 0
    assert scores["index_html"] is True
