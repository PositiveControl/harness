"""Framework-agnostic matrix runner.

For each (framework, run_idx): fresh isolated workspace -> health-gate gx10 ->
adapter.invoke -> run the metrics.yaml scorers -> append one attributed row +
persist artifacts. Knows nothing framework-specific; that lives in adapters/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from adapters.aider import AiderAdapter
from adapters.base import Adapter, Endpoint, RunArtifacts
from scorers.base import Scorer, Scores
from scorers.builds import BuildsScorer
from scorers.feature_checklist import FeatureChecklistScorer
from scorers.runs_headless import RunsHeadlessScorer
from store import ResultsStore, RunRecord

ROOT = Path(__file__).resolve().parent

# Registries. Add a framework / metric by adding a line here.
ADAPTERS: dict[str, type[Adapter]] = {
    "aider": AiderAdapter,
}
SCORERS: dict[str, type[Scorer]] = {
    "builds": BuildsScorer,
    "runs_headless": RunsHeadlessScorer,
    "feature_checklist": FeatureChecklistScorer,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="agent-bench matrix runner")
    ap.add_argument("--config", default=str(ROOT / "bench.yaml"))
    ap.add_argument("--framework", help="run only this framework (default: all in config)")
    ap.add_argument("--runs", type=int, help="override runs_per_framework")
    ap.add_argument("--no-health-gate", action="store_true", help="skip gx10 /v1/models check")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    gx10 = Endpoint(
        base_url=cfg["gx10"]["base_url"],
        model=cfg["gx10"]["model"],
        api_key=cfg["gx10"].get("api_key", "gx10"),
        temperature=float(cfg["gx10"].get("temperature", 0.0)),
    )
    runs = args.runs or int(cfg["runs_per_framework"])
    timeout_s = int(cfg["timeout_s"])
    spec_path = ROOT / cfg["spec"]
    spec = spec_path.read_text()
    spec_sha = hashlib.sha256(spec.encode()).hexdigest()[:12]
    metrics = yaml.safe_load((ROOT / cfg["metrics_file"]).read_text())["metrics"]
    store = ResultsStore(ROOT / cfg["store_db"])
    results_dir = ROOT / cfg["results_dir"]

    frameworks = [args.framework] if args.framework else list(cfg["frameworks"])
    _validate(frameworks, metrics)

    if not args.no_health_gate and not _health_ok(gx10):
        print(f"[bench] ABORT: gx10 health-gate failed for {gx10.base_url} / {gx10.model}")
        return 2

    scorers = [SCORERS[m]() for m in metrics]
    for fw in frameworks:
        adapter = ADAPTERS[fw]()
        for run_idx in range(runs):
            print(f"[bench] {fw} run {run_idx + 1}/{runs} ...")
            _do_run(
                adapter=adapter,
                gx10=gx10,
                spec=spec,
                spec_sha=spec_sha,
                timeout_s=timeout_s,
                run_idx=run_idx,
                scorers=scorers,
                results_dir=results_dir,
                store=store,
            )
    print(f"[bench] done. store: {ROOT / cfg['store_db']}")
    return 0


def _do_run(
    *,
    adapter: Adapter,
    gx10: Endpoint,
    spec: str,
    spec_sha: str,
    timeout_s: int,
    run_idx: int,
    scorers: list[Scorer],
    results_dir: Path,
    store: ResultsStore,
) -> None:
    run_dir = results_dir / adapter.name / str(run_idx)
    workspace = run_dir / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)  # fresh, isolated — never cross-contaminate runs
    workspace.mkdir(parents=True)

    adapter.prepare(workspace)
    artifacts = adapter.invoke(spec, gx10, workspace, timeout_s)

    scores: Scores = {}
    for scorer in scorers:
        scores.update(scorer.score(workspace, artifacts))

    _persist_artifacts(run_dir, artifacts, scores)
    store.append(
        RunRecord(
            framework=adapter.name,
            version=adapter.version,
            run_idx=run_idx,
            spec_sha=spec_sha,
            model_id=gx10.model,
            exit_ok=artifacts.exit_ok,
            duration_s=artifacts.duration_s,
            scores=dict(scores),
            manifest={
                "base_url": gx10.base_url,
                "temperature": gx10.temperature,
                "timeout_s": timeout_s,
                "tokens_prompt": artifacts.tokens_prompt,
                "tokens_completion": artifacts.tokens_completion,
                "turns": artifacts.turns,
            },
        )
    )


def _persist_artifacts(run_dir: Path, artifacts: RunArtifacts, scores: Scores) -> None:
    (run_dir / "transcript.txt").write_text(artifacts.transcript)
    (run_dir / "result.diff").write_text(artifacts.diff)
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))


def _validate(frameworks: list[str], metrics: list[str]) -> None:
    unknown_fw = [f for f in frameworks if f not in ADAPTERS]
    unknown_m = [m for m in metrics if m not in SCORERS]
    if unknown_fw:
        raise SystemExit(f"unknown framework(s): {unknown_fw}; known: {list(ADAPTERS)}")
    if unknown_m:
        raise SystemExit(f"unknown metric(s): {unknown_m}; known: {list(SCORERS)}")


def _health_ok(gx10: Endpoint) -> bool:
    """Probe /v1/models; require the served model id to match the pinned one."""
    url = gx10.base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {gx10.api_key}"})  # noqa: S310 — http(s) to the configured gx10 endpoint
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            served = {m.get("id") for m in json.load(resp).get("data", [])}
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"[bench] health probe error: {exc}")
        return False
    if gx10.model not in served:
        print(f"[bench] served models {served} do not include pinned {gx10.model!r}")
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
