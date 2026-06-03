# agent-bench

Benchmark rig: give each agent framework the same autonomous build task — a
GTA-style game — against one shared gx10 vLLM endpoint, then score the result.

See `../docs/agent-framework-bench.md` for the full plan.

## Layout

```
spec/gta-spec.md   the one task input (M1..M7 milestones)
bench.yaml         frameworks, runs-per-framework, gx10 endpoint, paths
metrics.yaml       ordered scorer plugins to run (the adjustable surface)
adapters/          one thin wrapper per framework (framework SDKs live ONLY here)
scorers/           pluggable metric collectors (drop-in, listed in metrics.yaml)
runner.py          framework-agnostic matrix runner
store.py           append-first SQLite + JSONL results store
report.py          aggregate -> tables + per-metric distributions
results/           per-run artifacts (transcript, diff, scores.json)
```

## Run

```bash
cd agent-bench
python runner.py --runs 5                 # full matrix from bench.yaml
python runner.py --framework aider --runs 1
python report.py                          # tables from the store
```

## Extend

- **New framework**: add `adapters/<name>.py` implementing the `Adapter` protocol;
  register it in `ADAPTERS` (runner.py).
- **New metric**: add `scorers/<name>.py` implementing the `Scorer` protocol;
  register it in `SCORERS` (runner.py) and list its name in `metrics.yaml`.
  No runner edit beyond the registry line; scores are schemaless JSON per row.

## Status

Scaffold. Wired end-to-end with the **aider** adapter and three scorers
(`builds`, `runs_headless`, `feature_checklist`). Remaining adapters (OpenHands,
OpenCode, Goose, mini-SWE-agent) and cost/process scorers are TODO — see the plan.
