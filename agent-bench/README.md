# agent-bench

Benchmark rig: give each agent framework the same autonomous build task — a
top-down **GTA2-style web game** (HTML5 canvas + vanilla JS: `index.html` +
`game.js`) — against one shared gx10 vLLM endpoint, then score the result.

See `../docs/agent-framework-bench.md` for the full plan.

## Prerequisites

- **node** (≥18) — JS syntax check in the `builds` scorer (`node --check`).
- **Playwright + chromium** — the `runs_headless` scorer loads the generated
  page in a real headless browser. Install once:
  `uv pip install playwright && uv run playwright install chromium`.
- A framework CLI on PATH for each adapter you run (`aider`, `opencode`, …).

## Layout

```
spec/gta-spec.md   the one task input (HTML/CSS/JS GTA2, milestones M1..M8)
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

## Scorers (web stack)

- `builds` — `node --check` over every `.js`; `index.html` has `<canvas>` + `<script>`.
- `runs_headless` — Playwright chromium loads `index.html`, runs ~3s, asserts:
  zero console errors (`loads_clean`), `requestAnimationFrame` ran (`loop_alive`),
  canvas drew non-blank pixels (`renders`).
- `feature_checklist` — static regex probe of milestones M1..M8 over `game.js` + `index.html`.
- `cost` — tokens / wall-clock / tokens·s⁻¹ / $ (env-priced).
- `process` — files touched/created, diff LOC, turns, tool calls, tracebacks.

## Status

Validated end-to-end: **aider** adapter builds the web game against gx10
(`Qwen/Qwen3-Coder-30B-A3B-Instruct`); all five scorers run, incl. the headless
browser. The reworked scorer suite scores the reference build
(`scratch/workspace/index.html` + `game.js`) at 8/8 milestones + runs_headless ✅.
**opencode** + **goose** adapters exist but opencode 1.x hangs on run-init in this
env; goose CLI not installed. OpenHands / mini-SWE adapters are TODO — see the plan.
