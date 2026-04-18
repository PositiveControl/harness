"""Benchmark router-on vs router-off on a realistic prompt mix.

Per prompt, runs the tool loop twice:
  1. Baseline — main model (Qwen 7B by default) with tools enabled,
     no router. This is what users hit when the small model fabricates
     instead of calling search_web.
  2. Router — same main model + an MLX ModelRouter at the 1.5B. The
     orchestrator pre-runs the tool before the 7B gets a turn.

Measurements per run:
  - wall_s  — end-to-end time through run_tool_loop.
  - rounds  — how many model calls were needed.
  - tool_called — did a tool actually execute.
  - nudges  — count of recovery nudges the orchestrator fed back
              (harness-q27 / harness-j1d fabrication-detection). A
              high number means the main model was looping on bad
              output without the router.
  - router_intent — 1 if the router picked a tool, else 0.

Peak RAM is sampled before loading any model and after both phases
so you can see the cost of holding the 1.5B in addition to the 7B.

Usage:
    uv run python scripts/bench_router.py
    uv run python scripts/bench_router.py --prompts 5
    uv run python scripts/bench_router.py \\
        --main-repo mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --router-repo mlx-community/Hermes-3-Llama-3.2-3B-4bit

Prompts are pulled from the shipped router_eval.yaml fixture so the
bench tracks the same ground truth as the accuracy eval. --prompts N
trims to the first N entries.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from harness.config import settings
from harness.evals.router import load_fixture
from harness.model.adapter import ChatMessage
from harness.orchestrator import run_tool_loop
from harness.router import ModelRouter
from harness.tools import (
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    SearchWebTool,
    ToolRegistry,
)

_DEFAULT_MAIN_REPO = "mlx-community/Qwen2.5-7B-Instruct-4bit"
_DEFAULT_ROUTER_REPO = "mlx-community/Hermes-3-Llama-3.2-3B-4bit"


@dataclass
class _RunMetrics:
    wall_s: float
    rounds: int
    tool_called: bool
    nudges: int
    router_intents: int
    reply_snippet: str


def _peak_rss_mb() -> float:
    """Peak resident set size in MB. macOS returns bytes; Linux
    returns kilobytes. getrusage is process-lifetime, so each call
    returns the max-ever observed — useful for 'did the 1.5B push
    us over the edge?'"""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Using sys.platform inline is fine here; mypy statically narrows
    # on the build host which can collapse the branch on non-darwin,
    # so we keep the division factor dynamic via a variable.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return raw / divisor


def _build_registry(workspace: Path) -> ToolRegistry:
    """Registry matches the 'research' profile so the bench is
    comparable to `harness eval router --tool-set research`."""
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=workspace))
    registry.register(ListDirTool(root=workspace))
    registry.register(GrepTool(root=workspace))
    registry.register(GlobTool(root=workspace))
    registry.register(SearchWebTool())
    return registry


def _run_one(
    adapter: object,
    prompt: str,
    registry: ToolRegistry,
    router: ModelRouter | None,
) -> _RunMetrics:
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are a concise assistant. When a user asks for "
                "information, call the appropriate tool and then answer "
                "in one or two sentences."
            ),
        ),
        ChatMessage(role="user", content=prompt),
    ]
    start = time.perf_counter()
    result = run_tool_loop(
        adapter,  # type: ignore[arg-type]
        messages,
        registry,
        max_rounds=4,
        max_tokens=512,
        temperature=0.2,
        router=router,
    )
    wall_s = time.perf_counter() - start

    tool_called = any(m.role == "tool" for m in result.messages)
    # Nudges are user-role messages the orchestrator itself appended
    # during _diagnose_bail (fabrication / teaser / meta-confirm etc).
    # Input had `len(messages)` turns; anything past that with role='user'
    # came from the recovery path.
    nudges = sum(1 for m in result.messages[len(messages) :] if m.role == "user")
    router_intents = sum(1 for e in result.events if e.kind == "router_intent")
    snippet = result.content[:80].replace("\n", " ")
    return _RunMetrics(
        wall_s=wall_s,
        rounds=result.rounds,
        tool_called=tool_called,
        nudges=nudges,
        router_intents=router_intents,
        reply_snippet=snippet,
    )


def _format_row(label: str, m: _RunMetrics) -> str:
    mark = "✓" if m.tool_called else "·"
    return (
        f"  {label:<10} {m.wall_s:>6.2f}s  rounds={m.rounds}  tool={mark}  "
        f"nudges={m.nudges}  router={m.router_intents}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--main-repo", default=_DEFAULT_MAIN_REPO)
    parser.add_argument("--router-repo", default=_DEFAULT_ROUTER_REPO)
    parser.add_argument(
        "--prompts",
        type=int,
        default=0,
        help="Cap the number of fixture prompts to run. 0 = all.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=settings.character_path / "router_eval.yaml",
        help="YAML fixture to draw prompts from.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Only run the router-on phase. Useful when the baseline is "
        "known-slow and you just want to verify router behavior.",
    )
    args = parser.parse_args()

    fixture = load_fixture(args.fixture)
    if args.prompts > 0:
        fixture = fixture[: args.prompts]
    prompts = [row[0] for row in fixture]
    expected_tools = [row[1] for row in fixture]

    print(f"Bench — {len(prompts)} prompts from {args.fixture}")
    print(f"  main:   {args.main_repo}")
    print(f"  router: {args.router_repo}")
    print()

    from harness.model.mlx import MLXAdapter

    baseline_peak = 0.0
    router_peak = 0.0

    base_rss = _peak_rss_mb()
    print(f"Baseline RAM before load: {base_rss:.0f} MB")

    main_adapter = MLXAdapter(repo=args.main_repo)
    # Warm the main adapter so the first prompt isn't a cold-start tax.
    main_adapter.complete([ChatMessage(role="user", content="hi")], max_tokens=4, temperature=0.0)
    after_main = _peak_rss_mb()
    print(f"After loading main:       {after_main:.0f} MB (+{after_main - base_rss:.0f})")

    workspace = settings.root
    registry = _build_registry(workspace)

    baseline_results: list[_RunMetrics] = []
    if not args.skip_baseline:
        print("\n── Baseline (router off) ──")
        for prompt, expected in zip(prompts, expected_tools, strict=False):
            m = _run_one(main_adapter, prompt, registry, router=None)
            baseline_results.append(m)
            tag = expected if expected is not None else "null"
            print(f"  {tag:<14} {_format_row('baseline', m)[2:]}")
        baseline_peak = _peak_rss_mb()

    print("\n── Router on ──")
    router_adapter = MLXAdapter(repo=args.router_repo)
    router_adapter.complete([ChatMessage(role="user", content="hi")], max_tokens=4, temperature=0.0)
    after_router_load = _peak_rss_mb()
    print(
        f"After loading router:     {after_router_load:.0f} MB "
        f"(+{after_router_load - after_main:.0f} over main only)"
    )
    router = ModelRouter(adapter=router_adapter)

    router_results: list[_RunMetrics] = []
    for prompt, expected in zip(prompts, expected_tools, strict=False):
        m = _run_one(main_adapter, prompt, registry, router=router)
        router_results.append(m)
        tag = expected if expected is not None else "null"
        print(f"  {tag:<14} {_format_row('router', m)[2:]}")
    router_peak = _peak_rss_mb()

    print("\n── Summary ──")
    if baseline_results:
        _summarize("baseline", baseline_results)
    _summarize("router", router_results)

    print()
    if baseline_results:
        print(f"Peak RAM baseline phase: {baseline_peak:.0f} MB")
    print(f"Peak RAM router phase:   {router_peak:.0f} MB")
    return 0


def _summarize(label: str, runs: list[_RunMetrics]) -> None:
    n = len(runs)
    total_wall = sum(r.wall_s for r in runs)
    tool_rate = sum(1 for r in runs if r.tool_called) / n if n else 0.0
    total_nudges = sum(r.nudges for r in runs)
    total_routed = sum(r.router_intents for r in runs)
    print(
        f"  {label:<10} {n} prompts · wall={total_wall:.1f}s · "
        f"avg={total_wall / max(n, 1):.2f}s · tool-rate={tool_rate:.0%} · "
        f"nudges={total_nudges} · routed={total_routed}"
    )


if __name__ == "__main__":
    sys.exit(main())
