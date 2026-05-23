"""Single-round-trip throughput probe for the vLLM server (harness-u70v).

Designed to be cheap to rerun after every gx10 config tweak — one
script invocation produces the four numbers that bound the diagnosis:

  - per-request tok/s (non-streaming, the harness's actual single-user load)
  - server-side ITL (from /metrics, before network)
  - TTFT (from server-side metric delta + client wall-clock)
  - aggregate tok/s at N=2 and N=4 concurrency (sanity-check batch path
    hasn't regressed)

Reads vllm's /metrics endpoint, takes a delta across the probe window,
so the reported ITL is from THIS run, not the lifetime histogram.

Usage:
    uv run python scripts/bench_vllm_remote.py
    uv run python scripts/bench_vllm_remote.py --base-url http://gx10-5fb9:8000
    uv run python scripts/bench_vllm_remote.py --concurrency 1,2,4,8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import time
from dataclasses import dataclass

import httpx

_PROMPT = "Count from 1 to 80 separated by spaces."
_MAX_TOKENS = 300


@dataclass(frozen=True)
class _ITLSummary:
    sum_sec: float
    count: int

    @property
    def mean_ms(self) -> float:
        return (self.sum_sec / self.count * 1000.0) if self.count else 0.0


def _scrape_itl(base_url: str) -> _ITLSummary:
    """Pull the lifetime ITL histogram sum + count. Caller takes a delta."""
    r = httpx.get(f"{base_url}/metrics", timeout=10.0)
    r.raise_for_status()
    sum_sec = 0.0
    count = 0
    for line in r.text.splitlines():
        if line.startswith("vllm:inter_token_latency_seconds_sum"):
            m = re.search(r"\s([0-9eE.+-]+)\s*$", line)
            if m:
                sum_sec = float(m.group(1))
        elif line.startswith("vllm:inter_token_latency_seconds_count"):
            m = re.search(r"\s([0-9eE.+-]+)\s*$", line)
            if m:
                count = int(float(m.group(1)))
    return _ITLSummary(sum_sec=sum_sec, count=count)


def _itl_delta(before: _ITLSummary, after: _ITLSummary) -> _ITLSummary:
    return _ITLSummary(sum_sec=after.sum_sec - before.sum_sec, count=after.count - before.count)


def _one_request(base_url: str, model: str, max_tokens: int) -> tuple[int, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.time()
    r = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180.0)
    r.raise_for_status()
    u = r.json()["usage"]
    return u["completion_tokens"], time.time() - t0


def _discover_model(base_url: str) -> str:
    r = httpx.get(f"{base_url}/v1/models", timeout=10.0)
    r.raise_for_status()
    served_id = r.json()["data"][0]["id"]
    assert isinstance(served_id, str)
    return served_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://gx10-5fb9:8000")
    parser.add_argument("--concurrency", default="1,2,4", help="comma-separated N values")
    parser.add_argument("--max-tokens", type=int, default=_MAX_TOKENS)
    args = parser.parse_args()

    model = _discover_model(args.base_url)
    print(f"server: {args.base_url}")
    print(f"model:  {model}")
    print(f"prompt: {_PROMPT!r}  max_tokens={args.max_tokens}")
    print()
    header = f"{'N':>3}  {'agg_tok/s':>10}  {'per_req_tok/s':>14}"
    header += f"  {'wall_s':>8}  {'itl_ms':>8}  {'n_steps':>8}"
    print(header)

    levels = [int(x) for x in args.concurrency.split(",")]
    for n in levels:
        itl_before = _scrape_itl(args.base_url)
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            results = list(
                ex.map(
                    lambda _: _one_request(args.base_url, model, args.max_tokens),
                    range(n),
                )
            )
        wall = time.time() - t0
        itl_after = _scrape_itl(args.base_url)
        delta = _itl_delta(itl_before, itl_after)
        total = sum(r[0] for r in results)
        agg = total / wall
        per_req = agg / n
        row = f"{n:3d}  {agg:10.1f}  {per_req:14.1f}"
        row += f"  {wall:8.2f}  {delta.mean_ms:8.1f}  {delta.count:8d}"
        print(row)


if __name__ == "__main__":
    main()
