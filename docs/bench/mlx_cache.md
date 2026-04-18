# MLX cache-limit tuning

Tracks the memory/latency sweep for `Settings.mlx_cache_limit_mb` (env:
`HARNESS_MLX_CACHE_LIMIT_MB`). Each row is one `uv run python
scripts/bench_ram.py` invocation against the current default stack
(Qwen 7B main + Hermes-3 3B router + bge-small embedder). Numbers come
from MLX's own counters (`mx.get_peak_memory` / `get_active_memory` /
`get_cache_memory`) — far more precise than process RSS for MLX
workloads, since sentence-transformers and mmap'd weights muddy RSS.

## Why this knob matters on a 32 GB box

Two MLX models are resident at once during a router-on turn (main +
router). Their quantized weights are ~5.8 GB active. On top, MLX
keeps a *cache*: buffers it allocated, the graph released, but it has
not returned to the system allocator so the next op can reuse them.
`set_cache_limit(N)` caps that pool — once the cache hits `N` bytes,
further deallocs hand buffers back to the OS.

Low cap = smaller peak RSS under memory pressure (harness-0kw rationale
for caring about this on the 32 GB box), at the cost of re-allocating
on the next turn. No cap = MLX's default, which optimizes for latency
but can hold onto several GB of scratch.

Active-memory (weights + live KV cache) is **not** affected by this
knob — that's set by what models you have loaded and how much context
you're holding. `set_cache_limit` only reclaims the unused pool.

## Measurement protocol

`scripts/bench_ram.py` snapshots MLX counters after each of:

1. startup
2. embedder loaded + one top-k
3. main loaded + one 8-token generation
4. router loaded + one 8-token classify
5. one more main generation (post-router, "steady-state" TUI turn)

The last row is the one that tracks cache behavior — by that point
both models are resident and a full turn has cycled buffers.

## Runs

| cache limit | mlx_peak (MB) | mlx_active (MB) | mlx_cache end (MB) | wall post-router (s) | commit |
|-------------|---------------|-----------------|--------------------|----------------------|--------|
| none (default) | 5892 | 5813 | 18 | 0.20 | baseline |

Subsequent iterations commit one row each with the raw JSON in
`docs/bench/mlx_cache_<limit>.json`.
