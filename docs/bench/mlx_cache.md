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

All numbers from the same box (M4 Pro 32 GB, MLX 0.x, quantized 4-bit
weights for both main + router). Heavy row adds two 256-token
generations; "heavy wall" is the per-turn wall time, post-router.

| cache limit | mlx_peak (MB) | mlx_active (MB) | mlx_cache end (MB) | heavy wall (s) | commit |
|-------------|---------------|-----------------|--------------------|----------------|--------|
| none (default) | 5923 | 5813 | 18 | 1.28 | baseline |
| 1024 MB | 5923 | 5813 | 18 | 1.28 | iter |
| 0 (disabled) | 5923 | 5813 | 0 | 1.27 | iter |

## Conclusion — `set_cache_limit` is a non-lever for this stack

The free-cache pool sits at 18 MB with the default; capping it at 0
saves exactly that. Peak (5923 MB) and wall time (1.28 s) don't move.

Why: quantized 4-bit weights keep active memory nearly flat after load
(~5813 MB with Qwen 7B + Hermes-3 3B). The MLX allocator doesn't build
a large reusable scratch pool under short/medium generations because
intermediate tensors are small and lifetimes short — the cache just
doesn't accumulate. For any cap ≥ ~50 MB, MLX's natural cache usage
never hits the ceiling, so the cap is inert. For caps below that,
you're reclaiming tens of MB for no measurable latency cost — real but
irrelevant next to the 5.8 GB of weights.

Default stays `None` (MLX's own default). The knob stays in Settings so
a future workload with long contexts + multi-round tool loops can flip
it on, but there's no reason to set a value today.

## Next memory levers (outside this issue)

Peak is dominated by `active_memory` = model weights + live KV cache.
To move peak meaningfully:

1. **Unload router between turns.** Free Hermes-3's ~2 GB when the
   turn isn't classifying. Cold reload on next turn = +10–30 s latency
   per first-classify, which a pre-warm worker hides partially. Would
   also need explicit `del router._model; mx.clear_cache()` to actually
   release.
2. **Quantize weights further.** 3-bit Qwen 7B (~3.5 GB) exists but
   voice quality regresses — needs voice-eval gate.
3. **Smaller main model.** Not the ask — user wants quality.
4. **KV-cache reuse / trimming.** MLX doesn't expose an easy knob for
   this yet. Long-turn bench would be needed first.

None cheaper than what's already landed (embedder swap, +1.2 GB win).
