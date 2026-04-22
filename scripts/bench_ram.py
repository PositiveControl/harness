"""Peak-RAM probe for the stack a TUI turn holds in memory at once.

Loads main MLX + (optionally) router MLX + embedder, runs one warm
classify + one turn so every model is paged in, then prints peak RSS.
Useful as a back-of-envelope for the memory-optimization work
(harness-e4m, harness-5b3) where we want to see "does swapping these
defaults actually shave the peak we'd hit on the 32GB box."

    uv run python scripts/bench_ram.py
    uv run python scripts/bench_ram.py --router-repo mlx-community/Hermes-3-Llama-3.2-3B-4bit
    uv run python scripts/bench_ram.py --embedder-repo mixedbread-ai/mxbai-embed-large-v1
    uv run python scripts/bench_ram.py --skip-router

Each flag takes the same kind of repo string Settings accepts, so the
same probe script covers every combination of (heavy/light) across the
three components. Results print as a small table with the peak-RAM
delta between configurations you'd have to run separately (peak is a
process-lifetime watermark)."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


def _peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes; Linux returns kilobytes.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return raw / divisor


def _mlx_memory_mb() -> tuple[float, float, float]:
    """MLX internal memory counters in MB: (peak, active, cache). Peak is
    the watermark since process start (or last reset_peak_memory). Active
    is memory currently backing live tensors. Cache is free buffers MLX
    has allocated but not yet returned to the system — the thing
    set_cache_limit caps. Returns zeros if MLX hasn't allocated anything
    yet (e.g. before the first model load) or if the APIs aren't
    available on this build."""
    try:
        import mlx.core as mx
    except ImportError:
        return 0.0, 0.0, 0.0
    scale = 1 / (1024 * 1024)
    return (
        mx.get_peak_memory() * scale,
        mx.get_active_memory() * scale,
        mx.get_cache_memory() * scale,
    )


@dataclass
class _Probe:
    stage: str
    peak_mb: float  # process RSS watermark from getrusage
    wall_s: float
    mlx_peak_mb: float = 0.0
    mlx_active_mb: float = 0.0
    mlx_cache_mb: float = 0.0
    notes: str = ""


@dataclass
class _Run:
    main_repo: str
    router_repo: str | None
    embedder_repo: str
    probes: list[_Probe] = field(default_factory=list)


def run(args: argparse.Namespace) -> _Run:
    from harness.character import load_character
    from harness.config import settings
    from harness.model.adapter import ChatMessage
    from harness.model.mlx import MLXAdapter
    from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    from harness.retrieval.voice_retriever import VoiceRetriever

    result = _Run(
        main_repo=args.main_repo,
        router_repo=None if args.skip_router else args.router_repo,
        embedder_repo=args.embedder_repo,
    )

    def _snapshot(stage: str, wall_s: float, notes: str = "") -> _Probe:
        mlx_peak, mlx_active, mlx_cache = _mlx_memory_mb()
        return _Probe(
            stage=stage,
            peak_mb=_peak_rss_mb(),
            wall_s=wall_s,
            mlx_peak_mb=mlx_peak,
            mlx_active_mb=mlx_active,
            mlx_cache_mb=mlx_cache,
            notes=notes,
        )

    result.probes.append(_snapshot("startup", 0.0))

    character = load_character(settings.character_path)

    t0 = time.monotonic()
    embedder = SentenceTransformersEmbedder(model_name=args.embedder_repo)
    retriever = VoiceRetriever(embedder=embedder, character=character)
    _ = retriever.top_k("test query", k=3)
    result.probes.append(
        _snapshot(
            f"embedder {args.embedder_repo}",
            time.monotonic() - t0,
            notes=f"dim={embedder.dimension}",
        )
    )

    t0 = time.monotonic()
    main = MLXAdapter(repo=args.main_repo)
    main.load()
    reply = main.complete([ChatMessage(role="user", content="hi")], max_tokens=8, temperature=0.0)
    result.probes.append(
        _snapshot(
            f"main {args.main_repo}",
            time.monotonic() - t0,
            notes=f"reply={reply[:30]!r}",
        )
    )

    if not args.skip_router:
        t0 = time.monotonic()
        router = MLXAdapter(repo=args.router_repo)
        router.load()
        _ = router.complete(
            [ChatMessage(role="user", content="ping")],
            max_tokens=8,
            temperature=0.0,
        )
        result.probes.append(_snapshot(f"router {args.router_repo}", time.monotonic() - t0))

        # One more main generation AFTER the router has run, so the
        # bench captures the "steady-state" pattern a real TUI turn
        # hits: router classify, then main wrap-up. Peak during this
        # stage is what set_cache_limit actually caps against.
        t0 = time.monotonic()
        reply2 = main.complete(
            [ChatMessage(role="user", content="tell me one word")],
            max_tokens=8,
            temperature=0.0,
        )
        result.probes.append(
            _snapshot(
                "main post-router turn",
                time.monotonic() - t0,
                notes=f"reply={reply2[:30]!r}",
            )
        )

    # Long-generation probe: 8-token completions barely exercise the
    # allocator. A 256-token reply with a non-trivial system prompt
    # churns many more transient tensors through the free-cache pool,
    # which is what set_cache_limit actually bounds. Repeat twice to
    # catch any cache-hot-vs-cold difference.
    if args.heavy:
        for i in range(args.heavy):
            t0 = time.monotonic()
            reply_long = main.complete(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "You are Airton, a gruff senior engineer. Answer in two sentences."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=(
                            "Walk through how a quantized Qwen 7B actually "
                            "stores its weights in memory."
                        ),
                    ),
                ],
                max_tokens=args.heavy_max_tokens,
                temperature=0.0,
            )
            result.probes.append(
                _snapshot(
                    f"main heavy turn {i + 1} ({args.heavy_max_tokens}t)",
                    time.monotonic() - t0,
                    notes=f"reply_len={len(reply_long)}",
                )
            )

    return result


def _print(run_: _Run) -> None:
    print(f"main   : {run_.main_repo}")
    print(f"router : {run_.router_repo or '(skipped)'}")
    print(f"embed  : {run_.embedder_repo}")
    print()
    header = (
        f"{'stage':<50} {'rss_MB':>8} {'wall_s':>7} "
        f"{'mlx_peak':>9} {'mlx_act':>8} {'mlx_cache':>10}  notes"
    )
    print(header)
    print("-" * len(header))
    for p in run_.probes:
        print(
            f"{p.stage:<50} {p.peak_mb:>8.1f} {p.wall_s:>7.2f} "
            f"{p.mlx_peak_mb:>9.1f} {p.mlx_active_mb:>8.1f} "
            f"{p.mlx_cache_mb:>10.1f}  {p.notes}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Peak-RAM probe for a full main + router + embedder stack."
    )
    parser.add_argument("--main-repo", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    from harness.config import settings as _settings

    parser.add_argument("--router-repo", default=_settings.router_repo)
    parser.add_argument("--embedder-repo", default=_settings.embedder_repo)
    parser.add_argument(
        "--skip-router",
        action="store_true",
        help="Only load main + embedder. Useful for measuring the floor.",
    )
    parser.add_argument(
        "--cache-limit-mb",
        type=int,
        default=None,
        help="MLX free-cache cap in MB. Applied to both main and router "
        "adapters (set_cache_limit is process-global). None = no cap, "
        "0 = disable cache entirely.",
    )
    parser.add_argument(
        "--heavy",
        type=int,
        default=0,
        help="Number of long-generation turns to run on the main adapter "
        "after the short probes. 0 = skip (default).",
    )
    parser.add_argument(
        "--heavy-max-tokens",
        type=int,
        default=256,
        help="Max tokens per heavy turn.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write the run as JSON to this path in addition to the table.",
    )
    args = parser.parse_args()

    if args.cache_limit_mb is not None:
        try:
            import mlx.core as mx

            mx.set_cache_limit(args.cache_limit_mb * 1024 * 1024)
        except Exception as exc:
            print(f"warning: could not set cache limit ({exc})", file=sys.stderr)

    run_ = run(args)
    _print(run_)

    if args.json_out is not None:
        payload = {
            "main_repo": run_.main_repo,
            "router_repo": run_.router_repo,
            "embedder_repo": run_.embedder_repo,
            "cache_limit_mb": args.cache_limit_mb,
            "probes": [
                {
                    "stage": p.stage,
                    "rss_mb": p.peak_mb,
                    "wall_s": p.wall_s,
                    "mlx_peak_mb": p.mlx_peak_mb,
                    "mlx_active_mb": p.mlx_active_mb,
                    "mlx_cache_mb": p.mlx_cache_mb,
                    "notes": p.notes,
                }
                for p in run_.probes
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
