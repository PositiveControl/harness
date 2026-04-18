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


@dataclass
class _Probe:
    stage: str
    peak_mb: float
    wall_s: float
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

    result.probes.append(_Probe(stage="startup", peak_mb=_peak_rss_mb(), wall_s=0.0))

    character = load_character(settings.character_path)

    t0 = time.monotonic()
    embedder = SentenceTransformersEmbedder(model_name=args.embedder_repo)
    retriever = VoiceRetriever(embedder=embedder, character=character)
    _ = retriever.top_k("test query", k=3)
    result.probes.append(
        _Probe(
            stage=f"embedder {args.embedder_repo}",
            peak_mb=_peak_rss_mb(),
            wall_s=time.monotonic() - t0,
            notes=f"dim={embedder.dimension}",
        )
    )

    t0 = time.monotonic()
    main = MLXAdapter(repo=args.main_repo)
    main.load()
    reply = main.complete([ChatMessage(role="user", content="hi")], max_tokens=8, temperature=0.0)
    result.probes.append(
        _Probe(
            stage=f"main {args.main_repo}",
            peak_mb=_peak_rss_mb(),
            wall_s=time.monotonic() - t0,
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
        result.probes.append(
            _Probe(
                stage=f"router {args.router_repo}",
                peak_mb=_peak_rss_mb(),
                wall_s=time.monotonic() - t0,
            )
        )

    return result


def _print(run_: _Run) -> None:
    print(f"main   : {run_.main_repo}")
    print(f"router : {run_.router_repo or '(skipped)'}")
    print(f"embed  : {run_.embedder_repo}")
    print()
    print(f"{'stage':<60} {'peak_MB':>10} {'wall_s':>8}  notes")
    print("-" * 100)
    for p in run_.probes:
        print(f"{p.stage:<60} {p.peak_mb:>10.1f} {p.wall_s:>8.2f}  {p.notes}")


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
        "--json-out",
        type=Path,
        help="Write the run as JSON to this path in addition to the table.",
    )
    args = parser.parse_args()

    run_ = run(args)
    _print(run_)

    if args.json_out is not None:
        payload = {
            "main_repo": run_.main_repo,
            "router_repo": run_.router_repo,
            "embedder_repo": run_.embedder_repo,
            "probes": [
                {
                    "stage": p.stage,
                    "peak_mb": p.peak_mb,
                    "wall_s": p.wall_s,
                    "notes": p.notes,
                }
                for p in run_.probes
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
