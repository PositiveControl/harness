"""Benchmark models on the actual pattern the harness uses: tooled chat.

Three measurements per config:

  1. `load_s` — cold-start time (weight load for MLX; /api/generate with
     empty prompt for Ollama).
  2. `warm_chars_per_s` — decode throughput on a short prompt once
     weights are warm. Characters/sec is a rough proxy for tokens/sec
     (~4 chars ≈ 1 token).
  3. `tool_success` — runs the ReadFileTool + `run_tool_loop` against a
     "read pyproject.toml and tell me the project name" prompt. Pass if
     the reply contains "harness" (the true name in this repo's
     pyproject.toml) AND the model actually called read_file.

Usage:
    uv run python scripts/bench_tool_use.py
    uv run python scripts/bench_tool_use.py --only ollama,qwen3-coder
    uv run python scripts/bench_tool_use.py --prompt-tokens 512

Skips any config whose model isn't locally available (HF cache for MLX,
`ollama list` for Ollama) so this is safe to run even if one model
still downloading in another terminal.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harness.config import settings
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.model.ollama import OllamaAdapter
from harness.orchestrator import run_tool_loop
from harness.tools import ReadFileTool, ToolRegistry

_WARM_SYSTEM = "You are a concise assistant. Reply in prose, no bullet lists."
_WARM_PROMPT = "In one paragraph of 4-6 sentences, explain what makes a good side project scope."
_TOOL_PROMPT = (
    "Read pyproject.toml in the workspace root and tell me the project name "
    "from the [project] table. Answer in one sentence that restates the name."
)


@dataclass
class BenchResult:
    label: str
    repo: str
    load_s: float | None
    warm_chars: int
    warm_s: float
    tool_success: bool
    tool_called_read_file: bool
    tool_elapsed_s: float
    tool_reply_snippet: str

    @property
    def warm_chars_per_s(self) -> float:
        return self.warm_chars / self.warm_s if self.warm_s > 0 else 0.0


def _mlx_repo_cached(repo: str) -> bool:
    """True if the HF cache has this repo's snapshot on disk."""
    slug = repo.replace("/", "--")
    cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{slug}"
    return cache.exists() and any(cache.glob("snapshots/*/config.json"))


def _ollama_model_present(tag: str) -> bool:
    try:
        out = subprocess.run(
            ["ollama", "list"],  # noqa: S607 — `ollama` on PATH is the documented CLI
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    return any(tag.split(":", 1)[0] in line for line in out.stdout.splitlines()[1:])


@dataclass
class Config:
    label: str
    repo: str
    build: Callable[[], ModelAdapter]
    available: Callable[[], bool]


def _make_configs() -> list[Config]:
    def mlx_adapter(repo: str) -> Callable[[], ModelAdapter]:
        def _build() -> ModelAdapter:
            from harness.model.mlx import MLXAdapter

            return MLXAdapter(repo=repo)

        return _build

    def ollama_adapter(tag: str) -> Callable[[], ModelAdapter]:
        def _build() -> ModelAdapter:
            return OllamaAdapter(model=tag)

        return _build

    qwen3_coder = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
    qwen25_32b = "mlx-community/Qwen2.5-32B-Instruct-4bit"
    return [
        Config(
            label="ollama:gemma4",
            repo="gemma4:latest",
            build=ollama_adapter("gemma4:latest"),
            available=lambda: _ollama_model_present("gemma4:latest"),
        ),
        Config(
            label="mlx:qwen3-coder-30b-a3b",
            repo=qwen3_coder,
            build=mlx_adapter(qwen3_coder),
            available=lambda: _mlx_repo_cached(qwen3_coder),
        ),
        Config(
            label="mlx:qwen2.5-32b",
            repo=qwen25_32b,
            build=mlx_adapter(qwen25_32b),
            available=lambda: _mlx_repo_cached(qwen25_32b),
        ),
    ]


def _time_load(adapter: ModelAdapter) -> float | None:
    loader = getattr(adapter, "load", None)
    if not callable(loader):
        return None
    start = time.perf_counter()
    loader()
    return time.perf_counter() - start


def _warm(adapter: ModelAdapter, max_tokens: int) -> tuple[int, float]:
    msgs = [
        ChatMessage(role="system", content=_WARM_SYSTEM),
        ChatMessage(role="user", content=_WARM_PROMPT),
    ]
    start = time.perf_counter()
    reply = adapter.complete(msgs, max_tokens=max_tokens, temperature=0.7)
    elapsed = time.perf_counter() - start
    return len(reply), elapsed


def _tool_probe(adapter: ModelAdapter) -> tuple[bool, bool, float, str]:
    registry = ToolRegistry()
    registry.register(ReadFileTool(root=settings.root))
    tool_called = False
    start = time.perf_counter()
    result = run_tool_loop(
        adapter,  # type: ignore[arg-type]
        [
            ChatMessage(
                role="system",
                content=(
                    "You can read files in the workspace via `read_file`. "
                    "After the tool returns content, answer the user in one "
                    "sentence. The user cannot see tool output, only your reply."
                ),
            ),
            ChatMessage(role="user", content=_TOOL_PROMPT),
        ],
        registry,
        max_rounds=4,
        max_tokens=512,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - start
    for msg in result.messages:
        if msg.role == "assistant" and any(tc.name == "read_file" for tc in msg.tool_calls):
            tool_called = True
            break
    reply = result.content.strip()
    success = tool_called and "harness" in reply.lower()
    return success, tool_called, elapsed, reply[:120].replace("\n", " ")


def run_bench(cfg: Config, *, warm_tokens: int) -> BenchResult:
    print(f"\n=== {cfg.label} · {cfg.repo} ===")
    adapter = cfg.build()
    load_s = _time_load(adapter)
    if load_s is not None:
        print(f"  load: {load_s:.2f}s")
    warm_chars, warm_s = _warm(adapter, max_tokens=warm_tokens)
    print(f"  warm: {warm_chars} chars in {warm_s:.2f}s ({warm_chars / warm_s:.0f} ch/s)")
    success, tool_called, tool_s, snippet = _tool_probe(adapter)
    marker = "✓" if success else ("✗ no-tool" if not tool_called else "✗ wrong-answer")
    print(f"  tool: {marker} in {tool_s:.2f}s — {snippet!r}")
    return BenchResult(
        label=cfg.label,
        repo=cfg.repo,
        load_s=load_s,
        warm_chars=warm_chars,
        warm_s=warm_s,
        tool_success=success,
        tool_called_read_file=tool_called,
        tool_elapsed_s=tool_s,
        tool_reply_snippet=snippet,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        help="Comma-separated labels to run (e.g. ollama:gemma4,mlx:qwen3-coder-30b-a3b).",
    )
    parser.add_argument(
        "--warm-tokens",
        type=int,
        default=256,
        help="max_tokens for the warm-throughput prompt (default 256).",
    )
    args = parser.parse_args()

    configs = _make_configs()
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        configs = [c for c in configs if c.label in wanted]

    results: list[BenchResult] = []
    for cfg in configs:
        if not cfg.available():
            print(f"\n=== {cfg.label} · SKIP (not downloaded / daemon off) ===")
            continue
        try:
            results.append(run_bench(cfg, warm_tokens=args.warm_tokens))
        except Exception as exc:  # keep going on individual failures
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    if not results:
        print("\nNo configs ran. Pull models or start `ollama serve`.")
        return

    print("\n=== summary ===")
    header = f"{'label':<28} {'load':>7} {'warm ch/s':>10} {'tool':>5} {'tool s':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        load = f"{r.load_s:.1f}s" if r.load_s is not None else "-"
        tool = "✓" if r.tool_success else "✗"
        print(
            f"{r.label:<28} {load:>7} {r.warm_chars_per_s:>10.0f} "
            f"{tool:>5} {r.tool_elapsed_s:>6.1f}s"
        )


if __name__ == "__main__":
    main()
