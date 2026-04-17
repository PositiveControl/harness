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

A fourth mode, `--measure-tokens`, loads only the tokenizer for a
chosen repo and prints per-tool schema + result costs. Fast (tokenizer
download is a few MB; no model weights needed) and useful for deciding
which tools to enable by default.

Usage:
    uv run python scripts/bench_tool_use.py
    uv run python scripts/bench_tool_use.py --only ollama,qwen3-coder
    uv run python scripts/bench_tool_use.py --prompt-tokens 512
    uv run python scripts/bench_tool_use.py --measure-tokens

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
from typing import Any

from harness.config import settings
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.model.ollama import OllamaAdapter
from harness.orchestrator import run_tool_loop
from harness.tools import (
    ReadFileTool,
    SearchFactsTool,
    SearchMemoryTool,
    ShellTool,
    ToolRegistry,
    WriteFileTool,
)

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


def _tool_spec_schema(spec: Any) -> dict[str, Any]:
    """Render a ToolSpec into the OpenAI-style schema the chat template expects."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _build_all_tools() -> list[Any]:
    """Instantiate every built-in tool with real dependencies.

    Memory tools need the live episodic + semantic stores so we can
    invoke them and measure realistic output. If the stores are empty
    the tools still return (the "no results" path), which is a valid
    minimum — the schema cost is what dominates anyway."""
    # Lazy imports so the throughput-only bench path doesn't pay for
    # sentence-transformers startup cost when it doesn't need memory.
    from harness.retrieval.st_embedder import SentenceTransformersEmbedder
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore
    from harness.tools import EditFileTool

    embedder = SentenceTransformersEmbedder()
    episodic = EpisodicStore(settings.db_path, embedder=embedder)
    semantic = SemanticStore(settings.db_path, embedder=embedder)
    return [
        ReadFileTool(root=settings.root),
        EditFileTool(root=settings.root),
        WriteFileTool(root=settings.root),
        ShellTool(cwd=settings.root),
        SearchMemoryTool(store=episodic, user_id="mark"),
        SearchFactsTool(store=semantic, user_id="mark"),
    ]


_BASE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi."},
]

_RESULT_PROBES: dict[str, tuple[str, dict[str, Any]]] = {
    # (tool_name, kwargs). write_file is skipped at call time — we
    # synthesize its return string from the documented format rather
    # than actually writing, to keep the bench side-effect-free.
    "read_file": ("read_file", {"path": "pyproject.toml"}),
    "shell": ("shell", {"cmd": "ls"}),
    "search_memory": ("search_memory", {"query": "first week priorities", "k": 3}),
    "search_facts": ("search_facts", {"query": "project", "k": 3}),
}


def measure_tokens(repo: str) -> None:
    """Tokenizer-only measurement path. Downloads tokenizer files (small)
    and applies the chat template with and without tools to compute the
    schema delta per tool, then invokes each tool with a representative
    input to measure result tokens."""
    print(f"\n=== token costs · {repo} ===")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover — defer to runtime
        print(f"  skipped: transformers not installed ({exc})")
        return
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    def count(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
        # Render to string, then encode — `tokenize=True` is inconsistent
        # across transformers versions (sometimes wraps in a BatchEncoding).
        # tokenizer stubs type `tools` narrowly; our JSON-schema dicts fit at
        # runtime — the templates just read string fields.
        prompt: str = tokenizer.apply_chat_template(
            messages,
            tools=tools,  # type: ignore[arg-type]
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer.encode(prompt))

    tools = _build_all_tools()
    specs = [t.spec for t in tools]
    schemas = [_tool_spec_schema(s) for s in specs]

    baseline = count(_BASE_MESSAGES, tools=None)
    total_with_all = count(_BASE_MESSAGES, tools=schemas)
    per_tool_costs: list[tuple[str, int]] = []
    for spec, schema in zip(specs, schemas, strict=True):
        one = count(_BASE_MESSAGES, tools=[schema])
        per_tool_costs.append((spec.name, one - baseline))

    print(f"  baseline (no tools):          {baseline:>5} tokens")
    print(
        f"  all {len(specs)} tools combined:        {total_with_all:>5} tokens "
        f"(delta {total_with_all - baseline})"
    )
    print()
    print("  per-tool schema cost (delta vs baseline):")
    print(f"  {'tool':<20} {'tokens':>7}")
    print(f"  {'-' * 20} {'-' * 7}")
    for name, cost in per_tool_costs:
        print(f"  {name:<20} {cost:>7}")

    # Tool results — actually invoke each tool, tokenize the output
    # as a tool-role message body (closest to what goes back to the model).
    print()
    print("  result cost on representative input:")
    print(f"  {'tool':<20} {'chars':>7} {'tokens':>7}")
    print(f"  {'-' * 20} {'-' * 7} {'-' * 7}")
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    for name, args in _RESULT_PROBES.values():
        try:
            res = registry.call(name, args)
            output = res.output if res.success else f"[error] {res.output}"
        except Exception as exc:
            output = f"[raised] {type(exc).__name__}: {exc}"
        # tokenize the output as a message body — this is what the model
        # sees next round as the tool-role result content.
        toks = len(tokenizer.encode(output, add_special_tokens=False))
        print(f"  {name:<20} {len(output):>7} {toks:>7}")
    # write_file is synthesized — its return is always "wrote N chars to PATH"
    synth = "wrote 1234 chars to src/harness/tools/new_tool.py"
    synth_tokens = len(tokenizer.encode(synth, add_special_tokens=False))
    print(f"  {'write_file (synth)':<20} {len(synth):>7} {synth_tokens:>7}")


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
    parser.add_argument(
        "--measure-tokens",
        action="store_true",
        help="Skip throughput / tool-probe; instead, load only the tokenizer "
        "for --token-repo and print per-tool schema + result token costs.",
    )
    parser.add_argument(
        "--token-repo",
        default="mlx-community/Qwen2.5-32B-Instruct-4bit",
        help="HF repo whose chat template + tokenizer to use for --measure-tokens.",
    )
    args = parser.parse_args()

    if args.measure_tokens:
        measure_tokens(args.token_repo)
        return

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
