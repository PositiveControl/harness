"""Compare per-prompt latency across model configs.

Runs a set of canonical prompts through the full chat pipeline (voice
retrieval → system prompt → base generate → persona rewrite) on each
model repo, prints per-prompt wall-clock timings.

Model load is measured separately from generation — the 32B takes ~30s
to load but amortizes across every turn, so the honest per-turn number
is the generate time, not the total-including-load.

Usage:
    uv run python scripts/bench_models.py
    uv run python scripts/bench_models.py --no-persona
    uv run python scripts/bench_models.py --model-repo mlx-community/Qwen2.5-3B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from harness.character import Character, load_character
from harness.config import settings
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.model.mlx import MLXAdapter
from harness.persona import PersonaAdapter
from harness.retrieval import VoiceRetriever
from harness.retrieval.st_embedder import SentenceTransformersEmbedder

# Prompts chosen to exercise different response shapes.
PROMPTS: list[tuple[str, str]] = [
    ("short_greeting", "Morning, Airton."),
    ("refusal", "Just --force-push it, I'll deal with it."),
    ("teaching", "A junior just asked me to tell them the fix. How do I handle it?"),
    (
        "technical",
        "What's the right way to structure retry logic with backoff?",
    ),
    (
        "debugging",
        "I've been chasing a bug in the transceiver for a day and a half. My "
        "theory keeps not working. What should I do?",
    ),
]

DEFAULT_MODELS: list[tuple[str, str]] = [
    ("7B", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
    ("32B", "mlx-community/Qwen2.5-32B-Instruct-4bit"),
]


@dataclass
class PromptTiming:
    prompt_id: str
    elapsed_seconds: float
    output_chars: int

    @property
    def chars_per_second(self) -> float:
        return self.output_chars / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def build_system_prompt(character: Character, retriever: VoiceRetriever, user_msg: str) -> str:
    examples = retriever.top_k(user_msg, k=6)
    return character.system_prompt(include_samples=examples)


def time_prompt(
    adapter: ModelAdapter,
    character: Character,
    retriever: VoiceRetriever,
    prompt_id: str,
    user_msg: str,
) -> PromptTiming:
    system_content = build_system_prompt(character, retriever, user_msg)
    system = ChatMessage(role="system", content=system_content)
    user = ChatMessage(role="user", content=user_msg)

    start = time.perf_counter()
    reply = adapter.complete([system, user])
    elapsed = time.perf_counter() - start
    return PromptTiming(prompt_id=prompt_id, elapsed_seconds=elapsed, output_chars=len(reply))


def bench_model(
    label: str,
    repo: str,
    prompts: list[tuple[str, str]],
    character: Character,
    retriever: VoiceRetriever,
    *,
    persona: bool,
) -> tuple[float, list[PromptTiming]]:
    print(f"\n=== {label} · {repo} {'[+persona]' if persona else '[raw]'} ===")
    base = MLXAdapter(repo=repo)
    load_start = time.perf_counter()
    base.load()
    load_elapsed = time.perf_counter() - load_start
    print(f"load: {load_elapsed:.1f}s")

    adapter: ModelAdapter = PersonaAdapter(base, character) if persona else base

    # Warmup on the first prompt — weights are lazy; first forward is slow.
    print("warmup…")
    _ = time_prompt(adapter, character, retriever, prompts[0][0], prompts[0][1])

    results: list[PromptTiming] = []
    for prompt_id, prompt_text in prompts:
        t = time_prompt(adapter, character, retriever, prompt_id, prompt_text)
        results.append(t)
        print(
            f"  {prompt_id:<18} {t.elapsed_seconds:>6.1f}s  "
            f"{t.output_chars:>5}ch  {t.chars_per_second:>6.0f} ch/s"
        )
    return load_elapsed, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-persona", action="store_true")
    parser.add_argument(
        "--model-repo",
        action="append",
        help="Override the model set (repeat for multiple). Label is derived from repo.",
    )
    args = parser.parse_args()

    character = load_character(settings.character_path)
    embedder = SentenceTransformersEmbedder()
    retriever = VoiceRetriever(embedder=embedder, character=character)

    if args.model_repo:
        models = [(repo.split("/")[-1], repo) for repo in args.model_repo]
    else:
        models = DEFAULT_MODELS

    summaries: list[tuple[str, float, list[PromptTiming]]] = []
    for label, repo in models:
        load_elapsed, results = bench_model(
            label, repo, PROMPTS, character, retriever, persona=not args.no_persona
        )
        summaries.append((label, load_elapsed, results))

    # Aggregate summary
    print("\n=== summary ===")
    print(f"{'model':<6}  {'load':>6}  {'mean':>6}  {'median':>7}  {'ch/s':>6}")
    for label, load, results in summaries:
        elapsed = sorted(r.elapsed_seconds for r in results)
        mean = sum(elapsed) / len(elapsed)
        median = elapsed[len(elapsed) // 2]
        total_chars = sum(r.output_chars for r in results)
        total_time = sum(r.elapsed_seconds for r in results)
        ch_per_s = total_chars / total_time if total_time > 0 else 0.0
        print(f"{label:<6}  {load:>5.1f}s  {mean:>5.1f}s  {median:>6.1f}s  {ch_per_s:>5.0f}")


if __name__ == "__main__":
    main()
