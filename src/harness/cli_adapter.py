"""Model-adapter composition for the CLI.

Step 6 of docs/cli-extraction-plan.md. `_resolve_adapter` turns the
model-selection flags into a live `ModelAdapter`: backend dispatch,
the MLX-only flag guards, the character's declared voice rewriter, and
the optional eager `.load()`.

Extracted last because it was already well-isolated. The payoff is
testability — a test can import the composition contract without
importing the whole Typer tree.
"""

from __future__ import annotations

from typing import cast

import typer
from rich.console import Console
from rich.status import Status

from harness.character import Character
from harness.config import settings
from harness.model import AdapterName, ModelAdapter, make_adapter
from harness.persona import PersonaAdapter
from harness.persona.caveman_rewriter import CavemanRewriter, load_register_map

console = Console()


def _resolve_adapter(
    name: str,
    *,
    persona: bool = False,
    character: Character | None = None,
    model_repo: str | None = None,
    lora_path: str | None = None,
    draft_repo: str | None = None,
    chain_rewrites: bool = False,
    rewriter_temperature: float | None = None,
) -> ModelAdapter:
    # Custom configs bypass the factory and instantiate the adapter
    # directly. --lora-path / --draft-repo are MLX-only; --model-repo
    # works for MLX (HF repo), Ollama (model tag like "gemma4:latest"),
    # and vLLM (base URL like "http://gx10-1.tailnet:8000/v1").
    if lora_path and name != "mlx":
        raise typer.BadParameter("--lora-path requires --model mlx.")
    if draft_repo and name != "mlx":
        raise typer.BadParameter("--draft-repo requires --model mlx.")

    adapter: ModelAdapter
    if model_repo or lora_path or draft_repo:
        if name == "mlx":
            from harness.model.mlx import MLXAdapter

            mlx_kwargs: dict[str, object] = {}
            if model_repo:
                mlx_kwargs["repo"] = model_repo
            if lora_path:
                mlx_kwargs["adapter_path"] = lora_path
            if draft_repo:
                mlx_kwargs["draft_repo"] = draft_repo
            adapter = MLXAdapter(**mlx_kwargs)  # type: ignore[arg-type]
        elif name == "ollama":
            from harness.model.ollama import OllamaAdapter

            adapter = OllamaAdapter(model=model_repo) if model_repo else OllamaAdapter()
        elif name == "vllm":
            from harness.model.vllm import VllmAdapter

            adapter = VllmAdapter(base_url=model_repo) if model_repo else VllmAdapter()
        else:
            raise typer.BadParameter(
                f"--model-repo not supported for --model {name}; use mlx, ollama, or vllm."
            )
    else:
        try:
            adapter = make_adapter(cast(AdapterName, name))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if persona:
        if character is None:
            raise typer.BadParameter("persona=True requires a character")
        # voice_rewriter is declared in core.yaml (harness-a2sa).
        # "caveman" wraps in CavemanRewriter (ab's per-surface intensity
        # map); "persona" wraps in PersonaAdapter (Airton's two-pass
        # voice rewrite); "none" leaves the adapter unwrapped — useful
        # when --persona was set but the character doesn't actually
        # ship a rewriter.
        if character.voice_rewriter == "caveman":
            # CavemanRewriter ships with the character data so the
            # register map evolves alongside the persona.
            register_map = load_register_map(
                settings.root / "character" / character.name / "register_map.yaml"
            )
            adapter = CavemanRewriter(
                adapter,
                intensity=settings.ab_register,
                register_map=register_map,
                rewrite_on_tools=settings.ab_rewrite_on_tools,
            )
        elif character.voice_rewriter == "persona":
            persona_kwargs: dict[str, object] = {"chain_rewrites": chain_rewrites}
            if rewriter_temperature is not None:
                persona_kwargs["rewriter_temperature"] = rewriter_temperature
            adapter = PersonaAdapter(adapter, character, **persona_kwargs)  # type: ignore[arg-type]
        # voice_rewriter == "none": leave adapter unwrapped.

    # Honor an optional eager `.load()` method without making it part of
    # the ModelAdapter Protocol — only some adapters need it.
    loader = getattr(adapter, "load", None)
    if callable(loader):
        with Status(f"loading {adapter.id}…", console=console):
            loader()
    return adapter
