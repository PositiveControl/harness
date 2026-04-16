from __future__ import annotations

import json
from typing import cast

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status
from rich.table import Table

from harness.character import Character, load_character
from harness.config import settings
from harness.evals.voice import run_voice_eval
from harness.model import AdapterName, ChatMessage, ModelAdapter, make_adapter
from harness.model.adapter import Role
from harness.persona import PersonaAdapter
from harness.store.transcript import Transcript

app = typer.Typer(add_completion=False, no_args_is_help=True)
eval_app = typer.Typer(help="Evaluations against the current character.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
console = Console()


def _resolve_adapter(
    name: str,
    *,
    persona: bool = False,
    character: Character | None = None,
) -> ModelAdapter:
    try:
        adapter: ModelAdapter = make_adapter(cast(AdapterName, name))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if persona:
        if character is None:
            raise typer.BadParameter("persona=True requires a character")
        adapter = PersonaAdapter(adapter, character)

    # Honor an optional eager `.load()` method without making it part of
    # the ModelAdapter Protocol — only some adapters need it.
    loader = getattr(adapter, "load", None)
    if callable(loader):
        with Status(f"loading {adapter.id}…", console=console):
            loader()
    return adapter


@app.command()
def chat(
    session: str = typer.Option("local", help="Session identifier"),
    channel: str = typer.Option("cli", help="Channel name"),
    speaker: str = typer.Option("mark", help="Your handle"),
    model: str = typer.Option("echo", help="Adapter: echo | mlx"),
    persona: bool = typer.Option(
        False,
        "--persona/--no-persona",
        help="Wrap the model with a voice-rewrite post-pass (Airton's register).",
    ),
) -> None:
    """CLI chat loop. Swap model runtimes with --model."""
    character = load_character(settings.character_path)
    adapter = _resolve_adapter(model, persona=persona, character=character)
    transcript = Transcript(settings.db_path)

    console.print(f"[bold]{character.name}[/bold] loaded. session={session} model={adapter.id}")
    console.print("[dim](ctrl-c to exit)[/dim]\n")

    system = ChatMessage(role="system", content=character.system_prompt())

    try:
        while True:
            user_input = console.input("[bold cyan]you › [/bold cyan]").strip()
            if not user_input:
                continue
            transcript.append(
                session=session,
                channel=channel,
                speaker=speaker,
                role="user",
                content=user_input,
            )
            history = [
                ChatMessage(role=cast(Role, m.role), content=m.content)
                for m in transcript.tail(session, limit=50)
            ]
            reply = adapter.complete([system, *history])
            transcript.append(
                session=session,
                channel=channel,
                speaker=character.name,
                role="assistant",
                content=reply,
            )
            console.print(f"[bold green]{character.name} ›[/bold green]")
            console.print(Markdown(reply))
            console.print()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]bye.[/dim]")
    finally:
        transcript.close()


@app.command()
def describe() -> None:
    """Print Airton's resolved character sheet (sanity check)."""
    character = load_character(settings.character_path)
    console.print(f"[bold]{character.name}[/bold] — {character.premise}\n")
    console.print("[bold]Values[/bold]")
    for v in character.values:
        console.print(f"  • {v.rule}")
    console.print("\n[bold]Taboos[/bold]")
    for t in character.taboos:
        console.print(f"  • {t}")
    console.print("\n[bold]Seed memories[/bold]")
    for s in character.seed_memories:
        console.print(f"  • {s.title} — {s.principle}")
    console.print("\n[bold]Voice samples[/bold]")
    for sample in character.voice_samples:
        console.print(f"  • {sample.id}")


@eval_app.command("voice")
def eval_voice(
    model: str = typer.Option("mlx", help="Adapter: echo | mlx"),
    sample: list[str] | None = typer.Option(
        None, "--sample", help="Limit to a specific sample id (repeatable)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    temperature: float = typer.Option(0.5, help="Sampling temperature"),
    leave_one_out: bool = typer.Option(
        True,
        "--leave-one-out/--no-leave-one-out",
        help="Exclude each sample from its own few-shot examples (default on). "
        "Disable to measure the ceiling with the full example set in view.",
    ),
    persona: bool = typer.Option(
        False,
        "--persona/--no-persona",
        help="Run the voice-rewrite post-pass after the substance pass.",
    ),
) -> None:
    """Run the canonical voice prompts and show model-vs-gold side by side."""
    character = load_character(settings.character_path)
    # eval runs persona inline in run_voice_eval so both passes stay
    # leave-one-out-consistent — do not wrap adapter here.
    adapter = _resolve_adapter(model)

    results = run_voice_eval(
        character,
        adapter,
        temperature=temperature,
        sample_ids=sample if sample else None,
        leave_one_out=leave_one_out,
        persona=persona,
    )

    if as_json:
        payload = [
            {
                "sample_id": r.sample_id,
                "prompt": r.prompt,
                "gold": r.gold,
                "actual": r.actual,
                **({"draft": r.draft} if r.draft is not None else {}),
            }
            for r in results
        ]
        console.print_json(json.dumps(payload))
        return

    table = Table(title=f"Voice eval — {adapter.id}", show_lines=True)
    table.add_column("id", style="bold")
    table.add_column("prompt")
    table.add_column("gold", style="green")
    table.add_column("actual", style="yellow")
    for r in results:
        table.add_row(r.sample_id, r.prompt, r.gold.strip(), r.actual.strip())
    console.print(table)


if __name__ == "__main__":
    app()
