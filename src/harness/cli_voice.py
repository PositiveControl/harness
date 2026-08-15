"""`harness voice` subcommands.

Step 5b of docs/cli-extraction-plan.md. Capture a corrected reply as a
new voice sample, and inspect what has been captured.

`_write_voice_capture` is shared with the in-chat /edit slash command,
which calls it directly rather than going through Typer — the reason
the capture command's decorator drifted onto the helper and broke the
shell path for four months (fixed in 3c26e5e).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from harness.cli_apps import voice_app
from harness.config import settings
from harness.store.transcript import Transcript

console = Console()


def _write_voice_capture(
    *,
    prompt: str,
    gold: str,
    session: str,
    original: str | None,
    sample_id: str | None = None,
) -> tuple[Path, str, int]:
    """Shared between `harness voice capture` and the in-chat /edit
    slash command. Appends a sample to `voice/captured.yaml` and
    returns (path, sample_id, total_sample_count)."""
    import yaml

    captured_path = settings.character_path / "voice" / "captured.yaml"
    captured_path.parent.mkdir(parents=True, exist_ok=True)

    if captured_path.exists():
        doc = yaml.safe_load(captured_path.read_text()) or {"samples": []}
    else:
        doc = {"version": 1, "samples": []}

    if sample_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        sample_id = f"captured-{stamp}"

    new_sample: dict[str, object] = {
        "id": sample_id,
        "prompt": prompt,
        "gold": gold.strip(),
        "captured_at": datetime.now(UTC).isoformat(),
        "captured_from": f"session={session}",
    }
    if original is not None:
        new_sample["original"] = original

    doc.setdefault("samples", []).append(new_sample)
    captured_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return captured_path, sample_id, len(doc["samples"])


@voice_app.command("capture")
def voice_capture(
    session: str = typer.Option("local", help="Session id to pull the exchange from."),
    gold: str = typer.Option(
        ...,
        "--gold",
        help="The corrected reply — what Airton should have said in response "
        "to the last user prompt in the session.",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Override the user prompt this sample is paired with. Defaults "
        "to the last user turn in the session.",
    ),
    sample_id: str | None = typer.Option(
        None,
        "--id",
        help="Custom sample id. Defaults to captured-<UTC timestamp>.",
    ),
) -> None:
    """Capture a user edit of Airton's reply as a new voice sample.

    The captured sample goes into `character/<name>/voice/captured.yaml`,
    a separate file from the curated canonical set, and will be loaded
    alongside canonical samples on the next character load. Over time
    this is how the voice corpus compounds from real use."""
    transcript = Transcript(settings.character_db_path)
    try:
        history = transcript.tail(session, limit=200)
    finally:
        transcript.close()

    if prompt is None:
        user_turns = [m for m in history if m.role == "user"]
        if not user_turns:
            console.print(
                f"[red]no user turns in session '{session}'. "
                "Pass --prompt to supply one explicitly.[/red]"
            )
            raise typer.Exit(code=1)
        prompt = user_turns[-1].content

    original: str | None = None
    assistant_turns = [m for m in history if m.role == "assistant"]
    if assistant_turns:
        original = assistant_turns[-1].content

    captured_path, final_id, total = _write_voice_capture(
        prompt=prompt,
        gold=gold,
        session=session,
        original=original,
        sample_id=sample_id,
    )

    console.print(
        f"[green]captured[/green] id={final_id!r} "
        f"→ {captured_path.relative_to(settings.root)} "
        f"(now {total} captured sample(s))"
    )


@voice_app.command("list-captured")
def voice_list_captured() -> None:
    """List every captured sample in the current character."""
    import yaml

    captured_path = settings.character_path / "voice" / "captured.yaml"
    if not captured_path.exists():
        console.print("[dim](no captured samples yet)[/dim]")
        return
    doc = yaml.safe_load(captured_path.read_text()) or {}
    samples = doc.get("samples", []) or []
    if not samples:
        console.print("[dim](no captured samples yet)[/dim]")
        return
    table = Table(title=f"Captured samples ({len(samples)})", show_lines=True)
    table.add_column("id", style="bold")
    table.add_column("captured_at")
    table.add_column("prompt")
    table.add_column("gold", style="green")
    for s in samples:
        table.add_row(
            str(s.get("id", "?")),
            str(s.get("captured_at", "?")),
            str(s.get("prompt", "?")),
            str(s.get("gold", "?")),
        )
    console.print(table)
