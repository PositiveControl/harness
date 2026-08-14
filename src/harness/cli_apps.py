"""The Typer application objects, and nothing else.

Step 5a of docs/cli-extraction-plan.md. Sub-apps are the one piece of
genuinely shared state in the CLI: `@sub_app.command()` captures the app
object at import time, so every module that registers a handler has to
import the SAME object the root app mounted. Two copies of `eval_app`
means half the commands silently never register.

Owning them here lets the handler modules (cli_memory, cli_voice,
cli_eval, ...) import their sub-app without importing `cli.py` itself —
which is what makes the split possible at all.

`harness.cli` re-exports every name below, so the `harness.cli:app`
entry point and existing `from harness.cli import app` call sites keep
working unchanged.

Nothing here may import `harness.cli`.
"""

from __future__ import annotations

import typer

from harness.driver.cli import drive_app as drive_app

app = typer.Typer(add_completion=False, no_args_is_help=True)

eval_app = typer.Typer(help="Evaluations against the current character.", no_args_is_help=True)
memory_app = typer.Typer(help="Inspect and manage episodic memory.", no_args_is_help=True)
voice_app = typer.Typer(help="Voice suite — capture and manage samples.", no_args_is_help=True)
session_app = typer.Typer(
    help="Browse and stream Airton's recorded chat sessions.",
    no_args_is_help=True,
)
phraseology_app = typer.Typer(
    help="Cite-grounded ATC transmission verifier (airton_c1, JO 7110.65).",
    no_args_is_help=True,
)
web_app = typer.Typer(
    help="Serve the current character over HTTP (harness.web factory).",
    no_args_is_help=True,
)
plan_app = typer.Typer(
    help="Inspect, bootstrap, and manage runtime-typed plans (harness-ptdw).",
    no_args_is_help=True,
)
tool_app = typer.Typer(
    help="Inspect + manage the tool catalog (harness-rqg0).",
    no_args_is_help=True,
)
denylist_app = typer.Typer(
    help="Inspect + manage the fetch_url denylist (harness-4dgm).",
    no_args_is_help=True,
)

app.add_typer(eval_app, name="eval")
app.add_typer(memory_app, name="memory")
app.add_typer(voice_app, name="voice")
app.add_typer(session_app, name="session")
app.add_typer(phraseology_app, name="phraseology")
app.add_typer(web_app, name="web")
app.add_typer(plan_app, name="plan")
app.add_typer(tool_app, name="tool")
app.add_typer(denylist_app, name="denylist")
# Multi-turn driver subcommands (harness-e9oq). `harness drive plan`
# and `harness drive loop` — namespaced under `drive` so they don't
# collide with the existing `harness plan` runtime-typed-plan tools.
app.add_typer(drive_app, name="drive")
