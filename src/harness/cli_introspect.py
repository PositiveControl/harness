"""Enumerate the Typer CLI's registered commands so the introspect tool
(harness-4f2) can describe the user-facing surface without drifting from
the actual `@app.command` decorators.

Typer exposes commands and sub-apps via two attributes on every
`Typer()` instance:
  - `registered_commands: list[CommandInfo]` — each carries the
    optional explicit `name` and the `callback` function.
  - `registered_groups: list[TyperInfo]` — each carries the sub-app
    `name` and `typer_instance` to recurse into.

When a command is declared with bare `@app.command()` (no explicit
name), Typer stores `name=None` and derives the public name from the
callback function name with underscores → dashes at CLI resolution
time. We apply the same derivation here.

A pinning test (`tests/test_cli_introspect.py`) compares the enumerated
set against a frozen expected list; a drift — new command, deleted
command, renamed command — fails loudly so the introspect tool's
`scope=commands` output stays truthful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


@dataclass(frozen=True)
class CommandInfo:
    """Public-facing summary of one CLI command.

    `path` is the space-separated invocation (e.g. 'chat', 'memory
    ingest'). `summary` is the first non-empty line of the callback's
    docstring, or '' if the callback has no docstring."""

    path: str
    summary: str


def _derive_name(name: str | None, callback_name: str) -> str:
    """Typer's public-name rule: explicit name when given, else the
    callback's function name with underscores swapped for dashes."""
    return name if name else callback_name.replace("_", "-")


def _first_docstring_line(callback: object) -> str:
    doc = getattr(callback, "__doc__", None)
    if not isinstance(doc, str) or not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def list_cli_commands(app: typer.Typer, *, prefix: str = "") -> list[CommandInfo]:
    """Walk `app`'s registered commands and sub-apps; return a flat
    list of `CommandInfo` sorted by `path`. The recursion threads a
    `prefix` so 'memory' + 'ingest' becomes 'memory ingest'."""
    out: list[CommandInfo] = []
    for cmd in app.registered_commands:
        callback = cmd.callback
        if callback is None:
            continue
        name = _derive_name(cmd.name, callback.__name__)
        path = f"{prefix} {name}".strip()
        out.append(CommandInfo(path=path, summary=_first_docstring_line(callback)))
    for group in app.registered_groups:
        sub_app = group.typer_instance
        if sub_app is None:
            continue
        group_name = group.name or ""
        sub_prefix = f"{prefix} {group_name}".strip()
        out.extend(list_cli_commands(sub_app, prefix=sub_prefix))
    out.sort(key=lambda c: c.path)
    return out
