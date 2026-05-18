"""Pinning test for the CLI commands enumerator (harness-jjg).

Hard-pins the set of user-facing commands so a new `@app.command`
without an introspection-tool update fails loudly. When you add or
rename a CLI command, update `_EXPECTED_COMMANDS` here.
"""

from __future__ import annotations

from harness.cli import app
from harness.cli_introspect import CommandInfo, list_cli_commands

# Flat, sorted list of all user-facing CLI commands. When this drifts
# from the enumerator output, the harness gained or lost a command —
# update deliberately and make sure the introspect tool's documented
# surface (and docs/usage.md) move together.
_EXPECTED_COMMANDS: tuple[str, ...] = (
    "chat",
    "daemon",
    "daemon-status",
    "describe",
    "eval atc",
    "eval atc-audio",
    "eval atc-retrieval",
    "eval phraseology",
    "eval router",
    "eval session-resume",
    "eval tfr",
    "eval tool-loop",
    "eval voice",
    "memory consolidate",
    "memory fact-add",
    "memory fact-list",
    "memory fact-search",
    "memory harvest-memories",
    "memory harvest-skills",
    "memory ingest",
    "memory list",
    "memory rebuild-embeddings",
    "memory scribe",
    "memory search",
    "memory wipe",
    "phraseology lint",
    "session compact-reset",
    "session list",
    "session reset",
    "session show",
    "voice capture",
    "voice list-captured",
    "web serve",
)


def test_enumerated_commands_match_pin() -> None:
    """The enumerator walks both top-level commands (chat, describe)
    and every sub-app (memory/eval/voice) and must match the pin."""
    enumerated = [c.path for c in list_cli_commands(app)]
    assert enumerated == list(_EXPECTED_COMMANDS), (
        "CLI commands changed. Update _EXPECTED_COMMANDS + introspect "
        f"tool output. Got: {enumerated}"
    )


def test_chat_command_exposes_chain_rewrites_flag() -> None:
    """Regression for harness-n9k: `--chain-rewrites` was advertised in
    docs/usage.md and CLAUDE.md but never wired into the chat command.
    Pin the flag's presence so a silent removal fails here. Pair with
    `test_resolve_adapter_threads_chain_rewrites_into_persona_adapter`
    in test_airton_b_integration.py (the orchestration half)."""
    import inspect

    from harness.cli import chat

    params = inspect.signature(chat).parameters
    assert "chain_rewrites" in params, (
        "chat command lost --chain-rewrites. Either re-add it or strip "
        "the flag from docs/usage.md and CLAUDE.md."
    )


def test_every_command_has_a_summary() -> None:
    """Introspection's scope=commands output relies on the summary
    line. Every registered callback must have a docstring so the
    user sees something more than the bare command name. If a new
    command lands without a docstring this fails — write one."""
    missing = [c.path for c in list_cli_commands(app) if not c.summary]
    assert not missing, f"commands missing docstring summaries: {missing}"


def test_list_cli_commands_returns_command_info_instances() -> None:
    """Light shape-check so downstream consumers (IntrospectTool) can
    rely on the dataclass fields rather than raw tuples."""
    enumerated = list_cli_commands(app)
    assert enumerated  # non-empty
    for c in enumerated:
        assert isinstance(c, CommandInfo)
        assert c.path
        assert isinstance(c.summary, str)
