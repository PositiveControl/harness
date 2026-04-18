"""Self-introspection tool (harness-4f2).

Single read-tier tool the agent can call when asked what it can do.
A `scope` enum selects which surface to describe — tools, model,
memory, character, CLI commands, or all of them concatenated.

The tool reads from an `IntrospectContext` wired up at registry-build
time rather than reaching back into `harness.cli` directly, so the
tool layer stays free of circular imports and the whole thing stays
unit-testable with stub dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from harness.tools.base import ToolSpec

if TYPE_CHECKING:
    from pathlib import Path

    from harness.character import Character
    from harness.cli_introspect import CommandInfo
    from harness.config import Settings
    from harness.model.adapter import ModelAdapter
    from harness.store.episodic import EpisodicStore
    from harness.store.semantic import SemanticStore
    from harness.tools.base import ToolRegistry


_VALID_SCOPES: tuple[str, ...] = (
    "tools",
    "model",
    "memory",
    "character",
    "commands",
    "all",
)


@dataclass
class IntrospectContext:
    """Dependencies the introspect tool needs to answer accurately.

    Built once at registry-wiring time. `registry` is the same
    ToolRegistry the introspect tool lives inside — see
    `src/harness/cli.py::_build_tool_registry_for_tui` for the
    chicken-and-egg dance (register every other tool first, then
    build this context referencing the populated registry, then
    register the introspect tool).

    `commands` is pre-enumerated by `cli_introspect.list_cli_commands`
    so the tool doesn't have to import `harness.cli` at runtime.
    `user_id` scopes memory counts the same way `search_memory` does.
    """

    registry: ToolRegistry
    adapter: ModelAdapter
    character: Character
    settings: Settings
    episodic: EpisodicStore | None = None
    semantic: SemanticStore | None = None
    workspace: Path | None = None
    user_id: str | None = None
    commands: tuple[CommandInfo, ...] = field(default_factory=tuple)


@dataclass
class IntrospectTool:
    """Describe the harness's current capabilities on demand.

    One-shot read-tier tool — costs nothing to call; the data it
    reports is already held in memory or cheaply reachable from
    SQLite. The system-prompt nudge (harness-u71) is what steers the
    model toward calling this instead of guessing."""

    context: IntrospectContext

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="introspect",
            description=(
                "Describe your own capabilities accurately instead of "
                "guessing. Call this when the user asks what tools you "
                "have, what model is running, how much you remember, "
                "what CLI commands exist, or what you can/can't do. "
                "`scope` picks which surface to report on; use 'all' "
                "for the full summary."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": list(_VALID_SCOPES),
                        "description": (
                            "Which surface to describe. tools=loaded "
                            "tools + tier; model=adapter/model info; "
                            "memory=episodic + semantic stats; "
                            "character=name, persona, voice corpus "
                            "size; commands=CLI commands; "
                            "all=everything."
                        ),
                    },
                },
                "required": ["scope"],
            },
            tier="read",
            display_name="Describe self",
        )

    def call(self, *, scope: str) -> str:
        if scope not in _VALID_SCOPES:
            return f"unknown scope {scope!r}; valid: {', '.join(_VALID_SCOPES)}"
        if scope == "all":
            return "\n\n".join(
                self._render(s) for s in ("tools", "model", "memory", "character", "commands")
            )
        return self._render(scope)

    def _render(self, scope: str) -> str:
        if scope == "tools":
            return self._render_tools()
        if scope == "model":
            return self._render_model()
        if scope == "memory":
            return self._render_memory()
        if scope == "character":
            return self._render_character()
        if scope == "commands":
            return self._render_commands()
        # Guarded by call()'s scope check above — kept for mypy.
        return f"unknown scope {scope!r}"

    def _render_tools(self) -> str:
        specs = self.context.registry.specs()
        if not specs:
            return "Tools: (none loaded)"
        lines = ["Tools (current session):"]
        for spec in sorted(specs, key=lambda s: s.name):
            first_line = (
                (spec.description or "").strip().splitlines()[0] if spec.description else ""
            )
            lines.append(f"  - {spec.name} [{spec.tier}] — {first_line}")
        return "\n".join(lines)

    def _render_model(self) -> str:
        adapter = self.context.adapter
        lines = ["Model:"]
        lines.append(f"  adapter id: {adapter.id}")
        lines.append(f"  context window: {adapter.context_window} tokens")
        repo = getattr(adapter, "repo", None)
        if repo:
            lines.append(f"  model repo: {repo}")
        adapter_path = getattr(adapter, "adapter_path", None)
        if adapter_path:
            lines.append(f"  LoRA adapter: {adapter_path}")
        lines.append(f"  embedder: {self.context.settings.embedder_repo}")
        return "\n".join(lines)

    def _render_memory(self) -> str:
        lines = ["Memory:"]
        ep = self.context.episodic
        if ep is None:
            lines.append("  episodic: (not enabled this session)")
        else:
            total = ep.count(user_id=self.context.user_id)
            seeds = ep.count(tier="seed", user_id=self.context.user_id)
            last = ep.last_created_at(user_id=self.context.user_id)
            last_str = last.isoformat(timespec="seconds") if last else "never"
            lines.append(
                f"  episodic: {total} active records ({seeds} seed); last write {last_str}"
            )
        sem = self.context.semantic
        if sem is None:
            lines.append("  semantic: (not enabled this session)")
        else:
            total = sem.count(user_id=self.context.user_id)
            last = sem.last_created_at(user_id=self.context.user_id)
            last_str = last.isoformat(timespec="seconds") if last else "never"
            lines.append(f"  semantic: {total} active facts; last write {last_str}")
        scope = (
            "shared only" if self.context.user_id is None else f"shared + {self.context.user_id}"
        )
        lines.append(f"  scope: {scope}")
        return "\n".join(lines)

    def _render_character(self) -> str:
        c = self.context.character
        lines = ["Character:"]
        lines.append(f"  name: {c.name}")
        lines.append(f"  pronouns: {c.pronouns}")
        lines.append(f"  era: {c.era}")
        lines.append(
            f"  voice samples: {c.canonical_voice_count} canonical + "
            f"{c.captured_voice_count} captured = {len(c.voice_samples)}"
        )
        lines.append(f"  seed memories: {len(c.seed_memories)}")
        lines.append(f"  values: {len(c.values)}, taboos: {len(c.taboos)}")
        return "\n".join(lines)

    def _render_commands(self) -> str:
        if not self.context.commands:
            return "CLI commands: (not enumerated — context.commands empty)"
        lines = ["CLI commands:"]
        for cmd in self.context.commands:
            summary = f" — {cmd.summary}" if cmd.summary else ""
            lines.append(f"  harness {cmd.path}{summary}")
        return "\n".join(lines)
