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
from typing import TYPE_CHECKING, Any

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

# Capability-gap detection (harness-8is). Each entry names a human-
# readable capability, the set of tool names that would provide it,
# and a one-line hint that tells the user how to turn it on. When
# NONE of the required tools are in the registry the gap is reported
# in scope=tools so the model can answer 'can you do X?' truthfully.
_CAPABILITY_GAPS: tuple[tuple[str, frozenset[str], str], ...] = (
    (
        "browse the web",
        frozenset({"search_web"}),
        "--tools-add search_web (or --tool-set research)",
    ),
    (
        "edit or write files",
        frozenset({"edit_file", "write_file"}),
        "read-only workspace this session; --tool-set coding for editing",
    ),
    (
        "run shell commands",
        frozenset({"shell"}),
        "--tools-add shell",
    ),
    (
        "inspect git history",
        frozenset({"git_status", "git_diff", "git_log"}),
        "--tool-set coding",
    ),
    (
        "record new memories or facts",
        frozenset({"remember_fact", "remember_event"}),
        "--tool-set memory",
    ),
    (
        "curate or consolidate memory",
        frozenset({"scribe_session", "consolidate_memory"}),
        "--tool-set memory",
    ),
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
    # Live references — read at call time, not snapshot at context
    # build. retrieval_health is the CLI's _RetrievalState dataclass
    # (voice_ok / episodic_ok / semantic_ok booleans). persona_active
    # mirrors the --persona flag's effective state; router_id is a
    # user-facing short label like 'grammar:Hermes-3-3B' or None when
    # the router isn't loaded. Kept as Any / bool / str rather than
    # importing CLI types to keep the tool layer cycle-free.
    retrieval_health: Any | None = None
    persona_active: bool = False
    router_id: str | None = None


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
        gaps = self._capability_gaps()
        if gaps:
            lines.append("")
            lines.append("Capability gaps (cannot do this session):")
            for capability, hint in gaps:
                lines.append(f"  - {capability} — {hint}")
        return "\n".join(lines)

    def _capability_gaps(self) -> list[tuple[str, str]]:
        """Walk the capability map and return (capability, hint) tuples
        for any whose required tools are entirely absent from the
        registry. Answers 'can you do X?' honestly."""
        loaded = set(self.context.registry.names())
        out: list[tuple[str, str]] = []
        for capability, required, hint in _CAPABILITY_GAPS:
            if required.isdisjoint(loaded):
                out.append((capability, hint))
        return out

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
        lines.append(f"  persona rewriter: {'on' if self.context.persona_active else 'off'}")
        lines.append(
            f"  intent router: {self.context.router_id}"
            if self.context.router_id
            else "  intent router: off"
        )
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
        health = self.context.retrieval_health
        if health is not None:
            # Duck-typed: CLI's _RetrievalState exposes voice_ok /
            # episodic_ok / semantic_ok booleans. Any source that
            # raised earlier this session is flipped False; we surface
            # those so 'why didn't you recall X' has an answer.
            disabled = [
                name
                for name, attr in (
                    ("voice", "voice_ok"),
                    ("episodic", "episodic_ok"),
                    ("semantic", "semantic_ok"),
                )
                if not getattr(health, attr, True)
            ]
            if disabled:
                lines.append(
                    f"  retrieval health: {', '.join(disabled)} disabled this session (prior error)"
                )
            else:
                lines.append("  retrieval health: all sources ok")
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
