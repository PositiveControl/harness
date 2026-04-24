from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema description of a tool, plus metadata the orchestrator
    needs (tier for authorization)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for arguments
    tier: str  # "read" | "write" — write-tier tools need user confirmation
    display_name: str | None = None  # human-readable label for UI; falls back to `name`
    # Tools flagged high_noise dump large, often-unhelpful bulk into
    # the message thread — grep / list_dir / search_web results are
    # classic offenders. When the tool-result summarizer hook is
    # registered (CLI --summarize-tool-results), outputs from these
    # tools above a size threshold get compressed before the model
    # sees them. Everything else passes through untouched.
    high_noise: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.name


@runtime_checkable
class Tool(Protocol):
    """A tool has a ToolSpec and a `call(**kwargs) -> str` method. The
    `call` signature is intentionally omitted from the protocol: each
    concrete tool has typed, per-tool keyword-only arguments (e.g.
    `call(self, *, path: str)`), which would fail strict structural
    typing against a generic `call(**kwargs: Any)` declaration. The
    registry invokes `call` with unpacked kwargs at runtime; tests
    cover the argument shape per tool."""

    @property
    def spec(self) -> ToolSpec: ...


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    output: str
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class ModelReply:
    """Richer return type for adapters that support tool use. `content`
    may be empty if the model only wants to call tools this round.

    `was_truncated` and `had_unparseable_call` are diagnostic hints the
    orchestrator uses to recover from common failure modes (token-limit
    truncation, malformed `<tool_call>` blocks). Adapters that can't
    cheaply detect these leave them False — the loop falls back to
    teaser-regex detection."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    was_truncated: bool = False
    had_unparseable_call: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True)
class StreamText:
    """Visible text delta emitted by a streaming adapter. The accumulation
    of every `StreamText.text` across a stream reconstructs the model's
    raw output (tool-call tags included — callers mask them for display)."""

    text: str


@dataclass(frozen=True)
class StreamComplete:
    """Terminal chunk of a tool-aware stream. Carries the parsed
    ModelReply so the tool loop can dispatch any tool calls the model
    emitted. Exactly one StreamComplete is yielded per stream, always
    last."""

    reply: ModelReply


StreamChunk = StreamText | StreamComplete


class ToolRegistry:
    """Holds tools by name, renders specs for the model, dispatches
    calls. Registry is mutable (CLI configures it per session); tool
    instances are read-only after construction."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # Per-registration description overrides. Applied at specs()
        # emission so every downstream consumer (router, model schema,
        # introspect) sees the override without wrapping Tool instances.
        # Populated via override_description(), typically from a
        # profile-level map like TOOL_PROFILE_DESCRIPTIONS (profiles.py)
        # so a character-specific profile can reframe generic tools
        # (e.g. atc's search_memory targets a rulebook, not "past events").
        self._description_overrides: dict[str, str] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"no tool named {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def override_description(self, name: str, description: str) -> None:
        """Replace the description surfaced in specs() for `name`. Raises
        KeyError if the tool isn't registered — overrides are meant to
        reframe real tools, not stub placeholders."""
        if name not in self._tools:
            raise KeyError(f"no tool named {name!r}")
        self._description_overrides[name] = description

    def specs(self) -> list[ToolSpec]:
        out: list[ToolSpec] = []
        for tool in self._tools.values():
            spec = tool.spec
            override = self._description_overrides.get(spec.name)
            out.append(replace(spec, description=override) if override else spec)
        return out

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute the named tool. Failures are returned as ToolResult,
        not raised — the caller is the orchestrator, which wants to
        feed errors back to the model for it to recover from.

        Unknown-keyword TypeErrors get a structured rewrite (harness-d7e)
        that lists the tool's accepted properties so the model can
        retry without the offending field instead of repeating the
        same call. Generic Python TypeError messages don't enumerate
        valid kwargs, so the model has no way to know what to drop."""
        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                output=f"unknown tool: {name!r}. Available: {self.names()}",
                success=False,
                error="unknown_tool",
            )
        tool = self._tools[name]
        try:
            # Tool's `call` is not on the Protocol (see Tool docstring); each
            # concrete implementation supplies it with typed kwargs.
            out: str = tool.call(**arguments)  # type: ignore[attr-defined]
        except TypeError as exc:
            unknown = _unknown_kwarg_from(exc)
            if unknown is not None:
                accepted = sorted((tool.spec.parameters.get("properties") or {}).keys())
                msg = (
                    f"tool {name!r} rejected unknown argument {unknown!r}. "
                    f"Accepts: {', '.join(accepted) or '(none)'}. "
                    "Retry without the unknown field."
                )
                return ToolResult(
                    tool_name=name,
                    output=msg,
                    success=False,
                    error=f"unknown_kwarg:{unknown}",
                )
            return ToolResult(
                tool_name=name,
                output=f"error calling {name}: {exc}",
                success=False,
                error=f"TypeError: {exc}",
            )
        except Exception as exc:  # any failure goes back to the model, not up the stack
            return ToolResult(
                tool_name=name,
                output=f"error calling {name}: {exc}",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(tool_name=name, output=out, success=True)


_UNKNOWN_KWARG_RE = re.compile(r"got an unexpected keyword argument ['\"]([^'\"]+)['\"]")


def _unknown_kwarg_from(exc: TypeError) -> str | None:
    """Pluck the offending kwarg name from a TypeError raised by a
    `call(**arguments)` invocation. Returns None when the TypeError
    came from something else (shape mismatch, missing required arg)
    — those flow through the generic error path."""
    match = _UNKNOWN_KWARG_RE.search(str(exc))
    return match.group(1) if match else None
