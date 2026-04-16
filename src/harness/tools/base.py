from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema description of a tool, plus metadata the orchestrator
    needs (tier for authorization)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for arguments
    tier: str  # "read" | "write" — write-tier tools need user confirmation


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
    may be empty if the model only wants to call tools this round."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ToolRegistry:
    """Holds tools by name, renders specs for the model, dispatches
    calls. Registry is mutable (CLI configures it per session); tool
    instances are read-only after construction."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

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

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute the named tool. Failures are returned as ToolResult,
        not raised — the caller is the orchestrator, which wants to
        feed errors back to the model for it to recover from."""
        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                output=f"unknown tool: {name!r}. Available: {self.names()}",
                success=False,
                error="unknown_tool",
            )
        try:
            # Tool's `call` is not on the Protocol (see Tool docstring); each
            # concrete implementation supplies it with typed kwargs.
            out: str = self._tools[name].call(**arguments)  # type: ignore[attr-defined]
        except Exception as exc:  # any failure goes back to the model, not up the stack
            return ToolResult(
                tool_name=name,
                output=f"error calling {name}: {exc}",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(tool_name=name, output=out, success=True)
