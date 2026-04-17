"""Router contract: given a user message and the available tools, pick
a tool + arguments (or decide no tool is needed).

Two distinct `None`-ish outcomes on purpose:
- `RouterIntent(tool_name=None, arguments={})` — router worked, says
  nothing to do. Orchestrator falls through to the normal loop with
  one fewer fabrication round wasted.
- `None` from `classify()` — router could not decide (unparseable
  output, model degenerated, etc.). Orchestrator also falls through.
  The distinction matters for evals + logging, not for routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from harness.tools.base import ToolSpec


@dataclass(frozen=True)
class RouterIntent:
    """The router's verdict for a single user turn.

    `tool_name=None` means the router confidently decided no tool is
    needed (casual chat, question the main model can answer directly).
    Non-null names are NOT validated against the registry here — that
    check lives in the orchestrator, which also verifies arguments
    match the tool's schema before executing."""

    tool_name: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Router(Protocol):
    """Structural type for anything that can classify a user turn into
    a tool-use intent. Implementations must be advisory: any failure to
    parse or classify returns `None` rather than raising."""

    def classify(
        self,
        user_message: str,
        tool_specs: Sequence[ToolSpec],
    ) -> RouterIntent | None: ...
