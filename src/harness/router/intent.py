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
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from harness.tools.base import ToolSpec


# Scope verdicts for character-bounded personas (harness-8dop). Only
# personas with a populated `scope_redirect_template` consume `out` —
# every other persona ignores the field. Defaulting to `unsure` keeps
# legacy callers + un-extended grammar routers fully back-compatible.
RouterScope = Literal["in", "out", "unsure"]


@dataclass(frozen=True)
class RouterIntent:
    """The router's verdict for a single user turn.

    `tool_name=None` means the router confidently decided no tool is
    needed (casual chat, question the main model can answer directly).
    Non-null names are NOT validated against the registry here — that
    check lives in the orchestrator, which also verifies arguments
    match the tool's schema before executing.

    `scope` (harness-8dop) — for personas with a bounded corpus
    (e.g. airton_c1 = JO 7110.65 only), the router classifies whether
    the question falls inside the persona's scope. The orchestrator
    short-circuits on `out` to a character-supplied scope-redirect
    template instead of running the model. `in` and `unsure` both
    proceed to the normal flow; `unsure` is the conservative default
    so a misclassification can't false-block a legitimate question."""

    tool_name: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    scope: RouterScope = "unsure"


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
