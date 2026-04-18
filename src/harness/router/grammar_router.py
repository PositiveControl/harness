"""Grammar-constrained Router: routes via an adapter that supports
JSON-schema-constrained decoding. Guarantees the output is valid JSON
and the `tool` field is one of the registered tool names (or null), by
construction rather than by post-validation.

What this buys vs ModelRouter:
- No parse failures — every classify() produces a well-typed RouterIntent
  (subject to adapter not raising).
- No hallucinated tool names — the sampler literally cannot emit one.
- Opens the door to args-schema enforcement when we're ready.

What this does NOT buy:
- Correct tool selection. If the model's *preferred* choice among the
  enumerated options is wrong, the grammar still lets it pick it. The
  residual eval failures (find/online → glob, search_memory vs
  search_facts, opinion → memory) are logic failures — no grammar
  can fix them.

Constraint is applied at the adapter layer (see
MLXAdapter.complete_grammar) so outlines / mlx-lm internals stay
inside the adapter boundary, consistent with the rest of
src/harness/model/."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from harness.model.adapter import ChatMessage
from harness.router.intent import RouterIntent
from harness.router.model_router import _build_system_prompt, parse_router_output
from harness.tools.base import ToolSpec


def build_router_schema(tool_specs: Sequence[ToolSpec]) -> dict[str, Any]:
    """Build the JSON schema that constrains the router output.

    Shape:
        {"tool": <enum of tool_names + null>, "arguments": <any object>}

    Tool names are exposed as a string enum; `null` is included via a
    `["string", "null"]` type union so the small model can confidently
    express 'no tool needed' without having to invent a sentinel name.
    `arguments` stays free-form here — we let the model write whatever
    shape it wants and post-validate required keys in the orchestrator
    (or in the args-schema extension later)."""
    tool_names = [s.name for s in tool_specs]
    if tool_names:
        tool_field: dict[str, Any] = {
            "type": ["string", "null"],
            "enum": [*tool_names, None],
        }
    else:
        # With no tools available the only valid answer is null. Keeping
        # the field typed as nullable-string (no enum) means the model
        # is also free to express that the request is unanswerable.
        tool_field = {"type": "null"}
    return {
        "type": "object",
        "properties": {
            "tool": tool_field,
            "arguments": {"type": "object"},
        },
        "required": ["tool", "arguments"],
        "additionalProperties": False,
    }


@dataclass
class GrammarRouter:
    """A Router backed by a grammar-capable adapter. The adapter must
    expose `complete_grammar(messages, schema, ...)`. If it doesn't,
    construction raises — callers should pick ModelRouter instead."""

    adapter: Any  # structurally typed; see GrammarCapableAdapter Protocol
    max_tokens: int = 256
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not callable(getattr(self.adapter, "complete_grammar", None)):
            raise TypeError(
                f"GrammarRouter requires an adapter with complete_grammar(); "
                f"{type(self.adapter).__name__} does not expose one. "
                f"Use ModelRouter for free-form adapters."
            )

    def classify(
        self,
        user_message: str,
        tool_specs: Sequence[ToolSpec],
    ) -> RouterIntent | None:
        schema = build_router_schema(tool_specs)
        system = ChatMessage(role="system", content=_build_system_prompt(tool_specs))
        user = ChatMessage(role="user", content=user_message)
        try:
            raw = self.adapter.complete_grammar(
                [system, user],
                schema,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception:
            # Advisory contract — never raise out of classify(). Adapter
            # failures (missing grammar extra, OOM, outlines bug) degrade
            # to 'router could not decide' and the orchestrator falls
            # through to the normal tool loop.
            return None
        return parse_router_output(raw)
