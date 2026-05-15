"""Schema-constrained JSON output for web endpoints.

`complete_json(adapter, messages, schema)` runs the adapter's
`complete_grammar()` method against a JSON schema and parses the
result. Returns a Python `dict[str, Any]` that conforms to the schema.

Adapter-boundary invariant preserved: this helper calls the adapter
through its public `GrammarCapableAdapter` protocol; outlines / mlx-lm
internals stay inside `harness.model.mlx`. The helper raises
`UnsupportedAdapterError` for adapters that don't expose
`complete_grammar()` so the caller can surface a clear error rather
than silently degrading.

Used by character extensions (the TFR explainer's `/explain` endpoint
in harness-3jz1.4 is the first concrete caller) that need the model to
emit prose under a tight structural contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from harness.model.adapter import ChatMessage


class UnsupportedAdapterError(RuntimeError):
    """Adapter doesn't expose `complete_grammar()`. Caller should pick a
    different adapter (MLX supports it; echo / Ollama do not as of
    harness-3jz1.9) or fall through to a non-grammar code path."""


def complete_json(
    adapter: object,
    messages: Iterable[ChatMessage],
    *,
    schema: dict[str, Any],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run schema-constrained generation and return the parsed JSON.

    `adapter` must expose `complete_grammar(messages, schema, *,
    max_tokens, temperature) -> str`. The returned string is parsed as
    JSON; the parsed dict is returned. `json.JSONDecodeError` raised by
    the parse propagates so callers can decide whether to retry, log,
    or surface a 502.

    Caller-side validation: the adapter is responsible for honoring the
    schema. We deliberately don't re-validate against the schema here
    — `complete_grammar` is the contract that the schema was honored;
    duplicating that with `jsonschema` would tie the helper to a heavy
    optional dep without buying much.
    """
    fn = getattr(adapter, "complete_grammar", None)
    if not callable(fn):
        raise UnsupportedAdapterError(
            f"adapter {type(adapter).__name__} does not expose "
            f"complete_grammar(); pick an MLX-family adapter or use a "
            f"non-grammar code path."
        )
    raw: object = fn(list(messages), schema, max_tokens=max_tokens, temperature=temperature)
    if not isinstance(raw, str):
        raise TypeError(f"complete_grammar returned {type(raw).__name__}, expected str")
    return _parse_json(raw)


def _parse_json(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError(
            f"expected a JSON object, got {type(parsed).__name__}",
            raw,
            0,
        )
    return parsed


__all__ = ["UnsupportedAdapterError", "complete_json"]
