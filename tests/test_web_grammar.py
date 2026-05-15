"""Tests for the JSON-schema-constrained reply helper (harness-3jz1.9).

The helper is a thin wrapper around `adapter.complete_grammar()`. We
exercise the failure path (echo adapter has no `complete_grammar`) and
the parse path (mock adapter returns a JSON string) without spinning
up a real MLX model.
"""

from __future__ import annotations

import json

import pytest

from harness.model.adapter import ChatMessage
from harness.model.echo import EchoAdapter
from harness.web.grammar import UnsupportedAdapterError, complete_json


class _MockGrammarAdapter:
    """Stand-in for an MLX-style grammar-capable adapter."""

    id = "mock-grammar"

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.received_schema: dict[str, object] | None = None

    def complete_grammar(
        self,
        messages: list[ChatMessage],
        schema: dict[str, object],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        self.received_schema = schema
        return json.dumps(self._payload)


def test_complete_json_raises_unsupported_for_echo() -> None:
    with pytest.raises(UnsupportedAdapterError, match="complete_grammar"):
        complete_json(
            EchoAdapter(),
            [ChatMessage(role="user", content="ping")],
            schema={"type": "object"},
        )


def test_complete_json_returns_parsed_payload_from_grammar_adapter() -> None:
    adapter = _MockGrammarAdapter(payload={"verdict": "Stadium TFR", "caveats": []})
    out = complete_json(
        adapter,
        [ChatMessage(role="user", content="explain")],
        schema={"type": "object", "properties": {"verdict": {"type": "string"}}},
    )
    assert out == {"verdict": "Stadium TFR", "caveats": []}
    # Schema is forwarded to the adapter unchanged.
    assert adapter.received_schema is not None
    assert "properties" in adapter.received_schema


def test_complete_json_rejects_non_object_payload() -> None:
    """If the adapter (somehow) returns a JSON array or scalar, the
    helper raises rather than silently returning a non-dict."""
    adapter = _MockGrammarAdapter(payload={})

    # Force a non-object reply via raw bytes.
    def _raw_array(
        messages: list[ChatMessage],
        schema: dict[str, object],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        return json.dumps([1, 2, 3])

    adapter.complete_grammar = _raw_array  # type: ignore[method-assign]
    with pytest.raises(json.JSONDecodeError):
        complete_json(adapter, [], schema={"type": "object"})
