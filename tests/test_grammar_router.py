"""Tests for GrammarRouter (harness-c99). Schema builder tests run
unconditionally. Integration tests against a real MLX model are gated
behind HARNESS_TEST_ROUTER_GRAMMAR to match the existing MLX gate —
they load outlines + a ~2GB model and aren't portable."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from harness.model.adapter import ChatMessage
from harness.router import GrammarRouter, RouterIntent, build_router_schema
from harness.tools.base import ToolSpec


def _spec(name: str = "search_web") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        tier="read",
    )


# ---------- build_router_schema ----------


def test_schema_exposes_tool_enum_of_names_plus_null() -> None:
    schema = build_router_schema([_spec("search_web"), _spec("read_file")])
    tool_field = schema["properties"]["tool"]
    assert tool_field["type"] == ["string", "null"]
    assert tool_field["enum"] == ["search_web", "read_file", None]


def test_schema_requires_tool_and_arguments() -> None:
    schema = build_router_schema([_spec()])
    assert schema["required"] == ["tool", "arguments"]
    # Guard against accidental schema laxness — the router contract is
    # that these two keys always appear.
    assert schema["additionalProperties"] is False


def test_schema_with_no_tools_returns_null_only_shape() -> None:
    """With nothing to route to, the only valid 'tool' value is null —
    express that directly in the schema so the FSM can short-circuit."""
    schema = build_router_schema([])
    assert schema["properties"]["tool"] == {"type": "null"}


def test_schema_arguments_field_is_free_form_object() -> None:
    """Post-validation (required-key check in the orchestrator) owns
    args correctness. The schema here only constrains that `arguments`
    is an object — not that any specific keys are present."""
    schema = build_router_schema([_spec()])
    assert schema["properties"]["arguments"] == {"type": "object"}


# ---------- GrammarRouter ----------


@dataclass
class _GrammarScriptedAdapter:
    """Adapter that records the (messages, schema) passed to
    complete_grammar and returns a canned JSON string. Enough for
    exercising the router wiring without loading outlines / MLX."""

    id: str = "scripted-grammar"
    context_window: int = 8192
    replies: list[str] = field(default_factory=list)
    calls: list[tuple[list[ChatMessage], dict[str, Any]]] = field(default_factory=list)

    def complete_grammar(
        self,
        messages: Iterable[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append((list(messages), schema))
        return self.replies.pop(0) if self.replies else ""


@dataclass
class _FreeFormOnlyAdapter:
    """Adapter that exposes `complete` but NOT `complete_grammar` —
    GrammarRouter should refuse to accept it at construction."""

    id: str = "free-only"
    context_window: int = 8192

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        _ = messages, max_tokens, temperature
        return "{}"


def test_grammar_router_rejects_non_grammar_adapter() -> None:
    with pytest.raises(TypeError, match="complete_grammar"):
        GrammarRouter(adapter=_FreeFormOnlyAdapter())


def test_grammar_router_happy_path_parses_constrained_output() -> None:
    adapter = _GrammarScriptedAdapter(
        replies=['{"tool": "search_web", "arguments": {"query": "bbq"}}']
    )
    router = GrammarRouter(adapter=adapter)
    intent = router.classify("search the web for bbq", [_spec("search_web")])
    assert intent == RouterIntent(tool_name="search_web", arguments={"query": "bbq"})


def test_grammar_router_passes_schema_to_adapter() -> None:
    """The schema built from tool_specs is exactly what the adapter
    sees — that's the point of the grammar path."""
    adapter = _GrammarScriptedAdapter(replies=['{"tool": null, "arguments": {}}'])
    router = GrammarRouter(adapter=adapter)
    router.classify("hey", [_spec("search_web"), _spec("read_file")])
    _, schema = adapter.calls[0]
    assert schema == build_router_schema([_spec("search_web"), _spec("read_file")])


def test_grammar_router_handles_null_tool() -> None:
    adapter = _GrammarScriptedAdapter(replies=['{"tool": null, "arguments": {}}'])
    router = GrammarRouter(adapter=adapter)
    intent = router.classify("hey", [_spec()])
    assert intent == RouterIntent(tool_name=None, arguments={})


def test_grammar_router_returns_none_when_adapter_raises() -> None:
    """Advisory contract — never raises out of classify(). Real-world
    causes: missing `grammar` extra (RuntimeError), outlines internal
    failure, MLX OOM, etc. We also emit a RuntimeWarning the first time
    so silent degradation (every call returning null) is visible."""
    import warnings as warnings_module

    @dataclass
    class _ExplodingAdapter:
        id: str = "boom"
        context_window: int = 8192

        def complete_grammar(
            self,
            messages: Iterable[ChatMessage],
            schema: dict[str, Any],
            *,
            max_tokens: int = 256,
            temperature: float = 0.0,
        ) -> str:
            _ = messages, schema, max_tokens, temperature
            raise RuntimeError("outlines unavailable")

    router = GrammarRouter(adapter=_ExplodingAdapter())
    with warnings_module.catch_warnings(record=True) as w:
        warnings_module.simplefilter("always")
        intent1 = router.classify("hey", [_spec()])
        intent2 = router.classify("also hey", [_spec()])
    assert intent1 is None
    assert intent2 is None
    # Warning fires once, not on every call — avoids spamming a chat
    # loop when outlines is mis-installed.
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1
    assert "outlines unavailable" in str(runtime_warnings[0].message)


def test_grammar_router_uses_low_temperature_by_default() -> None:
    """Grammar decoding doesn't need creativity — keep the temp at 0
    by default so given-same-input-same-output holds."""
    adapter = _GrammarScriptedAdapter(replies=['{"tool": null, "arguments": {}}'])
    router = GrammarRouter(adapter=adapter)
    assert router.temperature == 0.0


def test_grammar_router_degraded_output_still_returns_none() -> None:
    """Even with a grammar-capable adapter, an empty / malformed
    response degrades to None — doesn't raise."""
    adapter = _GrammarScriptedAdapter(replies=[""])  # no JSON at all
    router = GrammarRouter(adapter=adapter)
    intent = router.classify("hey", [_spec()])
    assert intent is None


# ---------- Protocol + structural check ----------


def test_grammar_router_uses_classify_signature_same_as_router_protocol() -> None:
    """Regression guard: GrammarRouter must be usable wherever Router is
    expected. The Router Protocol asserts classify(user_message, tool_specs)."""
    from harness.router.intent import Router

    adapter = _GrammarScriptedAdapter(replies=['{"tool": null, "arguments": {}}'])
    router: Router = GrammarRouter(adapter=adapter)  # mypy check at runtime too
    intent = router.classify("anything", [_spec()])
    assert intent is not None
    assert intent.tool_name is None


# Silence the Sequence import — used implicitly via tool_specs type in tests above.
_ = Sequence
