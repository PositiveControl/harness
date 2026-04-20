"""Router unit tests (harness-6db). Exercise ModelRouter and the parse
helpers via a scripted ModelAdapter. No MLX required."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from harness.model.adapter import ChatMessage
from harness.router import ModelRouter, Router, RouterIntent
from harness.router.model_router import (
    _build_system_prompt,
    _format_spec,
    parse_router_output,
)
from harness.tools.base import ToolSpec


@dataclass
class _ScriptedAdapter:
    """Returns queued raw strings from `complete()`. Records the
    messages + kwargs passed to each call so tests can assert on them."""

    id: str = "scripted"
    context_window: int = 8192
    replies: list[str] = field(default_factory=list)
    calls: list[tuple[list[ChatMessage], int, float]] = field(default_factory=list)

    def complete(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append((list(messages), max_tokens, temperature))
        return self.replies.pop(0) if self.replies else ""


def _spec(name: str = "search_web") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Search the public web via {name}.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "query"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
        tier="read",
    )


# ---------- parse_router_output ----------


def test_parse_clean_json() -> None:
    intent = parse_router_output('{"tool": "search_web", "arguments": {"query": "bbq"}}')
    assert intent == RouterIntent(tool_name="search_web", arguments={"query": "bbq"})


def test_parse_null_tool() -> None:
    intent = parse_router_output('{"tool": null, "arguments": {}}')
    assert intent == RouterIntent(tool_name=None, arguments={})


def test_parse_markdown_fenced_json() -> None:
    raw = '```json\n{"tool": "search_web", "arguments": {"query": "x"}}\n```'
    intent = parse_router_output(raw)
    assert intent is not None
    assert intent.tool_name == "search_web"
    assert intent.arguments == {"query": "x"}


def test_parse_tolerates_leading_prose() -> None:
    raw = 'Sure, here is the JSON:\n{"tool": "search_web", "arguments": {"query": "x"}}'
    intent = parse_router_output(raw)
    assert intent is not None
    assert intent.tool_name == "search_web"


def test_parse_tolerates_trailing_prose() -> None:
    raw = '{"tool": "search_web", "arguments": {"query": "x"}}\n\nHope that helps!'
    intent = parse_router_output(raw)
    assert intent is not None
    assert intent.tool_name == "search_web"


def test_parse_defaults_missing_arguments_to_empty() -> None:
    """Small models routinely drop `arguments` when none are needed.
    Treat that as an empty dict rather than a parse failure; the
    orchestrator validates against the tool schema downstream."""
    intent = parse_router_output('{"tool": "list_dir"}')
    assert intent == RouterIntent(tool_name="list_dir", arguments={})


def test_parse_strips_whitespace_tool_name() -> None:
    intent = parse_router_output('{"tool": "  search_web  ", "arguments": {}}')
    assert intent == RouterIntent(tool_name="search_web", arguments={})


def test_parse_empty_tool_string_becomes_none() -> None:
    intent = parse_router_output('{"tool": "", "arguments": {}}')
    assert intent == RouterIntent(tool_name=None, arguments={})


def test_parse_literal_null_string_becomes_none() -> None:
    """harness-0bu: Hermes-3-3B sometimes emits {"tool": "null", ...}
    (literal string) instead of JSON null. Treat case-folded 'null' and
    'none' as equivalent to Python None so these get scored correctly."""
    for variant in ('"null"', '"NULL"', '"None"', '"  null  "'):
        intent = parse_router_output(f'{{"tool": {variant}, "arguments": {{}}}}')
        assert intent == RouterIntent(tool_name=None, arguments={}), variant


def test_parse_returns_none_on_invalid_json() -> None:
    assert parse_router_output("just some prose, no json at all") is None


def test_parse_returns_none_on_malformed_json() -> None:
    assert parse_router_output('{"tool": "search_web", "arguments": {') is None


def test_parse_returns_none_on_missing_tool_key() -> None:
    assert parse_router_output('{"arguments": {"query": "x"}}') is None


def test_parse_returns_none_on_wrong_tool_type() -> None:
    assert parse_router_output('{"tool": 42, "arguments": {}}') is None


def test_parse_returns_none_on_non_dict_arguments() -> None:
    assert parse_router_output('{"tool": "search_web", "arguments": "not a dict"}') is None


def test_parse_returns_none_on_top_level_list() -> None:
    assert parse_router_output('[{"tool": "search_web"}]') is None


# ---------- _format_spec / _build_system_prompt ----------


def test_format_spec_marks_required_and_optional() -> None:
    line = _format_spec(_spec())
    assert "search_web(" in line
    assert "query: string" in line
    assert "max_results: integer?" in line  # optional → trailing ?
    assert "Search the public web" in line


def test_format_spec_renders_enum_inline() -> None:
    """harness-nom: small models ignore json-schema enum constraints
    they don't see, and the router's `name(args)` line is the only
    schema view they get. Render `enum[a|b|c]` inline so the router
    has the constraint in plain sight, not just `string`."""
    spec = ToolSpec(
        name="t",
        description="d",
        parameters={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": ["professional", "personal"]},
            },
        },
        tier="read",
    )
    line = _format_spec(spec)
    assert "scope: enum[professional|personal]?" in line
    assert "scope: string" not in line


def test_format_spec_emits_per_arg_hints_for_long_descriptions() -> None:
    """Per-arg `description` strings on the JSON schema were silently
    dropped pre-harness-nom — the router never saw them, so carefully
    written 'OMIT for default' guidance went nowhere. Pin the
    behavior: a non-trivial description (>30 chars) shows up as a
    `· name: hint` indented bullet under the spec line."""
    spec = ToolSpec(
        name="t",
        description="d",
        parameters={
            "type": "object",
            "properties": {
                "long_arg": {
                    "type": "string",
                    "description": "OMIT this when the user phrasing is vague.",
                },
                "short_arg": {"type": "string", "description": "tiny"},
            },
        },
        tier="read",
    )
    line = _format_spec(spec)
    assert "  · long_arg: OMIT this when the user phrasing is vague." in line
    # Tiny descriptions don't earn a hint — keeps the prompt tight.
    assert "short_arg:" not in line.split("\n", 1)[1]


def test_format_spec_collapses_multiline_hints_to_first_line() -> None:
    """Hints stay one line each so the router prompt doesn't explode.
    A description spanning 5 lines should surface only the first."""
    spec = ToolSpec(
        name="t",
        description="d",
        parameters={
            "type": "object",
            "properties": {
                "x": {
                    "type": "string",
                    "description": (
                        "FIRST LINE GUIDANCE that fits in one row.\n"
                        "Second line — important context but not for the router.\n"
                        "Third line — even more detail."
                    ),
                },
            },
        },
        tier="read",
    )
    line = _format_spec(spec)
    assert "  · x: FIRST LINE GUIDANCE that fits in one row." in line
    assert "Second line" not in line
    assert "Third line" not in line


def test_build_system_prompt_lists_tools() -> None:
    prompt = _build_system_prompt([_spec("search_web"), _spec("read_file")])
    assert "search_web(" in prompt
    assert "read_file(" in prompt
    assert "STRICT JSON" in prompt


def test_build_system_prompt_handles_empty_specs() -> None:
    prompt = _build_system_prompt([])
    assert "no tools available" in prompt


def test_build_system_prompt_has_tool_vs_null_rubric() -> None:
    """harness-ve7: the prompt must carry an explicit 'when to use a
    tool' vs 'when to return null' rubric so the 1.5B doesn't
    over-route creative / explanatory prompts to search_web."""
    prompt = _build_system_prompt([_spec("search_web")])
    assert "A tool is needed ONLY when" in prompt
    assert "Return null (no tool)" in prompt
    # Specific failure modes we're defending against:
    assert "Creative writing" in prompt  # haiku / joke over-routing
    assert "Explanations" in prompt  # merkle-tree over-routing
    assert "Greetings" in prompt


def test_build_system_prompt_has_tool_overlap_notes() -> None:
    """The router was collapsing read_file / list_dir / glob all to
    glob. The disambiguation notes in the prompt distinguish
    CONTENTS vs LISTING vs FINDING vs SEARCHING."""
    prompt = _build_system_prompt([_spec("search_web")])
    assert "CONTENTS" in prompt
    assert "LISTING" in prompt or "IN a directory" in prompt
    assert "FIND" in prompt
    assert "SEARCH INSIDE" in prompt


def test_build_system_prompt_covers_each_failure_mode_with_an_example() -> None:
    """Each of the three failure modes from the eval has a dedicated
    few-shot now. Small models imitate examples more reliably than they
    follow instructions — these lock in the desired behavior."""
    prompt = _build_system_prompt([_spec("search_web")])
    # Read-tool disambiguation
    assert "read_file" in prompt
    assert "list_dir" in prompt
    # Memory intent
    assert "search_memory" in prompt
    assert "recall" in prompt
    # Creative / explanatory null cases
    assert "haiku" in prompt
    assert "hash table" in prompt


def test_build_system_prompt_has_ops_browsing_few_shot() -> None:
    """harness-nom: small models invent args for ab ops tools when
    they have no positive example to imitate. The prompt must carry
    at least one 'browsing query → list with NO args' few-shot so the
    model has a template, plus a disambiguation paragraph that says
    'OMIT defaults; only fill args when the user names them'."""
    prompt = _build_system_prompt([_spec("search_web")])
    # Browsing example with the empty-args output — the load-bearing
    # template that breaks the in_progress / scope='ab' default-
    # invention pattern.
    assert "what tasks do we have coming up?" in prompt
    assert '"tool": "list", "arguments": {}' in prompt
    # Disambiguation paragraph addresses ops tools by name and the
    # specific anti-patterns we hit (in_progress for vague time, scope
    # for identity terms). Substring assertions tolerate prompt
    # rewraps — load-bearing tokens, not full phrases.
    assert "ab ops tools" in prompt
    assert "NEVER pass status='in_progress'" in prompt
    assert "in_progress signals" in prompt  # the "vague time != in_progress" rule


# ---------- ModelRouter ----------


def test_model_router_classifies_happy_path() -> None:
    adapter = _ScriptedAdapter(
        replies=['{"tool": "search_web", "arguments": {"query": "bbq near me"}}']
    )
    router: Router = ModelRouter(adapter=adapter)
    intent = router.classify("search the web for bbq near me", [_spec("search_web")])
    assert intent == RouterIntent(tool_name="search_web", arguments={"query": "bbq near me"})


def test_model_router_passes_tool_specs_in_system_prompt() -> None:
    adapter = _ScriptedAdapter(replies=['{"tool": null, "arguments": {}}'])
    router = ModelRouter(adapter=adapter)
    router.classify("hey", [_spec("search_web"), _spec("read_file")])
    system = adapter.calls[0][0][0]
    assert system.role == "system"
    assert "search_web" in system.content
    assert "read_file" in system.content


def test_model_router_uses_zero_temperature() -> None:
    adapter = _ScriptedAdapter(replies=['{"tool": null, "arguments": {}}'])
    ModelRouter(adapter=adapter).classify("hey", [_spec()])
    _, _, temperature = adapter.calls[0]
    assert temperature == 0.0


def test_model_router_returns_none_on_unparseable_output() -> None:
    adapter = _ScriptedAdapter(replies=["I don't know how to route this."])
    intent = ModelRouter(adapter=adapter).classify("search the web", [_spec()])
    assert intent is None


def test_model_router_returns_none_when_adapter_raises() -> None:
    """Advisory contract: classify() never raises. Adapter failures
    (OOM / transport / anything) degrade to 'could not decide'."""

    @dataclass
    class _ExplodingAdapter:
        id: str = "boom"
        context_window: int = 8192

        def complete(
            self,
            messages: Iterable[ChatMessage],
            *,
            max_tokens: int = 512,
            temperature: float = 0.7,
        ) -> str:
            raise RuntimeError("model went away")

    intent = ModelRouter(adapter=_ExplodingAdapter()).classify("hey", [_spec()])
    assert intent is None


def test_model_router_handles_empty_tool_specs() -> None:
    """With no tools available the router should still function — the
    small model is told to always return null."""
    adapter = _ScriptedAdapter(replies=['{"tool": null, "arguments": {}}'])
    intent = ModelRouter(adapter=adapter).classify("hey", [])
    assert intent == RouterIntent(tool_name=None, arguments={})
