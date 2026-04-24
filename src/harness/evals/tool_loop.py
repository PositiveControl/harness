"""Tool-loop failure-corpus eval (harness-cfm7).

Replays scripted adapter reply sequences through `run_tool_loop` and
scores each scenario against contains / not_contains assertions. Each
scenario is labelled with the orchestrator catcher it stresses so the
attribution harness can disable one catcher at a time and measure:

- uniquely saved: scenarios that pass only while this catcher is on
- redundant with: scenarios where another catcher still caught the
  failure when this one was off

Pure — no real model, no filesystem mutation. Scripted adapter
consumes queued `ModelReply`-shaped entries in order; a tiny mock
registry covers the `duplicate_call` + `paired_meta_confirm_strip`
scenarios that need a real tool execution.

Mirrors the shape of `harness.evals.router` / `harness.evals.session_resume`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from harness.model.adapter import ChatMessage, Role
from harness.orchestrator import run_tool_loop
from harness.orchestrator.tool_loop import _CATCHER_NAMES, _DISABLED_CATCHERS
from harness.tools.base import (
    ModelReply,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

# ---------- catcher toggle ----------


CATCHER_NAMES: tuple[str, ...] = _CATCHER_NAMES


@contextmanager
def disable_catchers(names: Iterable[str]) -> Iterator[None]:
    """Temporarily disable the named orchestrator catchers. Unknown
    names raise; silent typos would make attribution results
    meaningless."""
    wanted = tuple(names)
    for n in wanted:
        if n not in CATCHER_NAMES:
            raise ValueError(f"unknown catcher {n!r}; known: {sorted(CATCHER_NAMES)}")
    added: list[str] = []
    for n in wanted:
        if n not in _DISABLED_CATCHERS:
            _DISABLED_CATCHERS.add(n)
            added.append(n)
    try:
        yield
    finally:
        for n in added:
            _DISABLED_CATCHERS.discard(n)


# ---------- mock registry ----------


@dataclass(frozen=True)
class _MockToolSpec:
    """Fixture-declared tool. `output` is the string the tool returns
    on every call; `success` lets scenarios exercise all-errored-turn
    paths (harness-a0y) without needing a real failure shape."""

    name: str
    output: str = "ok"
    success: bool = True
    tier: str = "read"


class _MockTool:
    def __init__(self, mock: _MockToolSpec) -> None:
        self._mock = mock
        self.spec = ToolSpec(
            name=mock.name,
            description=f"mock {mock.name}",
            parameters={"type": "object", "properties": {}, "required": []},
            tier=mock.tier,
        )

    def call(self, **_: Any) -> str:
        # ToolRegistry.call wraps the return in a ToolResult(success=True).
        # For mock-failure scenarios we need success=False, so we
        # register via a custom dispatch path below.
        return self._mock.output


def _build_registry(tool_entries: list[dict[str, Any]]) -> ToolRegistry:
    registry = ToolRegistry()
    for raw in tool_entries:
        mock = _MockToolSpec(
            name=str(raw["name"]),
            output=str(raw.get("output", "ok")),
            success=bool(raw.get("success", True)),
            tier=str(raw.get("tier", "read")),
        )
        tool = _MockTool(mock)
        registry.register(tool)
        if not mock.success:
            # Override registry dispatch for this tool so its ToolResult
            # reports success=False — ToolRegistry's default `call`
            # path assumes any non-raising tool succeeded.
            _patch_registry_for_failure(registry, mock)
    return registry


def _patch_registry_for_failure(registry: ToolRegistry, mock: _MockToolSpec) -> None:
    """Rewrite registry.call to emit a ToolResult(success=False) for
    `mock.name`. Other tools fall through to the real dispatch.

    Kept out-of-band so we don't need a bespoke Tool subclass hierarchy
    for eval-only failure injection."""
    original = registry.call

    def dispatch(name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == mock.name:
            return ToolResult(
                tool_name=name,
                output=mock.output,
                success=False,
                error="mock_failure",
            )
        return original(name, arguments)

    registry.call = dispatch  # type: ignore[method-assign]


# ---------- scripted adapter ----------


@dataclass
class _ScriptedAdapter:
    """Returns pre-queued ModelReply objects in order. When the queue
    runs dry, returns a benign empty text reply so the loop terminates
    rather than IndexError'ing — scenarios that over-queue are a fixture
    authoring smell, not a crash."""

    replies: list[ModelReply]

    def complete_with_tools(
        self,
        messages: Iterable[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ModelReply:
        if self.replies:
            return self.replies.pop(0)
        return ModelReply(content="")


def _parse_reply(raw: dict[str, Any]) -> ModelReply:
    calls_raw = raw.get("tool_calls") or []
    calls = tuple(
        ToolCall(name=str(c["name"]), arguments=dict(c.get("arguments", {}))) for c in calls_raw
    )
    return ModelReply(
        content=str(raw.get("content", "")),
        tool_calls=calls,
        was_truncated=bool(raw.get("was_truncated", False)),
        had_unparseable_call=bool(raw.get("had_unparseable_call", False)),
    )


# ---------- scoring ----------


@dataclass(frozen=True)
class ToolLoopCase:
    id: str
    label: str
    description: str
    final_content: str
    rounds: int
    missing_contains: tuple[str, ...]
    unexpected_contains: tuple[str, ...]
    expected_fallback: bool
    fallback_triggered: bool
    missing_events: tuple[str, ...]
    unexpected_events: tuple[str, ...]
    missing_message_substrings: tuple[str, ...]
    unexpected_message_substrings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        if self.missing_contains or self.unexpected_contains:
            return False
        if self.missing_events or self.unexpected_events:
            return False
        if self.missing_message_substrings or self.unexpected_message_substrings:
            return False
        return self.expected_fallback == self.fallback_triggered


@dataclass(frozen=True)
class ToolLoopEvalResult:
    cases: tuple[ToolLoopCase, ...]

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    def failures(self) -> tuple[ToolLoopCase, ...]:
        return tuple(c for c in self.cases if not c.passed)

    def pass_set(self) -> frozenset[str]:
        return frozenset(c.id for c in self.cases if c.passed)


_FALLBACK_SENTINEL = "my attempts to call one didn't land cleanly"


def _run_scenario(scenario: dict[str, Any]) -> ToolLoopCase:
    scripted = [_parse_reply(r) for r in scenario.get("scripted_replies", [])]
    adapter = _ScriptedAdapter(replies=scripted)
    registry = _build_registry(scenario.get("registry", []))
    messages = [
        ChatMessage(role=cast(Role, str(m["role"])), content=str(m.get("content", "")))
        for m in scenario.get("messages", [{"role": "user", "content": "hi"}])
    ]
    max_rounds = int(scenario.get("max_rounds", 8))
    # `memory_block_attached` is the flag the ungrounded_citation
    # finalize hook consumes. Fixture rows that want to simulate "a
    # retrieval memory block landed in the prompt" set it to True —
    # otherwise the default (False) models the cold / out-of-scope
    # retrieval path where the catcher is armed.
    memory_block_attached = bool(scenario.get("memory_block_attached", False))
    result = run_tool_loop(
        adapter,
        messages,
        registry,
        max_rounds=max_rounds,
        confirm=lambda _call: True,
        memory_block_attached=memory_block_attached,
    )
    contains = tuple(str(s) for s in scenario.get("expected_contains", []))
    not_contains = tuple(str(s) for s in scenario.get("expected_not_contains", []))
    missing = tuple(s for s in contains if s not in result.content)
    unexpected = tuple(s for s in not_contains if s in result.content)
    expected_fallback = bool(scenario.get("expected_fallback", False))
    fallback_triggered = _FALLBACK_SENTINEL in result.content

    event_kinds = {e.kind for e in result.events}
    want_events = tuple(str(s) for s in scenario.get("expected_events_contain", []))
    block_events = tuple(str(s) for s in scenario.get("expected_events_not_contain", []))
    missing_events = tuple(k for k in want_events if k not in event_kinds)
    unexpected_events = tuple(k for k in block_events if k in event_kinds)

    all_content = "\n".join(m.content or "" for m in result.messages)
    want_msgs = tuple(str(s) for s in scenario.get("expected_messages_contain", []))
    block_msgs = tuple(str(s) for s in scenario.get("expected_messages_not_contain", []))
    missing_msgs = tuple(s for s in want_msgs if s not in all_content)
    unexpected_msgs = tuple(s for s in block_msgs if s in all_content)

    return ToolLoopCase(
        id=str(scenario["id"]),
        label=str(scenario.get("label", "")),
        description=str(scenario.get("description", "")).strip(),
        final_content=result.content,
        rounds=result.rounds,
        missing_contains=missing,
        unexpected_contains=unexpected,
        expected_fallback=expected_fallback,
        fallback_triggered=fallback_triggered,
        missing_events=missing_events,
        unexpected_events=unexpected_events,
        missing_message_substrings=missing_msgs,
        unexpected_message_substrings=unexpected_msgs,
    )


def load_fixture(path: Path) -> tuple[dict[str, Any], ...]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"tool-loop eval fixture {path} is not a YAML list")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{idx}] is not a mapping")
        if "id" not in entry:
            raise ValueError(f"{path}[{idx}] missing 'id'")
        sid = str(entry["id"])
        if sid in seen:
            raise ValueError(f"{path}[{idx}] duplicate id {sid!r}")
        seen.add(sid)
        label = entry.get("label")
        if label is not None and label not in CATCHER_NAMES and label != "control":
            raise ValueError(
                f"{path}[{idx}] label {label!r} is not a known catcher "
                f"(known: {sorted(CATCHER_NAMES)} or 'control')"
            )
        out.append(entry)
    return tuple(out)


def run_tool_loop_eval(
    fixtures: tuple[dict[str, Any], ...],
    *,
    disabled: Iterable[str] = (),
) -> ToolLoopEvalResult:
    """Run each scenario, optionally with the listed catchers disabled.
    Returns a ToolLoopEvalResult; caller decides pass/fail reporting."""
    with disable_catchers(disabled):
        cases = tuple(_run_scenario(fx) for fx in fixtures)
    return ToolLoopEvalResult(cases=cases)


# ---------- attribution ----------


@dataclass(frozen=True)
class CatcherAttribution:
    """What happens when a single catcher is disabled.

    `unique_saves`: scenario ids that pass when all catchers are on but
    fail when only this one is off. Load-bearing for those cases.

    `also_breaks`: scenario ids that fail with this catcher off AND
    with at least one other catcher off. Redundant coverage with
    another catcher.

    `no_effect`: True if disabling this catcher changes no scenario's
    pass/fail. Either the catcher is dead code or no scenario exercises
    it — inspect the fixture before concluding the former."""

    catcher: str
    unique_saves: tuple[str, ...]
    also_breaks: tuple[str, ...]

    @property
    def no_effect(self) -> bool:
        return not self.unique_saves and not self.also_breaks


@dataclass(frozen=True)
class AttributionResult:
    baseline: ToolLoopEvalResult
    per_catcher: dict[str, ToolLoopEvalResult]
    attributions: tuple[CatcherAttribution, ...]


def run_attribution(fixtures: tuple[dict[str, Any], ...]) -> AttributionResult:
    """Run the fixture once with all catchers on (baseline), then once
    per catcher with just that catcher disabled. Scores which scenarios
    each catcher uniquely saves vs shares with other catchers.

    Complexity is O(len(fixtures) * len(CATCHER_NAMES)). Each scripted
    run is sub-millisecond, so this stays cheap — a handful of ms for
    ~20 scenarios and 11 catchers."""
    baseline = run_tool_loop_eval(fixtures)
    baseline_pass = baseline.pass_set()

    per_catcher: dict[str, ToolLoopEvalResult] = {}
    for name in CATCHER_NAMES:
        per_catcher[name] = run_tool_loop_eval(fixtures, disabled=[name])

    # unique_saves[c] = baseline∩fail_when_c_off across all other catchers.
    attributions: list[CatcherAttribution] = []
    for name in CATCHER_NAMES:
        broken = baseline_pass - per_catcher[name].pass_set()
        also_broken_elsewhere: set[str] = set()
        unique: set[str] = set()
        for sid in broken:
            shared_with = [
                other
                for other in CATCHER_NAMES
                if other != name and sid not in per_catcher[other].pass_set()
            ]
            if shared_with:
                also_broken_elsewhere.add(sid)
            else:
                unique.add(sid)
        attributions.append(
            CatcherAttribution(
                catcher=name,
                unique_saves=tuple(sorted(unique)),
                also_breaks=tuple(sorted(also_broken_elsewhere)),
            )
        )

    return AttributionResult(
        baseline=baseline,
        per_catcher=per_catcher,
        attributions=tuple(attributions),
    )


def default_fixture_path(character_path: Path) -> Path:
    return character_path / "tool_loop_eval.yaml"


__all__ = [
    "CATCHER_NAMES",
    "AttributionResult",
    "CatcherAttribution",
    "ToolLoopCase",
    "ToolLoopEvalResult",
    "default_fixture_path",
    "disable_catchers",
    "load_fixture",
    "run_attribution",
    "run_tool_loop_eval",
]
