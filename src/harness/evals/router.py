"""Router eval — replays a fixture of `(prompt, expected_tool)` cases
through a Router and scores tool-selection accuracy plus a soft args
check.

Two things this is not:
- A latency benchmark. See scripts/bench_router.py (harness-64x) for
  wall-clock / RAM measurements.
- A test of the tool tier/schema validation in the orchestrator. Those
  live in tests/test_tool_loop.py under the router-integration block.

This eval only asks: given the fixture prompt + available tool specs,
does the router pick the right tool (or confidently null), and does it
include the required argument names?"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from harness.router.intent import Router
from harness.tools.base import ToolSpec


@dataclass(frozen=True)
class RouterEvalCase:
    prompt: str
    expected_tool: str | None
    expected_args: tuple[str, ...]
    actual_tool: str | None
    actual_args: dict[str, Any]
    tool_correct: bool
    args_correct: bool  # all expected_args appear in actual_args

    @property
    def passed(self) -> bool:
        return self.tool_correct and self.args_correct


@dataclass(frozen=True)
class RouterEvalResult:
    cases: tuple[RouterEvalCase, ...]

    @property
    def accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    @property
    def tool_accuracy(self) -> float:
        """Correct-tool rate ignoring args. Useful for spotting cases
        where the model picks the right tool but drops required args."""
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.tool_correct) / len(self.cases)

    def failures(self) -> tuple[RouterEvalCase, ...]:
        return tuple(c for c in self.cases if not c.passed)


def load_fixture(path: Path) -> tuple[tuple[str, str | None, tuple[str, ...]], ...]:
    """Parse a router-eval YAML file into (prompt, expected_tool, expected_args) rows.
    Public so the CLI can surface a row count before running."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"router eval fixture {path} is not a YAML list")
    rows: list[tuple[str, str | None, tuple[str, ...]]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}[{idx}] is not a mapping")
        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}[{idx}] missing or empty 'prompt'")
        expected_tool_raw = entry.get("expected_tool", ...)
        if expected_tool_raw is ...:
            raise ValueError(f"{path}[{idx}] missing 'expected_tool' (use null for no-tool)")
        if expected_tool_raw is not None and not isinstance(expected_tool_raw, str):
            raise ValueError(f"{path}[{idx}] 'expected_tool' must be a string or null")
        expected_tool: str | None = expected_tool_raw
        expected_args_raw = entry.get("expected_args", []) or []
        if not isinstance(expected_args_raw, list):
            raise ValueError(f"{path}[{idx}] 'expected_args' must be a list")
        expected_args = tuple(str(a) for a in expected_args_raw)
        rows.append((prompt.strip(), expected_tool, expected_args))
    return tuple(rows)


def run_router_eval(
    router: Router,
    tool_specs: Sequence[ToolSpec],
    fixture: Sequence[tuple[str, str | None, tuple[str, ...]]],
) -> RouterEvalResult:
    """Replay `fixture` through `router` and score each case.

    Pure beyond the router.classify() call — no filesystem, no network.
    CLI / bench wrap this; tests exercise it directly with a scripted
    Router."""
    cases: list[RouterEvalCase] = []
    for prompt, expected_tool, expected_args in fixture:
        intent = router.classify(prompt, list(tool_specs))
        actual_tool = intent.tool_name if intent is not None else None
        actual_args = dict(intent.arguments) if intent is not None else {}
        tool_correct = actual_tool == expected_tool
        args_correct = (
            all(a in actual_args for a in expected_args) if tool_correct and expected_tool else True
        )
        cases.append(
            RouterEvalCase(
                prompt=prompt,
                expected_tool=expected_tool,
                expected_args=expected_args,
                actual_tool=actual_tool,
                actual_args=actual_args,
                tool_correct=tool_correct,
                args_correct=args_correct,
            )
        )
    return RouterEvalResult(cases=tuple(cases))


def default_fixture_path(character_path: Path) -> Path:
    """Conventional location under character/<name>/ for the fixture.
    Kept as a function so tests and the CLI agree on the path."""
    return character_path / "router_eval.yaml"
