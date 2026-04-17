"""Tests for the router eval (harness-zgu). Exercises fixture loading,
scoring logic, and the canonical fixture shape without loading MLX."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from harness.evals.router import (
    RouterEvalCase,
    default_fixture_path,
    load_fixture,
    run_router_eval,
)
from harness.router.intent import RouterIntent
from harness.tools.base import ToolSpec


@dataclass
class _ScriptedRouter:
    """Returns a queued RouterIntent for each classify() call, in
    insertion order. When the queue runs dry, returns None."""

    intents: list[RouterIntent | None]
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def classify(self, user_message: str, tool_specs: Sequence[ToolSpec]) -> RouterIntent | None:
        self.calls.append((user_message, tuple(s.name for s in tool_specs)))
        return self.intents.pop(0) if self.intents else None


def _spec(name: str, required: tuple[str, ...] = ("query",)) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={
            "type": "object",
            "properties": {k: {"type": "string"} for k in required},
            "required": list(required),
        },
        tier="read",
    )


# ---------- load_fixture ----------


def test_load_fixture_parses_basic_entries(tmp_path: Path) -> None:
    f = tmp_path / "eval.yaml"
    f.write_text(
        "- prompt: search for bbq\n"
        "  expected_tool: search_web\n"
        "  expected_args: [query]\n"
        "- prompt: hey\n"
        "  expected_tool: null\n"
    )
    rows = load_fixture(f)
    assert len(rows) == 2
    assert rows[0] == ("search for bbq", "search_web", ("query",))
    assert rows[1] == ("hey", None, ())


def test_load_fixture_rejects_non_list(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("prompt: x\n")
    with pytest.raises(ValueError, match="not a YAML list"):
        load_fixture(f)


def test_load_fixture_rejects_missing_prompt(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("- expected_tool: search_web\n")
    with pytest.raises(ValueError, match="missing or empty 'prompt'"):
        load_fixture(f)


def test_load_fixture_rejects_missing_expected_tool(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("- prompt: hi\n")
    with pytest.raises(ValueError, match="missing 'expected_tool'"):
        load_fixture(f)


def test_load_fixture_allows_omitted_expected_args(tmp_path: Path) -> None:
    f = tmp_path / "ok.yaml"
    f.write_text("- prompt: list tools\n  expected_tool: list_dir\n")
    rows = load_fixture(f)
    assert rows[0][2] == ()


# ---------- run_router_eval ----------


def test_eval_scores_exact_tool_and_args_match() -> None:
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="search_web", arguments={"query": "bbq"})]
    )
    fixture = (("search for bbq", "search_web", ("query",)),)
    result = run_router_eval(router, [_spec("search_web")], fixture)
    assert result.accuracy == 1.0
    assert result.cases[0].passed


def test_eval_marks_wrong_tool_as_fail() -> None:
    router = _ScriptedRouter(intents=[RouterIntent(tool_name="read_file", arguments={"path": "x"})])
    fixture = (("search for bbq", "search_web", ("query",)),)
    result = run_router_eval(router, [_spec("search_web"), _spec("read_file", ("path",))], fixture)
    assert result.accuracy == 0.0
    case = result.cases[0]
    assert not case.tool_correct
    assert case.actual_tool == "read_file"


def test_eval_marks_missing_arg_as_args_fail() -> None:
    """Tool right, required arg missing → tool_correct but not
    args_correct; overall fails."""
    router = _ScriptedRouter(intents=[RouterIntent(tool_name="search_web", arguments={})])
    fixture = (("search for bbq", "search_web", ("query",)),)
    result = run_router_eval(router, [_spec("search_web")], fixture)
    case = result.cases[0]
    assert case.tool_correct
    assert not case.args_correct
    assert not case.passed
    assert result.tool_accuracy == 1.0
    assert result.accuracy == 0.0


def test_eval_accepts_null_match() -> None:
    """Expected null + actual null (either from {tool: null} or None
    return) scores as pass."""
    router = _ScriptedRouter(
        intents=[
            RouterIntent(tool_name=None, arguments={}),
            None,  # router gave up → also treated as null-match
        ]
    )
    fixture = (("hey", None, ()), ("good morning", None, ()))
    result = run_router_eval(router, [_spec("search_web")], fixture)
    assert result.accuracy == 1.0


def test_eval_partial_correct_tool_wrong_arg_reporting() -> None:
    """Aggregate metrics separate tool-selection from args-completeness."""
    router = _ScriptedRouter(
        intents=[
            RouterIntent(tool_name="search_web", arguments={"query": "x"}),
            RouterIntent(tool_name="search_web", arguments={}),  # right tool, no query
            RouterIntent(tool_name=None, arguments={}),
        ]
    )
    fixture = (
        ("search one", "search_web", ("query",)),
        ("search two", "search_web", ("query",)),
        ("hey", None, ()),
    )
    result = run_router_eval(router, [_spec("search_web")], fixture)
    assert result.tool_accuracy == 1.0  # all three tool-correct
    assert result.accuracy == pytest.approx(2 / 3)  # case 2 failed args
    failures = result.failures()
    assert len(failures) == 1
    assert failures[0].prompt == "search two"


def test_eval_case_exposes_actual_args_for_debugging() -> None:
    router = _ScriptedRouter(intents=[RouterIntent(tool_name="search_web", arguments={"q": "bbq"})])
    fixture = (("search", "search_web", ("query",)),)
    result = run_router_eval(router, [_spec("search_web")], fixture)
    case: RouterEvalCase = result.cases[0]
    # Actual args preserved so the CLI can show 'model used "q" instead of "query"'.
    assert case.actual_args == {"q": "bbq"}


# ---------- canonical fixture ----------


def test_canonical_fixture_loads_and_covers_tool_mix() -> None:
    """The shipped character/airton/router_eval.yaml parses cleanly and
    mixes positive + null cases. Guards against a fixture edit breaking
    the loader or accidentally making everything null."""
    path = default_fixture_path(Path(__file__).parent.parent / "character" / "airton")
    rows = load_fixture(path)
    assert len(rows) >= 15
    tools_used = {r[1] for r in rows}
    # Sanity: at least one search_web, at least one null, at least one read-tool.
    assert "search_web" in tools_used
    assert None in tools_used
    assert "read_file" in tools_used
    # Null cases should be a meaningful minority so the eval catches
    # over-routing regressions.
    null_count = sum(1 for r in rows if r[1] is None)
    assert null_count >= 5
