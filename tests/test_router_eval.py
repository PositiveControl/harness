"""Tests for the router eval (harness-zgu). Exercises fixture loading,
scoring logic, and the canonical fixture shape without loading MLX."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from harness.evals.router import (
    FixtureRow,
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
    assert rows[0] == ("search for bbq", "search_web", ("query",), (), None)
    assert rows[1] == ("hey", None, (), (), None)


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
    fixture = (("search for bbq", "search_web", ("query",), (), None),)
    result = run_router_eval(router, [_spec("search_web")], fixture)
    assert result.accuracy == 1.0
    assert result.cases[0].passed


def test_eval_marks_wrong_tool_as_fail() -> None:
    router = _ScriptedRouter(intents=[RouterIntent(tool_name="read_file", arguments={"path": "x"})])
    fixture = (("search for bbq", "search_web", ("query",), (), None),)
    result = run_router_eval(router, [_spec("search_web"), _spec("read_file", ("path",))], fixture)
    assert result.accuracy == 0.0
    case = result.cases[0]
    assert not case.tool_correct
    assert case.actual_tool == "read_file"


def test_eval_marks_missing_arg_as_args_fail() -> None:
    """Tool right, required arg missing → tool_correct but not
    args_correct; overall fails."""
    router = _ScriptedRouter(intents=[RouterIntent(tool_name="search_web", arguments={})])
    fixture = (("search for bbq", "search_web", ("query",), (), None),)
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
    fixture = (("hey", None, (), (), None), ("good morning", None, (), (), None))
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
        ("search one", "search_web", ("query",), (), None),
        ("search two", "search_web", ("query",), (), None),
        ("hey", None, (), (), None),
    )
    result = run_router_eval(router, [_spec("search_web")], fixture)
    assert result.tool_accuracy == 1.0  # all three tool-correct
    assert result.accuracy == pytest.approx(2 / 3)  # case 2 failed args
    failures = result.failures()
    assert len(failures) == 1
    assert failures[0].prompt == "search two"


def test_load_fixture_parses_expected_arg_values(tmp_path: Path) -> None:
    """`expected_arg_values` is an optional mapping {arg: required
    substring} parsed into a tuple of (arg, substring) pairs."""
    f = tmp_path / "ok.yaml"
    f.write_text(
        "- prompt: go to stackoverflow\n"
        "  expected_tool: fetch_url\n"
        "  expected_args: [url]\n"
        "  expected_arg_values:\n"
        "    url: stackoverflow\n"
    )
    rows = load_fixture(f)
    assert rows[0] == (
        "go to stackoverflow",
        "fetch_url",
        ("url",),
        (("url", "stackoverflow"),),
        None,
    )


def test_load_fixture_rejects_non_mapping_arg_values(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text(
        "- prompt: x\n  expected_tool: fetch_url\n  expected_arg_values: [not, a, mapping]\n"
    )
    with pytest.raises(ValueError, match="expected_arg_values"):
        load_fixture(f)


def test_eval_flags_value_mismatch_as_fail() -> None:
    """harness-w1z: router picks the right tool + right arg NAME but
    the arg VALUE is a leaked domain ('dailydrop.fm') rather than the
    entity the user named ('stackoverflow'). The expected_arg_values
    substring assertion catches it."""
    router = _ScriptedRouter(
        intents=[
            RouterIntent(
                tool_name="fetch_url",
                arguments={"url": "https://dailydrop.fm"},  # WRONG — leaked from router few-shot
            )
        ]
    )
    fixture = (
        (
            "go to stackoverflow and summarize the first question",
            "fetch_url",
            ("url",),
            (("url", "stackoverflow"),),
            None,
        ),
    )
    result = run_router_eval(router, [_spec("fetch_url", ("url",))], fixture)
    case = result.cases[0]
    assert case.tool_correct
    assert case.args_correct
    assert not case.arg_values_correct
    assert not case.passed


def test_eval_accepts_matching_arg_value_substring() -> None:
    """Counter-case: when the arg value contains the required
    substring (case-insensitive), arg_values_correct is True."""
    router = _ScriptedRouter(
        intents=[
            RouterIntent(
                tool_name="fetch_url", arguments={"url": "https://StackOverflow.com/questions"}
            )
        ]
    )
    fixture = (
        (
            "go to stackoverflow",
            "fetch_url",
            ("url",),
            (("url", "stackoverflow"),),
            None,
        ),
    )
    result = run_router_eval(router, [_spec("fetch_url", ("url",))], fixture)
    assert result.cases[0].passed


def test_eval_skips_value_check_when_tool_wrong() -> None:
    """When the router picked the wrong tool, arg_values_correct is
    True (the failure is already captured by tool_correct=False, and
    flagging both would double-count)."""
    router = _ScriptedRouter(intents=[RouterIntent(tool_name="read_file", arguments={"path": "x"})])
    fixture = (
        (
            "go to stackoverflow",
            "fetch_url",
            ("url",),
            (("url", "stackoverflow"),),
            None,
        ),
    )
    result = run_router_eval(
        router, [_spec("fetch_url", ("url",)), _spec("read_file", ("path",))], fixture
    )
    case = result.cases[0]
    assert not case.tool_correct
    assert case.arg_values_correct


def test_eval_case_exposes_actual_args_for_debugging() -> None:
    router = _ScriptedRouter(intents=[RouterIntent(tool_name="search_web", arguments={"q": "bbq"})])
    fixture = (("search", "search_web", ("query",), (), None),)
    result = run_router_eval(router, [_spec("search_web")], fixture)
    case: RouterEvalCase = result.cases[0]
    # Actual args preserved so the CLI can show 'model used "q" instead of "query"'.
    assert case.actual_args == {"q": "bbq"}


# ---------- scope scoring (harness-8dop) ----------


def test_eval_scope_correct_when_match() -> None:
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="search_memory", arguments={"query": "x"}, scope="in")]
    )
    fixture: tuple[FixtureRow, ...] = (("rule lookup", "search_memory", ("query",), (), "in"),)
    result = run_router_eval(router, [_spec("search_memory")], fixture)
    assert result.cases[0].scope_correct
    assert result.cases[0].passed
    assert result.scope_accuracy == 1.0
    assert result.scope_case_count == 1


def test_eval_scope_incorrect_when_mismatch_fails_case() -> None:
    """Scope mismatch alone fails the case even when tool + args are right."""
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="search_memory", arguments={"query": "x"}, scope="in")]
    )
    fixture: tuple[FixtureRow, ...] = (("pilot question", "search_memory", ("query",), (), "out"),)
    result = run_router_eval(router, [_spec("search_memory")], fixture)
    case = result.cases[0]
    assert case.tool_correct
    assert not case.scope_correct
    assert not case.passed


def test_eval_scope_skipped_when_no_expected_scope() -> None:
    """Rows without `expected_scope` (legacy fixtures) treat scope as
    a pass — they don't get penalized for unknown scope behavior."""
    router = _ScriptedRouter(
        intents=[RouterIntent(tool_name="search_web", arguments={"query": "x"}, scope="out")]
    )
    fixture = (("search", "search_web", ("query",), (), None),)
    result = run_router_eval(router, [_spec("search_web")], fixture)
    assert result.cases[0].scope_correct
    assert result.cases[0].passed
    # No scope rows → scope_case_count is 0; accuracy is 0.0 by convention.
    assert result.scope_case_count == 0
    assert result.scope_accuracy == 0.0


def test_eval_scope_accuracy_excludes_unscoped_rows() -> None:
    """Mixed fixture: scope_accuracy is over the scope-pinned rows only."""
    router = _ScriptedRouter(
        intents=[
            RouterIntent(tool_name="search_memory", arguments={"query": "x"}, scope="in"),
            RouterIntent(tool_name="search_memory", arguments={"query": "y"}, scope="out"),
            RouterIntent(tool_name="search_memory", arguments={"query": "z"}, scope="in"),
        ]
    )
    fixture: tuple[FixtureRow, ...] = (
        ("scope row 1 (correct)", "search_memory", ("query",), (), "in"),
        ("scope row 2 (wrong)", "search_memory", ("query",), (), "in"),
        ("legacy row, no scope", "search_memory", ("query",), (), None),
    )
    result = run_router_eval(router, [_spec("search_memory")], fixture)
    # 2 scope rows pinned, 1 correct → 50%.
    assert result.scope_case_count == 2
    assert result.scope_accuracy == 0.5
    # Tool accuracy stays at 100% — every row picked search_memory.
    assert result.tool_accuracy == 1.0


def test_load_fixture_parses_expected_scope() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "- prompt: pilot question\n"
            "  expected_tool: null\n"
            "  expected_scope: out\n"
            "- prompt: controller question\n"
            "  expected_tool: search_memory\n"
            "  expected_args: [query]\n"
            "  expected_scope: in\n"
            "- prompt: legacy row\n"
            "  expected_tool: null\n"
        )
        path = Path(f.name)

    rows = load_fixture(path)
    assert rows[0][4] == "out"
    assert rows[1][4] == "in"
    assert rows[2][4] is None  # legacy row, no scope


def test_load_fixture_rejects_invalid_scope() -> None:
    import tempfile
    from pathlib import Path

    import pytest

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "- prompt: bogus\n"
            "  expected_tool: null\n"
            "  expected_scope: maybe\n"  # not in {in, out, unsure}
        )
        path = Path(f.name)

    with pytest.raises(ValueError, match="expected_scope"):
        load_fixture(path)


# ---------- canonical fixture ----------


def test_canonical_fixture_loads_and_covers_tool_mix() -> None:
    """The shipped character/airton/router_eval.yaml parses cleanly and
    mixes positive + null cases. Guards against a fixture edit breaking
    the loader or accidentally making everything null."""
    path = default_fixture_path(Path(__file__).parent.parent / "character" / "airton")
    rows = load_fixture(path)
    assert len(rows) >= 15
    tools_used = {r[1] for r in rows}
    # Sanity: at least one search_web, at least one null, at least one
    # read-tool, at least one fetch_url (harness-057 — routing URL-shaped
    # prompts to fetch_url instead of mode-collapsing into read_file).
    assert "search_web" in tools_used
    assert None in tools_used
    assert "read_file" in tools_used
    assert "fetch_url" in tools_used
    # Null cases should be a meaningful minority so the eval catches
    # over-routing regressions.
    null_count = sum(1 for r in rows if r[1] is None)
    assert null_count >= 5


def test_airton_c1_fixture_pins_banter_to_null() -> None:
    """airton_c1's router fixture must include banter / empty-signal
    prompts mapped to null (harness-q7ff). Pre-retrieval on the
    JO 7110.65 corpus is unreliable; routing 'test' / 'ping' /
    'this page intentionally left blank' to search_memory makes the
    model fabricate a chunk from the top-1 hit at cosine 0.02."""
    path = default_fixture_path(Path(__file__).parent.parent / "character" / "airton_c1")
    rows = load_fixture(path)
    by_prompt = {prompt: expected for prompt, expected, _, _, _ in rows}
    # Spot-check the banter pin set.
    assert by_prompt.get("this page intentionally left blank") is None
    assert by_prompt.get("test") is None
    assert by_prompt.get("ping") is None
    assert by_prompt.get("are you alive") is None
    assert by_prompt.get("aaaa") is None
    assert by_prompt.get("lorem ipsum dolor sit amet") is None
    # Negative boundary — a §-anchor must still route to search_memory
    # so a future router-prompt tweak that over-broadens 'null' fails.
    assert by_prompt.get("§4-5-1") == "search_memory"
