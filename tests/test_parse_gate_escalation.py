"""Tests for the parse-gate escalation pair (harness-0v6d)."""

from __future__ import annotations

from harness.orchestrator.hooks import (
    Continue,
    PostToolContext,
    PreToolContext,
    Skip,
)
from harness.orchestrator.parse_gate_escalation import (
    DEFAULT_THRESHOLD,
    ParseGateEscalationHook,
    ParseGateFailureObserver,
    ParseGateState,
    _extract_path,
    _is_parse_gate_failure,
    make_parse_gate_escalation_pair,
)
from harness.tools.base import ToolCall, ToolResult, ToolSpec


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description="", parameters={}, tier="write")


def _post_ctx(name: str, args: dict[str, object], result: ToolResult) -> PostToolContext:
    return PostToolContext(
        call=ToolCall(name=name, arguments=dict(args)),
        result=result,
        spec=_spec(name),
    )


def _pre_ctx(name: str, args: dict[str, object]) -> PreToolContext:
    return PreToolContext(
        call=ToolCall(name=name, arguments=dict(args)),
        seen_calls={},
    )


def _parse_fail(path: str = "game.js") -> ToolResult:
    return ToolResult(
        tool_name="edit_file",
        output=(
            f"edit landed on disk but {path} no longer parses. Read the "
            f"file and either fix the syntax error or back out the bad edit."
        ),
        success=False,
        error="ValueError",
    )


def _other_fail() -> ToolResult:
    return ToolResult(
        tool_name="edit_file",
        output="some other failure",
        success=False,
        error="ValueError",
    )


def test_is_parse_gate_failure_matches_marker_in_output() -> None:
    assert _is_parse_gate_failure(_parse_fail())


def test_is_parse_gate_failure_matches_marker_in_error_field() -> None:
    r = ToolResult(
        tool_name="edit_file",
        output="x",
        success=False,
        error="thing no longer parses ok",
    )
    assert _is_parse_gate_failure(r)


def test_is_parse_gate_failure_rejects_success() -> None:
    assert not _is_parse_gate_failure(
        ToolResult(tool_name="edit_file", output="no longer parses", success=True)
    )


def test_is_parse_gate_failure_rejects_unrelated_failure() -> None:
    assert not _is_parse_gate_failure(_other_fail())


def test_extract_path_handles_path_string() -> None:
    assert _extract_path({"path": "game.js"}) == "game.js"


def test_extract_path_handles_paths_list() -> None:
    assert _extract_path({"paths": ["game.js"]}) == "game.js"


def test_extract_path_handles_paths_stringy() -> None:
    assert _extract_path({"paths": "\n\ngame.js\n"}) == "game.js"


def test_extract_path_returns_none_when_missing() -> None:
    assert _extract_path({"other": "x"}) is None


def test_observer_increments_counter_on_parse_gate_failure() -> None:
    state = ParseGateState()
    obs = ParseGateFailureObserver(state=state)
    obs.check(_post_ctx("edit_file", {"path": "game.js"}, _parse_fail()))
    assert state.failures_by_path == {"game.js": 1}
    assert state.blocked_paths == set()


def test_observer_blocks_path_at_threshold() -> None:
    state = ParseGateState()
    obs = ParseGateFailureObserver(state=state, threshold=2)
    obs.check(_post_ctx("edit_file", {"path": "game.js"}, _parse_fail()))
    obs.check(_post_ctx("edit_file", {"path": "game.js"}, _parse_fail()))
    assert state.blocked_paths == {"game.js"}


def test_observer_ignores_non_parse_gate_failure() -> None:
    state = ParseGateState()
    obs = ParseGateFailureObserver(state=state)
    obs.check(_post_ctx("edit_file", {"path": "game.js"}, _other_fail()))
    assert state.failures_by_path == {}


def test_observer_separates_paths() -> None:
    state = ParseGateState()
    obs = ParseGateFailureObserver(state=state, threshold=2)
    obs.check(_post_ctx("edit_file", {"path": "game.js"}, _parse_fail()))
    obs.check(_post_ctx("edit_file", {"path": "other.js"}, _parse_fail("other.js")))
    assert state.failures_by_path == {"game.js": 1, "other.js": 1}
    assert state.blocked_paths == set()


def test_escalation_lets_through_when_path_not_blocked() -> None:
    state = ParseGateState()
    hook = ParseGateEscalationHook(state=state)
    outcome = hook.check(_pre_ctx("edit_file", {"path": "game.js"}))
    assert isinstance(outcome, Continue)


def test_escalation_intercepts_edit_file_on_blocked_path() -> None:
    state = ParseGateState(
        failures_by_path={"game.js": 2},
        blocked_paths={"game.js"},
    )
    hook = ParseGateEscalationHook(state=state)
    outcome = hook.check(_pre_ctx("edit_file", {"path": "game.js"}))
    assert isinstance(outcome, Skip)
    assert outcome.result.success is False
    assert outcome.result.error == "parse_gate_escalation"
    assert "game.js" in outcome.result.output
    assert "read_file" in outcome.result.output
    assert "write_file" in outcome.result.output


def test_escalation_intercepts_stream_edit_on_blocked_path() -> None:
    state = ParseGateState(blocked_paths={"game.js"})
    hook = ParseGateEscalationHook(state=state)
    outcome = hook.check(_pre_ctx("stream_edit", {"paths": ["game.js"]}))
    assert isinstance(outcome, Skip)


def test_escalation_intercepts_python_stream_on_blocked_path() -> None:
    state = ParseGateState(blocked_paths={"game.js"})
    hook = ParseGateEscalationHook(state=state)
    outcome = hook.check(_pre_ctx("python_stream", {"paths": ["game.js"]}))
    assert isinstance(outcome, Skip)


def test_escalation_lets_through_write_file_on_blocked_path() -> None:
    state = ParseGateState(blocked_paths={"game.js"})
    hook = ParseGateEscalationHook(state=state)
    outcome = hook.check(_pre_ctx("write_file", {"path": "game.js"}))
    assert isinstance(outcome, Continue)


def test_escalation_lets_through_read_file_on_blocked_path() -> None:
    state = ParseGateState(blocked_paths={"game.js"})
    hook = ParseGateEscalationHook(state=state)
    outcome = hook.check(_pre_ctx("read_file", {"path": "game.js"}))
    assert isinstance(outcome, Continue)


def test_escalation_does_not_affect_other_paths() -> None:
    state = ParseGateState(blocked_paths={"game.js"})
    hook = ParseGateEscalationHook(state=state)
    outcome = hook.check(_pre_ctx("edit_file", {"path": "other.js"}))
    assert isinstance(outcome, Continue)


def test_pair_shares_state() -> None:
    """The factory wires both hooks to ONE ParseGateState so the
    observer's writes feed the escalation hook's reads."""
    obs, esc = make_parse_gate_escalation_pair(threshold=2)
    assert obs.state is esc.state
    obs.check(_post_ctx("edit_file", {"path": "game.js"}, _parse_fail()))
    obs.check(_post_ctx("edit_file", {"path": "game.js"}, _parse_fail()))
    outcome = esc.check(_pre_ctx("edit_file", {"path": "game.js"}))
    assert isinstance(outcome, Skip)


def test_pair_default_threshold_is_two() -> None:
    obs, _ = make_parse_gate_escalation_pair()
    assert obs.threshold == DEFAULT_THRESHOLD == 2
