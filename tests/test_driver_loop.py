"""Tests for src/harness/driver/loop.py — harness-ml66.

The executor loop is the integration point between LoopRunState,
DriverBd, build_handoff, and orchestrator.tool_loop.run_tool_loop. To
keep tests independent of MLX / Ollama / a live bd CLI, both
`run_tool_loop` and the bd client are mocked at the module boundary.
Tests pin the control flow: success closes the issue and continues,
first failure retries the same issue, second failure halts, dry-run
short-circuits, resume rehydrates state, exhaustion fires on max_turns.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from harness.driver.loop import (
    EXECUTOR_USER_MESSAGE,
    LoopConfig,
    LoopResult,
    run_loop,
)
from harness.driver.state import LoopRunState
from harness.orchestrator import ToolLoopResult
from harness.orchestrator.hooks import EXHAUSTED_FABRICATION_FALLBACK
from harness.store._bd_types import BeadsIssue, _issue_from_json

# --- fakes -----------------------------------------------------------


def _issue(
    issue_id: str,
    *,
    title: str = "",
    status: str = "open",
    priority: int = 2,
    labels: tuple[str, ...] = (),
    dependencies: list[dict[str, Any]] | None = None,
) -> BeadsIssue:
    raw: dict[str, Any] = {
        "id": issue_id,
        "title": title,
        "status": status,
        "priority": priority,
        "issue_type": "task",
        "labels": list(labels),
        "dependencies": dependencies or [],
        "created_at": "2026-05-20T00:00:00Z",
    }
    return _issue_from_json(raw)


@dataclass
class _BdLog:
    """Captures everything the loop did against bd, for test assertions."""

    closed_via_session_state: list[tuple[str, str]] = field(default_factory=list)
    human_flags: list[tuple[str, str]] = field(default_factory=list)
    ready_calls: int = 0
    show_calls: list[str] = field(default_factory=list)


class _ScenarioBd:
    """Bd fake driven by a scenario. The test sets up:
    - `ready_responses`: list of ready_under_epic responses, consumed
      in order. Default = repeat the last response.
    - `closed_after_turn`: set of bd-ids that should report
      status="closed" once `flip_closed(id)` is called by the test
      runner between turns (simulating the model closing the
      issue mid-turn).
    - `show_errors`: bd-ids that raise DriverBdError on show.
    """

    def __init__(
        self,
        *,
        ready_sequence: Sequence[list[BeadsIssue]],
        issues: dict[str, BeadsIssue] | None = None,
        show_errors: set[str] | None = None,
        ready_raises: int | None = None,
    ) -> None:
        self._ready_sequence = list(ready_sequence)
        self._issues = dict(issues or {})
        self._show_errors = show_errors or set()
        self._ready_raises = ready_raises
        self.log = _BdLog()

    def ready_under_epic(self, _epic_id: str) -> list[BeadsIssue]:
        self.log.ready_calls += 1
        if self._ready_raises is not None and self.log.ready_calls == self._ready_raises:
            from harness.driver.bd import DriverBdError

            raise DriverBdError("simulated ready failure")
        if not self._ready_sequence:
            return []
        if len(self._ready_sequence) == 1:
            return list(self._ready_sequence[0])
        return self._ready_sequence.pop(0)

    def show(self, issue_id: str) -> BeadsIssue:
        self.log.show_calls.append(issue_id)
        if issue_id in self._show_errors:
            from harness.driver.bd import DriverBdError

            raise DriverBdError(f"simulated show failure for {issue_id}")
        return self._issues[issue_id]

    def thoughts_in_loop_run(
        self,
        _loop_run_id: str,
        *,
        types: Sequence[str] | None = None,
    ) -> list[BeadsIssue]:
        return []

    def write_session_state(
        self,
        *,
        loop_run_id: str,
        current_issue_id: str,
        status: str,
        body: str,
    ) -> str:
        self.log.closed_via_session_state.append((current_issue_id, status))
        return "harness-session-state-bead"

    def flag_human(self, issue_id: str, *, reason: str) -> None:
        self.log.human_flags.append((issue_id, reason))

    def flip_closed(self, issue_id: str) -> None:
        """Test helper — flip an issue's status to closed mid-test."""
        existing = self._issues[issue_id]
        self._issues[issue_id] = _issue(
            existing.id,
            title=existing.title,
            status="closed",
            priority=existing.priority,
            labels=existing.labels,
        )


class _FakeCharacter:
    """Minimum surface build_loop needs from a Character: system_prompt."""

    def system_prompt(self, *, include_samples: Sequence[Any] = ()) -> str:
        return "you are an executor."


class _FakeAdapter:
    """Placeholder — never invoked because run_tool_loop is monkeypatched."""


# --- monkeypatch helpers --------------------------------------------


def _stub_git_head(monkeypatch: pytest.MonkeyPatch, sha: str = "deadbeef") -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=f"{sha}\n", stderr="")

    monkeypatch.setattr("harness.driver.loop.subprocess.run", fake_run)


def _stub_run_tool_loop(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[str],
    *,
    bd: _ScenarioBd | None = None,
    on_each_call: Any = None,
) -> list[int]:
    """Replace run_tool_loop so it returns the next `outcome` from
    the list each call. Outcomes are interpreted as:
      - "close <id>": the model "closed" the issue (flip on the bd
        fake) and the turn reply is benign.
      - "fail": turn reply is EXHAUSTED_FABRICATION_FALLBACK.
      - "open <id>": turn reply is benign but the issue stays open
        (so post-turn check classifies as failure).
    Returns the count list — caller can read `len(...)` post-test."""
    call_count = [0]

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, _registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        idx = call_count[0]
        if idx >= len(outcomes):
            pytest.fail(f"run_tool_loop called more than {len(outcomes)} times")
        outcome = outcomes[idx]
        call_count[0] += 1
        if outcome.startswith("close "):
            if bd is not None:
                bd.flip_closed(outcome.split(" ", 1)[1])
            content = "issue closed, done."
        elif outcome.startswith("open "):
            content = "I think I'm done but did not close the issue."
        elif outcome == "fail":
            content = EXHAUSTED_FABRICATION_FALLBACK
        else:
            pytest.fail(f"unknown outcome: {outcome!r}")
        if on_each_call is not None:
            on_each_call(idx, outcome)
        return ToolLoopResult(content=content, messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.loop.run_tool_loop", fake_run_tool_loop)
    return call_count


def _config(tmp_path: Path, **overrides: Any) -> LoopConfig:
    base: dict[str, Any] = {
        "epic_id": "harness-e9oq",
        "workspace": tmp_path,
        "character": _FakeCharacter(),
        "max_turns": 5,
    }
    base.update(overrides)
    return LoopConfig(**base)


# --- success path ---------------------------------------------------


def test_run_loop_closes_two_issues_then_exits_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    issue_a = _issue("harness-a", title="A", status="open")
    issue_b = _issue("harness-b", title="B", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a, issue_b], [issue_b], []],
        issues={
            "harness-a": issue_a,
            "harness-b": issue_b,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(
        monkeypatch,
        outcomes=["close harness-a", "close harness-b"],
        bd=bd,
    )

    result = run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]
    assert result.exit_reason == "success"
    assert result.closed == ["harness-a", "harness-b"]
    assert result.turns_used == 2
    assert bd.log.human_flags == []
    # session-state beads written for each success.
    statuses = [s for _, s in bd.log.closed_via_session_state]
    assert statuses == ["success", "success"]


def test_run_loop_state_file_persists_per_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(monkeypatch, outcomes=["close harness-a"], bd=bd)

    result = run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]

    state_path = LoopRunState.state_path(tmp_path, result.loop_run_id)
    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["closed_this_run"] == ["harness-a"]
    assert payload["turns_used"] == 1


# --- first-failure retry --------------------------------------------


def test_run_loop_first_failure_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Turn 1: open ack but doesn't close → first failure.
    # Turn 2: closes → success.
    # Turn 3: empty ready → exit_success.
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], [issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(
        monkeypatch,
        outcomes=["open harness-a", "close harness-a"],
        bd=bd,
    )

    result = run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]
    assert result.exit_reason == "success"
    assert result.closed == ["harness-a"]
    assert result.turns_used == 2
    # Human not flagged — we recovered on attempt #2.
    assert bd.log.human_flags == []


def test_run_loop_records_last_failure_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], [issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)

    # Capture state at the second call so we can assert last_failure[a] was set.
    state_snapshot: dict[str, Any] = {}

    def capture(idx: int, outcome: str) -> None:
        if idx == 1:
            # Locate the most-recently-saved state file and inspect.
            d = LoopRunState.state_dir(tmp_path)
            files = list(d.glob("*.json"))
            assert len(files) == 1
            state_snapshot.update(json.loads(files[0].read_text()))

    _stub_run_tool_loop(
        monkeypatch,
        outcomes=["fail", "close harness-a"],
        bd=bd,
        on_each_call=capture,
    )
    run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]
    assert "harness-a" in state_snapshot["last_failure"]
    assert "fabrication_fallback" in state_snapshot["last_failure"]["harness-a"]


# --- halt on second failure -----------------------------------------


def test_run_loop_halts_after_two_failures_on_same_issue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a]],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(monkeypatch, outcomes=["fail", "fail"], bd=bd)

    result = run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]
    assert result.exit_reason == "halted"
    assert result.halted_on == "harness-a"
    assert result.turns_used == 2
    assert bd.log.human_flags == [("harness-a", "loop halted: fabrication_fallback fired")]
    # session-state bead for the halt event written.
    assert ("harness-a", "halted") in bd.log.closed_via_session_state


# --- exhaustion -----------------------------------------------------


def test_run_loop_exits_exhausted_at_max_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # max_turns=2; two failing turns on different issues exhaust the budget.
    # First failure on issue_a, then because it's only the FIRST failure, the
    # loop retries — but max_turns hits before retry can run.
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a]],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(monkeypatch, outcomes=["open harness-a", "open harness-a"], bd=bd)
    cfg = _config(tmp_path, max_turns=2)

    result = run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    # After two failed attempts on the same issue, halt fires (attempt==2
    # is the halt rule, which happens BEFORE exhaustion check on iteration 3).
    # So this test actually exercises the halt path. Confirm.
    assert result.exit_reason == "halted"


def test_run_loop_exhausts_when_many_different_issues_fail_first_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Different issue each iteration so the retry rule doesn't trigger.
    issue_a = _issue("harness-a", title="A", status="open")
    issue_b = _issue("harness-b", title="B", status="open")
    issue_c = _issue("harness-c", title="C", status="open")
    issues = {
        "harness-a": issue_a,
        "harness-b": issue_b,
        "harness-c": issue_c,
        "harness-e9oq": _issue("harness-e9oq", title="epic"),
    }
    # Each iteration the top of ready is a fresh issue; first failures
    # don't halt because the issue changes between iterations.
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], [issue_b], [issue_c]],
        issues=issues,
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(
        monkeypatch,
        outcomes=["open harness-a", "open harness-b"],
        bd=bd,
    )
    cfg = _config(tmp_path, max_turns=2)
    result = run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "exhausted"
    assert result.turns_used == 2


# --- dry run --------------------------------------------------------


def test_run_loop_dry_run_returns_handoff_without_running_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a]],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)

    def boom(*_args: Any, **_kwargs: Any) -> ToolLoopResult:
        pytest.fail("run_tool_loop should not be called in --dry-run mode")

    monkeypatch.setattr("harness.driver.loop.run_tool_loop", boom)
    cfg = _config(tmp_path, dry_run=True)
    result = run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "dry_run"
    assert result.turns_used == 0
    assert len(result.handoffs) == 1
    assert "harness-a" in result.handoffs[0].current_issue


# --- resume ---------------------------------------------------------


def test_run_loop_resume_rehydrates_state_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Seed a pre-existing state file representing "loop already closed
    # one issue, paused, now resuming."
    s = LoopRunState.fresh(epic_id="harness-e9oq", max_turns=5, started_at_sha="seedsha")
    s.closed_this_run.append("harness-z")
    s.turns_used = 1
    s.save(LoopRunState.state_path(tmp_path, s.loop_run_id))

    issue_b = _issue("harness-b", title="B", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_b], []],
        issues={
            "harness-b": issue_b,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch, sha="should-not-be-used")
    _stub_run_tool_loop(monkeypatch, outcomes=["close harness-b"], bd=bd)
    cfg = _config(tmp_path, resume_from=s.loop_run_id)
    result = run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.loop_run_id == s.loop_run_id
    # Pre-existing close survived; new close appended.
    assert result.closed == ["harness-z", "harness-b"]
    # turns_used incremented from the pre-resume baseline.
    assert result.turns_used == 2


def test_run_loop_resume_max_turns_override_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-iai4: resuming with --max-turns N (config) must override
    the saved state's max_turns. Otherwise the user's resume + budget-
    bump intent is silently dropped."""
    # Seed: max_turns=1, already exhausted (turns_used=1).
    s = LoopRunState.fresh(epic_id="harness-e9oq", max_turns=1, started_at_sha="seedsha")
    s.turns_used = 1
    s.save(LoopRunState.state_path(tmp_path, s.loop_run_id))

    issue_b = _issue("harness-b", title="B", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_b], []],
        issues={
            "harness-b": issue_b,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(monkeypatch, outcomes=["close harness-b"], bd=bd)
    # Resume with --max-turns 3: should override saved max_turns=1.
    cfg = _config(tmp_path, resume_from=s.loop_run_id, max_turns=3)
    result = run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    # The override let turn 2 run; harness-b closed; ready empties; success.
    assert result.exit_reason == "success"
    assert result.turns_used == 2
    assert result.closed == ["harness-b"]


def test_run_loop_resume_persists_overridden_max_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-iai4: the override is persisted to the state file so a
    subsequent load sees the new value, not the original."""
    s = LoopRunState.fresh(epic_id="harness-e9oq", max_turns=1, started_at_sha="seedsha")
    s.turns_used = 1
    s.save(LoopRunState.state_path(tmp_path, s.loop_run_id))

    bd = _ScenarioBd(
        ready_sequence=[[]],
        issues={"harness-e9oq": _issue("harness-e9oq", title="epic")},
    )
    _stub_git_head(monkeypatch)
    cfg = _config(tmp_path, resume_from=s.loop_run_id, max_turns=5)
    run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    # Reload state from disk and verify the override stuck.
    reloaded = LoopRunState.load(LoopRunState.state_path(tmp_path, s.loop_run_id))
    assert reloaded.max_turns == 5


# --- ready_under_epic failure --------------------------------------


def test_run_loop_halts_on_persistent_ready_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bd = _ScenarioBd(
        ready_sequence=[[]],
        issues={"harness-e9oq": _issue("harness-e9oq", title="epic")},
        ready_raises=1,
    )
    _stub_git_head(monkeypatch)

    def boom(*_args: Any, **_kwargs: Any) -> ToolLoopResult:
        pytest.fail("run_tool_loop should not be called when ready fails")

    monkeypatch.setattr("harness.driver.loop.run_tool_loop", boom)
    result = run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]
    assert result.exit_reason == "halted"


# --- log file -------------------------------------------------------


def test_run_loop_writes_progress_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(monkeypatch, outcomes=["close harness-a"], bd=bd)
    result = run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]

    log_path = LoopRunState.state_dir(tmp_path) / f"{result.loop_run_id}.log"
    assert log_path.exists()
    content = log_path.read_text()
    assert f"loop_run={result.loop_run_id}" in content
    assert "harness-a CLOSED" in content
    assert "SUCCESS" in content


def test_run_loop_executor_observer_writes_to_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-9bpt: per-turn tool events get appended to the loop log
    with a `turn N |` prefix alongside the lifecycle markers."""
    from harness.orchestrator import ToolLoopEvent
    from harness.tools.base import ToolCall

    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)

    def fake_run_tool_loop(
        _adapter: Any,
        _messages: Any,
        _registry: Any,
        *,
        observe: Any = None,
        **_kwargs: Any,
    ) -> ToolLoopResult:
        if observe is not None:
            observe(
                ToolLoopEvent(
                    kind="tool_call_start",
                    call=ToolCall(name="read_file", arguments={"path": "x.md"}),
                )
            )
        bd.flip_closed("harness-a")
        return ToolLoopResult(content="done", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.loop.run_tool_loop", fake_run_tool_loop)
    result = run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]

    log_path = LoopRunState.state_dir(tmp_path) / f"{result.loop_run_id}.log"
    content = log_path.read_text()
    assert "turn 1 |" in content
    assert "tool_call_start" in content
    assert "call=read_file" in content


def test_run_loop_extra_observer_receives_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-9bpt: LoopConfig.extra_observer receives every per-turn
    event so the CLI's --verbose flag can mirror to stderr."""
    from harness.orchestrator import ToolLoopEvent

    captured: list[ToolLoopEvent] = []
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)

    def fake_run_tool_loop(
        _adapter: Any,
        _messages: Any,
        _registry: Any,
        *,
        observe: Any = None,
        **_kwargs: Any,
    ) -> ToolLoopResult:
        if observe is not None:
            observe(ToolLoopEvent(kind="round_start"))
        bd.flip_closed("harness-a")
        return ToolLoopResult(content="done", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.loop.run_tool_loop", fake_run_tool_loop)
    cfg = _config(tmp_path, extra_observer=captured.append)
    run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert any(e.kind == "round_start" for e in captured)


def test_run_loop_forbidden_pattern_fails_closed_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-k52f: a bd-closed issue with a TODO in a modified file
    fails the turn — closed-with-bad-code is worse than not-closed."""
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a]],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, _registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        (tmp_path / "game.js").write_text("// TODO: implement physics\nfunction gameLoop() {}\n")
        bd.flip_closed("harness-a")
        return ToolLoopResult(content="done", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.loop.run_tool_loop", fake_run_tool_loop)
    result = run_loop(_FakeAdapter(), bd, _config(tmp_path, max_turns=1))  # type: ignore[arg-type]
    # 1 turn, 1 failure, max_turns exhausted before retry.
    assert result.exit_reason == "exhausted"
    assert result.closed == []


def test_run_loop_no_verify_disables_forbidden_pattern_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-k52f: forbidden_patterns=() disables the check."""
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, _registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        (tmp_path / "game.js").write_text("// TODO: implement physics\n")
        bd.flip_closed("harness-a")
        return ToolLoopResult(content="done", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.loop.run_tool_loop", fake_run_tool_loop)
    cfg = _config(tmp_path, forbidden_patterns=())
    result = run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "success"
    assert result.closed == ["harness-a"]


def test_run_loop_forbidden_pattern_check_skips_hidden_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-k52f: hidden dirs (.git, .harness, .beads) are skipped
    even if they contain forbidden patterns — the loop's own log file
    must not trip the check on itself."""
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, _registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        hidden = tmp_path / ".harness" / "something.log"
        hidden.parent.mkdir(parents=True, exist_ok=True)
        hidden.write_text("TODO: should be ignored\n")
        bd.flip_closed("harness-a")
        return ToolLoopResult(content="done", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.loop.run_tool_loop", fake_run_tool_loop)
    result = run_loop(_FakeAdapter(), bd, _config(tmp_path))  # type: ignore[arg-type]
    assert result.exit_reason == "success"


def test_run_loop_uses_configured_log_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom_log = tmp_path / "custom.log"
    issue_a = _issue("harness-a", title="A", status="open")
    bd = _ScenarioBd(
        ready_sequence=[[issue_a], []],
        issues={
            "harness-a": issue_a,
            "harness-e9oq": _issue("harness-e9oq", title="epic"),
        },
    )
    _stub_git_head(monkeypatch)
    _stub_run_tool_loop(monkeypatch, outcomes=["close harness-a"], bd=bd)
    cfg = _config(tmp_path, log_path=custom_log)
    run_loop(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert custom_log.exists()


# --- constants ------------------------------------------------------


def test_executor_user_message_mentions_bd_close() -> None:
    # Pinned: the user message tells the model how to signal completion.
    # If this drifts, the loop's outcome detection still works (post-turn
    # bd.show is mechanical) but the convergence rate would suffer.
    assert "bd close" in EXECUTOR_USER_MESSAGE
    assert "acceptance criteria" in EXECUTOR_USER_MESSAGE


def test_loop_result_is_dataclass_like() -> None:
    # Smoke test on the public shape — guard against accidental rename.
    r = LoopResult(
        loop_run_id="x",
        epic_id="y",
        closed=[],
        halted_on=None,
        turns_used=0,
        exit_reason="success",
    )
    assert r.exit_reason == "success"
    assert r.handoffs == []
