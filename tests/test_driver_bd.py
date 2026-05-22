"""Tests for src/harness/driver/bd.py — harness-b4wx.

subprocess.run is monkeypatched end-to-end so the suite doesn't need a
real bd CLI or a running Dolt server. Tests pin the bd invocation shape
(args, cwd), the parse paths (single dict vs list-of-one, empty stdout,
missing created_at), and the driver-specific semantics (loop-run-<id>
labels, immediate-close for thought-graph beads, ready ∩ direct
dependencies for ready_under_epic).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from harness.driver.bd import (
    DEFAULT_THOUGHT_TYPES,
    DriverBd,
    DriverBdError,
    _extract_created_id,
)

# --- subprocess.run stub ---------------------------------------------


class _FakeProc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


_Response = _FakeProc | Callable[[tuple[str, ...]], _FakeProc]


class _RunSpy:
    """Captures bd subprocess calls and returns scripted responses.

    `responses` is a list of either _FakeProc instances (returned in
    order) or callables taking the args tuple and returning a _FakeProc.
    If exhausted, returns a default success with empty stdout."""

    def __init__(self, responses: Sequence[_Response] | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self._responses: list[_Response] = list(responses or [])

    def __call__(self, cmd: Sequence[str], **kwargs: Any) -> _FakeProc:
        args = tuple(cmd[1:])  # drop the bd executable
        self.calls.append((args, kwargs.get("cwd", Path("/"))))
        if not self._responses:
            return _FakeProc()
        next_response = self._responses.pop(0)
        if callable(next_response):
            return next_response(args)
        return next_response


def _install_run(monkeypatch: pytest.MonkeyPatch, spy: _RunSpy) -> None:
    monkeypatch.setattr("harness.driver.bd.subprocess.run", spy)


# --- fixtures --------------------------------------------------------


@pytest.fixture
def bd(tmp_path: Path) -> DriverBd:
    return DriverBd(bd_dir=tmp_path)


def _issue_json(
    issue_id: str,
    *,
    title: str = "",
    status: str = "open",
    priority: int = 2,
    labels: tuple[str, ...] = (),
    created_at: str = "2026-05-20T00:00:00Z",
    dependencies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "title": title,
        "status": status,
        "priority": priority,
        "issue_type": "task",
        "labels": list(labels),
        "created_at": created_at,
        "dependencies": dependencies or [],
    }


# --- _extract_created_id ---------------------------------------------


def test_extract_created_id_picks_first_match() -> None:
    stdout = """
    ✓ Created issue: harness-abcd — driver: foo
      Priority: P2
      Status: open
    """
    assert _extract_created_id(stdout) == "harness-abcd"


def test_extract_created_id_returns_none_when_missing() -> None:
    assert _extract_created_id("nothing relevant here") is None
    assert _extract_created_id("") is None


# --- show ------------------------------------------------------------


def test_show_parses_single_object(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    spy = _RunSpy([_FakeProc(stdout=json.dumps(_issue_json("harness-x", title="x")))])
    _install_run(monkeypatch, spy)
    issue = bd.show("harness-x")
    assert issue.id == "harness-x"
    assert issue.title == "x"
    assert spy.calls[0][0] == ("show", "harness-x", "--json")


def test_show_parses_list_of_one(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    spy = _RunSpy([_FakeProc(stdout=json.dumps([_issue_json("harness-y")]))])
    _install_run(monkeypatch, spy)
    issue = bd.show("harness-y")
    assert issue.id == "harness-y"


def test_show_raises_on_empty_list(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    spy = _RunSpy([_FakeProc(stdout="[]")])
    _install_run(monkeypatch, spy)
    with pytest.raises(DriverBdError, match="empty list"):
        bd.show("harness-missing")


def test_show_raises_with_stderr_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    spy = _RunSpy([_FakeProc(stderr="no such issue", returncode=1)])
    _install_run(monkeypatch, spy)
    with pytest.raises(DriverBdError, match="no such issue"):
        bd.show("harness-missing")


# --- ready / ready_under_epic ---------------------------------------


def test_ready_returns_parsed_list(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    payload = json.dumps([_issue_json("harness-a"), _issue_json("harness-b")])
    spy = _RunSpy([_FakeProc(stdout=payload)])
    _install_run(monkeypatch, spy)
    issues = bd.ready()
    assert [i.id for i in issues] == ["harness-a", "harness-b"]
    # harness-c6yu: -n 9999 bypasses bd ready's default cap of 10
    # results. Without it, ready_under_epic returned empty when an
    # epic's children sat past the global top-10 priority slot.
    assert spy.calls[0][0] == ("ready", "-n", "9999", "--json")


def test_ready_empty_stdout(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    spy = _RunSpy([_FakeProc(stdout="")])
    _install_run(monkeypatch, spy)
    assert bd.ready() == []


def test_ready_under_epic_intersects_with_direct_dependencies(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    # Epic has 3 direct dependencies. Globally ready: 4 issues, two of
    # which are children of the epic, plus one that lives elsewhere.
    epic = _issue_json(
        "harness-epic",
        dependencies=[
            {"id": "harness-a", "dependency_type": "blocks"},
            {"id": "harness-b", "dependency_type": "blocks"},
            {"id": "harness-c", "dependency_type": "blocks"},
        ],
    )
    ready_global = json.dumps(
        [
            _issue_json("harness-a"),
            _issue_json("harness-other"),
            _issue_json("harness-c"),
            _issue_json("harness-b"),
        ]
    )
    spy = _RunSpy([_FakeProc(stdout=json.dumps(epic)), _FakeProc(stdout=ready_global)])
    _install_run(monkeypatch, spy)
    ready = bd.ready_under_epic("harness-epic")
    # Preserves bd ready order.
    assert [i.id for i in ready] == ["harness-a", "harness-c", "harness-b"]


def test_ready_under_epic_empty_when_epic_has_no_dependencies(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    spy = _RunSpy([_FakeProc(stdout=json.dumps(_issue_json("harness-epic")))])
    _install_run(monkeypatch, spy)
    assert bd.ready_under_epic("harness-epic") == []
    # ready() should NOT be called when there are no children to intersect with.
    assert len(spy.calls) == 1


# --- thoughts_in_loop_run --------------------------------------------


def test_thoughts_in_loop_run_merges_dedupes_and_sorts_desc(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    # Three search calls, one per thought-type. A single bead carries
    # both thought:decision AND thought:question labels — must dedupe.
    decisions = [
        _issue_json("harness-d1", created_at="2026-05-19T00:00:00Z"),
        _issue_json("harness-d2", created_at="2026-05-21T00:00:00Z"),
    ]
    observations = [_issue_json("harness-o1", created_at="2026-05-20T00:00:00Z")]
    # harness-d1 also matched the thought:question query — should appear once.
    questions = [
        _issue_json("harness-d1", created_at="2026-05-19T00:00:00Z"),
        _issue_json("harness-q1", created_at="2026-05-22T00:00:00Z"),
    ]
    spy = _RunSpy(
        [
            _FakeProc(stdout=json.dumps(decisions)),
            _FakeProc(stdout=json.dumps(observations)),
            _FakeProc(stdout=json.dumps(questions)),
        ]
    )
    _install_run(monkeypatch, spy)
    results = bd.thoughts_in_loop_run("abcd1234")
    # Newest-first, deduped on id.
    assert [i.id for i in results] == ["harness-q1", "harness-d2", "harness-o1", "harness-d1"]
    # Verify the label filter shape on each search call.
    expected_loop_label = "loop-run-abcd1234"
    for call_args, _ in spy.calls:
        assert "search" in call_args
        assert "--label" in call_args
        assert expected_loop_label in call_args


def test_thoughts_in_loop_run_tolerates_search_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    # Some bd versions exit non-zero when a search has no matches —
    # the driver must not raise, just return whatever did match.
    decisions = [_issue_json("harness-d1")]
    spy = _RunSpy(
        [
            _FakeProc(stdout=json.dumps(decisions)),
            _FakeProc(stderr="no results", returncode=1),
            _FakeProc(stderr="no results", returncode=1),
        ]
    )
    _install_run(monkeypatch, spy)
    results = bd.thoughts_in_loop_run("abcd1234")
    assert [i.id for i in results] == ["harness-d1"]


def test_thoughts_in_loop_run_custom_types(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    # Override the default thought-type set; verify only the supplied
    # types are queried.
    spy = _RunSpy([_FakeProc(stdout="[]")])
    _install_run(monkeypatch, spy)
    bd.thoughts_in_loop_run("abcd1234", types=("thought:hypothesis",))
    assert len(spy.calls) == 1
    args, _ = spy.calls[0]
    assert "thought:hypothesis" in args


# --- create_with_labels ----------------------------------------------


def test_create_with_labels_emits_each_label_flag(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    confirmation = "✓ Created issue: harness-new — title here"
    spy = _RunSpy([_FakeProc(stdout=confirmation)])
    _install_run(monkeypatch, spy)
    new_id = bd.create_with_labels(
        title="title",
        description="desc",
        labels=("loop-run-abcd1234", "thought:decision"),
    )
    assert new_id == "harness-new"
    args, _ = spy.calls[0]
    assert "create" in args
    assert "--label=loop-run-abcd1234" in args
    assert "--label=thought:decision" in args


def test_create_with_labels_includes_acceptance_when_provided(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    spy = _RunSpy([_FakeProc(stdout="✓ Created issue: harness-new — t")])
    _install_run(monkeypatch, spy)
    bd.create_with_labels(title="t", description="d", acceptance="must do X")
    args, _ = spy.calls[0]
    assert "--acceptance=must do X" in args


def test_create_with_labels_raises_when_no_id_returned(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    spy = _RunSpy([_FakeProc(stdout="something else entirely")])
    _install_run(monkeypatch, spy)
    with pytest.raises(DriverBdError, match="not return a parseable id"):
        bd.create_with_labels(title="t", description="d")


# --- close / dep_add / flag_human ------------------------------------


def test_close_with_reason(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    spy = _RunSpy([_FakeProc()])
    _install_run(monkeypatch, spy)
    bd.close("harness-x", reason="done")
    assert spy.calls[0][0] == ("close", "harness-x", "--reason=done")


def test_close_without_reason(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    spy = _RunSpy([_FakeProc()])
    _install_run(monkeypatch, spy)
    bd.close("harness-x")
    assert spy.calls[0][0] == ("close", "harness-x")


def test_dep_add(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    spy = _RunSpy([_FakeProc()])
    _install_run(monkeypatch, spy)
    bd.dep_add(blocked="harness-child", blocker="harness-parent")
    assert spy.calls[0][0] == ("dep", "add", "harness-child", "harness-parent")


def test_flag_human(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    spy = _RunSpy([_FakeProc()])
    _install_run(monkeypatch, spy)
    bd.flag_human("harness-x", reason="halted after 2 attempts")
    assert spy.calls[0][0] == ("human", "harness-x", "--reason=halted after 2 attempts")


def test_reopen_sets_status_open(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    """harness-xfh2: DriverBd.reopen shells `bd update <id> --status=open`
    so the loop's verify gate can put a model-closed issue back in
    ready_under_epic when verify fails."""
    spy = _RunSpy([_FakeProc()])
    _install_run(monkeypatch, spy)
    bd.reopen("harness-x")
    assert spy.calls[0][0] == ("update", "harness-x", "--status=open")


# --- write_thought / write_session_state -----------------------------


def test_write_thought_creates_then_closes_with_both_labels(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    spy = _RunSpy(
        [
            _FakeProc(stdout="✓ Created issue: harness-thought1 — t"),
            _FakeProc(),  # close
        ]
    )
    _install_run(monkeypatch, spy)
    new_id = bd.write_thought(
        thought_type="thought:decision",
        title="picked plan A",
        body="full body of the decision",
        loop_run_id="abcd1234",
    )
    assert new_id == "harness-thought1"
    create_args, _ = spy.calls[0]
    assert "--label=thought:decision" in create_args
    assert "--label=loop-run-abcd1234" in create_args
    close_args, _ = spy.calls[1]
    assert close_args[0] == "close"
    assert close_args[1] == "harness-thought1"
    # Reason references the loop run for audit.
    assert any("loop driver" in a for a in close_args)


def test_write_session_state_uses_session_state_label_and_structured_header(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    spy = _RunSpy(
        [
            _FakeProc(stdout="✓ Created issue: harness-state1 — s"),
            _FakeProc(),
        ]
    )
    _install_run(monkeypatch, spy)
    bd.write_session_state(
        loop_run_id="abcd1234",
        current_issue_id="harness-task",
        status="success",
        body="closed cleanly",
    )
    create_args, _ = spy.calls[0]
    assert "--label=thought:session-state" in create_args
    # Body carries the structured header so harvested rows are machine-readable.
    description_args = [a for a in create_args if a.startswith("--description=")]
    assert len(description_args) == 1
    description = description_args[0].removeprefix("--description=")
    assert "loop_run=abcd1234" in description
    assert "issue=harness-task" in description
    assert "status=success" in description


# --- verify ----------------------------------------------------------


def test_verify_raises_when_bd_missing(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    monkeypatch.setattr("harness.driver.bd.shutil.which", lambda _name: None)
    with pytest.raises(DriverBdError, match="not found"):
        bd.verify()


def test_verify_raises_when_no_beads_dir(monkeypatch: pytest.MonkeyPatch, bd: DriverBd) -> None:
    monkeypatch.setattr("harness.driver.bd.shutil.which", lambda _name: "/usr/local/bin/bd")
    with pytest.raises(DriverBdError, match=r"\.beads/"):
        bd.verify()


def test_verify_succeeds_when_beads_dir_exists(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd, tmp_path: Path
) -> None:
    (tmp_path / ".beads").mkdir()
    monkeypatch.setattr("harness.driver.bd.shutil.which", lambda _name: "/usr/local/bin/bd")
    bd.verify()  # no raise


# --- subprocess plumbing edge cases ----------------------------------


def test_filenotfound_surfaces_as_driverbderror(
    monkeypatch: pytest.MonkeyPatch, bd: DriverBd
) -> None:
    def boom(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("no bd")

    monkeypatch.setattr("harness.driver.bd.subprocess.run", boom)
    with pytest.raises(DriverBdError, match="not found"):
        bd.ready()


def test_cwd_is_workspace_root_on_every_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    bd = DriverBd(bd_dir=workspace)
    spy = _RunSpy([_FakeProc(stdout="[]"), _FakeProc()])
    _install_run(monkeypatch, spy)
    bd.ready()
    bd.close("harness-x")
    for _args, cwd in spy.calls:
        assert cwd == workspace


def test_default_thought_types_matches_spec() -> None:
    # Pinned: changing this default is a contract change for handoff.
    assert DEFAULT_THOUGHT_TYPES == (
        "thought:decision",
        "thought:observation",
        "thought:question",
    )
