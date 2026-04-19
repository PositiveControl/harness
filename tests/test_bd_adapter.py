"""Tests for src/harness/store/bd_adapter.py — harness-inj.4.

subprocess is mocked end-to-end so the suite doesn't require bd or a
running Dolt SQL server. The tests pin:

- every bd invocation runs with cwd=<ab_bd_dir> (isolation invariant)
- scope label is always applied on create
- scope + type validation rejects disallowed values before spawning bd
- custom-type registration is idempotent and merges with pre-existing
  user config
- verify() surfaces repair hints when bd isn't runnable or the dir is
  uninitialized
- JSON parsing handles both list and single-object bd output
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from harness.store.bd_adapter import (
    AB_CUSTOM_TYPES,
    ALLOWED_SCOPES,
    BeadsAdapter,
    BeadsAdapterError,
    BeadsIssue,
    _extract_created_id,
)


@dataclass
class FakeCompletedProcess:
    """Minimal stand-in for subprocess.CompletedProcess that BeadsAdapter
    consumes. `check` in the adapter is post-call, so this deliberately
    leaves returncode controllable."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeRunner:
    """Records the sequence of subprocess.run calls the adapter makes
    and hands back canned responses in order. Tests assert both what
    got called and what got returned."""

    def __init__(self, responses: list[FakeCompletedProcess] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses or [])

    def __call__(
        self,
        cmd: list[str],
        *,
        cwd: Path | str | None = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
    ) -> FakeCompletedProcess:
        self.calls.append({"cmd": cmd, "cwd": cwd, "capture_output": capture_output, "text": text})
        if self._responses:
            return self._responses.pop(0)
        return FakeCompletedProcess()

    def queue(self, *responses: FakeCompletedProcess) -> None:
        self._responses.extend(responses)


@pytest.fixture
def bd_dir(tmp_path: Path) -> Path:
    """Minimum viable bd_dir: the directory exists and has a .beads/
    subdir so verify() passes without bd actually running."""
    (tmp_path / ".beads").mkdir()
    return tmp_path


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    fake = FakeRunner()
    monkeypatch.setattr(
        "harness.store.bd_adapter.subprocess.run",
        fake,
    )
    return fake


def test_create_applies_scope_label(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(
        FakeCompletedProcess(stdout="project,event,habit\n"),
        FakeCompletedProcess(stdout="✓ Created issue: harness-abc — test title\n  Priority: P2\n"),
    )
    adapter = BeadsAdapter(bd_dir)

    issue_id = adapter.create(
        title="test title",
        scope="professional",
        issue_type="task",
        description="why",
    )

    assert issue_id == "harness-abc"
    # First call is the types config get, second is the create.
    create_call = runner.calls[-1]
    cmd = create_call["cmd"]
    assert cmd[0] == "bd"
    assert cmd[1] == "create"
    # Labels are passed comma-separated via --labels (not --add-label;
    # that flag is update-only).
    assert "--labels" in cmd
    labels_idx = cmd.index("--labels")
    label_values = cmd[labels_idx + 1].split(",")
    assert "scope:professional" in label_values
    # Every call's cwd must pin to the adapter's bd_dir.
    for call in runner.calls:
        assert call["cwd"] == bd_dir


def test_create_rejects_unknown_scope(bd_dir: Path, runner: FakeRunner) -> None:
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(ValueError, match="scope"):
        adapter.create(title="x", scope="social", issue_type="task")
    # Validation runs before any subprocess dispatch.
    assert runner.calls == []


def test_create_rejects_unknown_type(bd_dir: Path, runner: FakeRunner) -> None:
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(ValueError, match="issue_type"):
        adapter.create(title="x", scope="personal", issue_type="bug")
    assert runner.calls == []


def test_create_with_parent_and_deps(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(
        FakeCompletedProcess(stdout="project,event,habit\n"),
        FakeCompletedProcess(stdout="✓ Created issue: harness-new — child\n"),
        FakeCompletedProcess(),  # dep_add #1
        FakeCompletedProcess(),  # dep_add #2
    )
    adapter = BeadsAdapter(bd_dir)

    issue_id = adapter.create(
        title="child",
        scope="personal",
        issue_type="project",
        parent="harness-parent",
        deps=["harness-a", "harness-b"],
    )

    assert issue_id == "harness-new"
    create_cmd = runner.calls[1]["cmd"]
    assert "--parent" in create_cmd
    assert create_cmd[create_cmd.index("--parent") + 1] == "harness-parent"
    dep_cmds = [c["cmd"] for c in runner.calls[2:]]
    assert ["bd", "dep", "add", "harness-new", "harness-a"] in dep_cmds
    assert ["bd", "dep", "add", "harness-new", "harness-b"] in dep_cmds


def test_ensure_custom_types_skips_when_present(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(
        FakeCompletedProcess(stdout="project,event,habit\n"),
    )
    adapter = BeadsAdapter(bd_dir)

    adapter.ensure_custom_types()
    adapter.ensure_custom_types()  # idempotent — should not re-dispatch

    # Exactly one `config get` call, no `config set`.
    assert len(runner.calls) == 1
    assert runner.calls[0]["cmd"][1:] == ["config", "get", "types.custom"]


def test_ensure_custom_types_merges_with_existing(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(
        FakeCompletedProcess(stdout="awaiting_review\n"),
        FakeCompletedProcess(),  # config set
    )
    adapter = BeadsAdapter(bd_dir)

    adapter.ensure_custom_types()

    set_cmd = runner.calls[1]["cmd"]
    assert set_cmd[:3] == ["bd", "config", "set"]
    assert set_cmd[3] == "types.custom"
    merged = set(set_cmd[4].split(","))
    # Pre-existing user value preserved AND ab types added.
    assert "awaiting_review" in merged
    for t in AB_CUSTOM_TYPES:
        assert t in merged


def test_ensure_custom_types_adds_when_empty(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(
        FakeCompletedProcess(stdout=""),
        FakeCompletedProcess(),
    )
    adapter = BeadsAdapter(bd_dir)
    adapter.ensure_custom_types()
    set_cmd = runner.calls[1]["cmd"]
    merged = set(set_cmd[4].split(","))
    assert merged == set(AB_CUSTOM_TYPES)


def test_list_filters_by_scope(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-1",
                "title": "prof item",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": ["scope:professional"],
            },
            {
                "id": "harness-2",
                "title": "pers item",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": ["scope:personal"],
            },
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)

    only_pers = adapter.list_issues(scope="personal")

    assert len(only_pers) == 1
    assert only_pers[0].id == "harness-2"
    assert only_pers[0].scope == "personal"
    list_cmd = runner.calls[0]["cmd"]
    assert list_cmd[:3] == ["bd", "list", "--json"]


def test_ready_returns_empty_list_for_empty_stdout(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout=""))
    adapter = BeadsAdapter(bd_dir)
    assert adapter.ready() == []


def test_show_unwraps_single_issue(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-x",
                "title": "one",
                "status": "open",
                "priority": 3,
                "issue_type": "project",
                "labels": ["scope:professional"],
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)

    issue = adapter.show("harness-x")

    assert isinstance(issue, BeadsIssue)
    assert issue.id == "harness-x"
    assert issue.scope == "professional"


def test_show_raises_when_no_issue(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="[]"))
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(BeadsAdapterError, match="no issue"):
        adapter.show("harness-missing")


def test_close_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.close("harness-x", reason="done")
    cmd = runner.calls[0]["cmd"]
    assert cmd == ["bd", "close", "harness-x", "--reason", "done"]


def test_reopen_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.reopen("harness-x", reason="still relevant")
    cmd = runner.calls[0]["cmd"]
    assert cmd == ["bd", "reopen", "harness-x", "--reason", "still relevant"]


def test_reopen_command_shape_no_reason(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.reopen("harness-x")
    cmd = runner.calls[0]["cmd"]
    assert cmd == ["bd", "reopen", "harness-x"]


def test_delete_command_shape_default(bd_dir: Path, runner: FakeRunner) -> None:
    """bd delete always runs with --force because the agent-driven path
    already gates on a write-tier confirmation. Cascade defaults off."""
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.delete("harness-x")
    cmd = runner.calls[0]["cmd"]
    assert cmd == ["bd", "delete", "harness-x", "--force"]
    assert "--cascade" not in cmd


def test_delete_command_shape_cascade(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.delete("harness-x", cascade=True)
    cmd = runner.calls[0]["cmd"]
    assert cmd == ["bd", "delete", "harness-x", "--force", "--cascade"]


def test_delete_surfaces_bd_error(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(returncode=1, stderr="bd: issue not found"))
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(BeadsAdapterError, match="not found"):
        adapter.delete("harness-missing")


def test_update_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.update("harness-x", priority="1", status="in_progress")
    cmd = runner.calls[0]["cmd"]
    assert cmd[0:2] == ["bd", "update"]
    assert cmd[2] == "harness-x"
    # key args preserved, underscores converted to dashes.
    assert "--priority" in cmd
    assert "--status" in cmd


def test_dep_add_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.dep_add("harness-a", "harness-b")
    assert runner.calls[0]["cmd"] == ["bd", "dep", "add", "harness-a", "harness-b"]


def test_dep_rm_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.dep_rm("harness-a", "harness-b")
    assert runner.calls[0]["cmd"] == ["bd", "dep", "remove", "harness-a", "harness-b"]


def test_search_passes_query_and_flags(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-1",
                "title": "auth refactor",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": ["scope:professional"],
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)

    hits = adapter.search("auth", status="all", limit=5)

    assert len(hits) == 1
    assert hits[0].id == "harness-1"
    cmd = runner.calls[0]["cmd"]
    assert cmd[:3] == ["bd", "search", "auth"]
    assert "--json" in cmd
    assert cmd[cmd.index("--status") + 1] == "all"
    assert cmd[cmd.index("--limit") + 1] == "5"


def test_search_rejects_empty_query(bd_dir: Path, runner: FakeRunner) -> None:
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(ValueError, match="non-empty"):
        adapter.search("   ")
    assert runner.calls == []


def test_list_passes_priority_and_type(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="[]"))
    adapter = BeadsAdapter(bd_dir)

    adapter.list_issues(status="open", priority="2", issue_type="task", limit=10)

    cmd = runner.calls[0]["cmd"]
    assert cmd[:3] == ["bd", "list", "--json"]
    assert cmd[cmd.index("--status") + 1] == "open"
    assert cmd[cmd.index("--priority") + 1] == "2"
    assert cmd[cmd.index("--type") + 1] == "task"
    assert cmd[cmd.index("--limit") + 1] == "10"


def test_forget_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.forget("dolt-phantoms")
    assert runner.calls[0]["cmd"] == ["bd", "forget", "dolt-phantoms"]


def test_forget_rejects_empty_key(bd_dir: Path, runner: FakeRunner) -> None:
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(ValueError, match="non-empty"):
        adapter.forget("   ")
    assert runner.calls == []


def test_memories_passes_query(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="memory: dolt phantoms\n"))
    adapter = BeadsAdapter(bd_dir)
    out = adapter.memories("dolt")
    assert "dolt phantoms" in out
    assert runner.calls[0]["cmd"] == ["bd", "memories", "dolt"]


def test_label_add_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.label_add("harness-x", "tech-debt")
    assert runner.calls[0]["cmd"] == ["bd", "label", "add", "harness-x", "tech-debt"]


def test_label_rm_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.label_rm("harness-x", "tech-debt")
    assert runner.calls[0]["cmd"] == ["bd", "label", "remove", "harness-x", "tech-debt"]


def test_label_rejects_empty_label(bd_dir: Path, runner: FakeRunner) -> None:
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(ValueError, match="non-empty"):
        adapter.label_add("harness-x", "   ")
    assert runner.calls == []


def test_label_list_returns_stdout(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="scope:professional\ntech-debt\n"))
    adapter = BeadsAdapter(bd_dir)
    out = adapter.label_list("harness-x")
    assert "tech-debt" in out
    assert runner.calls[0]["cmd"] == ["bd", "label", "list", "harness-x"]


def test_comment_add_command_shape(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)
    adapter.comment_add("harness-x", "looking into this")
    assert runner.calls[0]["cmd"] == ["bd", "comments", "add", "harness-x", "looking into this"]


def test_comments_list_returns_stdout(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="2026-04-18: looking into this\n"))
    adapter = BeadsAdapter(bd_dir)
    out = adapter.comments_list("harness-x")
    assert "looking into this" in out
    assert runner.calls[0]["cmd"] == ["bd", "comments", "harness-x"]


def test_comment_add_rejects_empty_text(bd_dir: Path, runner: FakeRunner) -> None:
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(ValueError, match="non-empty"):
        adapter.comment_add("harness-x", "   ")
    assert runner.calls == []


def test_find_duplicates_parses_json(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "a_id": "harness-a",
                "b_id": "harness-b",
                "a_title": "auth refactor",
                "b_title": "refactor the auth path",
                "similarity": 0.78,
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)

    pairs = adapter.find_duplicates(threshold=0.4, limit=10, status="open")

    assert len(pairs) == 1
    assert pairs[0]["a_id"] == "harness-a"
    cmd = runner.calls[0]["cmd"]
    assert cmd[:2] == ["bd", "find-duplicates"]
    assert "--json" in cmd
    # Mechanical method is forced; AI method would leak to cloud.
    assert cmd[cmd.index("--method") + 1] == "mechanical"
    assert cmd[cmd.index("--threshold") + 1] == "0.4"
    assert cmd[cmd.index("--limit") + 1] == "10"
    assert cmd[cmd.index("--status") + 1] == "open"


def test_find_duplicates_empty_stdout_returns_empty_list(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout=""))
    adapter = BeadsAdapter(bd_dir)
    assert adapter.find_duplicates() == []


def test_verify_raises_when_bd_not_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".beads").mkdir()
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: None)
    adapter = BeadsAdapter(tmp_path)
    with pytest.raises(BeadsAdapterError, match="not found"):
        adapter.verify()


def test_verify_raises_when_dir_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "not_there"
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/u/b/bd")
    adapter = BeadsAdapter(missing)
    with pytest.raises(BeadsAdapterError, match="does not exist"):
        adapter.verify()


def test_verify_raises_when_beads_subdir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/u/b/bd")
    adapter = BeadsAdapter(tmp_path)
    with pytest.raises(BeadsAdapterError, match=r"\.beads"):
        adapter.verify()


def test_verify_raises_when_server_unreachable(
    bd_dir: Path, runner: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verify() by default probes `bd dolt test`. Non-zero exit from
    the probe must surface a repair hint pointing the user at
    `bd dolt start`."""
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/u/b/bd")
    runner.queue(FakeCompletedProcess(returncode=1, stderr="connection refused"))
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(BeadsAdapterError, match="bd dolt start"):
        adapter.verify()
    # The probe went to `bd dolt test` (not some other subcommand).
    assert runner.calls[0]["cmd"][-2:] == ["dolt", "test"]


def test_verify_passes_when_server_reachable(
    bd_dir: Path, runner: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/u/b/bd")
    runner.queue(FakeCompletedProcess(returncode=0, stdout="✓ Connection successful"))
    adapter = BeadsAdapter(bd_dir)
    adapter.verify()  # should not raise


def test_verify_check_server_false_skips_probe(
    bd_dir: Path, runner: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check_server=False is the escape hatch for test paths that have
    already stubbed subprocess and don't need a second round-trip."""
    monkeypatch.setattr("harness.store.bd_adapter.shutil.which", lambda _: "/u/b/bd")
    adapter = BeadsAdapter(bd_dir)
    adapter.verify(check_server=False)
    # No subprocess calls — server probe skipped.
    assert runner.calls == []


def test_run_raises_beads_error_when_bd_missing(
    bd_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("bd not found")

    monkeypatch.setattr("harness.store.bd_adapter.subprocess.run", boom)
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(BeadsAdapterError, match="not found"):
        adapter.close("harness-x")


def test_run_raises_on_nonzero_exit(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(returncode=1, stderr="bd: issue not found"))
    adapter = BeadsAdapter(bd_dir)
    with pytest.raises(BeadsAdapterError, match="not found"):
        adapter.close("harness-missing")


def test_extract_created_id_handles_varied_shapes() -> None:
    assert _extract_created_id("✓ Created issue: harness-abc — title\n") == "harness-abc"
    # Without the em-dash / title.
    assert _extract_created_id("✓ Created issue: harness-xyz") == "harness-xyz"
    # Multi-line with noise above and below.
    out = "Some preamble\n✓ Created issue: harness-def — neat title\n  Priority: P2\n"
    assert _extract_created_id(out) == "harness-def"
    # Empty / non-matching.
    assert _extract_created_id("") is None
    assert _extract_created_id("nothing to see") is None


def test_allowed_scopes_constant() -> None:
    assert ALLOWED_SCOPES == ("professional", "personal")


def test_beads_issue_scope_returns_none_when_unlabeled() -> None:
    issue = BeadsIssue(
        id="harness-x",
        title="t",
        status="open",
        priority=2,
        issue_type="task",
        labels=(),
        raw={},
    )
    assert issue.scope is None


def test_beads_issue_parses_assignee_from_json(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-a",
                "title": "own thought",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": ["scope:personal"],
                "assignee": "airton_b",
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)

    issue = adapter.show("harness-a")

    assert issue.assignee == "airton_b"


def test_beads_issue_assignee_defaults_none_when_absent() -> None:
    issue = BeadsIssue(
        id="harness-x",
        title="t",
        status="open",
        priority=2,
        issue_type="task",
        labels=(),
        raw={},
    )
    assert issue.assignee is None


def test_beads_issue_assignee_empty_string_normalizes_to_none(
    bd_dir: Path, runner: FakeRunner
) -> None:
    """bd sometimes emits an empty-string assignee for unassigned beads;
    normalize to None so `issue.assignee is None` reads cleanly in
    downstream filters."""
    payload = json.dumps(
        [
            {
                "id": "harness-u",
                "title": "unassigned",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "",
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)

    issue = adapter.show("harness-u")

    assert issue.assignee is None


def test_create_passes_assignee_flag(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(
        FakeCompletedProcess(stdout="project,event,habit\n"),
        FakeCompletedProcess(stdout="✓ Created issue: harness-own — own thought\n"),
    )
    adapter = BeadsAdapter(bd_dir)

    issue_id = adapter.create(
        title="own thought",
        scope="personal",
        assignee="airton_b",
    )

    assert issue_id == "harness-own"
    create_cmd = runner.calls[-1]["cmd"]
    assert "--assignee" in create_cmd
    assert create_cmd[create_cmd.index("--assignee") + 1] == "airton_b"


def test_create_omits_assignee_flag_when_unset(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(
        FakeCompletedProcess(stdout="project,event,habit\n"),
        FakeCompletedProcess(stdout="✓ Created issue: harness-anon — untagged\n"),
    )
    adapter = BeadsAdapter(bd_dir)

    adapter.create(title="untagged", scope="personal")

    create_cmd = runner.calls[-1]["cmd"]
    assert "--assignee" not in create_cmd


def test_list_passes_assignee_flag(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="[]"))
    adapter = BeadsAdapter(bd_dir)

    adapter.list_issues(assignee="airton_b")

    cmd = runner.calls[0]["cmd"]
    assert cmd[cmd.index("--assignee") + 1] == "airton_b"


def test_ready_passes_assignee_flag(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="[]"))
    adapter = BeadsAdapter(bd_dir)

    adapter.ready(assignee="airton_b")

    cmd = runner.calls[0]["cmd"]
    assert cmd[cmd.index("--assignee") + 1] == "airton_b"


def test_search_passes_assignee_flag(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="[]"))
    adapter = BeadsAdapter(bd_dir)

    adapter.search("focus", assignee="airton_b")

    cmd = runner.calls[0]["cmd"]
    assert cmd[cmd.index("--assignee") + 1] == "airton_b"


def test_default_exclude_drops_matching_assignee_from_list(
    bd_dir: Path, runner: FakeRunner
) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-user",
                "title": "user work",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "mark",
            },
            {
                "id": "harness-ab",
                "title": "ab internal",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
            },
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir, default_exclude_assignee="airton_b")

    issues = adapter.list_issues()

    assert [i.id for i in issues] == ["harness-user"]


def test_default_exclude_skipped_when_explicit_assignee_passed(
    bd_dir: Path, runner: FakeRunner
) -> None:
    """A positive `assignee=X` filter opts in explicitly — the default
    exclude gets out of the way. Without this, querying for
    airton_b-owned beads on an adapter that defaults to hiding them
    would return empty."""
    payload = json.dumps(
        [
            {
                "id": "harness-ab",
                "title": "ab internal",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir, default_exclude_assignee="airton_b")

    issues = adapter.list_issues(assignee="airton_b")

    assert [i.id for i in issues] == ["harness-ab"]


def test_default_exclude_applies_to_ready(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-ab",
                "title": "ab internal",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir, default_exclude_assignee="airton_b")

    assert adapter.ready() == []


def test_default_exclude_applies_to_search(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-ab",
                "title": "ab internal",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir, default_exclude_assignee="airton_b")

    assert adapter.search("foo") == []


def test_default_exclude_applies_to_stale(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-ab",
                "title": "ab internal",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
            },
            {
                "id": "harness-user",
                "title": "user",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "mark",
            },
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir, default_exclude_assignee="airton_b")

    issues = adapter.stale()

    assert [i.id for i in issues] == ["harness-user"]


def test_no_default_exclude_returns_everything(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-ab",
                "title": "ab internal",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
            },
            {
                "id": "harness-user",
                "title": "user",
                "status": "open",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "mark",
            },
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)  # no default_exclude_assignee

    issues = adapter.list_issues()

    assert {i.id for i in issues} == {"harness-ab", "harness-user"}


def test_get_focus_returns_none_when_no_in_progress(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(FakeCompletedProcess(stdout="[]"))
    adapter = BeadsAdapter(bd_dir)

    focus = adapter.get_focus("airton_b")

    assert focus is None
    cmd = runner.calls[0]["cmd"]
    assert cmd[:3] == ["bd", "list", "--json"]
    assert cmd[cmd.index("--status") + 1] == "in_progress"
    assert cmd[cmd.index("--assignee") + 1] == "airton_b"


def test_get_focus_returns_single_in_progress(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-f",
                "title": "focused",
                "status": "in_progress",
                "priority": 2,
                "issue_type": "task",
                "labels": ["scope:personal"],
                "assignee": "airton_b",
                "updated_at": "2026-04-18T10:00:00Z",
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)

    focus = adapter.get_focus("airton_b")

    assert focus is not None
    assert focus.id == "harness-f"


def test_get_focus_warns_and_demotes_when_multiple(bd_dir: Path, runner: FakeRunner) -> None:
    """Lazy reconciliation: if bd's state has >1 in_progress for the
    same assignee (crash mid-switch, manual edit) get_focus keeps the
    most-recently-updated and demotes the rest. RuntimeWarning fires."""
    payload = json.dumps(
        [
            {
                "id": "harness-old",
                "title": "stale focus",
                "status": "in_progress",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
                "updated_at": "2026-04-18T08:00:00Z",
            },
            {
                "id": "harness-new",
                "title": "recent focus",
                "status": "in_progress",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
                "updated_at": "2026-04-18T12:00:00Z",
            },
            {
                "id": "harness-mid",
                "title": "middle focus",
                "status": "in_progress",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
                "updated_at": "2026-04-18T10:00:00Z",
            },
        ]
    )
    runner.queue(
        FakeCompletedProcess(stdout=payload),
        FakeCompletedProcess(),  # demote harness-mid
        FakeCompletedProcess(),  # demote harness-old
    )
    adapter = BeadsAdapter(bd_dir)

    with pytest.warns(RuntimeWarning, match="keeping most-recent harness-new"):
        focus = adapter.get_focus("airton_b")

    assert focus is not None
    assert focus.id == "harness-new"
    demote_cmds = [c["cmd"] for c in runner.calls[1:]]
    assert ["bd", "update", "harness-mid", "--status", "open"] in demote_cmds
    assert ["bd", "update", "harness-old", "--status", "open"] in demote_cmds
    # harness-new (most-recent) is never demoted.
    for cmd in demote_cmds:
        assert "harness-new" not in cmd


def test_set_focus_promotes_when_no_prior(bd_dir: Path, runner: FakeRunner) -> None:
    runner.queue(
        FakeCompletedProcess(stdout="[]"),  # get_focus sees none
        FakeCompletedProcess(),  # promote issue_id
    )
    adapter = BeadsAdapter(bd_dir)

    prior = adapter.set_focus("harness-new", assignee="airton_b")

    assert prior is None
    promote_cmd = runner.calls[-1]["cmd"]
    assert promote_cmd == ["bd", "update", "harness-new", "--status", "in_progress"]


def test_set_focus_demotes_prior_then_promotes_new(bd_dir: Path, runner: FakeRunner) -> None:
    payload = json.dumps(
        [
            {
                "id": "harness-old",
                "title": "prior focus",
                "status": "in_progress",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
                "updated_at": "2026-04-18T10:00:00Z",
            }
        ]
    )
    runner.queue(
        FakeCompletedProcess(stdout=payload),  # get_focus returns prior
        FakeCompletedProcess(),  # demote prior
        FakeCompletedProcess(),  # promote new
    )
    adapter = BeadsAdapter(bd_dir)

    prior_id = adapter.set_focus("harness-new", assignee="airton_b")

    assert prior_id == "harness-old"
    cmds = [c["cmd"] for c in runner.calls]
    assert cmds[1] == ["bd", "update", "harness-old", "--status", "open"]
    assert cmds[2] == ["bd", "update", "harness-new", "--status", "in_progress"]


def test_set_focus_noop_when_already_focused(bd_dir: Path, runner: FakeRunner) -> None:
    """set_focus to the currently-focused id should not demote or
    re-promote — it's already in the right state."""
    payload = json.dumps(
        [
            {
                "id": "harness-same",
                "title": "already focused",
                "status": "in_progress",
                "priority": 2,
                "issue_type": "task",
                "labels": [],
                "assignee": "airton_b",
                "updated_at": "2026-04-18T10:00:00Z",
            }
        ]
    )
    runner.queue(FakeCompletedProcess(stdout=payload))
    adapter = BeadsAdapter(bd_dir)

    prior_id = adapter.set_focus("harness-same", assignee="airton_b")

    assert prior_id is None
    # Only the get_focus list call; no update dispatched.
    assert len(runner.calls) == 1
    assert runner.calls[0]["cmd"][:3] == ["bd", "list", "--json"]


def test_update_accepts_assignee_kwarg(bd_dir: Path, runner: FakeRunner) -> None:
    """update() dispatches kwargs as --flag pairs; assignee comes along
    for free via the generic path. Pin it so the convention doesn't
    silently regress."""
    runner.queue(FakeCompletedProcess())
    adapter = BeadsAdapter(bd_dir)

    adapter.update("harness-x", assignee="airton_b")

    cmd = runner.calls[0]["cmd"]
    assert cmd[:3] == ["bd", "update", "harness-x"]
    assert cmd[cmd.index("--assignee") + 1] == "airton_b"
