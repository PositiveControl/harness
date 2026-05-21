"""Tests for src/harness/driver/cli.py — harness-s83d.

Uses Typer's CliRunner to exercise the `harness drive plan` / `harness
drive loop` subcommands without spawning subprocesses. Heavy components
(LLM adapter, bd CLI, run_loop / commit_plan) are monkeypatched at the
module boundary so tests run fast and offline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from harness.driver import cli as cli_mod
from harness.driver.cli import drive_app
from harness.driver.loop import LoopResult
from harness.driver.planner import PlanDraft, PlannerError
from harness.driver.state import LoopRunState

runner = CliRunner()


# --- fixtures --------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_make_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub make_adapter so CLI tests don't try to load MLX. The driver
    CLI passes the adapter through; tests that monkeypatch run_loop /
    run_planner ignore the adapter object."""
    monkeypatch.setattr("harness.driver.cli.make_adapter", lambda _name: object())


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    # Initialize an empty git repo so the dirty-tree check has something
    # consistent to query (no uncommitted changes by default).
    subprocess.run(
        ["git", "init"],  # noqa: S607
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    return tmp_path


# --- harness drive plan --------------------------------------------


def test_drive_plan_runs_and_writes_draft(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    spec = workspace / "spec.md"
    spec.write_text("some content here that the spec contains")
    draft_path = workspace / "plan-draft.yaml"

    def fake_run_planner(_adapter: Any, config: Any, **_kwargs: Any) -> PlanDraft:
        return PlanDraft(
            epic_title=config.epic_title,
            epic_description="from fake planner",
            items=[],
        )

    monkeypatch.setattr("harness.driver.cli.run_planner", fake_run_planner)
    result = runner.invoke(
        drive_app,
        [
            "plan",
            "--spec",
            str(spec),
            "--epic-title",
            "fake workplan",
            "--draft",
            str(draft_path),
            "--workspace",
            str(workspace),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wrote plan draft" in result.stdout
    assert draft_path.exists()


def test_drive_plan_requires_epic_title_when_running_planner(
    workspace: Path,
) -> None:
    spec = workspace / "spec.md"
    spec.write_text("content")
    result = runner.invoke(
        drive_app,
        ["plan", "--spec", str(spec), "--workspace", str(workspace)],
    )
    assert result.exit_code != 0
    assert "epic-title" in result.stderr.lower() or "epic-title" in result.output.lower()


def test_drive_plan_commit_mode_calls_commit_plan(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    spec = workspace / "spec.md"
    spec.write_text("content")
    draft_path = workspace / "draft.yaml"
    draft_path.write_text("epic_title: t\nepic_description: d\nitems: []\n")

    called: dict[str, Any] = {}

    def fake_commit_plan(draft: Path, _bd: Any, spec_arg: Path) -> str:
        called["draft"] = draft
        called["spec"] = spec_arg
        return "harness-new-epic"

    monkeypatch.setattr("harness.driver.cli.commit_plan", fake_commit_plan)
    result = runner.invoke(
        drive_app,
        [
            "plan",
            "--commit",
            "--draft",
            str(draft_path),
            "--spec",
            str(spec),
            "--workspace",
            str(workspace),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "harness-new-epic" in result.stdout
    assert called["draft"] == draft_path
    assert called["spec"] == spec


def test_drive_plan_commit_surfaces_planner_error(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    spec = workspace / "spec.md"
    spec.write_text("content")
    draft_path = workspace / "draft.yaml"
    draft_path.write_text("epic_title: t\nepic_description: d\nitems: []\n")

    def bad_commit(*_args: Any, **_kwargs: Any) -> str:
        raise PlannerError("item #1: spec_quote not found in spec.md")

    monkeypatch.setattr("harness.driver.cli.commit_plan", bad_commit)
    result = runner.invoke(
        drive_app,
        [
            "plan",
            "--commit",
            "--draft",
            str(draft_path),
            "--spec",
            str(spec),
            "--workspace",
            str(workspace),
        ],
    )
    assert result.exit_code == 2
    assert "spec_quote not found" in result.stderr


# --- harness drive loop --------------------------------------------


def _stub_run_loop(monkeypatch: pytest.MonkeyPatch, result: LoopResult) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run_loop(_adapter: Any, _bd: Any, config: Any) -> LoopResult:
        captured["config"] = config
        return result

    monkeypatch.setattr("harness.driver.cli.run_loop", fake_run_loop)
    return captured


def _stub_load_character(monkeypatch: pytest.MonkeyPatch) -> None:
    class _C:
        name = "fake"

        def system_prompt(self, *, include_samples: Any = ()) -> str:
            return "you are an executor."

    monkeypatch.setattr("harness.driver.cli.load_character", lambda _p: _C())


def test_drive_loop_runs_against_epic(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    _stub_load_character(monkeypatch)
    captured = _stub_run_loop(
        monkeypatch,
        LoopResult(
            loop_run_id="abc12345",
            epic_id="harness-e9oq",
            closed=["harness-a"],
            halted_on=None,
            turns_used=1,
            exit_reason="success",
        ),
    )
    result = runner.invoke(
        drive_app,
        [
            "loop",
            "--epic",
            "harness-e9oq",
            "--workspace",
            str(workspace),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "exit=success" in result.stdout
    assert captured["config"].epic_id == "harness-e9oq"


def test_drive_loop_halted_exits_2_and_prints_to_stderr(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    _stub_load_character(monkeypatch)
    _stub_run_loop(
        monkeypatch,
        LoopResult(
            loop_run_id="abc12345",
            epic_id="harness-e9oq",
            closed=[],
            halted_on="harness-a",
            turns_used=2,
            exit_reason="halted",
        ),
    )
    result = runner.invoke(
        drive_app,
        ["loop", "--epic", "harness-e9oq", "--workspace", str(workspace)],
    )
    assert result.exit_code == 2
    assert "HALTED on harness-a" in result.stderr


def test_drive_loop_interrupted_exits_1(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    _stub_load_character(monkeypatch)
    _stub_run_loop(
        monkeypatch,
        LoopResult(
            loop_run_id="abc12345",
            epic_id="harness-e9oq",
            closed=[],
            halted_on=None,
            turns_used=1,
            exit_reason="interrupted",
        ),
    )
    result = runner.invoke(
        drive_app,
        ["loop", "--epic", "harness-e9oq", "--workspace", str(workspace)],
    )
    assert result.exit_code == 1
    assert "INTERRUPTED" in result.stderr
    assert "abc12345" in result.stderr


def test_drive_loop_rejects_dirty_tree_by_default(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    _stub_load_character(monkeypatch)
    monkeypatch.setattr(
        "harness.driver.cli.run_loop", lambda *a, **k: pytest.fail("should not run")
    )
    # Force the dirty-tree check positive without touching the real fs.
    monkeypatch.setattr("harness.driver.cli._git_tree_is_dirty", lambda _ws: True)
    result = runner.invoke(
        drive_app,
        ["loop", "--epic", "harness-e9oq", "--workspace", str(workspace)],
    )
    assert result.exit_code == 2
    assert "uncommitted changes" in result.stderr
    assert "--allow-dirty" in result.stderr


def test_drive_loop_allow_dirty_overrides(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    _stub_load_character(monkeypatch)
    _stub_run_loop(
        monkeypatch,
        LoopResult(
            loop_run_id="abc12345",
            epic_id="harness-e9oq",
            closed=[],
            halted_on=None,
            turns_used=0,
            exit_reason="success",
        ),
    )
    monkeypatch.setattr("harness.driver.cli._git_tree_is_dirty", lambda _ws: True)
    result = runner.invoke(
        drive_app,
        [
            "loop",
            "--epic",
            "harness-e9oq",
            "--workspace",
            str(workspace),
            "--allow-dirty",
        ],
    )
    assert result.exit_code == 0, result.output


def test_drive_loop_dry_run_renders_handoff_to_stdout(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """harness-2pj3: --dry-run must render the assembled handoff so the
    operator can inspect it before burning real turns. Previously the
    handoff lived in LoopResult.handoffs[0] but was never printed."""
    from harness.driver.handoff import Handoff

    _stub_load_character(monkeypatch)
    handoff = Handoff(
        loop_run_id="abc12345",
        epic_id="harness-e9oq",
        current_issue="harness-x (P2 task)\nTitle: dry-run smoke",
        parent_epic_summary="harness-e9oq — [epic] test",
        files_touched=(),
        closed_this_run=(),
        decisions=(),
        observations=(),
        open_questions=(),
        prior_attempt_failure=None,
    )
    _stub_run_loop(
        monkeypatch,
        LoopResult(
            loop_run_id="abc12345",
            epic_id="harness-e9oq",
            closed=[],
            halted_on=None,
            turns_used=0,
            exit_reason="dry_run",
            handoffs=[handoff],
        ),
    )
    result = runner.invoke(
        drive_app,
        [
            "loop",
            "--epic",
            "harness-e9oq",
            "--workspace",
            str(workspace),
            "--dry-run",
            "--allow-dirty",
        ],
    )
    assert result.exit_code == 0, result.output
    # Summary line still present.
    assert "exit=dry_run" in result.stdout
    # Handoff block appears.
    assert "[SESSION HANDOFF — loop_run=abc12345 epic=harness-e9oq]" in result.stdout
    assert "harness-x (P2 task)" in result.stdout
    assert "[END HANDOFF]" in result.stdout


def test_drive_loop_rejects_epic_and_resume_together(workspace: Path) -> None:
    result = runner.invoke(
        drive_app,
        [
            "loop",
            "--epic",
            "x",
            "--resume",
            "y",
            "--workspace",
            str(workspace),
        ],
    )
    assert result.exit_code != 0


def test_drive_loop_requires_epic_or_resume(workspace: Path) -> None:
    result = runner.invoke(drive_app, ["loop", "--workspace", str(workspace)])
    assert result.exit_code != 0


# --- --list-runs -----------------------------------------------------


def test_drive_loop_list_runs_empty(workspace: Path) -> None:
    result = runner.invoke(
        drive_app,
        ["loop", "--list-runs", "--workspace", str(workspace)],
    )
    assert result.exit_code == 0
    assert "no loop runs yet" in result.stdout


def test_drive_loop_list_runs_enumerates_state_files(
    workspace: Path,
) -> None:
    s = LoopRunState.fresh(epic_id="harness-e9oq", max_turns=5, started_at_sha="sha")
    s.closed_this_run.extend(["harness-a", "harness-b"])
    s.turns_used = 2
    s.save(LoopRunState.state_path(workspace, s.loop_run_id))

    result = runner.invoke(
        drive_app,
        ["loop", "--list-runs", "--workspace", str(workspace)],
    )
    assert result.exit_code == 0
    assert s.loop_run_id in result.stdout
    assert "epic=harness-e9oq" in result.stdout
    assert "turns=2" in result.stdout
    assert "closed=2" in result.stdout


# --- helpers ---------------------------------------------------------


def test_git_tree_dirty_check_runs_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real _git_tree_is_dirty shells out to git diff (tracked-only,
    no untracked noise — harness-m0g4)."""
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("harness.driver.cli.subprocess.run", fake_run)
    assert not cli_mod._git_tree_is_dirty(tmp_path)
    assert calls == [["git", "diff", "--name-only", "HEAD"]]


def test_git_tree_dirty_check_returns_false_when_git_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("no git")

    monkeypatch.setattr("harness.driver.cli.subprocess.run", boom)
    assert not cli_mod._git_tree_is_dirty(tmp_path)


def test_git_tree_dirty_check_ignores_beads_only_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-m0g4: .beads/ files are bd-managed state. The executor
    writes to them via shell `bd close <id>` every successful turn —
    treating them as user-facing dirty would gate every loop run."""
    diff_output = ".beads/export-state.json\n.beads/interactions.jsonl\n.beads/issues.jsonl\n"

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout=diff_output, stderr=""
        )

    monkeypatch.setattr("harness.driver.cli.subprocess.run", fake_run)
    assert not cli_mod._git_tree_is_dirty(tmp_path)


def test_git_tree_dirty_check_trips_on_non_beads_tracked_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-m0g4: a non-.beads/ tracked modification still trips the
    gate. Mixing .beads/ noise with a real change must NOT mask the
    real change."""
    diff_output = ".beads/export-state.json\nsrc/harness/driver/cli.py\n"

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout=diff_output, stderr=""
        )

    monkeypatch.setattr("harness.driver.cli.subprocess.run", fake_run)
    assert cli_mod._git_tree_is_dirty(tmp_path)


def test_git_tree_dirty_check_ignores_untracked_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-m0g4: untracked files (`.coverage`, scratch outputs) are
    invisible to the handoff's `git diff --name-status` and shouldn't
    gate. `git diff --name-only HEAD` excludes them by design — verify
    the empty-stdout path returns False."""

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Untracked files don't appear in `git diff --name-only HEAD`
        # — they appear in `git status --porcelain`, which we no longer use.
        return subprocess.CompletedProcess(args=["git"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("harness.driver.cli.subprocess.run", fake_run)
    assert not cli_mod._git_tree_is_dirty(tmp_path)


def test_validate_model_rejects_unknown_name() -> None:
    import typer as typer_mod

    with pytest.raises(typer_mod.BadParameter):
        cli_mod._validate_model("gpt-4")


def test_validate_model_accepts_known_names() -> None:
    assert cli_mod._validate_model("echo") == "echo"
    assert cli_mod._validate_model("mlx") == "mlx"
    assert cli_mod._validate_model("ollama") == "ollama"
