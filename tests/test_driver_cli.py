"""Tests for src/harness/driver/cli.py — harness-s83d.

Uses Typer's CliRunner to exercise the `harness drive plan` / `harness
drive loop` subcommands without spawning subprocesses. Heavy components
(LLM adapter, bd CLI, run_loop / commit_plan) are monkeypatched at the
module boundary so tests run fast and offline.
"""

from __future__ import annotations

import os
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


def _clean_git_env() -> dict[str, str]:
    """Strip ``GIT_*`` env vars before invoking git in tmp_path. The
    pre-push hook sets ``GIT_DIR`` / ``GIT_WORK_TREE`` so its own git
    operations talk to the outer repo; child subprocesses inherit the
    env by default, which would silently override our ``cwd=tmp_path``.
    Same fix as ``tests/test_git_tools.py``."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


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
        env=_clean_git_env(),
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
    # Default per-issue retry budget when --max-attempts is omitted.
    assert captured["config"].max_attempts_per_issue == 3


def test_drive_loop_max_attempts_flag_flows_into_config(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """harness-8fvh: --max-attempts overrides the per-issue retry budget
    (previously fixed at LoopConfig's default of 3)."""
    _stub_load_character(monkeypatch)
    captured = _stub_run_loop(
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
    result = runner.invoke(
        drive_app,
        [
            "loop",
            "--epic",
            "harness-e9oq",
            "--workspace",
            str(workspace),
            "--max-attempts",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["config"].max_attempts_per_issue == 5


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


def test_drive_loop_partial_exits_1(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    """harness-iljv: a run that parked issues exits "partial" → exit
    code 1 (not 0), and the summary names the stranded parked issues so
    the operator knows the epic isn't actually done."""
    _stub_load_character(monkeypatch)
    _stub_run_loop(
        monkeypatch,
        LoopResult(
            loop_run_id="abc12345",
            epic_id="harness-e9oq",
            closed=["harness-b"],
            halted_on=None,
            turns_used=4,
            exit_reason="partial",
            parked_issues=["harness-a", "harness-c"],
        ),
    )
    result = runner.invoke(
        drive_app,
        ["loop", "--epic", "harness-e9oq", "--workspace", str(workspace)],
    )
    assert result.exit_code == 1
    assert "PARTIAL" in result.stderr
    assert "harness-a" in result.stderr
    assert "harness-c" in result.stderr


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


def test_resolve_driver_adapter_passes_model_repo_to_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harness-gu6k: --model-repo with --model mlx should instantiate
    MLXAdapter with repo=<value>, NOT go through make_adapter."""
    seen: dict[str, Any] = {}

    class _FakeMLX:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    import harness.model.mlx as mlx_mod

    monkeypatch.setattr(mlx_mod, "MLXAdapter", _FakeMLX)
    cli_mod._resolve_driver_adapter(
        "mlx",
        model_repo="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2",
        lora_path=None,
        draft_repo=None,
    )
    assert seen == {"repo": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-dwq-v2"}


def test_resolve_driver_adapter_rejects_lora_with_ollama() -> None:
    """harness-gu6k: --lora-path is MLX-only; raise on mismatch."""
    import typer as typer_mod

    with pytest.raises(typer_mod.BadParameter, match="lora-path requires --model mlx"):
        cli_mod._resolve_driver_adapter(
            "ollama",
            model_repo=None,
            lora_path="/some/lora",
            draft_repo=None,
        )


def test_resolve_driver_adapter_rejects_draft_with_echo() -> None:
    """harness-gu6k: --draft-repo is MLX-only; --model echo can't use it."""
    import typer as typer_mod

    with pytest.raises(typer_mod.BadParameter, match="draft-repo requires --model mlx"):
        cli_mod._resolve_driver_adapter(
            "echo",
            model_repo=None,
            lora_path=None,
            draft_repo="some/draft-repo",
        )


def test_resolve_driver_adapter_default_path_calls_make_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harness-gu6k: without --model-repo / --lora-path / --draft-repo,
    fall through to the factory's default path."""
    called: list[str] = []

    def fake_make_adapter(name: str) -> object:
        called.append(name)
        return object()

    monkeypatch.setattr("harness.driver.cli.make_adapter", fake_make_adapter)
    cli_mod._resolve_driver_adapter("mlx", model_repo=None, lora_path=None, draft_repo=None)
    assert called == ["mlx"]


def test_validate_model_rejects_unknown_name() -> None:
    import typer as typer_mod

    with pytest.raises(typer_mod.BadParameter):
        cli_mod._validate_model("gpt-4")


def test_validate_model_accepts_known_names() -> None:
    assert cli_mod._validate_model("echo") == "echo"
    assert cli_mod._validate_model("mlx") == "mlx"
    assert cli_mod._validate_model("ollama") == "ollama"
    assert cli_mod._validate_model("vllm") == "vllm"


def test_resolve_driver_adapter_passes_base_url_to_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harness-zie7: --model vllm + --model-repo should instantiate
    VllmAdapter with base_url=<value>, mirroring cli._resolve_adapter."""
    seen: dict[str, Any] = {}

    class _FakeVllm:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    import harness.model.vllm as vllm_mod

    monkeypatch.setattr(vllm_mod, "VllmAdapter", _FakeVllm)
    cli_mod._resolve_driver_adapter(
        "vllm",
        model_repo="http://gx10-5fb9:8000/v1",
        lora_path=None,
        draft_repo=None,
    )
    assert seen == {"base_url": "http://gx10-5fb9:8000/v1"}


# --- harness drive logs (harness-830a) ------------------------------


def _seed_loop_run(workspace: Path, run_id: str, started: str) -> None:
    """Build a fake loop-run on disk: .json + .log + _workspace.tar.gz
    so the prune machinery has all three files to relocate."""
    from datetime import datetime

    state_dir = LoopRunState.state_dir(workspace)
    state_dir.mkdir(parents=True, exist_ok=True)
    state = LoopRunState(
        loop_run_id=run_id,
        started_at_sha="deadbeef",
        started_at=datetime.fromisoformat(started),
        epic_id="harness-epic",
        max_turns=5,
        turns_used=1,
    )
    state.save(state_dir / f"{run_id}.json")
    (state_dir / f"{run_id}.log").write_text(f"turn 1 | {run_id} | round_start\n" * 5)
    (state_dir / f"{run_id}_workspace.tar.gz").write_bytes(b"\x1f\x8b\x08fake")


def test_logs_list_shows_runs_newest_first(tmp_path: Path) -> None:
    """harness-830a: `harness drive logs list` enumerates runs by
    started_at descending — newest first matches what the operator
    cares about (most recent run + its halt reason)."""
    _seed_loop_run(tmp_path, "oldest", "2026-05-20T00:00:00+00:00")
    _seed_loop_run(tmp_path, "newest", "2026-05-22T00:00:00+00:00")
    _seed_loop_run(tmp_path, "middle", "2026-05-21T00:00:00+00:00")

    result = CliRunner().invoke(drive_app, ["logs", "list", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line]
    ids = [line.split()[0] for line in lines]
    assert ids == ["newest", "middle", "oldest"]


def test_logs_prune_archives_older_runs(tmp_path: Path) -> None:
    """harness-830a: prune --keep N moves the oldest len-N runs into
    archive/ and gzips their .log. The N newest stay in the active
    dir untouched."""
    _seed_loop_run(tmp_path, "oldest", "2026-05-20T00:00:00+00:00")
    _seed_loop_run(tmp_path, "middle", "2026-05-21T00:00:00+00:00")
    _seed_loop_run(tmp_path, "newest", "2026-05-22T00:00:00+00:00")

    result = CliRunner().invoke(
        drive_app, ["logs", "prune", "--workspace", str(tmp_path), "--keep", "1"]
    )
    assert result.exit_code == 0
    assert "oldest" in result.stdout
    assert "middle" in result.stdout
    assert "newest" not in result.stdout.split("archived")[0]  # newest not pruned

    state_dir = LoopRunState.state_dir(tmp_path)
    archive_dir = state_dir / "archive"
    # Newest stays in the active dir.
    assert (state_dir / "newest.json").exists()
    assert (state_dir / "newest.log").exists()
    # Older runs move to archive with .log gzipped.
    assert (archive_dir / "oldest.json").exists()
    assert (archive_dir / "oldest.log.gz").exists()
    assert (archive_dir / "oldest_workspace.tar.gz").exists()
    assert not (state_dir / "oldest.log").exists(), "active log should be removed after archive"


def test_logs_prune_dry_run_makes_no_filesystem_changes(tmp_path: Path) -> None:
    """harness-830a: --dry-run reports what would be archived but
    leaves the filesystem untouched."""
    _seed_loop_run(tmp_path, "a", "2026-05-20T00:00:00+00:00")
    _seed_loop_run(tmp_path, "b", "2026-05-21T00:00:00+00:00")
    _seed_loop_run(tmp_path, "c", "2026-05-22T00:00:00+00:00")

    state_dir = LoopRunState.state_dir(tmp_path)
    before = {p.name for p in state_dir.iterdir()}
    result = CliRunner().invoke(
        drive_app,
        ["logs", "prune", "--workspace", str(tmp_path), "--keep", "1", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "would archive" in result.stdout
    after = {p.name for p in state_dir.iterdir()}
    assert before == after, "dry-run must not change the filesystem"


def test_logs_prune_under_keep_does_nothing(tmp_path: Path) -> None:
    """harness-830a: when there are fewer runs than --keep, the
    command reports 'nothing to prune' and leaves the dir alone."""
    _seed_loop_run(tmp_path, "only", "2026-05-22T00:00:00+00:00")

    result = CliRunner().invoke(
        drive_app, ["logs", "prune", "--workspace", str(tmp_path), "--keep", "5"]
    )
    assert result.exit_code == 0
    assert "nothing to prune" in result.stdout


def test_logs_list_handles_empty_dir(tmp_path: Path) -> None:
    """harness-830a: list on a workspace with no loop runs is a clean
    no-op, not an error."""
    result = CliRunner().invoke(drive_app, ["logs", "list", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "no loop runs" in result.stdout


# --- harness drive lint-epic (harness-bpix) -------------------------


class _LintBd:
    """Minimal bd fake for lint-epic: an epic with two children — one
    over-scoped (many §refs + long), one right-sized."""

    def __init__(self, bd_dir: Any) -> None:
        from harness.store._bd_types import _issue_from_json

        self._issues = {
            "harness-epic": _issue_from_json(
                {
                    "id": "harness-epic",
                    "title": "epic",
                    "status": "open",
                    "dependencies": [{"id": "harness-big"}, {"id": "harness-small"}],
                }
            ),
            "harness-big": _issue_from_json(
                {
                    "id": "harness-big",
                    "title": "§7 Police",
                    "status": "open",
                    "description": (
                        "Implement §7:\n- §7.1 state\n- §7.2 spawn\n- §7.3 chase AI\n"
                        "- §7.4 visual per §3.3\n- §7.5 siren\n" + ("detail " * 300)
                    ),
                }
            ),
            "harness-small": _issue_from_json(
                {
                    "id": "harness-small",
                    "title": "§7a spawn",
                    "status": "open",
                    "description": "Add a cop-state factory and spawn function.",
                }
            ),
        }

    def show(self, issue_id: str) -> Any:
        return self._issues[issue_id]


def test_lint_epic_flags_overscoped_and_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harness.driver.cli.DriverBd", _LintBd)
    result = runner.invoke(drive_app, ["lint-epic", "--epic", "harness-epic"])
    assert result.exit_code == 1  # at least one flagged → gate fails
    assert "harness-big" in result.stdout
    assert "Decomposition candidates" in result.stdout
    assert "sub-section" in result.stdout


def test_lint_epic_clean_epic_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CleanBd(_LintBd):
        def __init__(self, bd_dir: Any) -> None:
            super().__init__(bd_dir)
            # Drop the over-scoped child; only the right-sized one remains.
            self._issues["harness-epic"].raw["dependencies"] = [{"id": "harness-small"}]

    monkeypatch.setattr("harness.driver.cli.DriverBd", _CleanBd)
    result = runner.invoke(drive_app, ["lint-epic", "--epic", "harness-epic"])
    assert result.exit_code == 0, result.output
    assert "0 flagged" in result.stdout


# --- harness-9ugc: missing-smoke refuse -----------------------------


def test_drive_loop_refuses_browser_js_without_index(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """harness-9ugc: a browser-JS workspace with no index.html → the
    runtime smoke gate is OFF → refuse (exit 2) rather than false-close."""
    _stub_load_character(monkeypatch)
    (workspace / "game.js").write_text("const ctx = canvas.getContext('2d');\n")
    monkeypatch.setattr(
        "harness.driver.cli.run_loop", lambda *a, **k: pytest.fail("should not run")
    )
    result = runner.invoke(
        drive_app, ["loop", "--epic", "harness-e9oq", "--workspace", str(workspace)]
    )
    assert result.exit_code == 2
    assert "index.html" in result.stderr
    assert "--allow-missing-smoke" in result.stderr


def test_drive_loop_allow_missing_smoke_overrides(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    _stub_load_character(monkeypatch)
    (workspace / "game.js").write_text("const ctx = canvas.getContext('2d');\n")
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
    result = runner.invoke(
        drive_app,
        [
            "loop",
            "--epic",
            "harness-e9oq",
            "--workspace",
            str(workspace),
            "--allow-missing-smoke",
        ],
    )
    assert result.exit_code == 0, result.output
