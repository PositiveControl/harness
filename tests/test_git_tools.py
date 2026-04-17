from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.tools.git import GitDiffTool, GitLogTool, GitStatusTool


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)  # noqa: S603, S607


def _init_repo(path: Path) -> None:
    _git("init", "-q", cwd=path)
    _git("config", "user.email", "bench@example.com", cwd=path)
    _git("config", "user.name", "Bench", cwd=path)
    _git("config", "commit.gpgsign", "false", cwd=path)


def _commit(path: Path, filename: str, content: str, msg: str) -> None:
    (path / filename).write_text(content)
    _git("add", filename, cwd=path)
    _git("commit", "-q", "-m", msg, cwd=path)


def test_git_status_clean_tree(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "a.txt", "hi\n", "init")
    out = GitStatusTool(root=tmp_path).call()
    assert "clean" in out


def test_git_status_detects_modifications(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "a.txt", "hi\n", "init")
    (tmp_path / "a.txt").write_text("bye\n")
    (tmp_path / "new.txt").write_text("new\n")
    out = GitStatusTool(root=tmp_path).call()
    assert "a.txt" in out
    assert "new.txt" in out


def test_git_diff_shows_unstaged(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "a.txt", "hi\n", "init")
    (tmp_path / "a.txt").write_text("bye\n")
    out = GitDiffTool(root=tmp_path).call()
    assert "-hi" in out
    assert "+bye" in out


def test_git_diff_staged_flag(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "a.txt", "hi\n", "init")
    (tmp_path / "a.txt").write_text("bye\n")
    _git("add", "a.txt", cwd=tmp_path)
    # Unstaged diff is now empty (change is staged).
    assert "-hi" not in GitDiffTool(root=tmp_path).call()
    # Staged diff shows the change.
    staged = GitDiffTool(root=tmp_path).call(staged=True)
    assert "-hi" in staged
    assert "+bye" in staged


def test_git_diff_truncation(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "big.txt", "line\n" * 2000, "init")
    (tmp_path / "big.txt").write_text("other\n" * 2000)
    out = GitDiffTool(root=tmp_path).call(max_lines=10)
    assert "truncated at 10" in out


def test_git_log_shows_recent_commits(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "a.txt", "1\n", "first")
    _commit(tmp_path, "b.txt", "2\n", "second")
    _commit(tmp_path, "c.txt", "3\n", "third")
    out = GitLogTool(root=tmp_path).call(n=2)
    assert "third" in out
    assert "second" in out
    assert "first" not in out  # capped at 2


def test_git_log_author_filter(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, "a.txt", "1\n", "mine")
    _git(
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "alien",
        "--author=Alien <a@b>",
        cwd=tmp_path,
    )
    out = GitLogTool(root=tmp_path).call(author="Alien")
    assert "alien" in out
    assert "mine" not in out


def test_all_three_read_tier(tmp_path: Path) -> None:
    for tool in (
        GitStatusTool(root=tmp_path),
        GitDiffTool(root=tmp_path),
        GitLogTool(root=tmp_path),
    ):
        assert tool.spec.tier == "read"


def test_git_not_a_repo_returns_message(tmp_path: Path) -> None:
    """Not fatal — falls through with the git complaint as the tool's
    return, so the model can reason about it."""
    out = GitStatusTool(root=tmp_path).call()
    assert "not a git repository" in out.lower() or "fatal" in out.lower()


@pytest.mark.skip(reason="requires a system without git; not portable")
def test_git_missing_executable_raises() -> None:  # pragma: no cover
    pass
