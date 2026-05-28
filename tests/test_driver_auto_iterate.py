"""Tests for the drive + critic auto-iterate wrapper (harness-3f8e)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from harness.driver.auto_iterate import (
    AutoIterateConfig,
    _open_titles_under_epic,
    _resolve_spec,
    _snapshot_source_files,
    run_auto_iterate,
)
from harness.driver.bd import DriverBdError
from harness.driver.critic import CriticFinding
from harness.driver.loop import LoopConfig, LoopResult
from harness.store._bd_types import BeadsIssue

# ---- helpers -----------------------------------------------------------


def _issue(
    id_: str,
    *,
    title: str,
    status: str = "open",
    labels: Sequence[str] = (),
    deps: Sequence[str] = (),
) -> BeadsIssue:
    return BeadsIssue(
        id=id_,
        title=title,
        status=status,
        priority=2,
        issue_type="task",
        assignee=None,
        labels=tuple(labels),
        raw={
            "id": id_,
            "title": title,
            "status": status,
            "dependencies": [{"id": d} for d in deps],
        },
    )


class _FakeAdapter:
    """Adapter never called — run_loop and run_critic are both
    monkeypatched in these tests."""

    id = "fake"
    context_window = 32000


@dataclass
class _FakeBd:
    """In-memory bd stub. Supports show, create_with_labels, dep_add,
    and the calls the auto-iterate path needs. Records every write so
    tests can assert on them."""

    issues: dict[str, BeadsIssue] = field(default_factory=dict)
    creates: list[dict[str, Any]] = field(default_factory=list)
    dep_adds: list[tuple[str, str]] = field(default_factory=list)
    show_failures: set[str] = field(default_factory=set)
    _next_id: int = 0

    def show(self, issue_id: str) -> BeadsIssue:
        if issue_id in self.show_failures:
            raise DriverBdError(f"show fail {issue_id}")
        if issue_id not in self.issues:
            raise DriverBdError(f"no such issue {issue_id}")
        return self.issues[issue_id]

    def create_with_labels(
        self,
        *,
        title: str,
        description: str,
        issue_type: str = "task",
        priority: int = 2,
        labels: Sequence[str] = (),
        acceptance: str | None = None,
    ) -> str:
        self._next_id += 1
        new_id = f"harness-new{self._next_id:02d}"
        self.creates.append(
            {
                "id": new_id,
                "title": title,
                "description": description,
                "issue_type": issue_type,
                "priority": priority,
                "labels": tuple(labels),
                "acceptance": acceptance,
            }
        )
        self.issues[new_id] = _issue(new_id, title=title, labels=labels)
        return new_id

    def dep_add(self, blocked: str, blocker: str) -> None:
        self.dep_adds.append((blocked, blocker))


def _finding(
    title: str = "keys map never written",
    description: str = "see game.js:42",
    acceptance: str = "controls respond to keypresses",
    priority: int = 0,
    spec_quote: str = "both must update the same map",
    evidence_path: str = "game.js:42",
) -> CriticFinding:
    return CriticFinding(
        title=title,
        description=description,
        acceptance=acceptance,
        priority=priority,
        spec_quote=spec_quote,
        evidence_path=evidence_path,
    )


def _loop_result(
    *,
    closed: Sequence[str] = (),
    exit_reason: str = "success",
    loop_run_id: str = "run01",
) -> LoopResult:
    return LoopResult(
        loop_run_id=loop_run_id,
        epic_id="harness-epic",
        closed=list(closed),
        halted_on=None,
        turns_used=1,
        exit_reason=exit_reason,  # type: ignore[arg-type]
    )


def _config(
    workspace: Path,
    *,
    spec_path: Path | None = None,
    max_passes: int = 8,
    convergence_streak: int = 2,
    critic_max_findings: int = 10,
) -> AutoIterateConfig:
    loop_cfg = LoopConfig(
        epic_id="harness-epic",
        workspace=workspace,
        character=object(),  # type: ignore[arg-type]  # Character protocol; not called in these tests
        max_turns=5,
    )
    return AutoIterateConfig(
        loop_config=loop_cfg,
        spec_path=spec_path,
        max_passes=max_passes,
        convergence_streak=convergence_streak,
        critic_max_findings=critic_max_findings,
    )


# ---- _snapshot_source_files -------------------------------------------


def test_snapshot_collects_source_files(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("function gameLoop() {}")
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "thing.py").write_text("def foo(): pass\n")
    (tmp_path / "README.md").write_text("not collected")
    snap = _snapshot_source_files(tmp_path)
    assert set(snap.keys()) == {"game.js", "index.html", "thing.py"}
    assert "gameLoop" in snap["game.js"]


def test_snapshot_skips_excluded_dirs(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "vendor.js").write_text("don't collect me")
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".harness" / "skip.js").write_text("also skip")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.js").write_text("skip")
    (tmp_path / "game.js").write_text("collected")
    snap = _snapshot_source_files(tmp_path)
    assert snap == {"game.js": "collected"}


def test_snapshot_drops_files_over_size_cap(tmp_path: Path) -> None:
    (tmp_path / "game.js").write_text("small file")
    (tmp_path / "vendor.js").write_text("x" * (256 * 1024 + 100))
    snap = _snapshot_source_files(tmp_path)
    assert "game.js" in snap
    assert "vendor.js" not in snap


def test_snapshot_recurses_into_subdirectories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "game.js").write_text("nested")
    snap = _snapshot_source_files(tmp_path)
    assert "src/game.js" in snap


# ---- _resolve_spec ----------------------------------------------------


def test_resolve_spec_returns_explicit_path_content(tmp_path: Path) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text("the spec")
    cfg = _config(tmp_path, spec_path=spec)
    assert _resolve_spec(cfg, _FakeBd()) == "the spec"  # type: ignore[arg-type]


def test_resolve_spec_from_plan_source_label(tmp_path: Path) -> None:
    """When --spec is absent, read the epic's plan-source label and
    look up the file under the workspace."""
    (tmp_path / "gta2-spec.md").write_text("from label")
    cfg = _config(tmp_path)
    bd = _FakeBd(
        issues={
            "harness-epic": _issue(
                "harness-epic",
                title="GTAII",
                labels=("plan-source:gta2-spec.md",),
            )
        }
    )
    assert _resolve_spec(cfg, bd) == "from label"  # type: ignore[arg-type]


def test_resolve_spec_explicit_wins_over_label(tmp_path: Path) -> None:
    """Operator's --spec value is respected even when the epic also
    carries a plan-source label."""
    explicit = tmp_path / "explicit.md"
    explicit.write_text("explicit content")
    (tmp_path / "label.md").write_text("label content")
    cfg = _config(tmp_path, spec_path=explicit)
    bd = _FakeBd(
        issues={
            "harness-epic": _issue(
                "harness-epic",
                title="GTAII",
                labels=("plan-source:label.md",),
            )
        }
    )
    assert _resolve_spec(cfg, bd) == "explicit content"  # type: ignore[arg-type]


def test_resolve_spec_returns_none_when_no_spec_available(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    bd = _FakeBd(issues={"harness-epic": _issue("harness-epic", title="GTAII")})
    assert _resolve_spec(cfg, bd) is None  # type: ignore[arg-type]


def test_resolve_spec_walks_workspace_parents_for_label(tmp_path: Path) -> None:
    """When plan-source file isn't under the workspace, walk parents.
    Matches the GTA case where the spec lives at <repo>/scratch/gta/
    spec/ but the workspace is <repo>/scratch/workspace/."""
    workspace = tmp_path / "scratch" / "workspace"
    workspace.mkdir(parents=True)
    (tmp_path / "spec.md").write_text("parent-relative content")
    cfg = _config(workspace)
    bd = _FakeBd(
        issues={
            "harness-epic": _issue(
                "harness-epic",
                title="GTAII",
                labels=("plan-source:spec.md",),
            )
        }
    )
    assert _resolve_spec(cfg, bd) == "parent-relative content"  # type: ignore[arg-type]


# ---- _open_titles_under_epic ------------------------------------------


def test_open_titles_returns_open_children(tmp_path: Path) -> None:
    bd = _FakeBd(
        issues={
            "harness-epic": _issue("harness-epic", title="GTAII", deps=("harness-a", "harness-b")),
            "harness-a": _issue("harness-a", title="A", status="open"),
            "harness-b": _issue("harness-b", title="B", status="closed"),
        }
    )
    titles = _open_titles_under_epic(bd, "harness-epic")  # type: ignore[arg-type]
    assert titles == ("A",)


def test_open_titles_returns_empty_on_show_failure(tmp_path: Path) -> None:
    bd = _FakeBd(show_failures={"harness-epic"})
    assert _open_titles_under_epic(bd, "harness-epic") == ()  # type: ignore[arg-type]


# ---- run_auto_iterate -------------------------------------------------


def test_run_auto_iterate_converges_after_streak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two empty critic passes in a row → converged."""
    bd = _FakeBd(issues={"harness-epic": _issue("harness-epic", title="GTAII")})
    cfg = _config(tmp_path, convergence_streak=2)

    monkeypatch.setattr(
        "harness.driver.auto_iterate.run_loop",
        lambda _a, _b, _c: _loop_result(closed=["harness-x"]),
    )
    monkeypatch.setattr("harness.driver.auto_iterate.run_critic", lambda **_: [])

    result = run_auto_iterate(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "converged"
    assert result.passes_run == 2
    assert result.critic_findings_total == 0


def test_run_auto_iterate_exits_drive_halted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A halted drive returns drive_halted without consulting the critic."""
    bd = _FakeBd(issues={"harness-epic": _issue("harness-epic", title="GTAII")})
    cfg = _config(tmp_path)

    monkeypatch.setattr(
        "harness.driver.auto_iterate.run_loop",
        lambda _a, _b, _c: _loop_result(exit_reason="halted"),
    )
    critic_calls = [0]

    def critic_spy(**_: Any) -> list[CriticFinding]:
        critic_calls[0] += 1
        return []

    monkeypatch.setattr("harness.driver.auto_iterate.run_critic", critic_spy)
    result = run_auto_iterate(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "drive_halted"
    assert result.passes_run == 1
    assert critic_calls[0] == 0


def test_run_auto_iterate_files_and_autoblocks_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each filed finding gets bd create + dep_add against the epic."""
    bd = _FakeBd(issues={"harness-epic": _issue("harness-epic", title="GTAII")})
    cfg = _config(tmp_path, max_passes=3, convergence_streak=2)

    monkeypatch.setattr(
        "harness.driver.auto_iterate.run_loop",
        lambda _a, _b, _c: _loop_result(loop_run_id="loopXY"),
    )
    # Pass 1: two findings. Pass 2 + 3: empty.
    pass_calls = [0]
    findings_per_pass = [
        [_finding(title="bug one"), _finding(title="bug two")],
        [],
        [],
    ]

    def critic_returns(**_: Any) -> list[CriticFinding]:
        out = findings_per_pass[pass_calls[0]]
        pass_calls[0] += 1
        return out

    monkeypatch.setattr("harness.driver.auto_iterate.run_critic", critic_returns)

    result = run_auto_iterate(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "converged"
    assert result.critic_findings_total == 2
    assert len(bd.creates) == 2
    # Each created bead is auto-blocked against the epic.
    new_ids = [c["id"] for c in bd.creates]
    assert bd.dep_adds == [("harness-epic", new_ids[0]), ("harness-epic", new_ids[1])]
    # The critic:<loop_run_id> label is present on every filing.
    for created in bd.creates:
        assert "critic:loopXY" in created["labels"]


def test_run_auto_iterate_exits_passes_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Critic keeps finding things — hard cap stops the loop."""
    bd = _FakeBd(issues={"harness-epic": _issue("harness-epic", title="GTAII")})
    cfg = _config(tmp_path, max_passes=3, convergence_streak=10)

    monkeypatch.setattr("harness.driver.auto_iterate.run_loop", lambda _a, _b, _c: _loop_result())
    monkeypatch.setattr(
        "harness.driver.auto_iterate.run_critic",
        lambda **_: [_finding(title=f"bug {n}") for n in ("a", "b")],
    )
    result = run_auto_iterate(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "passes_exhausted"
    assert result.passes_run == 3
    # Three passes * 2 findings each = 6 total filings.
    assert result.critic_findings_total == 6


def test_run_auto_iterate_resets_streak_on_new_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty pass mid-run must NOT count toward convergence if a
    later pass finds more bugs — streak resets to 0 on non-empty."""
    bd = _FakeBd(issues={"harness-epic": _issue("harness-epic", title="GTAII")})
    cfg = _config(tmp_path, max_passes=8, convergence_streak=2)
    findings_per_pass = [
        [_finding(title="first")],
        [],  # streak=1
        [_finding(title="second")],  # streak=0
        [],  # streak=1
        [],  # streak=2 → converge
    ]
    pass_calls = [0]
    monkeypatch.setattr("harness.driver.auto_iterate.run_loop", lambda _a, _b, _c: _loop_result())

    def critic_returns(**_: Any) -> list[CriticFinding]:
        out = findings_per_pass[pass_calls[0]]
        pass_calls[0] += 1
        return out

    monkeypatch.setattr("harness.driver.auto_iterate.run_critic", critic_returns)
    result = run_auto_iterate(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "converged"
    assert result.passes_run == 5
    assert result.critic_findings_total == 2


def test_run_auto_iterate_continues_after_bd_create_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed bd write on one finding shouldn't poison the rest of
    the pass — skip and keep going."""
    bd = _FakeBd(issues={"harness-epic": _issue("harness-epic", title="GTAII")})
    cfg = _config(tmp_path, max_passes=2, convergence_streak=1)

    original_create = bd.create_with_labels
    create_calls = [0]

    def flaky_create(**kwargs: Any) -> str:
        create_calls[0] += 1
        if create_calls[0] == 1:
            raise DriverBdError("simulated transient bd error")
        return original_create(**kwargs)

    bd.create_with_labels = flaky_create  # type: ignore[method-assign]
    monkeypatch.setattr("harness.driver.auto_iterate.run_loop", lambda _a, _b, _c: _loop_result())
    pass_calls = [0]
    findings_per_pass = [[_finding(title="bug a"), _finding(title="bug b")], []]

    def critic_returns(**_: Any) -> list[CriticFinding]:
        out = findings_per_pass[pass_calls[0]]
        pass_calls[0] += 1
        return out

    monkeypatch.setattr("harness.driver.auto_iterate.run_critic", critic_returns)
    result = run_auto_iterate(_FakeAdapter(), bd, cfg)  # type: ignore[arg-type]
    assert result.exit_reason == "converged"
    # One finding errored, one succeeded.
    assert result.critic_findings_total == 1
    assert len(result.filed_beads) == 1
