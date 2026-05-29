"""Tests for src/harness/driver/handoff.py — harness-2gut."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from harness.driver import handoff as handoff_mod
from harness.driver.handoff import (
    MAX_DECISIONS,
    MAX_OBSERVATIONS,
    RENDER_CHAR_CAP,
    Handoff,
    build_handoff,
    substantial_artifacts,
)
from harness.driver.state import LoopRunState
from harness.store._bd_types import BeadsIssue, _issue_from_json

# --- helpers ---------------------------------------------------------


def _issue(
    issue_id: str,
    *,
    title: str = "",
    description: str = "",
    acceptance: str = "",
    notes: str = "",
    status: str = "open",
    priority: int = 2,
    labels: tuple[str, ...] = (),
    created_at: str = "2026-05-20T00:00:00Z",
) -> BeadsIssue:
    raw: dict[str, Any] = {
        "id": issue_id,
        "title": title,
        "status": status,
        "priority": priority,
        "issue_type": "task",
        "labels": list(labels),
        "description": description,
        "acceptance_criteria": acceptance,
        "notes": notes,
        "created_at": created_at,
    }
    return _issue_from_json(raw)


def _state(
    *,
    loop_run_id: str = "abcd1234",
    epic_id: str = "harness-e9oq",
    started_at_sha: str = "deadbeef",
    closed: Sequence[str] = (),
) -> LoopRunState:
    s = LoopRunState.fresh(
        epic_id=epic_id,
        max_turns=20,
        started_at_sha=started_at_sha,
    )
    # Reach in: tests want a deterministic id.
    s.loop_run_id = loop_run_id
    s.closed_this_run.extend(closed)
    return s


class _FakeBd:
    """Drop-in for DriverBd that returns pre-baked records.

    Only implements the surface build_handoff calls: `show` and
    `thoughts_in_loop_run`. Extra methods would surface as
    AttributeError so tests catch unexpected drift."""

    def __init__(
        self,
        *,
        issues: dict[str, BeadsIssue] | None = None,
        thoughts: Sequence[BeadsIssue] = (),
        show_errors: set[str] | None = None,
    ) -> None:
        self._issues = issues or {}
        self._thoughts = list(thoughts)
        self._show_errors = show_errors or set()
        self.show_calls: list[str] = []

    def show(self, issue_id: str) -> BeadsIssue:
        self.show_calls.append(issue_id)
        if issue_id in self._show_errors:
            from harness.driver.bd import DriverBdError

            raise DriverBdError(f"simulated bd error for {issue_id}")
        return self._issues[issue_id]

    def thoughts_in_loop_run(
        self,
        loop_run_id: str,
        *,
        types: Sequence[str] | None = None,
    ) -> list[BeadsIssue]:
        return list(self._thoughts)


def _install_git_diff(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> None:
    """Stub `subprocess.run` to return a fixed git diff result."""

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr("harness.driver.handoff.subprocess.run", fake_run)


# --- Handoff.render() ------------------------------------------------


def _base_handoff(**overrides: Any) -> Handoff:
    base: dict[str, Any] = {
        "loop_run_id": "abcd1234",
        "epic_id": "harness-e9oq",
        "current_issue": "harness-x (P2 task)\nTitle: do the thing",
        "parent_epic_summary": "harness-e9oq — [epic] harness loop",
        "files_touched": ("M src/foo.py", "A tests/test_foo.py"),
        "closed_this_run": ("harness-a",),
        "decisions": ("decision one", "decision two"),
        "observations": ("observation one",),
        "open_questions": ("question one",),
        "prior_attempt_failure": None,
    }
    base.update(overrides)
    return Handoff(**base)


def test_render_includes_all_sections() -> None:
    out = _base_handoff().render()
    assert "[SESSION HANDOFF — loop_run=abcd1234 epic=harness-e9oq]" in out
    assert "Current issue:" in out
    assert "harness-x (P2 task)" in out
    assert "Parent epic:" in out
    assert "harness-e9oq — [epic] harness loop" in out
    assert "Files touched this loop run:" in out
    assert "  M src/foo.py" in out
    assert "  A tests/test_foo.py" in out
    assert "Closed this loop run:" in out
    assert "  harness-a" in out
    assert "Decisions:" in out
    assert "  - decision one" in out
    assert "Observations:" in out
    assert "  - observation one" in out
    assert "Open questions:" in out
    assert "  - question one" in out
    assert "[END HANDOFF]" in out


def test_render_omits_parent_epic_when_none() -> None:
    out = _base_handoff(parent_epic_summary=None).render()
    assert "Parent epic:" not in out


def test_render_inserts_prior_attempt_block_when_set() -> None:
    out = _base_handoff(prior_attempt_failure="fabrication_fallback fired").render()
    assert "[PRIOR ATTEMPT FAILED]" in out
    assert "fabrication_fallback fired" in out
    # And it appears BEFORE the current-issue block.
    assert out.index("[PRIOR ATTEMPT FAILED]") < out.index("Current issue:")


def test_render_prior_attempt_block_absent_on_first_attempt() -> None:
    out = _base_handoff().render()
    assert "[PRIOR ATTEMPT FAILED]" not in out


def test_render_inserts_forbidden_patterns_block_when_set() -> None:
    """harness-d8e3: when the operator sets forbidden_patterns on
    LoopConfig (default 'TODO', 'FIXME', 'XXX', 'HACK'), every
    handoff carries a [FORBIDDEN PATTERNS] block so the model
    knows up-front the harness will reject any close that
    introduces these strings into workspace files. Preventive
    layer for the post-close audit."""
    out = _base_handoff(
        forbidden_patterns=("TODO", "FIXME", "XXX", "HACK"),
    ).render()
    assert "[FORBIDDEN PATTERNS]" in out
    assert "'TODO'" in out
    assert "'FIXME'" in out
    assert "'XXX'" in out
    assert "'HACK'" in out
    # And the block appears BEFORE the current-issue block so the
    # model reads the constraint before deciding what to write.
    assert out.index("[FORBIDDEN PATTERNS]") < out.index("Current issue:")


def test_render_forbidden_patterns_block_absent_when_empty() -> None:
    """Empty forbidden_patterns tuple means the operator disabled
    the check (or no patterns configured) — no block renders, no
    noise in the handoff."""
    out = _base_handoff(forbidden_patterns=()).render()
    assert "[FORBIDDEN PATTERNS]" not in out


def test_render_uses_placeholders_when_lists_are_empty() -> None:
    out = _base_handoff(
        files_touched=(),
        closed_this_run=(),
        decisions=(),
        observations=(),
        open_questions=(),
    ).render()
    assert "(none yet)" in out
    assert "(none recorded this run)" in out
    assert "(none open)" in out


def test_render_drops_oldest_decisions_when_over_budget() -> None:
    # Construct decisions / observations large enough to blow the cap;
    # render() should drop the OLDEST (tail) entries until under cap.
    # Decisions list is newest-first per the contract.
    big_decision_lines = tuple(f"decision-{i}: " + ("x" * 200) for i in range(MAX_DECISIONS))
    big_observation_lines = tuple(
        f"observation-{i}: " + ("y" * 200) for i in range(MAX_OBSERVATIONS)
    )
    out = _base_handoff(
        decisions=big_decision_lines,
        observations=big_observation_lines,
    ).render()
    assert len(out) <= RENDER_CHAR_CAP
    # Newest entry must survive.
    assert "decision-0" in out
    # Oldest entry must have been dropped.
    assert "decision-9" not in out


def test_render_never_drops_open_questions_even_over_budget() -> None:
    # Open questions are load-bearing. Pile on questions large enough
    # to push past cap, render anyway, confirm every question survives.
    huge_questions = tuple(f"q-{i}: " + ("z" * 400) for i in range(20))
    out = _base_handoff(
        decisions=(),
        observations=(),
        open_questions=huge_questions,
    ).render()
    # Length may exceed cap (open questions can't be dropped) — that's
    # intentional, accept the bloat rather than lose load-bearing data.
    for i in range(20):
        assert f"q-{i}:" in out


def test_render_drops_observations_after_decisions_when_decisions_already_empty() -> None:
    out = _base_handoff(
        decisions=(),
        observations=tuple(f"obs-{i}: " + ("y" * 400) for i in range(MAX_OBSERVATIONS)),
    ).render()
    assert len(out) <= RENDER_CHAR_CAP
    # Newest observation survives, oldest does not.
    assert "obs-0" in out
    assert "obs-9" not in out


# --- build_handoff ---------------------------------------------------


def test_render_includes_workspace_block_when_set() -> None:
    """harness-po0v + harness-cb7c: Workspace path + contents render
    between parent epic and files-touched blocks, using ls -F format
    (just name + trailing `/` for dirs — no `f`/`d` type prefix that
    small models misparse as path components)."""
    out = _base_handoff(
        workspace_path=Path("/some/workspace"),
        workspace_contents=("game.js", "gta/", "index.html"),
    ).render()
    assert "Workspace (your cwd; tool paths are relative to this): /some/workspace" in out
    assert "  game.js" in out
    assert "  gta/" in out
    assert "  index.html" in out
    # The OLD f/d prefix shape (harness-cb7c) must NOT appear — models
    # misparsed it as `d/<name>` / `f/<name>` paths.
    assert "f game.js" not in out
    assert "d gta/" not in out
    # Lives before files-touched so model orientation comes before the diff.
    assert out.index("Workspace") < out.index("Files touched this loop run:")


def test_render_omits_workspace_block_when_path_none() -> None:
    """harness-po0v: backward-compat — handoffs built without a
    workspace render normally (no orphan empty block)."""
    out = _base_handoff().render()
    assert "Workspace" not in out


def test_render_includes_targeted_fix_banner_when_set() -> None:
    """harness-lefw: the MODE:TARGETED-FIX banner renders at the top of
    the handoff (before [PRIOR ATTEMPT FAILED] and Current issue:) so
    it's the first thing the model reads. The banner explicitly forbids
    write_file on existing paths and steers to edit_file."""
    out = _base_handoff(
        targeted_fix=True,
        prior_attempt_failure="forbidden-pattern hit: TODO",
    ).render()
    assert "[MODE: TARGETED-FIX]" in out
    assert "edit_file" in out
    assert "write_file on an existing path" in out
    # harness-8tjnv: the banner now states the hard-block enforcement.
    assert "REFUSED by the harness" in out
    assert 'reason="rewrite-required"' in out
    # Banner precedes both prior-attempt and current-issue.
    assert out.index("[MODE: TARGETED-FIX]") < out.index("[PRIOR ATTEMPT FAILED]")
    assert out.index("[MODE: TARGETED-FIX]") < out.index("Current issue:")


def test_render_omits_targeted_fix_banner_by_default() -> None:
    """harness-lefw: targeted_fix=False (the default for first-attempt
    issues without a REGRESSION marker) leaves the handoff banner-free."""
    out = _base_handoff().render()
    assert "[MODE: TARGETED-FIX]" not in out
    assert "rewrite-required" not in out


def test_render_targeted_fix_banner_names_built_artifacts() -> None:
    """harness-8tjnv: when existing_artifacts is populated, the banner
    names each built file + line count so 'the artifact already exists'
    is concrete (the generic banner was ignored on run b085854e)."""
    out = _base_handoff(
        targeted_fix=True,
        existing_artifacts=(("game.js", 2094), ("index.html", 60)),
    ).render()
    assert "[MODE: TARGETED-FIX]" in out
    assert "ALREADY BUILT" in out
    assert "game.js — 2094 lines (built)" in out
    assert "index.html — 60 lines (built)" in out
    # The generic fallback wording is replaced by the concrete list.
    assert "The artifact already exists from a prior implementation" not in out


def test_substantial_artifacts_finds_built_source_above_floor(tmp_path: Path) -> None:
    """harness-8tjnv: a >=50-line source file counts as built; small
    stubs and hidden/excluded dirs are ignored; result is largest-first."""
    (tmp_path / "game.js").write_text("// game\n" + ("x();\n" * 200))  # 201 lines
    (tmp_path / "util.js").write_text("// util\n" + ("y();\n" * 60))  # 61 lines
    (tmp_path / "stub.js").write_text("let a;\n")  # 2 lines — below floor
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("z();\n" * 500)  # excluded
    arts = substantial_artifacts(tmp_path)
    names = [rel for rel, _ in arts]
    assert names == ["game.js", "util.js"]  # largest-first, stub + node_modules dropped
    assert dict(arts)["game.js"] == 202  # count("\n")+1 incl. trailing newline


def test_substantial_artifacts_empty_workspace_or_none(tmp_path: Path) -> None:
    """No source files → (); None workspace → () (back-compat for
    test/chat callers without a workspace)."""
    assert substantial_artifacts(tmp_path) == ()
    assert substantial_artifacts(None) == ()


def test_render_empty_workspace_contents_shows_placeholder() -> None:
    """harness-po0v: workspace set but empty contents → '(empty)' marker
    so the model sees the workspace orientation but knows there's
    nothing in it yet."""
    out = _base_handoff(
        workspace_path=Path("/empty/ws"),
        workspace_contents=(),
    ).render()
    assert "Workspace (your cwd; tool paths are relative to this): /empty/ws" in out
    assert "  (empty)" in out


def test_build_handoff_lists_workspace_top_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-po0v: build_handoff populates workspace_path +
    workspace_contents from the supplied workspace dir."""
    (tmp_path / "game.js").write_text("// js")
    (tmp_path / "index.html").write_text("<!doctype html>")
    (tmp_path / "gta").mkdir()
    # Hidden entries skipped:
    (tmp_path / ".harness").mkdir()
    (tmp_path / ".coverage").write_text("")

    current = _issue("harness-x", title="t", description="d")
    epic = _issue("harness-e9oq", title="e")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="")
    handoff = build_handoff(
        _state(),
        "harness-x",
        bd,
        git_root=tmp_path,
        workspace=tmp_path,
    )
    assert handoff.workspace_path == tmp_path.resolve()
    # Sorted: dirs first then files, alphabetical within each group.
    # ls -F format: trailing `/` for dirs, just the name for files
    # (harness-cb7c — no f/d type prefix that small models misparse).
    assert handoff.workspace_contents == (
        "gta/",
        "game.js",
        "index.html",
    )


def test_build_handoff_skips_workspace_when_not_supplied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-po0v: build_handoff without `workspace=` produces
    workspace_path=None + empty contents (back-compat)."""
    current = _issue("harness-x", title="t")
    epic = _issue("harness-e9oq", title="e")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="")
    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path)
    assert handoff.workspace_path is None
    assert handoff.workspace_contents == ()


def test_build_handoff_pulls_current_issue_and_epic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = _issue(
        "harness-x",
        title="do the thing",
        description="why this matters",
        acceptance="(1) thing done",
        priority=2,
    )
    epic = _issue("harness-e9oq", title="[epic] harness loop")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="M\tsrc/foo.py\n")

    state = _state()
    handoff = build_handoff(state, "harness-x", bd, git_root=tmp_path)

    assert "harness-x (P2 task)" in handoff.current_issue
    assert "Title: do the thing" in handoff.current_issue
    assert "why this matters" in handoff.current_issue
    assert "(1) thing done" in handoff.current_issue
    assert handoff.parent_epic_summary == "harness-e9oq — [epic] harness loop"
    assert handoff.files_touched == ("M\tsrc/foo.py",)


def test_build_handoff_plumbs_targeted_fix_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-lefw: the targeted_fix arg threads through build_handoff
    into the Handoff dataclass so the caller (loop.run_loop) can flip
    MODE banner rendering on/off without touching render() internals."""
    current = _issue("harness-x", title="do the thing")
    epic = _issue("harness-e9oq", title="[epic] harness loop")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="")

    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path, targeted_fix=True)
    assert handoff.targeted_fix is True
    assert "[MODE: TARGETED-FIX]" in handoff.render()


def test_build_handoff_defaults_targeted_fix_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """harness-lefw: targeted_fix defaults to False so existing callers
    (tests, future bindings) get the historical "no banner" behavior
    without opting in."""
    current = _issue("harness-x", title="do the thing")
    epic = _issue("harness-e9oq", title="[epic] harness loop")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="")

    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path)
    assert handoff.targeted_fix is False
    assert "[MODE: TARGETED-FIX]" not in handoff.render()


def test_build_handoff_surfaces_bd_notes_in_current_issue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Operator-supplied bd notes (added when reopening an issue with
    feedback) MUST reach the executor via the current_issue block.
    Without this the violation list the operator wrote is invisible."""
    current = _issue(
        "harness-x",
        title="do the thing",
        description="why this matters",
        acceptance="(1) thing done",
        notes="REOPENED: prior attempt used e.key instead of e.code; fix per §15.4",
    )
    epic = _issue("harness-e9oq", title="[epic] harness loop")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="")
    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path)
    assert "Notes" in handoff.current_issue
    assert "REOPENED" in handoff.current_issue
    assert "e.key instead of e.code" in handoff.current_issue


def test_build_handoff_categorises_thoughts_by_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = _issue("harness-x", title="t")
    epic = _issue("harness-e9oq", title="e")
    decision = _issue(
        "harness-d1",
        title="picked plan A",
        labels=("thought:decision",),
        status="closed",
    )
    observation = _issue(
        "harness-o1",
        title="saw a thing",
        labels=("thought:observation",),
        status="closed",
    )
    open_q = _issue("harness-q1", title="why?", labels=("thought:question",), status="open")
    closed_q = _issue(
        "harness-q2",
        title="answered",
        labels=("thought:question",),
        status="closed",
    )
    bd = _FakeBd(
        issues={"harness-x": current, "harness-e9oq": epic},
        thoughts=[decision, observation, open_q, closed_q],
    )
    _install_git_diff(monkeypatch, stdout="")

    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path)

    assert handoff.decisions == ("picked plan A",)
    assert handoff.observations == ("saw a thing",)
    # Closed questions don't show — only open ones.
    assert handoff.open_questions == ("why?",)


def test_build_handoff_truncates_decisions_to_top_n(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = _issue("harness-x", title="t")
    epic = _issue("harness-e9oq", title="e")
    # 20 decisions; thoughts_in_loop_run returns them newest-first
    # (the fake mimics DriverBd's sort).
    decisions = [
        _issue(
            f"harness-d{i}",
            title=f"decision-{i}",
            labels=("thought:decision",),
            status="closed",
        )
        for i in range(20)
    ]
    bd = _FakeBd(
        issues={"harness-x": current, "harness-e9oq": epic},
        thoughts=decisions,
    )
    _install_git_diff(monkeypatch, stdout="")

    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path)
    assert len(handoff.decisions) == MAX_DECISIONS
    # The newest (decision-0) survives; oldest (decision-19) does not.
    assert "decision-0" in handoff.decisions
    assert "decision-19" not in handoff.decisions


def test_build_handoff_open_questions_never_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = _issue("harness-x", title="t")
    epic = _issue("harness-e9oq", title="e")
    questions = [
        _issue(
            f"harness-q{i}",
            title=f"q-{i}",
            labels=("thought:question",),
            status="open",
        )
        for i in range(30)
    ]
    bd = _FakeBd(
        issues={"harness-x": current, "harness-e9oq": epic},
        thoughts=questions,
    )
    _install_git_diff(monkeypatch, stdout="")

    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path)
    assert len(handoff.open_questions) == 30


def test_build_handoff_carries_state_closed_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = _issue("harness-x", title="t")
    epic = _issue("harness-e9oq", title="e")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="")

    state = _state(closed=("harness-a", "harness-b"))
    handoff = build_handoff(state, "harness-x", bd, git_root=tmp_path)
    assert handoff.closed_this_run == ("harness-a", "harness-b")


def test_build_handoff_swallows_epic_errors_to_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = _issue("harness-x", title="t")
    bd = _FakeBd(
        issues={"harness-x": current},
        show_errors={"harness-e9oq"},
    )
    _install_git_diff(monkeypatch, stdout="")
    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path)
    assert handoff.parent_epic_summary is None


def test_build_handoff_propagates_current_issue_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from harness.driver.bd import DriverBdError

    bd = _FakeBd(show_errors={"harness-x"})
    _install_git_diff(monkeypatch, stdout="")
    with pytest.raises(DriverBdError):
        build_handoff(_state(), "harness-x", bd, git_root=tmp_path)


def test_build_handoff_returns_empty_files_on_git_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = _issue("harness-x", title="t")
    epic = _issue("harness-e9oq", title="e")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="", returncode=128)
    handoff = build_handoff(_state(), "harness-x", bd, git_root=tmp_path)
    assert handoff.files_touched == ()


def test_build_handoff_threads_prior_attempt_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = _issue("harness-x", title="t")
    epic = _issue("harness-e9oq", title="e")
    bd = _FakeBd(issues={"harness-x": current, "harness-e9oq": epic})
    _install_git_diff(monkeypatch, stdout="")
    handoff = build_handoff(
        _state(),
        "harness-x",
        bd,
        git_root=tmp_path,
        prior_attempt_failure="fab fired",
    )
    assert handoff.prior_attempt_failure == "fab fired"
    # And it lands in the rendered output as a top block.
    rendered = handoff.render()
    assert "[PRIOR ATTEMPT FAILED]" in rendered
    assert "fab fired" in rendered


# --- git stub fallback ---------------------------------------------


def test_git_diff_filenotfound_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("no git")

    monkeypatch.setattr("harness.driver.handoff.subprocess.run", boom)
    result = handoff_mod._git_diff_name_status(tmp_path, "deadbeef")
    assert result == ()
