"""Tests for src/harness/driver/planner.py — harness-dp0t."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from harness.driver.planner import (
    PLANNER_SYSTEM_PROMPT,
    PlanAddTool,
    PlanDraft,
    PlanFinishTool,
    PlanItem,
    PlannerConfig,
    PlannerError,
    _description_has_blockquote,
    _PlannerState,
    commit_plan,
    run_planner,
    write_draft,
)
from harness.orchestrator import ToolLoopResult
from harness.store._bd_types import _issue_from_json

# --- bd fake ---------------------------------------------------------


class _CapturingBd:
    """Bd fake that records every create / dep_add and returns
    deterministic ids."""

    def __init__(
        self,
        *,
        create_errors: dict[str, str] | None = None,
        dep_errors: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.creates: list[dict[str, Any]] = []
        self.deps: list[tuple[str, str]] = []
        self._create_errors = create_errors or {}
        self._dep_errors = dep_errors or {}
        self._next_id = 0

    def _mint_id(self, title: str) -> str:
        self._next_id += 1
        return f"harness-fake-{self._next_id:03d}"

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
        if title in self._create_errors:
            from harness.driver.bd import DriverBdError

            raise DriverBdError(self._create_errors[title])
        issue_id = self._mint_id(title)
        self.creates.append(
            {
                "id": issue_id,
                "title": title,
                "description": description,
                "type": issue_type,
                "priority": priority,
                "labels": list(labels),
                "acceptance": acceptance,
            }
        )
        return issue_id

    def dep_add(self, *, blocked: str, blocker: str) -> None:
        if (blocked, blocker) in self._dep_errors:
            from harness.driver.bd import DriverBdError

            raise DriverBdError(self._dep_errors[(blocked, blocker)])
        self.deps.append((blocked, blocker))


# --- PlanItem / PlanDraft round-trips -------------------------------


def test_plan_item_roundtrip_yaml() -> None:
    item = PlanItem(
        title="implement foo",
        description="body\n\n> quote\n",
        spec_quote="quote here",
        issue_type="feature",
        priority=1,
        acceptance="must do X",
        depends_on=["other"],
    )
    raw = item.to_yaml_dict()
    rebuilt = PlanItem.from_yaml_dict(raw)
    assert rebuilt == item


def test_plan_draft_roundtrip_yaml() -> None:
    draft = PlanDraft(
        epic_title="GTA2 clone",
        epic_description="implementation per spec",
        items=[
            PlanItem(
                title="index.html",
                description="> some quote",
                spec_quote="some quote",
            ),
            PlanItem(
                title="game.js",
                description="> another quote",
                spec_quote="another quote",
                depends_on=["index.html"],
            ),
        ],
    )
    rebuilt = PlanDraft.from_yaml(draft.to_yaml())
    assert rebuilt == draft


def test_plan_draft_from_yaml_rejects_non_mapping() -> None:
    with pytest.raises(PlannerError, match="not a mapping"):
        PlanDraft.from_yaml("- just a list\n")


def test_plan_draft_from_yaml_rejects_missing_fields() -> None:
    with pytest.raises(PlannerError, match="missing required field"):
        PlanDraft.from_yaml("items: []\n")


# --- write_draft round trip -----------------------------------------


def test_write_draft_writes_atomic_and_readable(tmp_path: Path) -> None:
    draft = PlanDraft(
        epic_title="t",
        epic_description="d",
        items=[PlanItem(title="a", description="> q", spec_quote="q here yes")],
    )
    path = tmp_path / "out.yaml"
    write_draft(draft, path)
    assert path.exists()
    rebuilt = PlanDraft.from_yaml(path.read_text())
    assert rebuilt == draft


# --- _description_has_blockquote ------------------------------------


def test_blockquote_match_normalizes_whitespace() -> None:
    assert _description_has_blockquote(
        description="some intro\n> The   quick   brown fox\n",
        spec_quote="The quick brown fox",
    )


def test_blockquote_no_blockquote_line_fails() -> None:
    assert not _description_has_blockquote(
        description="The quick brown fox jumps over",
        spec_quote="quick brown fox",
    )


def test_blockquote_empty_quote_fails() -> None:
    assert not _description_has_blockquote(description="> anything", spec_quote="")


# --- PlanAddTool ----------------------------------------------------


def test_plan_add_appends_on_success() -> None:
    state = _PlannerState()
    tool = PlanAddTool(state)
    result = tool.call(
        title="implement foo",
        description="reason this matters\n\n> here is the spec quote\n",
        spec_quote="here is the spec quote",
        priority=1,
        depends_on=[],
    )
    assert result.success
    assert "added: implement foo" in result.output
    assert len(state.items) == 1
    assert state.items[0].priority == 1


def test_plan_add_rejects_empty_title() -> None:
    state = _PlannerState()
    tool = PlanAddTool(state)
    result = tool.call(title="   ", description="> q here yes", spec_quote="q here yes plus extra")
    assert not result.success
    assert "title is empty" in result.output
    assert state.items == []


def test_plan_add_rejects_short_quote() -> None:
    state = _PlannerState()
    tool = PlanAddTool(state)
    result = tool.call(title="x", description="> short", spec_quote="short")
    assert not result.success
    assert "at least" in result.output


def test_plan_add_rejects_missing_blockquote() -> None:
    state = _PlannerState()
    tool = PlanAddTool(state)
    result = tool.call(
        title="x",
        description="no blockquote in body but mentions the quote",
        spec_quote="the quote is here",
    )
    assert not result.success
    assert "blockquote" in result.output
    assert state.items == []


def test_plan_add_tool_spec_marks_required_fields() -> None:
    spec = PlanAddTool(_PlannerState()).spec
    assert spec.name == "plan_add"
    assert spec.tier == "write"
    required = spec.parameters["required"]
    assert "title" in required
    assert "description" in required
    assert "spec_quote" in required


# --- PlanFinishTool -------------------------------------------------


def test_plan_finish_sets_state_flag() -> None:
    state = _PlannerState()
    tool = PlanFinishTool(state)
    assert not state.finished
    result = tool.call()
    assert result.success
    assert state.finished


def test_plan_finish_tool_spec_takes_no_args() -> None:
    spec = PlanFinishTool(_PlannerState()).spec
    assert spec.name == "plan_finish"
    assert spec.parameters["properties"] == {}


# --- commit_plan ----------------------------------------------------


def _write_spec(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "spec.md"
    path.write_text(content)
    return path


def _write_draft(tmp_path: Path, draft: PlanDraft) -> Path:
    path = tmp_path / "draft.yaml"
    write_draft(draft, path)
    return path


def test_commit_plan_creates_epic_children_and_deps(tmp_path: Path) -> None:
    spec = _write_spec(
        tmp_path,
        "Section 1\nThe quick brown fox jumps.\n\nSection 2\nLazy dog sleeps soundly.\n",
    )
    draft = PlanDraft(
        epic_title="example workplan",
        epic_description="from spec",
        items=[
            PlanItem(
                title="quick fox task",
                description="> The quick brown fox jumps.",
                spec_quote="The quick brown fox jumps.",
                acceptance="must implement A",
            ),
            PlanItem(
                title="lazy dog task",
                description="> Lazy dog sleeps soundly.",
                spec_quote="Lazy dog sleeps soundly.",
                depends_on=["quick fox task"],
            ),
        ],
    )
    draft_path = _write_draft(tmp_path, draft)
    bd = _CapturingBd()

    epic_id = commit_plan(draft_path, bd, spec)  # type: ignore[arg-type]
    assert epic_id == "harness-fake-001"
    # 1 epic + 2 children = 3 creates.
    titles = [c["title"] for c in bd.creates]
    assert titles == ["example workplan", "quick fox task", "lazy dog task"]
    # epic depends on each child + intra-draft depends_on.
    assert ("harness-fake-001", "harness-fake-002") in bd.deps  # epic ← quick fox
    assert ("harness-fake-001", "harness-fake-003") in bd.deps  # epic ← lazy dog
    assert ("harness-fake-003", "harness-fake-002") in bd.deps  # lazy dog ← quick fox
    # Spec-source label applied.
    for created in bd.creates:
        assert "plan-source:spec.md" in created["labels"]


def test_commit_plan_rejects_hallucinated_quote(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, "Real content here.\n")
    draft = PlanDraft(
        epic_title="bad",
        epic_description="d",
        items=[
            PlanItem(
                title="hallucinated",
                description="> Imaginary spec text not in the file.",
                spec_quote="Imaginary spec text not in the file.",
            )
        ],
    )
    draft_path = _write_draft(tmp_path, draft)
    bd = _CapturingBd()
    with pytest.raises(PlannerError, match="spec_quote not found"):
        commit_plan(draft_path, bd, spec)  # type: ignore[arg-type]
    # No bd writes attempted.
    assert bd.creates == []


def test_commit_plan_rejects_duplicate_titles(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, "foo bar baz\n")
    draft = PlanDraft(
        epic_title="dup",
        epic_description="d",
        items=[
            PlanItem(title="same", description="> foo bar baz", spec_quote="foo bar baz"),
            PlanItem(title="same", description="> foo bar baz", spec_quote="foo bar baz"),
        ],
    )
    draft_path = _write_draft(tmp_path, draft)
    bd = _CapturingBd()
    with pytest.raises(PlannerError, match="duplicate title"):
        commit_plan(draft_path, bd, spec)  # type: ignore[arg-type]


def test_commit_plan_rejects_unknown_depends_on(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, "foo bar baz here\n")
    draft = PlanDraft(
        epic_title="bad-dep",
        epic_description="d",
        items=[
            PlanItem(
                title="real",
                description="> foo bar baz here",
                spec_quote="foo bar baz here",
                depends_on=["nonexistent"],
            )
        ],
    )
    draft_path = _write_draft(tmp_path, draft)
    bd = _CapturingBd()
    with pytest.raises(PlannerError, match="unknown title"):
        commit_plan(draft_path, bd, spec)  # type: ignore[arg-type]


def test_commit_plan_reports_all_errors_in_one_pass(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, "real content here\n")
    draft = PlanDraft(
        epic_title="multi",
        epic_description="d",
        items=[
            PlanItem(title="a", description="> bad quote", spec_quote="hallucinated quote"),
            PlanItem(title="a", description="> dup", spec_quote="real content here"),
        ],
    )
    draft_path = _write_draft(tmp_path, draft)
    bd = _CapturingBd()
    with pytest.raises(PlannerError) as exc_info:
        commit_plan(draft_path, bd, spec)  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "spec_quote not found" in msg
    assert "duplicate title" in msg


def test_commit_plan_raises_when_draft_missing(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path, "content")
    bd = _CapturingBd()
    with pytest.raises(PlannerError, match="does not exist"):
        commit_plan(tmp_path / "missing.yaml", bd, spec)  # type: ignore[arg-type]


def test_commit_plan_raises_when_spec_missing(tmp_path: Path) -> None:
    draft = PlanDraft(epic_title="t", epic_description="d", items=[])
    draft_path = _write_draft(tmp_path, draft)
    bd = _CapturingBd()
    with pytest.raises(PlannerError, match="spec file"):
        commit_plan(draft_path, bd, tmp_path / "missing.md")  # type: ignore[arg-type]


def test_commit_plan_normalizes_whitespace_in_quote_check(tmp_path: Path) -> None:
    # Spec has a multi-line phrase that the draft quotes with different
    # whitespace — normalization should make them match.
    spec = _write_spec(
        tmp_path,
        "preamble\nA truly\n  multi-line\nphrase    in the spec.\n",
    )
    draft = PlanDraft(
        epic_title="ws",
        epic_description="d",
        items=[
            PlanItem(
                title="t",
                description="> A truly multi-line phrase in the spec.",
                spec_quote="A truly multi-line phrase in the spec.",
            )
        ],
    )
    draft_path = _write_draft(tmp_path, draft)
    bd = _CapturingBd()
    commit_plan(draft_path, bd, spec)  # type: ignore[arg-type]
    assert len(bd.creates) == 2  # epic + 1 child


# --- run_planner ----------------------------------------------------


def test_run_planner_exits_when_plan_finish_called(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fake adapter, via the mocked run_tool_loop, calls
    plan_add then plan_finish on the FIRST iteration. run_planner
    should exit the outer loop without running the second iteration."""
    call_count = [0]

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        call_count[0] += 1
        plan_add = registry.get("plan_add")
        plan_finish = registry.get("plan_finish")
        plan_add.call(
            title="task one",
            description="> the spec text appears here",
            spec_quote="the spec text appears here",
        )
        plan_finish.call()
        return ToolLoopResult(content="planning done", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.planner.run_tool_loop", fake_run_tool_loop)
    config = PlannerConfig(
        spec_path=tmp_path / "spec.md",
        epic_title="my workplan",
        workspace=tmp_path,
        max_plan_turns=5,
    )
    draft = run_planner(adapter=None, config=config)  # type: ignore[arg-type]
    assert call_count[0] == 1
    assert draft.epic_title == "my workplan"
    assert [i.title for i in draft.items] == ["task one"]


def test_run_planner_respects_max_plan_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Adapter never calls plan_finish; run_planner exits after
    `max_plan_turns` iterations and returns whatever it has."""
    call_count = [0]

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, **_kwargs: Any
    ) -> ToolLoopResult:
        call_count[0] += 1
        plan_add = registry.get("plan_add")
        plan_add.call(
            title=f"item {call_count[0]}",
            description="> the same spec quote shows up here every time",
            spec_quote="the same spec quote shows up here every time",
        )
        return ToolLoopResult(content="", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.planner.run_tool_loop", fake_run_tool_loop)
    config = PlannerConfig(
        spec_path=tmp_path / "spec.md",
        epic_title="t",
        workspace=tmp_path,
        max_plan_turns=3,
    )
    draft = run_planner(adapter=None, config=config)  # type: ignore[arg-type]
    assert call_count[0] == 3
    assert len(draft.items) == 3


# --- constants ------------------------------------------------------


def test_planner_system_prompt_pins_load_bearing_phrases() -> None:
    # Pinned: the planner system prompt's load-bearing instructions.
    # If these drift, decomposition quality is the regression vector.
    assert "plan_add" in PLANNER_SYSTEM_PROMPT
    assert "plan_finish" in PLANNER_SYSTEM_PROMPT
    assert "blockquote" in PLANNER_SYSTEM_PROMPT
    assert "do NOT write code" in PLANNER_SYSTEM_PROMPT.lower() or (
        "do not write code" in PLANNER_SYSTEM_PROMPT.lower()
    )


def test_planner_system_prompt_demands_tool_call_shape() -> None:
    # harness-mzce: the load-bearing fix for the empty-draft failure
    # was telling the model NOT to emit prose. Pin both halves of the
    # directive so a future cleanup doesn't strip them.
    lowered = PLANNER_SYSTEM_PROMPT.lower()
    assert "every reply must be a tool call" in lowered or (
        "every reply MUST be a tool call".lower() in lowered
    )
    assert "do not respond with prose" in lowered or (
        "do NOT respond with prose".lower() in lowered
    )


# --- planner observer + event log -----------------------------------


def test_run_planner_invokes_observer_with_tool_loop_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The observer wired into run_planner receives events from the
    inner run_tool_loop AND the planner's own lifecycle markers."""
    received: list[Any] = []

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, *, observe: Any = None, **_kwargs: Any
    ) -> ToolLoopResult:
        # Emit one synthetic tool-call-shaped event so the observer is
        # exercised end-to-end, then drive plan_add + plan_finish to
        # complete the planner.
        from harness.orchestrator import ToolLoopEvent
        from harness.tools.base import ToolCall

        if observe is not None:
            observe(
                ToolLoopEvent(
                    kind="tool_call_start",
                    call=ToolCall(name="read_file", arguments={"path": "spec.md"}),
                )
            )
        plan_add = registry.get("plan_add")
        plan_finish = registry.get("plan_finish")
        plan_add.call(
            title="t",
            description="> a real quote from the spec lives here",
            spec_quote="a real quote from the spec lives here",
        )
        plan_finish.call()
        return ToolLoopResult(content="", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.planner.run_tool_loop", fake_run_tool_loop)
    cfg = PlannerConfig(
        spec_path=tmp_path / "spec.md",
        epic_title="t",
        workspace=tmp_path,
        max_plan_turns=2,
    )
    run_planner(adapter=None, config=cfg, observe=received.append)  # type: ignore[arg-type]
    kinds = [e.kind for e in received]
    # Lifecycle markers from the planner itself.
    assert "planner_start" in kinds
    assert "planner_iteration_start" in kinds
    assert "planner_finish" in kinds
    assert "planner_summary" in kinds
    # And the synthetic tool-call event from the fake inner loop.
    assert "tool_call_start" in kinds


def test_run_planner_writes_event_log_to_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The planner ALWAYS writes a per-run event log under
    `<workspace>/.harness/planner_<ts>.log`, regardless of whether an
    external observer was supplied."""

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, registry: Any, *, observe: Any = None, **_kwargs: Any
    ) -> ToolLoopResult:
        plan_finish = registry.get("plan_finish")
        plan_finish.call()
        return ToolLoopResult(content="", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.planner.run_tool_loop", fake_run_tool_loop)
    cfg = PlannerConfig(
        spec_path=tmp_path / "spec.md",
        epic_title="t",
        workspace=tmp_path,
        max_plan_turns=2,
    )
    run_planner(adapter=None, config=cfg)  # type: ignore[arg-type]
    logs = list((tmp_path / ".harness").glob("planner_*.log"))
    assert len(logs) == 1
    content = logs[0].read_text()
    assert "planner_start" in content
    assert "planner_finish" in content
    assert "planner_summary" in content


def test_run_planner_finish_reason_max_plan_turns_when_not_called(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When max_plan_turns elapses without plan_finish, the summary
    event records `reason=max_plan_turns`."""
    received: list[Any] = []

    def fake_run_tool_loop(
        _adapter: Any, _messages: Any, _registry: Any, *, observe: Any = None, **_kwargs: Any
    ) -> ToolLoopResult:
        return ToolLoopResult(content="", messages=[], rounds=1, events=[])

    monkeypatch.setattr("harness.driver.planner.run_tool_loop", fake_run_tool_loop)
    cfg = PlannerConfig(
        spec_path=tmp_path / "spec.md",
        epic_title="t",
        workspace=tmp_path,
        max_plan_turns=2,
    )
    run_planner(adapter=None, config=cfg, observe=received.append)  # type: ignore[arg-type]
    finish_events = [e for e in received if e.kind == "planner_finish"]
    assert len(finish_events) == 1
    assert "reason=max_plan_turns" in (finish_events[0].delta or "")


def test_issue_from_json_imported() -> None:
    # Smoke test — the import only matters that it resolves; we don't
    # actually use _issue_from_json in planner tests, just confirming
    # nothing accidental.
    assert _issue_from_json is not None
