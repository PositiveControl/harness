"""Planner phase — harness-dp0t.

Converts a free-form spec doc into a bd workplan, gated by a YAML
approval step. The planner is intentionally separate from the executor
(`loop.py`) — its job is *decomposition*, not *implementation*. Output
lands in a YAML draft the operator reviews + edits before `commit_plan`
materializes it into bd epic + child issues + dep edges.

Two phases:

  1. `run_planner(adapter, config)` — drives an LLM through up to
     `max_plan_turns` orchestrator turns with two driver-scoped tools
     (`plan_add`, `plan_finish`). Returns a `PlanDraft`.
  2. `commit_plan(draft_path, bd, spec_path)` — validates every item's
     `spec_quote` actually appears in the spec file, then materializes
     epic + items + deps in bd.

Driver-scoped tools (NOT in `profiles.py` — these only exist in the
planner session):
  - `plan_add(title, description, type, priority, acceptance,
    depends_on=[])` — appends to the in-memory list. Rejects
    descriptions without a `> ` blockquote of the spec section being
    implemented.
  - `plan_finish()` — signals planning is done; the run loop exits on
    the next iteration even if max_plan_turns hasn't been reached.

YAML schema (stable ordering for diff-friendly review):

```yaml
epic_title: ...
epic_description: ...
items:
  - title: ...
    description: ...
    spec_quote: ...
    type: task | feature | bug
    priority: 2
    acceptance: ...
    depends_on:
      - <title of another item in the draft>
```

Spec-quote validation is substring after whitespace normalization (every
run of whitespace collapses to a single space). Anything stricter
generates spurious rejections; anything looser opens the door to
hallucinated quotes — the YAML draft is the gate, but the gate has a
contract.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness.driver.bd import DriverBd, DriverBdError
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.orchestrator import run_tool_loop
from harness.tools import (
    GlobTool,
    GrepTool,
    ReadFileTool,
    ToolCatalog,
    ToolRegistry,
    seed_builtins_into,
)
from harness.tools.base import ToolResult, ToolSpec

# Whitespace-collapse pattern shared by plan_add validation and
# commit-time quote verification. Public so tests can pin it.
_WHITESPACE_RE = re.compile(r"\s+")


# Minimum quote length the validator demands. Anything shorter is too
# generic to verify usefully (and probably indicates the model
# blockquoted a section header). Tuned for "one phrase or sentence."
_MIN_QUOTE_CHARS = 12


@dataclass
class PlanItem:
    """One child issue in the workplan.

    - `title`: bd issue title.
    - `description`: free-form body. MUST contain a `> ` blockquote of
      `spec_quote`. Stored verbatim.
    - `spec_quote`: the quoted text, normalized. Validated against the
      spec file at commit time.
    - `issue_type`: bd type (task / feature / bug).
    - `priority`: 0..4, matching bd's P0..P4.
    - `acceptance`: bd acceptance_criteria field.
    - `depends_on`: other PlanItem titles in the same draft. Resolved
      to bd ids at commit time; an unknown title halts the commit.
    """

    title: str
    description: str
    spec_quote: str
    issue_type: str = "task"
    priority: int = 2
    acceptance: str = ""
    depends_on: list[str] = field(default_factory=list)

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "spec_quote": self.spec_quote,
            "type": self.issue_type,
            "priority": self.priority,
            "acceptance": self.acceptance,
            "depends_on": list(self.depends_on),
        }

    @classmethod
    def from_yaml_dict(cls, raw: dict[str, Any]) -> PlanItem:
        return cls(
            title=str(raw["title"]),
            description=str(raw["description"]),
            spec_quote=str(raw["spec_quote"]),
            issue_type=str(raw.get("type", "task")),
            priority=int(raw.get("priority", 2)),
            acceptance=str(raw.get("acceptance", "")),
            depends_on=list(raw.get("depends_on", [])),
        )


@dataclass
class PlanDraft:
    """The planner's output. One epic, N child items, dep edges
    between items (resolved to titles, not bd ids — bd ids don't
    exist until commit time)."""

    epic_title: str
    epic_description: str
    items: list[PlanItem] = field(default_factory=list)

    def to_yaml(self) -> str:
        payload = {
            "epic_title": self.epic_title,
            "epic_description": self.epic_description,
            "items": [item.to_yaml_dict() for item in self.items],
        }
        # sort_keys=False so the operator sees fields in the schema's
        # natural order rather than alphabetical. The list order within
        # `items` is preserved.
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> PlanDraft:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise PlannerError(f"plan draft is not a mapping; got {type(raw).__name__}")
        try:
            epic_title = str(raw["epic_title"])
            epic_description = str(raw["epic_description"])
            items_raw = list(raw.get("items", []))
        except KeyError as exc:
            raise PlannerError(f"plan draft missing required field: {exc}") from exc
        items = [PlanItem.from_yaml_dict(d) for d in items_raw]
        return cls(epic_title=epic_title, epic_description=epic_description, items=items)


class PlannerError(RuntimeError):
    """Raised when the planner can't build / commit a draft. Distinct
    from `DriverBdError` so callers can tell the difference between
    'bd misbehaved' and 'the draft is malformed.'"""


@dataclass
class PlannerConfig:
    """Configuration for `run_planner`.

    - `spec_path`: the spec doc the planner decomposes.
    - `epic_title`: high-level title for the parent epic.
    - `workspace`: passed to the planner's filesystem tools so
      `read_file` can resolve relative paths.
    - `max_plan_turns`: outer-loop cap on planner iterations. Each
      iteration is one full `run_tool_loop` call.
    - `draft_path`: where `run_planner` writes the YAML draft (and
      where `commit_plan` reads it back).
    """

    spec_path: Path
    epic_title: str
    workspace: Path
    max_plan_turns: int = 5
    draft_path: Path = Path("./harness-plan-draft.yaml")


# --- driver-scoped tools ---------------------------------------------


@dataclass
class _PlannerState:
    """Mutable accumulator the planner tools write into. Lives only for
    the duration of a `run_planner` call; not persisted between
    invocations."""

    items: list[PlanItem] = field(default_factory=list)
    finished: bool = False
    epic_description: str = ""


class PlanAddTool:
    """Driver-scoped tool: append an item to the in-progress plan
    draft. Validates `description` contains a `> ` blockquote of
    `spec_quote` so the operator-review step has something concrete to
    verify against the source spec."""

    def __init__(self, state: _PlannerState) -> None:
        self._state = state

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="plan_add",
            description=(
                "Add one issue to the workplan being assembled. Description "
                "MUST contain a '> ' blockquote of the spec section the "
                "issue implements. Returns 'added: <title>' on success."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "spec_quote": {
                        "type": "string",
                        "description": (
                            "The verbatim spec text the description blockquotes. "
                            "Must be at least 12 characters and appear in the spec file."
                        ),
                    },
                    "issue_type": {
                        "type": "string",
                        "enum": ["task", "feature", "bug"],
                        "default": "task",
                    },
                    "priority": {"type": "integer", "default": 2},
                    "acceptance": {"type": "string", "default": ""},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                        "description": "Titles of other items in this draft that must close first.",
                    },
                },
                "required": ["title", "description", "spec_quote"],
            },
            tier="write",
        )

    def call(
        self,
        *,
        title: str,
        description: str,
        spec_quote: str,
        issue_type: str = "task",
        priority: int = 2,
        acceptance: str = "",
        depends_on: Sequence[str] | None = None,
    ) -> ToolResult:
        title = title.strip()
        spec_quote = spec_quote.strip()
        if not title:
            return ToolResult(
                tool_name="plan_add", output="plan_add: title is empty", success=False
            )
        if len(spec_quote) < _MIN_QUOTE_CHARS:
            return ToolResult(
                tool_name="plan_add",
                output=(
                    f"plan_add: spec_quote must be at least {_MIN_QUOTE_CHARS} characters; "
                    f"got {len(spec_quote)}."
                ),
                success=False,
            )
        if not _description_has_blockquote(description, spec_quote):
            return ToolResult(
                tool_name="plan_add",
                output=(
                    "plan_add: description must include a '> ' blockquote of spec_quote. "
                    "Format: '> <the spec text>'."
                ),
                success=False,
            )
        item = PlanItem(
            title=title,
            description=description,
            spec_quote=spec_quote,
            issue_type=issue_type,
            priority=priority,
            acceptance=acceptance,
            depends_on=list(depends_on or []),
        )
        self._state.items.append(item)
        return ToolResult.text("plan_add", f"added: {title}")


class PlanFinishTool:
    """Driver-scoped tool: signal the planner is done. The run_planner
    loop checks this between iterations and exits the outer loop
    immediately."""

    def __init__(self, state: _PlannerState) -> None:
        self._state = state

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="plan_finish",
            description=(
                "Signal that the plan is complete. Call this when every "
                "section of the spec has been decomposed into plan_add "
                "calls. The planner exits its outer loop on the next "
                "iteration."
            ),
            parameters={"type": "object", "properties": {}},
            tier="write",
        )

    def call(self) -> ToolResult:
        self._state.finished = True
        return ToolResult.text("plan_finish", "plan_finish: planner exit signaled.")


# --- planner system prompt ------------------------------------------


PLANNER_SYSTEM_PROMPT = """\
You are a planner. Your job is to read a spec document and decompose
it into a list of bd issues that an executor can drive to closure one
at a time. You do NOT write code. You do NOT suggest implementations.
You produce a workplan.

Process:
  1. Read the spec via `read_file`.
  2. For each discrete deliverable, call `plan_add(...)` with:
     - title: short bd-style title
     - description: full body. MUST include a '> ' blockquote of the
       relevant spec section.
     - spec_quote: the verbatim text the description blockquotes.
     - issue_type: task | feature | bug (default task).
     - priority: 0..4 (default 2).
     - acceptance: testable closure criteria.
     - depends_on: titles of other items that must close first (within
       this same draft).
  3. Call `plan_finish` when every section of the spec has been
     covered.

Granularity rules:
  - Each item should fit in ONE executor turn — roughly one file, one
    feature, or one test fixture.
  - If a section is too large for one item, split it into multiple
    sub-items with explicit depends_on relationships.
  - Do NOT add items that aren't in the spec. The spec_quote is your
    contract; the validator rejects items whose quotes don't appear
    in the source.
"""


# --- run_planner -----------------------------------------------------


def run_planner(adapter: ModelAdapter, config: PlannerConfig) -> PlanDraft:
    """Drive the LLM through up to `max_plan_turns` orchestrator turns
    until it calls `plan_finish` (or the budget runs out). Returns the
    assembled `PlanDraft`. Does NOT touch bd — that's `commit_plan`.

    The planner registry is intentionally tiny: `read_file`, `grep`,
    `glob`, plus the two driver-scoped tools. No write/edit/shell — the
    planner doesn't touch code."""
    state = _PlannerState()
    registry = _build_planner_registry(config.workspace, state)
    user_message = (
        f"Decompose the spec at `{config.spec_path}` into a workplan. "
        f"Read it first, then call `plan_add` for each item, then "
        f"`plan_finish` when done."
    )
    messages = [
        ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_message),
    ]
    for _ in range(config.max_plan_turns):
        run_tool_loop(
            adapter,  # type: ignore[arg-type]  # narrower _ToolCapableAdapter, checked at runtime
            messages,
            registry,
        )
        if state.finished:
            break
        # Append a continue-nudge so the next turn picks up where this
        # one left off. Keeps the conversation grounded without re-
        # rendering the spec-reading turn.
        messages.append(
            ChatMessage(
                role="user",
                content=(
                    "Continue decomposing the spec. Call plan_add for any "
                    "remaining items; call plan_finish when done."
                ),
            )
        )
    return PlanDraft(
        epic_title=config.epic_title,
        epic_description=state.epic_description or f"Workplan derived from {config.spec_path}.",
        items=state.items,
    )


def write_draft(draft: PlanDraft, path: Path) -> None:
    """Persist a draft to `path`. Atomic via tempfile + rename so a
    crash mid-write doesn't corrupt an existing draft."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(draft.to_yaml())
    tmp.replace(path)


# --- commit_plan -----------------------------------------------------


def commit_plan(draft_path: Path, bd: DriverBd, spec_path: Path) -> str:
    """Materialize a YAML draft into bd. Returns the epic's bd id.

    Validation gate (raises PlannerError on any failure — bd writes
    have NOT started yet):
      1. Every PlanItem.spec_quote must appear in spec_path (substring
         after whitespace normalization).
      2. Every depends_on entry must reference another item in this
         draft.
      3. Item titles must be unique (depends_on lookup is by title).

    Bd materialization (sequential — failure mid-stream leaves a
    partial epic and surfaces a PlannerError. The operator can clean
    up via `bd close` or retry with a corrected draft):
      1. Create the epic.
      2. Create each child issue (with epic-source label).
      3. Add the epic-as-blocked-by-each-child dep so the epic stays
         open until children close.
      4. Add intra-draft depends_on edges.
    """
    if not draft_path.exists():
        raise PlannerError(f"plan draft {draft_path} does not exist")
    if not spec_path.exists():
        raise PlannerError(f"spec file {spec_path} does not exist (needed for quote validation)")

    draft = PlanDraft.from_yaml(draft_path.read_text())
    spec_text_normalized = _WHITESPACE_RE.sub(" ", spec_path.read_text())

    # Validation pre-pass — collect ALL errors before failing so the
    # operator gets a complete picture on first try, not death by 100
    # rejections.
    errors: list[str] = []
    titles_seen: set[str] = set()
    for idx, item in enumerate(draft.items, start=1):
        if item.title in titles_seen:
            errors.append(f"item #{idx}: duplicate title {item.title!r}")
        titles_seen.add(item.title)
        normalized_quote = _WHITESPACE_RE.sub(" ", item.spec_quote).strip()
        if normalized_quote not in spec_text_normalized:
            errors.append(f"item #{idx} ({item.title!r}): spec_quote not found in {spec_path}")
    for idx, item in enumerate(draft.items, start=1):
        for dep_title in item.depends_on:
            if dep_title not in titles_seen:
                errors.append(
                    f"item #{idx} ({item.title!r}): depends_on references "
                    f"unknown title {dep_title!r}"
                )
    if errors:
        raise PlannerError(
            f"plan draft has {len(errors)} validation error(s):\n  - " + "\n  - ".join(errors)
        )

    # Materialize — epic first so children can reference it.
    epic_label = f"plan-source:{spec_path.name}"
    try:
        epic_id = bd.create_with_labels(
            title=draft.epic_title,
            description=draft.epic_description,
            issue_type="feature",
            priority=2,
            labels=[epic_label],
        )
    except DriverBdError as exc:
        raise PlannerError(f"bd create (epic) failed: {exc}") from exc

    title_to_id: dict[str, str] = {}
    for item in draft.items:
        try:
            issue_id = bd.create_with_labels(
                title=item.title,
                description=item.description,
                issue_type=item.issue_type,
                priority=item.priority,
                labels=[epic_label],
                acceptance=item.acceptance or None,
            )
        except DriverBdError as exc:
            raise PlannerError(
                f"bd create ({item.title!r}) failed; epic {epic_id} partially committed: {exc}"
            ) from exc
        title_to_id[item.title] = issue_id
        # Wire the epic-as-blocked-by-child relationship so bd ready
        # correctly walks the graph.
        try:
            bd.dep_add(blocked=epic_id, blocker=issue_id)
        except DriverBdError as exc:
            raise PlannerError(f"bd dep_add (epic→{item.title!r}) failed: {exc}") from exc

    # Intra-draft depends_on edges — done after all children exist so
    # title→id resolution is complete.
    for item in draft.items:
        if not item.depends_on:
            continue
        blocked_id = title_to_id[item.title]
        for dep_title in item.depends_on:
            blocker_id = title_to_id[dep_title]
            try:
                bd.dep_add(blocked=blocked_id, blocker=blocker_id)
            except DriverBdError as exc:
                raise PlannerError(
                    f"bd dep_add ({item.title!r}→{dep_title!r}) failed: {exc}"
                ) from exc

    return epic_id


# --- helpers ---------------------------------------------------------


def _description_has_blockquote(description: str, spec_quote: str) -> bool:
    """True iff `description` contains a markdown blockquote line that
    matches `spec_quote` after whitespace normalization. The match is
    'normalized blockquote line contains normalized quote' so the
    model has some latitude in formatting (extra indentation, etc.)."""
    normalized_quote = _WHITESPACE_RE.sub(" ", spec_quote).strip()
    if not normalized_quote:
        return False
    for raw_line in description.splitlines():
        stripped = raw_line.lstrip()
        if not stripped.startswith(">"):
            continue
        # Drop the '>' prefix + leading space; normalize the rest.
        quote_body = stripped[1:].strip()
        normalized_line = _WHITESPACE_RE.sub(" ", quote_body)
        if normalized_quote in normalized_line:
            return True
    return False


def _build_planner_registry(workspace: Path, state: _PlannerState) -> ToolRegistry:
    """Tiny tool registry for the planner session. No write/shell tools
    — the planner reads + plans, nothing else."""
    catalog = ToolCatalog()
    seed_builtins_into(catalog, now_iso="")
    registry = ToolRegistry(catalog=catalog)
    registry.register(ReadFileTool(root=workspace))
    registry.register(GrepTool(root=workspace))
    registry.register(GlobTool(root=workspace))
    registry.register(PlanAddTool(state))
    registry.register(PlanFinishTool(state))
    return registry


__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "PlanAddTool",
    "PlanDraft",
    "PlanFinishTool",
    "PlanItem",
    "PlannerConfig",
    "PlannerError",
    "commit_plan",
    "run_planner",
    "write_draft",
]
