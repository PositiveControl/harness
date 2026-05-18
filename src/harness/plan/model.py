"""Plan / Subgoal / Action / Precondition value types — harness-qm9n.

Frozen dataclasses + serialization helpers. No I/O, no external state,
no imports outside stdlib. Other Phase 2 slices compose these:

  ptdw.2 — PlanStore that persists Plan instances.
  ptdw.3 — bd → Plan adapter that builds a Plan from bd state.
  ptdw.4 — revise_plan(plan, world) that evaluates preconditions and
            advances Subgoal statuses.
  ptdw.5 — diff_plans(old, new) writeback to bd.

Status values are kept as a Literal alias (not Enum) so the JSON
round-trip is just strings and frozen dataclasses can compare cheaply.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

Status = Literal["pending", "active", "achieved", "abandoned"]
STATUS_VALUES: tuple[Status, ...] = ("pending", "active", "achieved", "abandoned")


@dataclass(frozen=True)
class Precondition:
    """A condition that must be satisfied before a Subgoal can move
    from `pending` to `active`. The evaluator (ptdw.4) interprets
    `kind` + `payload` against a WorldSnapshot — this record carries
    the data, not the logic.

    Stable kind names (extend without breaking the data file):
      - 'bd_closed': payload {'bead': '<bd-id>'}
      - 'bd_open':   payload {'bead': '<bd-id>'}
      - 'timestamp_past': payload {'when': '<ISO-8601>'}

    Unknown kinds are tolerated at load time so a newer revision of
    the evaluator can ship preconditions an older runtime would treat
    as 'never satisfied' rather than crashing.
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """A planned tool call (or other side-effecting move) the agent
    intends to take once the parent Subgoal is `active`. Carried for
    serialization + future plan-revision use; the orchestrator
    doesn't execute Actions directly in v0 (it picks tools through
    the existing tool loop).

    Stable kind names:
      - 'tool_call': payload {'tool': '<name>', 'args': {...}}
      - 'note':      payload {'text': '<free-form>'}
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Subgoal:
    """One node in the plan tree. The plan-revision pass mutates
    Subgoals only through `dataclasses.replace()` so audit-trail
    consumers can diff (old, new) without missing a mid-state.

    `parent_id` is None for the root Subgoal; every other Subgoal
    must point at an existing one in the same Plan. `Plan.with_subgoal()`
    validates the parent reference on add; here we accept any string.
    """

    id: str
    title: str
    status: Status = "pending"
    preconditions: tuple[Precondition, ...] = ()
    actions: tuple[Action, ...] = ()
    parent_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class Plan:
    """A tree of Subgoals rooted at `root_subgoal_id`. The subgoals
    dict is keyed by Subgoal.id so lookups are O(1); the tree shape
    lives in each Subgoal's `parent_id`.

    `with_subgoal()` returns a new Plan instance with the given
    Subgoal merged in (insert or replace). `subgoal()` raises KeyError
    on unknown ids — the caller asked for a guarantee.

    `active()` and `achievable()` are pure views over the existing
    Subgoals. `achievable()` is a stub here that returns the empty
    list; the real precondition evaluator is ptdw.4 (the stub keeps
    the API surface stable so callers can wire it now)."""

    id: str
    title: str
    root_subgoal_id: str
    subgoals: dict[str, Subgoal] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    # --- lookups -----------------------------------------------------

    def subgoal(self, subgoal_id: str) -> Subgoal:
        if subgoal_id not in self.subgoals:
            raise KeyError(f"no subgoal {subgoal_id!r} in plan {self.id!r}")
        return self.subgoals[subgoal_id]

    def children_of(self, subgoal_id: str) -> list[Subgoal]:
        """Subgoals whose `parent_id` matches. Order is insertion-stable
        because Python dicts preserve insertion order."""
        return [s for s in self.subgoals.values() if s.parent_id == subgoal_id]

    def active(self) -> list[Subgoal]:
        """Every Subgoal currently in `status='active'`. Insertion-stable
        ordering."""
        return [s for s in self.subgoals.values() if s.status == "active"]

    def achievable(self) -> list[Subgoal]:
        """Stub: full evaluator lands in ptdw.4. Keeps the API surface
        stable so downstream callers can wire it today. Currently
        returns the empty list — a Subgoal can't transition without
        the real precondition logic."""
        return []

    # --- mutation (returns new Plan) ---------------------------------

    def with_subgoal(self, updated: Subgoal) -> Plan:
        """Insert or replace `updated` keyed on its id. Validates the
        parent reference: if `parent_id` is non-None it must already
        exist in the plan (or be `updated.id` itself, which we reject
        to disallow self-cycles)."""
        if updated.parent_id is not None:
            if updated.parent_id == updated.id:
                raise ValueError(f"subgoal {updated.id!r} cannot be its own parent")
            if updated.parent_id not in self.subgoals:
                raise ValueError(
                    f"subgoal {updated.id!r} parent {updated.parent_id!r} is not in the plan"
                )
        new_subgoals = dict(self.subgoals)
        new_subgoals[updated.id] = updated
        return replace(self, subgoals=new_subgoals, updated_at=_now_iso())

    def with_subgoal_removed(self, subgoal_id: str) -> Plan:
        """Drop a Subgoal by id. Raises KeyError if absent; raises
        ValueError if dropping it would orphan children (caller should
        re-parent first)."""
        if subgoal_id not in self.subgoals:
            raise KeyError(f"no subgoal {subgoal_id!r} in plan {self.id!r}")
        if any(s.parent_id == subgoal_id for s in self.subgoals.values()):
            raise ValueError(f"subgoal {subgoal_id!r} has children; re-parent before removing")
        new_subgoals = {k: v for k, v in self.subgoals.items() if k != subgoal_id}
        return replace(self, subgoals=new_subgoals, updated_at=_now_iso())

    # --- serialization -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict. Subgoals serialize as a dict-of-dicts so the
        on-disk shape stays stable as fields change."""
        return {
            "id": self.id,
            "title": self.title,
            "root_subgoal_id": self.root_subgoal_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "subgoals": {sid: asdict(s) for sid, s in self.subgoals.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Inverse of to_dict. Unknown precondition/action kinds are
        preserved verbatim; missing optional fields default to empty
        tuples / empty strings."""
        if not isinstance(data, dict):
            raise ValueError(f"Plan.from_dict: expected dict, got {type(data).__name__}")
        raw_subgoals = data.get("subgoals", {}) or {}
        if not isinstance(raw_subgoals, dict):
            raise ValueError(
                f"Plan.from_dict: subgoals must be a mapping, got {type(raw_subgoals).__name__}"
            )
        subgoals: dict[str, Subgoal] = {}
        for sid, s_raw in raw_subgoals.items():
            if not isinstance(s_raw, dict):
                raise ValueError(
                    f"Plan.from_dict: subgoal {sid!r} must be a mapping, got {type(s_raw).__name__}"
                )
            subgoals[str(sid)] = _subgoal_from_dict(s_raw)
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            root_subgoal_id=str(data.get("root_subgoal_id", "")),
            subgoals=subgoals,
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


# --- helpers ---------------------------------------------------------


def _now_iso() -> str:
    """Wall-clock ISO-8601 UTC. Centralized so tests + clock injection
    can swap it in one place."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _subgoal_from_dict(data: dict[str, Any]) -> Subgoal:
    preconditions = tuple(
        Precondition(kind=str(p.get("kind", "")), payload=dict(p.get("payload") or {}))
        for p in (data.get("preconditions") or ())
        if isinstance(p, dict)
    )
    actions = tuple(
        Action(kind=str(a.get("kind", "")), payload=dict(a.get("payload") or {}))
        for a in (data.get("actions") or ())
        if isinstance(a, dict)
    )
    status_raw = data.get("status", "pending")
    status: Status = status_raw if status_raw in STATUS_VALUES else "pending"
    return Subgoal(
        id=str(data.get("id", "")),
        title=str(data.get("title", "")),
        status=status,
        preconditions=preconditions,
        actions=actions,
        parent_id=(str(data["parent_id"]) if data.get("parent_id") else None),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
    )


def new_subgoal(
    title: str,
    *,
    parent_id: str | None = None,
    status: Status = "pending",
    preconditions: tuple[Precondition, ...] = (),
    actions: tuple[Action, ...] = (),
    subgoal_id: str | None = None,
) -> Subgoal:
    """Mint a fresh Subgoal with auto-id + timestamps. Use this rather
    than calling the dataclass constructor directly so id-generation
    + timestamp behavior live in one place."""
    now = _now_iso()
    return Subgoal(
        id=subgoal_id or f"sg-{uuid.uuid4().hex[:12]}",
        title=title,
        status=status,
        preconditions=preconditions,
        actions=actions,
        parent_id=parent_id,
        created_at=now,
        updated_at=now,
    )


def new_plan(
    title: str,
    *,
    root_title: str = "root",
    plan_id: str | None = None,
) -> Plan:
    """Mint a fresh Plan with one root Subgoal. The root id is stable
    within the plan (`<plan_id>:root`) so external references survive
    Plan revisions."""
    now = _now_iso()
    pid = plan_id or f"plan-{uuid.uuid4().hex[:12]}"
    root_id = f"{pid}:root"
    root = Subgoal(
        id=root_id,
        title=root_title,
        status="active",
        parent_id=None,
        created_at=now,
        updated_at=now,
    )
    return Plan(
        id=pid,
        title=title,
        root_subgoal_id=root_id,
        subgoals={root_id: root},
        created_at=now,
        updated_at=now,
    )
