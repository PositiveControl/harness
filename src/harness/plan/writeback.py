"""Plan → bd writeback — harness-wxq4.

Closes the bd ↔ Plan loop. Given an `old` Plan and a `new` Plan
(typically the output of `revise_plan`), `diff_plans` computes the
set of bd operations that would mirror the Subgoal-state delta:
closes, reopens, new-bead creates, and free-form comments. The diff
is a pure function — no I/O. `apply_writeback` then executes the
ops against a BeadsAdapter and returns a per-op success/failure
record so callers can retry, surface, or quarantine on partial
failure.

The bd-id round-trip relies on the Subgoal-id convention from
ptdw.3 (`bd:<bead-id>`); `_bd_id_of` is the splitter. Non-bd
Subgoals (the agent created them directly) skip the bd writeback —
they live entirely in the Plan until ptdw.5's create-side fills
in a bd id.

Idempotency comes from two layers:
  1. `diff_plans` only emits ops for actual transitions (status
     mismatch between old and new), so re-applying the same
     `(old, new)` pair on top of a partially-applied state produces
     no spurious ops on the rows that already advanced.
  2. `apply_writeback` swallows the bd-CLI "already closed" /
     "already open" errors. Mid-tick crash + retry is safe.

Partial-failure model: each op is recorded individually in the
result. One failing op (network blip, bd lock contention) doesn't
abort the rest. Callers compose retry policy from the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.plan.model import Plan, Subgoal


@dataclass(frozen=True)
class BeadsIssueDraft:
    """Fields the writeback needs to create a brand-new bead. Matches
    BeadsAdapter.create's signature (title, scope, issue_type,
    description, priority, deps, assignee) — minus scope which is set
    by the caller-supplied default at apply time.

    `subgoal_id` is the Plan-side id the new bead corresponds to;
    after creation the caller can update the Plan to use
    `bd:<new-bd-id>` as the Subgoal id."""

    subgoal_id: str
    title: str
    description: str = ""
    issue_type: str = "task"
    priority: int = 2
    deps: tuple[str, ...] = ()
    extra_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanWriteback:
    """Aggregate of bd ops the writeback would perform.

    Tuples (not lists) for hashability + immutability. Empty tuples
    mean 'no ops of this kind' — callers can short-circuit on
    `not any((wb.to_close, wb.to_reopen, ...))`.

    `comments` carries free-form annotations the caller can attach
    to a bead — `(bead_id, body)` pairs. Currently unused by
    diff_plans (we don't translate Subgoal title/description edits
    into comments yet); the field is here so a future plan-revision
    can emit operator-facing notes without changing the writeback
    schema.
    """

    to_close: tuple[str, ...] = ()
    to_reopen: tuple[str, ...] = ()
    to_create: tuple[BeadsIssueDraft, ...] = ()
    comments: tuple[tuple[str, str], ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.to_close or self.to_reopen or self.to_create or self.comments)


@dataclass(frozen=True)
class WritebackOpResult:
    """One bd op's outcome. `kind` groups results in the aggregate
    summary; `target` is whatever string identifies the op (bead id
    for close/reopen/comment, subgoal id for create); `created_id`
    is set only on successful create ops (the new bd id the caller
    can use to update the Plan)."""

    kind: str
    target: str
    success: bool
    error: str | None = None
    created_id: str | None = None


@dataclass(frozen=True)
class WritebackResult:
    """Aggregate of per-op outcomes from a single `apply_writeback`
    call. The caller iterates `ops` to retry / surface failures; the
    convenience flags + counts are derived views."""

    ops: tuple[WritebackOpResult, ...] = field(default_factory=tuple)

    @property
    def all_succeeded(self) -> bool:
        return all(op.success for op in self.ops)

    @property
    def errors(self) -> list[WritebackOpResult]:
        return [op for op in self.ops if not op.success]

    def created_ids(self) -> dict[str, str]:
        """Map of subgoal_id -> new bd id for the create ops that
        succeeded. Caller uses this to re-id Subgoals in the Plan."""
        return {
            op.target: op.created_id for op in self.ops if op.kind == "create" and op.created_id
        }


# --- diff -----------------------------------------------------------------


def _bd_id_of(subgoal: Subgoal) -> str | None:
    """Extract the bd id from a `bd:<id>` Subgoal id. Same convention
    as revise.py — root Subgoals (':root' suffix) are excluded so the
    root never becomes a bd target."""
    if subgoal.id.startswith("bd:") and not subgoal.id.endswith(":root"):
        return subgoal.id[len("bd:") :]
    return None


def diff_plans(old: Plan, new: Plan) -> PlanWriteback:
    """Compute the bd writeback that would mirror the Subgoal-state
    delta between `old` and `new`.

    Op selection (per-Subgoal in BOTH plans):
      * old.status == 'active' and new.status == 'achieved' (bd-sourced):
          to_close
      * old.status == 'achieved' and new.status in ('pending', 'active')
          (bd-sourced): to_reopen   — rare: a Plan revision undid an
          achievement, e.g. the bead got reopened in bd
      * other transitions: no bd op (pending<->active is purely a
          Plan-side bookkeeping move; bd doesn't model 'active' vs
          'pending' the same way)

    Op selection (Subgoals in `new` but not `old`):
      * non-bd id (no 'bd:' prefix): to_create with a BeadsIssueDraft
      * bd id: skipped — a bd-prefixed Subgoal already corresponds to
        an existing bead, so it shouldn't be 'new' to bd

    Subgoals in `old` but not `new` are skipped — Plan revisions
    don't auto-delete beads. The operator deletes via bd directly.

    Determinism: ops are sorted by target id so two diffs of the
    same (old, new) pair produce identical PlanWriteback values.
    """
    to_close: list[str] = []
    to_reopen: list[str] = []
    to_create: list[BeadsIssueDraft] = []

    old_ids = set(old.subgoals)
    new_ids = set(new.subgoals)

    for sid in old_ids & new_ids:
        old_sg = old.subgoals[sid]
        new_sg = new.subgoals[sid]
        if old_sg.status == new_sg.status:
            continue
        bd_id = _bd_id_of(new_sg)
        if bd_id is None:
            continue
        if old_sg.status == "active" and new_sg.status == "achieved":
            to_close.append(bd_id)
        elif old_sg.status == "achieved" and new_sg.status in ("pending", "active"):
            to_reopen.append(bd_id)
        # Other transitions (pending<->active, ->abandoned) don't map
        # to bd ops. Abandonment is captured Plan-side; bd has no
        # native "abandoned" status.

    for sid in new_ids - old_ids:
        new_sg = new.subgoals[sid]
        if _bd_id_of(new_sg) is not None:
            # Plan claims this Subgoal already has a bd backing; nothing
            # to create.
            continue
        if sid == new.root_subgoal_id:
            # Root is a Plan-only synthesized node, not a bd target.
            continue
        to_create.append(
            BeadsIssueDraft(
                subgoal_id=sid,
                title=new_sg.title,
                description="",
            )
        )

    return PlanWriteback(
        to_close=tuple(sorted(to_close)),
        to_reopen=tuple(sorted(to_reopen)),
        to_create=tuple(sorted(to_create, key=lambda d: d.subgoal_id)),
    )


# --- apply ----------------------------------------------------------------


# Substring tags bd emits in its error messages when the requested
# transition is already in place. Treated as success so writeback
# retries on a partially-applied tick are idempotent.
_ALREADY_CLOSED_MARKERS: tuple[str, ...] = ("already closed", "is closed")
_ALREADY_OPEN_MARKERS: tuple[str, ...] = ("already open", "is open")


def _is_already_closed_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _ALREADY_CLOSED_MARKERS)


def _is_already_open_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _ALREADY_OPEN_MARKERS)


def apply_writeback(
    adapter: Any,
    writeback: PlanWriteback,
    *,
    scope: str = "professional",
    assignee: str | None = None,
) -> WritebackResult:
    """Execute `writeback` against `adapter`. Returns a per-op
    `WritebackResult`.

    Args:
        adapter: BeadsAdapter (or duck-typed equivalent with close /
            reopen / create / comment_add methods).
        scope: bd scope label applied to newly-created beads. Default
            'professional' matches the most common case; pass
            'personal' for personal-scope ops.
        assignee: bd assignee for newly-created beads. None lets bd
            pick its default (typically the user).
    """
    ops: list[WritebackOpResult] = []

    for bead_id in writeback.to_close:
        try:
            adapter.close(bead_id, reason="closed via plan writeback")
            ops.append(WritebackOpResult(kind="close", target=bead_id, success=True))
        except Exception as exc:
            if _is_already_closed_error(exc):
                # Idempotent — already closed counts as success.
                ops.append(WritebackOpResult(kind="close", target=bead_id, success=True))
            else:
                ops.append(
                    WritebackOpResult(kind="close", target=bead_id, success=False, error=repr(exc))
                )

    for bead_id in writeback.to_reopen:
        try:
            adapter.reopen(bead_id, reason="reopened via plan writeback")
            ops.append(WritebackOpResult(kind="reopen", target=bead_id, success=True))
        except Exception as exc:
            if _is_already_open_error(exc):
                ops.append(WritebackOpResult(kind="reopen", target=bead_id, success=True))
            else:
                ops.append(
                    WritebackOpResult(kind="reopen", target=bead_id, success=False, error=repr(exc))
                )

    for draft in writeback.to_create:
        try:
            new_id = adapter.create(
                title=draft.title,
                scope=scope,
                issue_type=draft.issue_type,
                description=draft.description,
                priority=draft.priority,
                deps=draft.deps,
                extra_labels=draft.extra_labels,
                assignee=assignee,
            )
            ops.append(
                WritebackOpResult(
                    kind="create",
                    target=draft.subgoal_id,
                    success=True,
                    created_id=str(new_id),
                )
            )
        except Exception as exc:
            ops.append(
                WritebackOpResult(
                    kind="create",
                    target=draft.subgoal_id,
                    success=False,
                    error=repr(exc),
                )
            )

    for bead_id, body in writeback.comments:
        try:
            adapter.comment_add(bead_id, body)
            ops.append(WritebackOpResult(kind="comment", target=bead_id, success=True))
        except Exception as exc:
            ops.append(
                WritebackOpResult(kind="comment", target=bead_id, success=False, error=repr(exc))
            )

    return WritebackResult(ops=tuple(ops))
