"""PlanStore protocol + JSON sidecar implementation — harness-yrwi.

The persistence abstraction for Plan instances. Three implementations
will exist by the end of Phase 2:

  * JsonPlanStore  — this slice. One JSON file per plan under a root
    directory. Atomic write via the .tmp + replace pattern used
    elsewhere in the runtime (state.py, scheduled_tools.py).
  * BdPlanStore    — read bd state as a Plan (ptdw.3).
  * SqlitePlanStore — graduation path past JSON for larger plans
    (ptdw.7, deferred until proven necessary).

All three implement `PlanStore`, so the runtime never sees which
backend is wired in. JSON is the default — small, debuggable, no
schema migrations needed when Plan fields evolve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from harness.plan.model import Plan


class PlanStoreError(Exception):
    """Raised when an on-disk Plan file is malformed in a way that
    can't be tolerated (corrupt JSON, wrong top-level shape). Missing
    files are NOT errors — they're returned as None from `load`."""


class PlanStore(Protocol):
    """Per-plan persistence. Implementations may use JSON, SQLite, or
    a bd-bridge — the Protocol is the contract every Phase 2 consumer
    codes against."""

    def save(self, plan: Plan) -> None: ...
    def load(self, plan_id: str) -> Plan | None: ...
    def list_ids(self) -> list[str]: ...
    def delete(self, plan_id: str) -> None: ...


class JsonPlanStore:
    """One JSON file per plan under `root`.

    File layout:
      <root>/<plan_id>.json

    Writes are atomic: serialize to a string, write to `<path>.tmp`,
    `Path.replace()` to swap. Crash mid-write leaves the previous
    file (or nothing, if first-ever save). The root directory is
    created lazily on first save — first-launch daemon doesn't have
    to mkdir explicitly.

    Plan ids are used directly as file stems, so they must be
    filesystem-safe. The constructor does no validation on existing
    files — `list_ids` walks `<root>/*.json` and trusts the names.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    # --- save -------------------------------------------------------

    def save(self, plan: Plan) -> None:
        """Persist `plan` atomically. Overwrites any existing file."""
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._path_for(plan.id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)

    # --- load -------------------------------------------------------

    def load(self, plan_id: str) -> Plan | None:
        """Read `plan_id`. Returns None if the file doesn't exist.
        Raises PlanStoreError on corrupt JSON / wrong shape — the
        caller can decide whether to delete + rebuild or surface
        the corruption."""
        path = self._path_for(plan_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise PlanStoreError(f"plan {plan_id!r} at {path}: malformed JSON: {exc}") from exc
        try:
            return Plan.from_dict(raw)
        except ValueError as exc:
            raise PlanStoreError(
                f"plan {plan_id!r} at {path}: malformed Plan shape: {exc}"
            ) from exc

    # --- list -------------------------------------------------------

    def list_ids(self) -> list[str]:
        """Return every plan id with a file on disk, sorted alphabetically.

        Walks `<root>/*.json` filenames; does NOT parse the files
        (cheap enumeration so the daemon-status command can list
        plans without paying load cost)."""
        if not self._root.exists():
            return []
        return sorted(p.stem for p in self._root.glob("*.json"))

    # --- delete -----------------------------------------------------

    def delete(self, plan_id: str) -> None:
        """Remove the plan's file. Idempotent — deleting an already-
        absent plan is a no-op, not an error."""
        path = self._path_for(plan_id)
        path.unlink(missing_ok=True)

    # --- internals --------------------------------------------------

    def _path_for(self, plan_id: str) -> Path:
        return self._root / f"{plan_id}.json"
