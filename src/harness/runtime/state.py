"""Heartbeat state persistence — harness-m64i.

A JSON sidecar at `<character>/data/heartbeat_state.json` (or wherever
the daemon points its `--state-path`) records per-task wall-clock
timestamps + error count + quarantine flag. Survives daemon restarts:
on startup the daemon loads the file and re-applies the persisted
fields to the registered tasks via `Heartbeat.restore_from_disk()`.

The state file is **not** authoritative for scheduling. Monotonic
timestamps reset across processes, so a restored task always re-anchors
its next-fire-time to `now + interval`. The persisted fields are the
human-readable wall-clock timestamps that the `daemon-status` command
displays + the consecutive_errors/quarantined flags that need to
survive restart (otherwise a daemon crash-restart-loop would mask a
broken task instead of keeping it quarantined).

Atomic write: `save_state` writes to a `.tmp` sibling and then
`Path.replace()` swaps. A crash mid-write leaves either the old file
or the new file intact, never a half-written one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TaskState:
    """One task's persisted record.

    `last_success_at` / `last_error_at` are ISO-8601 UTC strings (not
    datetime objects) because JSON doesn't natively serialize them and
    a string round-trip is the cheapest schema. `None` means 'never
    fired in any process that wrote this file'.

    `last_error_msg` captures `repr(exc)` so the daemon-status command
    can surface the most recent failure without parsing logs.

    `consecutive_errors` / `quarantined` are the load-bearing flags
    that survive restart — a daemon that's been crash-looping should
    not silently un-quarantine its busted tasks on every restart.
    """

    last_success_at: str | None = None
    last_error_at: str | None = None
    last_error_msg: str | None = None
    consecutive_errors: int = 0
    quarantined: bool = False


@dataclass
class HeartbeatState:
    """Top-level snapshot. Maps task name -> TaskState."""

    tasks: dict[str, TaskState] = field(default_factory=dict)

    def for_task(self, name: str) -> TaskState:
        """Return the existing record for `name` or a fresh empty one
        if the task isn't in the snapshot. Lets the daemon-status
        command treat 'never-seen task' and 'task with no history'
        identically without a None-check."""
        return self.tasks.get(name, TaskState())


def load_state(path: Path) -> HeartbeatState:
    """Read state JSON.

    A missing file returns an empty `HeartbeatState` — first-ever
    daemon launch is a normal case, not an error.

    A malformed file (truncated JSON, partial write, hand-edit gone
    wrong) also returns empty rather than raising, because crashing
    the daemon over a sidecar bug would be a worse outcome than
    rebuilding state from scratch. The malformation is silent here;
    the daemon log line on first save will make the regeneration
    visible.
    """
    if not path.exists():
        return HeartbeatState()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return HeartbeatState()
    raw_tasks = raw.get("tasks", {}) if isinstance(raw, dict) else {}
    tasks: dict[str, TaskState] = {}
    if isinstance(raw_tasks, dict):
        for name, state in raw_tasks.items():
            if not isinstance(name, str) or not isinstance(state, dict):
                continue
            # Drop unknown keys (forward-compat: a future field added
            # by a newer daemon shouldn't crash an older one reading
            # the same file).
            allowed = {
                "last_success_at",
                "last_error_at",
                "last_error_msg",
                "consecutive_errors",
                "quarantined",
            }
            filtered = {k: v for k, v in state.items() if k in allowed}
            try:
                tasks[name] = TaskState(**filtered)
            except TypeError:
                # Type mismatch on a known field — skip rather than crash.
                continue
    return HeartbeatState(tasks=tasks)


def save_state(state: HeartbeatState, path: Path) -> None:
    """Write state JSON atomically.

    Strategy: serialize to a string, write to `<path>.tmp`, then
    `Path.replace()` to swap. On POSIX `replace` is atomic — a crash
    mid-write leaves the previous file untouched. If the parent
    directory doesn't exist, create it (first-ever daemon launch
    won't have made `<character>/data/` yet for a brand-new character)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tasks": {name: asdict(rec) for name, rec in state.tasks.items()}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)
