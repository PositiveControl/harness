"""Scheduled tool-call heartbeat task — harness-6dnf.

Scope for the first slice: human-curated YAML schedule, interval-only
firing (cron expressions are a follow-up — the hand-rolled scheduler
covers the common 'every N hours' case without pulling in croniter).

Schedule file shape (default
`<character>/data/heartbeat_schedule.yaml`)::

    - id: notam-news        # optional; defaults to "<tool>#<index>"
      tool: search_web
      args:
        query: "FAA NOTAM news"
      interval: 21600       # seconds; 6 h
      description: "Daily NOTAM news check"

Each tick the task:

  1. Loads the schedule (re-reads the file every tick so the operator
     can edit without restarting the daemon).
  2. Loads the per-entry last-fire state from a JSON sidecar (atomic
     write, same pattern as harness-m64i).
  3. Fires every entry whose `now - last_fire >= interval` (or which
     has never fired) and records the new last-fire timestamp.
  4. Persists state, emits a `ScheduledToolsTaskOutcome` to the sink.

Tool execution goes through a caller-supplied `tool_executor` callable
so the task doesn't need a full ToolRegistry import — the daemon
wraps its registry into the executor at wiring time, and tests pass
a mock executor.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml


class ScheduleError(Exception):
    """Raised at schedule-load time for malformed entries. Surfaced at
    daemon startup (or first tick) so a typo doesn't sit silently in
    the schedule file."""


@dataclass(frozen=True)
class ScheduleEntry:
    """One scheduled tool call. `id` must be unique within the schedule;
    when not supplied in YAML it defaults to `<tool>#<index>`."""

    id: str
    tool: str
    args: dict[str, Any]
    interval_s: float
    description: str | None = None


@dataclass(frozen=True)
class ScheduledFire:
    """One fire's audit record: which entry, when it fired, and either
    the tool's output (truncated) or an error message."""

    entry_id: str
    tool: str
    fired_at: datetime
    success: bool
    output_preview: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ScheduledToolsTaskOutcome:
    tick_at: datetime
    entries_loaded: int
    fires: list[ScheduledFire] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)
    load_error: str | None = None


# --- schedule + state I/O ---------------------------------------------------


def load_schedule(path: Path) -> list[ScheduleEntry]:
    """Read the YAML schedule and produce validated entries.

    A missing file returns an empty schedule (first launch, no entries
    yet). A malformed entry (missing required field, non-dict, bad
    interval) raises `ScheduleError` with a message naming the offset
    so the operator can locate it. Validation happens at load time —
    we don't want bad entries to silently sit in the file and fail
    only at fire time.
    """
    if not path.exists():
        return []
    try:
        raw = yaml.safe_load(path.read_text()) or []
    except yaml.YAMLError as exc:
        raise ScheduleError(f"schedule {path}: YAML parse error: {exc}") from exc
    if not isinstance(raw, list):
        raise ScheduleError(f"schedule {path}: top-level must be a list, got {type(raw).__name__}")

    entries: list[ScheduleEntry] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ScheduleError(
                f"schedule {path}[{index}]: entry must be a mapping, got {type(entry).__name__}"
            )
        tool = entry.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ScheduleError(f"schedule {path}[{index}]: 'tool' must be a non-empty string")
        interval = entry.get("interval")
        if not isinstance(interval, (int, float)) or isinstance(interval, bool):
            raise ScheduleError(
                f"schedule {path}[{index}] (tool={tool!r}): 'interval' must be a number"
            )
        if interval <= 0:
            raise ScheduleError(
                f"schedule {path}[{index}] (tool={tool!r}): 'interval' must be positive"
            )
        args = entry.get("args", {})
        if not isinstance(args, dict):
            raise ScheduleError(
                f"schedule {path}[{index}] (tool={tool!r}): 'args' must be a mapping"
            )
        entry_id = entry.get("id") or f"{tool}#{index}"
        if not isinstance(entry_id, str) or not entry_id:
            raise ScheduleError(
                f"schedule {path}[{index}] (tool={tool!r}): 'id' must be a non-empty string"
            )
        if entry_id in seen_ids:
            raise ScheduleError(f"schedule {path}[{index}]: duplicate id {entry_id!r}")
        seen_ids.add(entry_id)
        description = entry.get("description")
        if description is not None and not isinstance(description, str):
            raise ScheduleError(
                f"schedule {path}[{index}] (tool={tool!r}): 'description' must be a string"
            )
        entries.append(
            ScheduleEntry(
                id=entry_id,
                tool=tool,
                args=args,
                interval_s=float(interval),
                description=description,
            )
        )
    return entries


@dataclass
class _ScheduleState:
    """Persisted last-fire timestamps per schedule entry id. ISO-string
    serialization to match the heartbeat state sidecar style; `None`
    means 'never fired'."""

    last_fire_at: dict[str, str | None] = field(default_factory=dict)


def load_schedule_state(path: Path) -> _ScheduleState:
    """Read the per-entry last-fire sidecar. Missing / malformed file
    yields empty state (same defensive policy as heartbeat-state)."""
    if not path.exists():
        return _ScheduleState()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return _ScheduleState()
    if not isinstance(raw, dict):
        return _ScheduleState()
    last = raw.get("last_fire_at", {})
    if not isinstance(last, dict):
        return _ScheduleState()
    return _ScheduleState(
        last_fire_at={
            k: v
            for k, v in last.items()
            if isinstance(k, str) and (v is None or isinstance(v, str))
        }
    )


def save_schedule_state(state: _ScheduleState, path: Path) -> None:
    """Atomic write — .tmp + replace, same pattern as runtime/state.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"last_fire_at": dict(state.last_fire_at)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


# --- task builder ----------------------------------------------------------


def build_scheduled_tools_task(
    *,
    schedule_path: Path,
    state_path: Path,
    tool_executor: Callable[[str, dict[str, Any]], str],
    sink: Callable[[ScheduledToolsTaskOutcome], None] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    preview_chars: int = 200,
) -> Callable[[], None]:
    """Return a zero-arg heartbeat task that fires due schedule entries.

    The task RE-READS the schedule file every tick so the operator can
    edit it without restarting the daemon. A malformed file lands in
    `load_error` on the outcome — every entry's last-fire is preserved
    and the next tick will try again after the operator fixes the
    file. Tool failures are per-entry: one bad call doesn't stop the
    rest of the tick.

    Args:
        schedule_path: YAML file with the schedule entries.
        state_path: JSON sidecar where per-entry last-fire timestamps
            persist.
        tool_executor: callable(tool_name, args) -> output string.
            Wraps the daemon's ToolRegistry; tests pass mocks.
        sink: optional outcome callback.
        clock: injected for tests.
        preview_chars: truncate tool output stored in the outcome
            record to this many characters (full output stays in
            whatever logger / store the tool writes to).
    """

    def task() -> None:
        now = clock()
        try:
            entries = load_schedule(schedule_path)
        except ScheduleError as exc:
            if sink is not None:
                sink(
                    ScheduledToolsTaskOutcome(
                        tick_at=now,
                        entries_loaded=0,
                        load_error=str(exc),
                    )
                )
            return

        state = load_schedule_state(state_path)
        fires: list[ScheduledFire] = []
        skipped: list[str] = []

        for entry in entries:
            last_iso = state.last_fire_at.get(entry.id)
            last_dt = _parse_iso_or_none(last_iso) if last_iso else None
            if last_dt is not None and (now - last_dt) < timedelta(seconds=entry.interval_s):
                skipped.append(entry.id)
                continue
            try:
                output = tool_executor(entry.tool, dict(entry.args))
            except Exception as exc:
                fires.append(
                    ScheduledFire(
                        entry_id=entry.id,
                        tool=entry.tool,
                        fired_at=now,
                        success=False,
                        error=repr(exc),
                    )
                )
                # On error we STILL advance the timestamp — otherwise
                # a permanently broken entry hammers the loop every
                # tick. Quarantine via repeated-failure tracking can
                # come in a follow-up; for now the operator sees the
                # error in the outcome record.
                state.last_fire_at[entry.id] = now.isoformat()
                continue
            preview = output if len(output) <= preview_chars else output[: preview_chars - 1] + "…"
            fires.append(
                ScheduledFire(
                    entry_id=entry.id,
                    tool=entry.tool,
                    fired_at=now,
                    success=True,
                    output_preview=preview,
                )
            )
            state.last_fire_at[entry.id] = now.isoformat()

        if fires:
            save_schedule_state(state, state_path)

        if sink is not None:
            sink(
                ScheduledToolsTaskOutcome(
                    tick_at=now,
                    entries_loaded=len(entries),
                    fires=fires,
                    skipped_ids=skipped,
                )
            )

    return task


def _parse_iso_or_none(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# asdict helper for callers / tests that want to JSON-dump fires.
def fire_to_dict(fire: ScheduledFire) -> dict[str, Any]:
    out = asdict(fire)
    out["fired_at"] = fire.fired_at.isoformat()
    return out
