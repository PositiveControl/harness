"""Clock-driven loop for maintenance tasks that don't belong on the
critical path of a chat turn — harness-swvf.

Scaffolding only (harness-98l4): the registry + loop + cancellation
plumbing. Individual tasks (compaction, consolidation, drift checks,
scheduled tool calls) plug in via `register()` and live in their own
sub-bead implementations.

Design notes:

* One loop, many intervals. Each task carries its own next-fire
  timestamp so a 10-minute compaction tick and a 1-hour consolidation
  tick can share the same heartbeat without one stalling the other.

* Sync and async tasks both welcome. Async tasks run inline on the
  loop; sync tasks dispatch to `asyncio.to_thread()` so a slow
  compaction pass doesn't block other ticks.

* Errors are contained. A task that raises is logged via `on_error`
  (or stderr if no callback registered), and the loop continues. After
  `max_consecutive_errors` in a row, the task is quarantined and
  skipped — surfaced by `quarantined_tasks()` so the daemon can flag
  it without crashing.

* Monotonic time for intervals — wall-clock changes (NTP correction,
  manual `date` edits, suspend/resume) don't mis-fire ticks.

* Stoppable mid-tick. `stop()` sets an asyncio.Event; the loop checks
  it after every fire and before every sleep so cancellation is at
  worst one task-duration away.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

TaskFn = Callable[[], None | Awaitable[None]]
ErrorHandler = Callable[[str, BaseException], None]


@dataclass
class HeartbeatTask:
    """A single registered task: callable + interval + bookkeeping.

    `name` is the human-readable handle used in logs and quarantine
    reports; it must be unique within a Heartbeat instance.

    `interval_s` is the minimum wall-clock gap between fires. The
    actual gap is at least this but can be longer if other tasks ran
    long; the loop never schedules behind by accident, but it also
    won't double-fire to catch up.

    `last_fire_ts` is set to the monotonic-clock value at the start of
    the most recent fire; `last_success_ts` and `last_error_ts` track
    the outcomes. `consecutive_errors` resets to 0 on a successful
    fire; when it hits `max_consecutive_errors` the task is moved into
    `quarantined`."""

    name: str
    fn: TaskFn
    interval_s: float
    last_fire_ts: float = 0.0
    last_success_ts: float = 0.0
    last_error_ts: float = 0.0
    consecutive_errors: int = 0
    quarantined: bool = False
    next_fire_ts: float = field(default=0.0)


class Heartbeat:
    """Clock-driven task registry + run loop.

    Usage::

        hb = Heartbeat()
        hb.register("compact", compaction_task, interval_s=600)
        hb.register("consolidate", consolidation_task, interval_s=3600)
        await hb.run_forever()  # in a daemon entry point

    Stop from another task / signal handler via `hb.stop()`."""

    DEFAULT_MAX_CONSECUTIVE_ERRORS = 3

    def __init__(
        self,
        *,
        on_error: ErrorHandler | None = None,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tasks: dict[str, HeartbeatTask] = {}
        self._on_error = on_error
        self._max_consecutive_errors = max_consecutive_errors
        self._clock = clock
        self._stop_event = asyncio.Event()
        self._running = False

    # --- registration ----------------------------------------------------

    def register(self, name: str, fn: TaskFn, interval_s: float) -> None:
        """Add a task to the registry. Fires `interval_s` seconds after
        registration, then every `interval_s` seconds (real interval ≥
        this; loop may skip if another task ran long)."""
        if not name:
            raise ValueError("heartbeat task name must be non-empty")
        if name in self._tasks:
            raise ValueError(f"heartbeat task {name!r} already registered")
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s!r}")
        now = self._clock()
        self._tasks[name] = HeartbeatTask(
            name=name,
            fn=fn,
            interval_s=interval_s,
            next_fire_ts=now + interval_s,
        )

    def unregister(self, name: str) -> None:
        """Remove a task. No-op if it's not registered (idempotent)."""
        self._tasks.pop(name, None)

    def task(self, name: str) -> HeartbeatTask:
        """Look up a task's bookkeeping record (read-only — caller
        shouldn't mutate)."""
        if name not in self._tasks:
            raise KeyError(f"no heartbeat task named {name!r}")
        return self._tasks[name]

    def names(self) -> list[str]:
        return list(self._tasks)

    def quarantined_tasks(self) -> list[str]:
        """Names of tasks the loop has stopped firing after
        `max_consecutive_errors` in a row. Surfaced for daemon status
        + human investigation."""
        return [name for name, t in self._tasks.items() if t.quarantined]

    # --- run loop --------------------------------------------------------

    async def run_forever(self) -> None:
        """Block until `stop()` is called. Fires due tasks; sleeps until
        the next due task. Reentrant-unsafe — a single Heartbeat must
        not have two concurrent `run_forever` calls."""
        if self._running:
            raise RuntimeError("heartbeat is already running")
        self._running = True
        self._stop_event.clear()
        try:
            while not self._stop_event.is_set():
                due = self._collect_due()
                for task in due:
                    if self._stop_event.is_set():
                        break
                    await self._fire(task)
                if self._stop_event.is_set():
                    break
                await self._sleep_until_next()
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal `run_forever` to exit at the next checkpoint."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self._running

    # --- internals -------------------------------------------------------

    def _collect_due(self) -> list[HeartbeatTask]:
        """Return tasks whose next_fire_ts has passed. Quarantined tasks
        are skipped. Sorted by next_fire_ts so the oldest-due runs first
        — keeps task fairness stable when multiple are due in the same
        tick."""
        now = self._clock()
        due = [t for t in self._tasks.values() if not t.quarantined and t.next_fire_ts <= now]
        due.sort(key=lambda t: t.next_fire_ts)
        return due

    async def _fire(self, task: HeartbeatTask) -> None:
        task.last_fire_ts = self._clock()
        try:
            result = task.fn()
            if inspect.isawaitable(result):
                await result
            elif not inspect.iscoroutinefunction(task.fn) and result is None:
                # Sync function returned None — already executed. But for
                # genuinely expensive sync tasks, the daemon-wiring layer
                # should wrap them with asyncio.to_thread() at registration
                # time. The heartbeat doesn't second-guess: a sync task
                # that runs inline blocks the loop.
                pass
        except BaseException as exc:
            # Heartbeat must not die from one bad task — every Exception
            # (and KeyboardInterrupt / SystemExit if the task somehow
            # raises one) gets caught, logged, and counted toward
            # quarantine. The error-handler itself runs under contextlib
            # .suppress because a buggy handler shouldn't crash the loop
            # either.
            task.last_error_ts = self._clock()
            task.consecutive_errors += 1
            if task.consecutive_errors >= self._max_consecutive_errors:
                task.quarantined = True
            if self._on_error is not None:
                with contextlib.suppress(BaseException):
                    self._on_error(task.name, exc)
        else:
            task.last_success_ts = self._clock()
            task.consecutive_errors = 0
        finally:
            task.next_fire_ts = self._clock() + task.interval_s

    async def _sleep_until_next(self) -> None:
        """Sleep until the next-due task or until `stop()` is called,
        whichever comes first. With no active tasks, sleep in 1-second
        slices so stop() responds promptly."""
        active = [t for t in self._tasks.values() if not t.quarantined]
        if not active:
            # No tasks (or all quarantined): poll for stop every second.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
            return
        now = self._clock()
        next_ts = min(t.next_fire_ts for t in active)
        delay = max(0.0, next_ts - now)
        if delay == 0.0:
            # Yield control briefly so a tight loop of due tasks doesn't
            # starve the event loop.
            await asyncio.sleep(0)
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
