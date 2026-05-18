"""Tests for the heartbeat scaffolding — harness-98l4."""

from __future__ import annotations

import asyncio

import pytest

from harness.runtime.heartbeat import Heartbeat, HeartbeatTask


@pytest.mark.asyncio
async def test_register_and_run_two_tasks_with_different_intervals() -> None:
    """Two tasks with different intervals share one loop. Over a 0.6s
    window with 0.1s and 0.2s intervals, expect ~6 and ~3 fires
    respectively, ±1 for scheduling jitter."""
    hb = Heartbeat()
    fires_a = 0
    fires_b = 0

    async def task_a() -> None:
        nonlocal fires_a
        fires_a += 1

    async def task_b() -> None:
        nonlocal fires_b
        fires_b += 1

    hb.register("a", task_a, interval_s=0.1)
    hb.register("b", task_b, interval_s=0.2)

    run_task = asyncio.create_task(hb.run_forever())
    await asyncio.sleep(0.65)
    hb.stop()
    await run_task

    # task_a fires at t=0.1, 0.2, 0.3, 0.4, 0.5, 0.6 -> 5-6 fires
    # task_b fires at t=0.2, 0.4, 0.6 -> 2-3 fires
    assert 4 <= fires_a <= 7, f"task_a fired {fires_a} times"
    assert 2 <= fires_b <= 4, f"task_b fired {fires_b} times"


@pytest.mark.asyncio
async def test_stop_halts_run_loop_within_one_tick() -> None:
    """stop() exits the loop without waiting for the next interval."""
    hb = Heartbeat()

    async def noop() -> None:
        pass

    hb.register("noop", noop, interval_s=10.0)  # long interval — only stop() unblocks
    run_task = asyncio.create_task(hb.run_forever())
    await asyncio.sleep(0.05)
    hb.stop()
    # Should resolve well under the 10s interval — give it 1s grace.
    await asyncio.wait_for(run_task, timeout=1.0)
    assert not hb.is_running


@pytest.mark.asyncio
async def test_task_that_raises_does_not_crash_loop() -> None:
    """A task that throws is caught; on_error sees the exception; the
    loop keeps firing other tasks."""
    hb = Heartbeat()
    errors: list[tuple[str, BaseException]] = []
    healthy_fires = 0

    def boom() -> None:
        raise RuntimeError("intentional")

    async def healthy() -> None:
        nonlocal healthy_fires
        healthy_fires += 1

    hb = Heartbeat(on_error=lambda name, exc: errors.append((name, exc)))
    hb.register("boom", boom, interval_s=0.1)
    hb.register("healthy", healthy, interval_s=0.1)

    run_task = asyncio.create_task(hb.run_forever())
    await asyncio.sleep(0.35)
    hb.stop()
    await run_task

    assert healthy_fires >= 2, f"healthy task fired only {healthy_fires} times"
    assert all(name == "boom" for name, _ in errors)
    assert all(isinstance(exc, RuntimeError) for _, exc in errors)


@pytest.mark.asyncio
async def test_sync_task_signature_supported() -> None:
    """Sync `def task(): ...` works alongside `async def`."""
    hb = Heartbeat()
    fires = 0

    def sync_task() -> None:
        nonlocal fires
        fires += 1

    hb.register("sync", sync_task, interval_s=0.05)
    run_task = asyncio.create_task(hb.run_forever())
    await asyncio.sleep(0.3)
    hb.stop()
    await run_task

    assert fires >= 3, f"sync task fired only {fires} times"


@pytest.mark.asyncio
async def test_max_consecutive_errors_quarantines_task() -> None:
    """A task that fails `max_consecutive_errors` times in a row is
    moved into the quarantine list and skipped on subsequent ticks."""
    hb = Heartbeat(max_consecutive_errors=2)

    def always_fails() -> None:
        raise ValueError("dead task")

    hb.register("always_fails", always_fails, interval_s=0.05)
    run_task = asyncio.create_task(hb.run_forever())
    await asyncio.sleep(0.3)
    hb.stop()
    await run_task

    quarantined = hb.quarantined_tasks()
    assert "always_fails" in quarantined
    record = hb.task("always_fails")
    # consecutive_errors stops incrementing after quarantine since the
    # task no longer fires. Lower bound = max_consecutive_errors.
    assert record.consecutive_errors >= 2


@pytest.mark.asyncio
async def test_successful_fire_resets_error_counter() -> None:
    """An error count that hasn't hit the quarantine threshold resets
    on the next successful fire."""
    hb = Heartbeat(max_consecutive_errors=5)
    counter = {"n": 0}

    def flaky() -> None:
        counter["n"] += 1
        if counter["n"] == 1:
            raise RuntimeError("transient")

    hb.register("flaky", flaky, interval_s=0.05)
    run_task = asyncio.create_task(hb.run_forever())
    await asyncio.sleep(0.25)
    hb.stop()
    await run_task

    record = hb.task("flaky")
    # First fire raised, subsequent fires succeeded -> counter reset to 0.
    assert record.consecutive_errors == 0
    assert not record.quarantined


def test_register_rejects_duplicate_name() -> None:
    hb = Heartbeat()

    async def t() -> None:
        pass

    hb.register("a", t, interval_s=1.0)
    with pytest.raises(ValueError, match="already registered"):
        hb.register("a", t, interval_s=1.0)


def test_register_rejects_non_positive_interval() -> None:
    hb = Heartbeat()

    async def t() -> None:
        pass

    with pytest.raises(ValueError, match="must be positive"):
        hb.register("a", t, interval_s=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        hb.register("a", t, interval_s=-1.0)


def test_register_rejects_empty_name() -> None:
    hb = Heartbeat()

    async def t() -> None:
        pass

    with pytest.raises(ValueError, match="non-empty"):
        hb.register("", t, interval_s=1.0)


def test_unregister_is_idempotent() -> None:
    hb = Heartbeat()

    async def t() -> None:
        pass

    hb.register("a", t, interval_s=1.0)
    hb.unregister("a")
    hb.unregister("a")  # second call must not raise
    assert "a" not in hb.names()


def test_task_lookup_raises_for_unknown() -> None:
    hb = Heartbeat()
    with pytest.raises(KeyError, match="no heartbeat task"):
        hb.task("missing")


@pytest.mark.asyncio
async def test_run_forever_is_not_reentrant() -> None:
    """Calling run_forever while it's already running raises — a single
    Heartbeat instance must drive one loop at a time."""
    hb = Heartbeat()
    run_task = asyncio.create_task(hb.run_forever())
    await asyncio.sleep(0.05)
    with pytest.raises(RuntimeError, match="already running"):
        await hb.run_forever()
    hb.stop()
    await run_task


@pytest.mark.asyncio
async def test_run_forever_with_no_tasks_is_stoppable() -> None:
    """An empty heartbeat doesn't busy-loop; stop() still wakes it."""
    hb = Heartbeat()
    run_task = asyncio.create_task(hb.run_forever())
    await asyncio.sleep(0.05)
    hb.stop()
    # No-task loop polls every 1s for stop; allow 2s grace.
    await asyncio.wait_for(run_task, timeout=2.0)


def test_heartbeat_task_dataclass_shape() -> None:
    """Pin the public bookkeeping shape so consumers (daemon status,
    observability bead m64i) can rely on the field set."""
    task = HeartbeatTask(name="t", fn=lambda: None, interval_s=1.0)
    assert task.name == "t"
    assert task.interval_s == 1.0
    assert task.last_fire_ts == 0.0
    assert task.last_success_ts == 0.0
    assert task.last_error_ts == 0.0
    assert task.consecutive_errors == 0
    assert task.quarantined is False
    assert task.next_fire_ts == 0.0
