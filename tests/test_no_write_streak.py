"""Unit tests for orchestrator/no_write_streak.py — harness-41b3.

The detector is a small per-turn counter the tool loop consults after
each tool dispatch when the caller (today: driver IMPLEMENT phase)
arms it. These tests pin the threshold semantics, the write-tool reset
behavior, and the one-shot-per-turn nudge contract. The end-to-end
"loop appends nudge to working thread" behavior is covered separately
in test_tool_loop.py."""

from __future__ import annotations

from harness.orchestrator.no_write_streak import (
    DEFAULT_NO_WRITE_STREAK_THRESHOLD,
    NoWriteStreakDetector,
    build_nudge_text,
)
from harness.tools.base import ToolCall, ToolResult


def _read(path: str = "x.txt") -> tuple[ToolCall, ToolResult]:
    """Helper: build a successful read_file call+result pair."""
    call = ToolCall(name="read_file", arguments={"path": path})
    result = ToolResult(tool_name="read_file", output="ok", success=True)
    return call, result


def _write(path: str = "x.txt", success: bool = True) -> tuple[ToolCall, ToolResult]:
    """Helper: build an edit_file call+result pair (success toggleable)."""
    call = ToolCall(
        name="edit_file",
        arguments={"path": path, "old_string": "a", "new_string": "b"},
    )
    result = ToolResult(
        tool_name="edit_file",
        output="wrote 1 byte" if success else "old_string not found",
        success=success,
    )
    return call, result


def test_default_threshold_is_four() -> None:
    """Pin the constant — the integration test + driver behavior both
    assume 4. If we change the default, both call sites need re-tuning."""
    assert DEFAULT_NO_WRITE_STREAK_THRESHOLD == 4


def test_detector_fires_only_at_threshold() -> None:
    """First three non-writes return False; the FOURTH (threshold)
    returns True. One-shot per turn — subsequent non-writes return
    False even though the streak keeps growing."""
    detector = NoWriteStreakDetector()  # default threshold=4
    call, result = _read()

    assert detector.observe(call, result) is False  # streak=1
    assert detector.observe(call, result) is False  # streak=2
    assert detector.observe(call, result) is False  # streak=3
    assert detector.observe(call, result) is True  # streak=4, first cross
    assert detector.observe(call, result) is False  # streak=5, already fired


def test_streak_property_tracks_count() -> None:
    """The `streak` property exposes the running count — callers
    (the tool loop) need it to render the magnitude in the nudge."""
    detector = NoWriteStreakDetector()
    call, result = _read()

    assert detector.streak == 0
    detector.observe(call, result)
    assert detector.streak == 1
    detector.observe(call, result)
    assert detector.streak == 2


def test_successful_write_resets_streak() -> None:
    """A successful edit_file zeros the counter — the spiral broke,
    real progress landed. After the reset the model gets a fresh
    threshold-worth of investigation before the detector fires
    again. Multi-write turns shouldn't false-positive."""
    detector = NoWriteStreakDetector()
    read_call, read_result = _read()
    write_call, write_result = _write()

    detector.observe(read_call, read_result)  # streak=1
    detector.observe(read_call, read_result)  # streak=2
    detector.observe(read_call, read_result)  # streak=3
    assert detector.streak == 3
    fired = detector.observe(write_call, write_result)
    assert fired is False
    assert detector.streak == 0


def test_successful_write_resets_but_one_shot_holds() -> None:
    """Once fired this turn, a later write resets the streak but the
    detector still won't fire again. One-shot is per-turn, not per-
    streak-segment — fresh detector instance per turn means the
    'next turn' case is covered by the driver instantiating new state."""
    detector = NoWriteStreakDetector()
    read_call, read_result = _read()
    write_call, write_result = _write()

    # Cross threshold once.
    for _ in range(4):
        detector.observe(read_call, read_result)

    # Reset via successful write.
    detector.observe(write_call, write_result)
    assert detector.streak == 0

    # Run another full streak — must NOT fire again this turn.
    for _ in range(5):
        result = detector.observe(read_call, read_result)
        assert result is False


def test_failed_write_does_not_reset_streak() -> None:
    """A failed edit_file (old_string not found, dedup rejection, etc.)
    is not progress and must not reset the counter. The edit_file
    hard-stop (harness-a9f6) owns the 'tried but kept failing' path;
    this detector continues to count it as part of the spiral so
    the model gets the broader 'change approach' nudge if it never
    breaks out."""
    detector = NoWriteStreakDetector()
    read_call, read_result = _read()
    failed_write = _write(success=False)

    detector.observe(read_call, read_result)  # streak=1
    detector.observe(*failed_write)  # streak=2 (no reset on failed write)
    detector.observe(read_call, read_result)  # streak=3
    assert detector.observe(read_call, read_result) is True  # streak=4
    assert detector.streak == 4


def test_write_file_also_counts_as_write() -> None:
    """`write_file` is the other write tool — successful write_file
    must reset the streak just like edit_file. Mirrors the set checked
    by `fsm_turn._resolve_implement_outcome`."""
    detector = NoWriteStreakDetector()
    read_call, read_result = _read()
    write_call = ToolCall(name="write_file", arguments={"path": "x.txt", "content": "hi"})
    write_result = ToolResult(tool_name="write_file", output="wrote", success=True)

    detector.observe(read_call, read_result)
    detector.observe(read_call, read_result)
    assert detector.streak == 2

    detector.observe(write_call, write_result)
    assert detector.streak == 0


def test_custom_threshold_lower_for_tight_test() -> None:
    """`threshold` is constructor-configurable so callers (today: the
    driver, tomorrow: bench scripts measuring sensitivity) can tune
    independently of the default. Pin the construction path with a
    threshold of 2 — a tight setting useful for tests and not much
    else."""
    detector = NoWriteStreakDetector(threshold=2)
    call, result = _read()

    assert detector.observe(call, result) is False  # streak=1
    assert detector.observe(call, result) is True  # streak=2


def test_nudge_text_includes_count() -> None:
    """The nudge string interpolates the streak count verbatim so the
    model can see the magnitude of what it just did. Also pins the
    three-option (a)(b)(c) escape-hatch shape — the model is
    conditioned to react to that shape by other catchers."""
    text = build_nudge_text(4)
    assert "4 consecutive tool calls" in text
    assert "(a)" in text
    assert "(b)" in text
    assert "(c)" in text
    assert "edit_file" in text
    assert "submit_implementation_complete" in text
