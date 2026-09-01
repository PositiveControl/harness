"""Unit tests for orchestrator/no_write_streak.py — harness-41b3.

The detector is a small per-turn counter the tool loop consults after
each tool dispatch when the caller (today: driver IMPLEMENT phase)
arms it. These tests pin the threshold semantics, the write-tool reset
behavior, and the one-shot-per-turn nudge contract. The end-to-end
"loop appends nudge to working thread" behavior is covered separately
in test_tool_loop.py."""

from __future__ import annotations

from harness.orchestrator.no_write_streak import (
    DEFAULT_NO_SUBMIT_STREAK_THRESHOLD,
    DEFAULT_NO_TEST_SUBMIT_STREAK_THRESHOLD,
    DEFAULT_NO_WRITE_STREAK_THRESHOLD,
    DEFAULT_READ_RESERVE,
    NoSubmitStreakDetector,
    NoTestSubmitStreakDetector,
    NoWriteStreakDetector,
    ReadReservation,
    build_nudge_text,
)
from harness.tools.base import ToolCall, ToolResult


def _submit(name: str = "submit_assessment") -> tuple[ToolCall, ToolResult]:
    return (
        ToolCall(name=name, arguments={}),
        ToolResult(tool_name=name, output="recorded", success=True),
    )


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


def test_detector_fires_at_threshold_then_refires_on_cadence() -> None:
    """First three non-writes return False; the FOURTH (threshold) fires.
    Then it re-fires every `refire_every` (2) further non-writes —
    streak 6, 8, … — instead of latching silent (loop_run=135f0d99: the
    model ignored the single one-shot nudge)."""
    detector = NoWriteStreakDetector()  # threshold=4, refire_every=2

    def obs() -> bool:
        return detector.observe(*_read())

    assert obs() is False  # streak=1
    assert obs() is False  # streak=2
    assert obs() is False  # streak=3
    assert obs() is True  # streak=4, first cross (tier 1)
    assert obs() is False  # streak=5
    assert obs() is True  # streak=6, re-fire (tier 2)
    assert obs() is False  # streak=7
    assert obs() is True  # streak=8, re-fire (tier 3)
    assert detector.fires == 3


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


def test_successful_write_resets_streak_and_escalation_tier() -> None:
    """A successful write breaks the spiral: it zeros the streak AND the
    escalation tier. A later relapse fires again from tier 1 — real
    progress earns the model a fresh threshold-worth of investigation
    before the (soft) nudge returns."""
    detector = NoWriteStreakDetector()
    read_call, read_result = _read()
    write_call, write_result = _write()

    # Cross threshold once (tier 1).
    for _ in range(4):
        detector.observe(read_call, read_result)
    assert detector.fires == 1

    # Reset via successful write.
    detector.observe(write_call, write_result)
    assert detector.streak == 0
    assert detector.fires == 0

    # A fresh full streak fires again — and from tier 1, not a carried tier.
    assert detector.observe(read_call, read_result) is False  # 1
    assert detector.observe(read_call, read_result) is False  # 2
    assert detector.observe(read_call, read_result) is False  # 3
    assert detector.observe(read_call, read_result) is True  # 4 → fire
    assert detector.fires == 1


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
    by `fsm_executor._resolve_implement_outcome`."""
    detector = NoWriteStreakDetector()
    read_call, read_result = _read()
    write_call = ToolCall(name="write_file", arguments={"path": "x.txt", "content": "hi"})
    write_result = ToolResult(tool_name="write_file", output="wrote", success=True)

    detector.observe(read_call, read_result)
    detector.observe(read_call, read_result)
    assert detector.streak == 2

    detector.observe(write_call, write_result)
    assert detector.streak == 0


def test_stream_edit_counts_as_a_write() -> None:
    """`stream_edit` is the driver IMPLEMENT roster's in-place edit path
    (awk/sed/cut/tr). A productive stream_edit is real progress and must
    reset the streak — otherwise the model gets falsely nudged for doing
    the right thing. Mirrors WRITE_TOOL_NAMES shared with
    fsm_executor._resolve_implement_outcome."""
    detector = NoWriteStreakDetector()
    read_call, read_result = _read()
    stream_call = ToolCall(name="stream_edit", arguments={"tool": "sed", "args": ["s/a/b/"]})
    stream_result = ToolResult(tool_name="stream_edit", output="edited", success=True)

    detector.observe(read_call, read_result)
    detector.observe(read_call, read_result)
    assert detector.streak == 2

    detector.observe(stream_call, stream_result)
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


def test_first_fire_renders_soft_menu() -> None:
    """fires=1 (default) is the soft tier — no ESCALATING framing yet."""
    text = build_nudge_text(4, fires=1)
    assert "ESCALATING" not in text
    assert "NO-WRITE STREAK" in text


def test_escalated_nudge_is_mandatory_and_counts_the_warning() -> None:
    """fires>=2 drops the optional menu framing for a mandatory directive
    that names which warning this is (loop_run=135f0d99: the soft nudge was
    ignored, so the re-fire must read as a wall, not a suggestion)."""
    text = build_nudge_text(6, fires=2)
    assert "ESCALATING" in text
    assert "2nd time" in text
    assert "STOP READING" in text
    # Still routes to the same three legal moves.
    assert "edit_file" in text
    assert "submit_implementation_complete" in text
    assert "flag_blocked" in text


def test_detector_nudge_escalates_with_fires() -> None:
    """The detector's own nudge() rises in tier as it re-fires."""
    detector = NoWriteStreakDetector()
    fired_texts = [detector.nudge() for _ in range(8) if detector.observe(*_read())]
    assert len(fired_texts) == 3  # streak 4, 6, 8
    assert "ESCALATING" not in fired_texts[0]
    assert "2nd time" in fired_texts[1]
    assert "3rd time" in fired_texts[2]


# --- ReadReservation (IMPLEMENT hard backstop, loop_run=135f0d99) -----


def test_read_reserve_default_is_two() -> None:
    assert DEFAULT_READ_RESERVE == 2
    assert ReadReservation().reserve == 2


def test_reservation_blocks_read_only_in_reserve_window() -> None:
    """With rounds_left at or below the reserve, read-only exploration is
    blocked so the remaining budget goes to a write or a decision."""
    res = ReadReservation()  # reserve=2
    read_call, _ = _read("game.js")

    assert res.should_block(read_call, rounds_left=3) is False  # outside window
    assert res.should_block(read_call, rounds_left=2) is True  # at the wall
    assert res.should_block(read_call, rounds_left=1) is True
    assert res.should_block(read_call, rounds_left=0) is True


def test_reservation_never_blocks_writes_or_decisions() -> None:
    """The wall forces a move — it must leave the write tools and the
    phase-exit decisions reachable even at zero rounds left."""
    res = ReadReservation()
    write_call, _ = _write("game.js")
    submit_call = ToolCall(name="submit_implementation_complete", arguments={"summary": "x"})
    flag_call = ToolCall(name="flag_blocked", arguments={"missing": "x", "reason": "y"})

    for call in (write_call, submit_call, flag_call):
        assert res.should_block(call, rounds_left=0) is False


def test_reservation_block_result_is_a_forcing_failure() -> None:
    """The substituted result fails the call and names the legal next moves
    so the model reads it as a redirect, not an opaque error."""
    res = ReadReservation()
    read_call, _ = _read("game.js")

    result = res.block_result(read_call, rounds_left=1)
    assert result.success is False
    assert result.error == "read_budget_reserved"
    assert "read-only tools are now locked" in result.output
    assert "read_file" in result.output  # names the refused tool
    assert "submit_implementation_complete" in result.output


# --- NoSubmitStreakDetector (ASSESS twin, loop_run=ad30d9ad) ----------


def test_no_submit_default_threshold_is_three() -> None:
    assert DEFAULT_NO_SUBMIT_STREAK_THRESHOLD == 3
    assert NoSubmitStreakDetector().threshold == 3


def test_no_submit_fires_at_threshold() -> None:
    """Three read-only calls in ASSESS without submitting → fire once."""
    det = NoSubmitStreakDetector()
    read_call, read_result = _read("game.js")
    assert det.observe(read_call, read_result) is False  # 1
    assert det.observe(read_call, read_result) is False  # 2
    assert det.observe(read_call, read_result) is True  # 3 → fire
    # one-shot
    assert det.observe(read_call, read_result) is False


def test_no_submit_reset_by_submit_assessment() -> None:
    """A successful submit_assessment is ASSESS progress — resets."""
    det = NoSubmitStreakDetector()
    read_call, read_result = _read("game.js")
    det.observe(read_call, read_result)
    det.observe(read_call, read_result)
    assert det.streak == 2
    det.observe(*_submit("submit_assessment"))
    assert det.streak == 0


def test_no_submit_reset_by_flag_blocked() -> None:
    """flag_blocked is also ASSESS progress (premise-unmet decision)."""
    det = NoSubmitStreakDetector()
    read_call, read_result = _read("game.js")
    det.observe(read_call, read_result)
    det.observe(*_submit("flag_blocked"))
    assert det.streak == 0


def test_no_submit_nudge_text_and_event_kind() -> None:
    """The nudge steers to submit_assessment / flag_blocked; event_kind
    distinguishes it from the write streak in the drive log."""
    det = NoSubmitStreakDetector()
    text = det.nudge()
    assert "submit_assessment" in text
    assert "flag_blocked" in text
    assert det.event_kind == "no_submit_streak_detected"


def test_no_write_detector_exposes_nudge_and_event_kind() -> None:
    """Paired interface: NoWriteStreakDetector.nudge() matches
    build_nudge_text, event_kind is the write-streak marker, and the
    nudge now points at stream_edit as the edit_file fallback."""
    det = NoWriteStreakDetector()
    det._streak = DEFAULT_NO_WRITE_STREAK_THRESHOLD
    assert det.nudge() == build_nudge_text(DEFAULT_NO_WRITE_STREAK_THRESHOLD)
    assert det.event_kind == "no_write_streak_detected"
    assert "stream_edit" in det.nudge()


# --- NoTestSubmitStreakDetector (WRITE_TEST twin, loop_run=3a0f6368) --


def test_no_test_submit_default_threshold_is_three() -> None:
    """Pin the constant: WRITE_TEST's round budget is the tightest of
    the working phases (4), so the nudge must fire by call 3 to leave
    rounds for the model to act on it."""
    assert DEFAULT_NO_TEST_SUBMIT_STREAK_THRESHOLD == 3
    assert NoTestSubmitStreakDetector().threshold == 3


def test_no_test_submit_fires_at_threshold() -> None:
    det = NoTestSubmitStreakDetector()
    read_call, read_result = _read("game.js")
    assert det.observe(read_call, read_result) is False  # 1
    assert det.observe(read_call, read_result) is False  # 2
    assert det.observe(read_call, read_result) is True  # 3 → fire
    # one-shot
    assert det.observe(read_call, read_result) is False


def test_no_test_submit_write_file_does_not_reset() -> None:
    """write_file + shell are the expected pre-submit work in
    WRITE_TEST, but the phase's deliverable is the submit call — the
    observed stall (loop_run=3a0f6368 turn 2) wrote and ran a genuine
    red test, then never submitted. They count toward the streak."""
    det = NoTestSubmitStreakDetector()
    wf_call = ToolCall(name="write_file", arguments={"path": "tests/t.js", "content": "x"})
    wf_result = ToolResult(tool_name="write_file", output="wrote", success=True)
    sh_call = ToolCall(name="shell", arguments={"cmd": "node tests/t.js"})
    sh_result = ToolResult(tool_name="shell", output="exit=1 FAIL", success=True)
    assert det.observe(wf_call, wf_result) is False  # 1
    assert det.observe(sh_call, sh_result) is False  # 2
    assert det.streak == 2
    read_call, read_result = _read("game.js")
    assert det.observe(read_call, read_result) is True  # 3 → fire


def test_no_test_submit_reset_by_submit_failing_test() -> None:
    det = NoTestSubmitStreakDetector()
    read_call, read_result = _read("game.js")
    det.observe(read_call, read_result)
    det.observe(*_submit("submit_failing_test"))
    assert det.streak == 0


def test_no_test_submit_reset_by_skip_test_phase() -> None:
    det = NoTestSubmitStreakDetector()
    read_call, read_result = _read("game.js")
    det.observe(read_call, read_result)
    det.observe(*_submit("skip_test_phase"))
    assert det.streak == 0


def test_no_test_submit_nudge_text_and_event_kind() -> None:
    """The nudge steers to submit_failing_test / skip_test_phase and
    names the IMPLEMENT-phase trap (trying to edit source mid-phase)."""
    det = NoTestSubmitStreakDetector()
    text = det.nudge()
    assert "submit_failing_test" in text
    assert "skip_test_phase" in text
    assert "IMPLEMENT" in text
    assert det.event_kind == "no_test_submit_streak_detected"
