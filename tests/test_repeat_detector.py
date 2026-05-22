"""Unit tests for orchestrator/repeat_detector.py — harness-cna0.

The detector is a small per-turn counter the tool loop consults after
each tool dispatch. These tests pin the coarse-fingerprint extraction,
the threshold-crossing semantics, and the one-shot-per-fingerprint
nudge contract. The end-to-end "loop appends nudge to working thread"
behavior is covered separately in test_tool_loop.py."""

from __future__ import annotations

from harness.orchestrator.repeat_detector import (
    DEFAULT_REPEAT_THRESHOLD,
    RepeatCounter,
    build_nudge_text,
    fingerprint,
)
from harness.tools.base import ToolCall

# --- fingerprint extraction ---------------------------------------


def test_fingerprint_edit_file_uses_path() -> None:
    """harness-cna0: file-mutating tools fingerprint on `path`. Two
    edit_file calls on the same path with different old/new strings
    share a fingerprint — the d4e01d68 failure pattern."""
    call_a = ToolCall(
        name="edit_file",
        arguments={"path": "game.js", "old_string": "A", "new_string": "A1"},
    )
    call_b = ToolCall(
        name="edit_file",
        arguments={"path": "game.js", "old_string": "B", "new_string": "B1"},
    )
    assert fingerprint(call_a) == fingerprint(call_b) == ("edit_file", "game.js")


def test_fingerprint_edit_file_different_paths_differ() -> None:
    """Different target paths must produce different fingerprints —
    legitimate iterative work across multiple files doesn't get
    flagged as stuck."""
    call_a = ToolCall(name="edit_file", arguments={"path": "a.js", "old_string": "x"})
    call_b = ToolCall(name="edit_file", arguments={"path": "b.js", "old_string": "x"})
    assert fingerprint(call_a) != fingerprint(call_b)


def test_fingerprint_shell_uses_first_word() -> None:
    """Shell calls fingerprint on the command verb. `node script_a`
    and `node script_b` share a fingerprint — the d4e01d68 case
    where successive node validator invocations had different inline
    -e bodies."""
    call_a = ToolCall(name="shell", arguments={"cmd": "node -e 'console.log(1)'"})
    call_b = ToolCall(name="shell", arguments={"cmd": "node -e 'console.log(2)'"})
    assert fingerprint(call_a) == fingerprint(call_b) == ("shell", "node")


def test_fingerprint_shell_different_verbs_differ() -> None:
    """`grep`, `cat`, `git status` are distinct fingerprints — the
    detector doesn't false-positive when the model legitimately
    runs different commands."""
    grep_call = ToolCall(name="shell", arguments={"cmd": "grep foo bar.js"})
    cat_call = ToolCall(name="shell", arguments={"cmd": "cat bar.js"})
    assert fingerprint(grep_call) != fingerprint(cat_call)


def test_fingerprint_non_whitelisted_tool_returns_none() -> None:
    """harness-qbu3: tools NOT in the extractor whitelist (calc,
    grep, glob, git_*, fetch_url, now, search_*, meta-tools) return
    None — RepeatCounter skips them entirely. Each call is
    independent work, repetition is not stuckness."""
    for name in ("calc", "grep", "glob", "git_status", "fetch_url", "now"):
        assert fingerprint(ToolCall(name=name, arguments={})) is None, (
            f"{name} must be exempt from repeat-detection"
        )


def test_fingerprint_missing_path_falls_back_to_empty() -> None:
    """Malformed edit_file with no path arg falls back to (name, '')
    instead of crashing. Whitelisted tool stays IN detection (the
    counter still tracks it), with an empty target. The detector is
    a safety net — bad calls shouldn't break the harness."""
    call = ToolCall(name="edit_file", arguments={"old_string": "x"})  # no path
    assert fingerprint(call) == ("edit_file", "")


# --- RepeatCounter ------------------------------------------------


def test_counter_returns_true_only_at_threshold() -> None:
    """harness-cna0: record() returns True ONLY on the call that first
    hits threshold. Below threshold and after-first-trip both
    return False — one nudge per fingerprint per turn. Uses shell
    (default threshold 3) since edit_file got bumped to 5 (harness-qbu3)."""
    counter = RepeatCounter(threshold=3)
    call = ToolCall(name="shell", arguments={"cmd": "node x.js"})

    assert counter.record(call) is False  # count=1
    assert counter.record(call) is False  # count=2
    assert counter.record(call) is True  # count=3, first cross
    assert counter.record(call) is False  # count=4, already fired


def test_counter_independent_fingerprints_each_fires_at_own_threshold() -> None:
    """Each (name, target) pair has its own counter — counts on one
    fingerprint don't interfere with another's threshold. Assert
    each fingerprint fires exactly once when ITS threshold is
    reached, regardless of interleaving. Uses read_file (default
    threshold 3) since edit_file got bumped to 5 (harness-qbu3)."""
    counter = RepeatCounter(threshold=3)
    a = ToolCall(name="read_file", arguments={"path": "game.js"})
    b = ToolCall(name="read_file", arguments={"path": "index.html"})

    fires: list[tuple[str, bool]] = []
    # Interleave 4 of each; each fingerprint should fire exactly once
    # (on its 3rd recording).
    for _ in range(4):
        fires.append(("a", counter.record(a)))
        fires.append(("b", counter.record(b)))

    fire_counts = {
        label: [r for lbl, r in fires if lbl == label].count(True) for label in {"a", "b"}
    }
    assert fire_counts == {"a": 1, "b": 1}


def test_counter_default_threshold_is_3() -> None:
    """DEFAULT_REPEAT_THRESHOLD is the documented operating point
    for tools that don't have a per-tool override (harness-qbu3).
    Pin it so a future bump is a deliberate change with a visible
    test diff. Shell is the canonical default-threshold case."""
    assert DEFAULT_REPEAT_THRESHOLD == 3
    counter = RepeatCounter()  # default
    call = ToolCall(name="shell", arguments={"cmd": "node x.js"})
    assert counter.record(call) is False
    assert counter.record(call) is False
    assert counter.record(call) is True


def test_counter_non_whitelisted_tool_never_fires() -> None:
    """harness-qbu3 (motivating bug): calc with different exprs is
    legitimate independent work, not stuckness. The counter never
    fires for non-whitelisted tools, no matter how many times they're
    called. Surfaced in loop run b024ee25 turn 1."""
    counter = RepeatCounter()
    for i in range(10):
        call = ToolCall(name="calc", arguments={"expr": f"len({i})"})
        assert counter.record(call) is False, f"calc call {i + 1} tripped detector"


def test_counter_edit_file_threshold_is_5_not_3() -> None:
    """harness-qbu3: file-mutating tools get an elevated threshold
    to accommodate legitimate iterative work (map rows, multi-line
    edits, etc.). 3 same-path edits no longer trigger; 5 does."""
    counter = RepeatCounter()
    call = ToolCall(name="edit_file", arguments={"path": "game.js"})
    # First three calls (below the old default threshold) — no fire.
    assert counter.record(call) is False
    assert counter.record(call) is False
    assert counter.record(call) is False
    # Fourth — still below new threshold of 5.
    assert counter.record(call) is False
    # Fifth — threshold met.
    assert counter.record(call) is True


def test_counter_write_file_threshold_is_5() -> None:
    """harness-qbu3: write_file gets the same elevated threshold
    as edit_file — same class of legitimate iterative work."""
    counter = RepeatCounter()
    call = ToolCall(name="write_file", arguments={"path": "game.js", "content": "x"})
    for _ in range(4):
        assert counter.record(call) is False
    assert counter.record(call) is True


def test_threshold_for_returns_per_tool_override_or_default() -> None:
    """harness-qbu3: threshold_for() is the public lookup the
    operator log can format ('this nudge fired at threshold 5').
    Pin both the override and the default-fallback paths."""
    from harness.orchestrator.repeat_detector import threshold_for

    assert threshold_for(ToolCall(name="edit_file", arguments={"path": "x"})) == 5
    assert threshold_for(ToolCall(name="write_file", arguments={"path": "x"})) == 5
    assert threshold_for(ToolCall(name="shell", arguments={"cmd": "node"})) == 3
    assert threshold_for(ToolCall(name="read_file", arguments={"path": "x"})) == 3


def test_counter_count_inspect_works_without_recording() -> None:
    """count() is read-only — querying doesn't increment. Used by
    build_nudge_text to mention the actual count in the nudge. For
    non-whitelisted tools count() returns 0 since the counter never
    tracks them (harness-qbu3)."""
    counter = RepeatCounter(threshold=3)
    call = ToolCall(name="edit_file", arguments={"path": "x.js"})
    counter.record(call)
    counter.record(call)
    assert counter.count(call) == 2
    assert counter.count(call) == 2  # idempotent
    # Non-whitelisted tools: count is always 0.
    calc_call = ToolCall(name="calc", arguments={"expr": "1+1"})
    counter.record(calc_call)
    assert counter.count(calc_call) == 0


# --- nudge text ---------------------------------------------------


def test_nudge_mentions_tool_name_and_count() -> None:
    """The nudge is the model-facing artifact — it must name the
    specific tool + target the model is fixating on and the actual
    count. Otherwise the model can't tell what 'stuck' refers to."""
    call = ToolCall(name="edit_file", arguments={"path": "game.js"})
    text = build_nudge_text(call, count=3)
    assert "edit_file" in text
    assert "game.js" in text
    assert "3 times" in text


def test_nudge_for_edit_file_suggests_read_file() -> None:
    """harness-cna0: edit_file/write_file get a tool-specific
    suggestion — re-read the file to refresh the mental model.
    Half the d4e01d68 failures were 'edits diverged from the
    actual file state'; read_file is the cheap corrective."""
    call = ToolCall(name="edit_file", arguments={"path": "game.js"})
    text = build_nudge_text(call, count=3)
    assert "read_file" in text


def test_nudge_for_non_file_tool_omits_read_file_suggestion() -> None:
    """The read_file-specific advice only applies to file-mutating
    tools. A `shell` lock-in (model running `node` 4 times) gets
    a different (b) option."""
    call = ToolCall(name="shell", arguments={"cmd": "node -e x"})
    text = build_nudge_text(call, count=3)
    assert "read_file" not in text
