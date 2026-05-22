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


def test_fingerprint_unknown_tool_uses_empty_target() -> None:
    """Tools without an explicit extractor fall back to (name, '').
    Repeated calls to such tools still share a fingerprint — catches
    'model called git_status 5 times' patterns even without natural
    target args."""
    call_a = ToolCall(name="git_status", arguments={})
    call_b = ToolCall(name="git_status", arguments={})
    assert fingerprint(call_a) == fingerprint(call_b) == ("git_status", "")


def test_fingerprint_missing_path_falls_back_to_empty() -> None:
    """Malformed edit_file with no path arg falls back to (name, '')
    instead of crashing. The detector is a safety net — bad calls
    shouldn't break the harness."""
    call = ToolCall(name="edit_file", arguments={"old_string": "x"})  # no path
    assert fingerprint(call) == ("edit_file", "")


# --- RepeatCounter ------------------------------------------------


def test_counter_returns_true_only_at_threshold() -> None:
    """harness-cna0: record() returns True ONLY on the call that first
    hits threshold. Below threshold and after-first-trip both
    return False — one nudge per fingerprint per turn."""
    counter = RepeatCounter(threshold=3)
    call = ToolCall(name="edit_file", arguments={"path": "game.js"})

    assert counter.record(call) is False  # count=1
    assert counter.record(call) is False  # count=2
    assert counter.record(call) is True  # count=3, first cross
    assert counter.record(call) is False  # count=4, already fired


def test_counter_independent_fingerprints_each_fires_at_own_threshold() -> None:
    """Each (name, target) pair has its own counter — counts on one
    fingerprint don't interfere with another's threshold. Assert
    each fingerprint fires exactly once when ITS threshold is
    reached, regardless of interleaving."""
    counter = RepeatCounter(threshold=3)
    game = ToolCall(name="edit_file", arguments={"path": "game.js"})
    index = ToolCall(name="edit_file", arguments={"path": "index.html"})

    fires: list[tuple[str, bool]] = []
    # Interleave 4 of each; each fingerprint should fire exactly once
    # (on its 3rd recording).
    for _ in range(4):
        fires.append(("game", counter.record(game)))
        fires.append(("index", counter.record(index)))

    fire_counts = {
        label: [r for lbl, r in fires if lbl == label].count(True) for label in {"game", "index"}
    }
    assert fire_counts == {"game": 1, "index": 1}


def test_counter_default_threshold_is_3() -> None:
    """DEFAULT_REPEAT_THRESHOLD is the documented operating point.
    Test pins it so a future bump is a deliberate change with a
    visible test diff."""
    assert DEFAULT_REPEAT_THRESHOLD == 3
    counter = RepeatCounter()  # default
    call = ToolCall(name="shell", arguments={"cmd": "node x.js"})
    assert counter.record(call) is False
    assert counter.record(call) is False
    assert counter.record(call) is True


def test_counter_count_inspect_works_without_recording() -> None:
    """count() is read-only — querying doesn't increment. Used by
    build_nudge_text to mention the actual count in the nudge."""
    counter = RepeatCounter(threshold=3)
    call = ToolCall(name="edit_file", arguments={"path": "x.js"})
    counter.record(call)
    counter.record(call)
    assert counter.count(call) == 2
    assert counter.count(call) == 2  # idempotent


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
