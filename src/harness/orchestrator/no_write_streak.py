"""Per-turn no-write streak detection for the tool loop — harness-41b3.

Surfaced in loop run 26c39558 turn 15 (harness-3jo1, §15 Controls).
Model spent 5 rounds doing read_file / grep / list_dir during the
IMPLEMENT phase without producing a single write, then halted as
`implement->halted (no writes)`. The phase-no-progress halt fires
only at end-of-budget; this detector nudges mid-turn so the model can
course-correct before the budget runs out.

Complements RepeatCounter (harness-cna0): RepeatCounter catches
'same kind of work, no convergence'; this catches 'looking instead of
writing'. The two run side-by-side in the same tool loop, each
appending at most one nudge per turn.

Intentionally caller-supplied: only the driver's IMPLEMENT phase
constructs and passes a detector to `run_tool_loop`. Other callers
(chat REPL, evals, ASSESS / WRITE_TEST / VERIFY phases) pass None and
see no behavior change — read-only investigation is correct outside
IMPLEMENT.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.tools.base import ToolCall, ToolResult

# Streak length that trips the nudge. Set above the conservative
# RepeatCounter default (3) so legitimate 'read three files before
# editing' doesn't false-positive, and below the default IMPLEMENT
# round budget so the nudge has rounds to actually land before the
# phase halts. One-shot per turn.
DEFAULT_NO_WRITE_STREAK_THRESHOLD: int = 4

# Tools that count as a write. Succeeding on any of these resets the
# streak. `fsm_turn._resolve_implement_outcome` imports this same set so
# what the detector counts and what the phase outcome checks stay in
# lock-step. stream_edit is here because the driver's IMPLEMENT roster
# offers it as an in-place edit path (awk/sed/cut/tr) — a productive
# stream_edit is real progress and must reset the streak, not trip it.
WRITE_TOOL_NAMES: frozenset[str] = frozenset({"edit_file", "write_file", "stream_edit"})
# Back-compat alias for the original private name.
_WRITE_TOOL_NAMES = WRITE_TOOL_NAMES


@dataclass
class NoWriteStreakDetector:
    """Per-turn counter; one instance per IMPLEMENT-phase
    `run_tool_loop` invocation.

    `observe(call, result)` runs after each tool result lands. It:

    - resets the streak when a write tool succeeds (real progress);
    - increments the streak on every other tool result (including
      failed writes — a model that emits edit_file 3 times and gets
      'old_string not found' each time is still the spiral pattern,
      just expressed via failed writes; the edit_file hard-stop at
      3 failures (harness-a9f6) covers that case, but only after the
      writes were attempted; this detector catches the 'never even
      tried' case);
    - returns True the FIRST time the streak crosses threshold, at
      which point the caller appends a nudge. Subsequent calls
      always return False — one nudge per turn is enough.

    Failed writes do NOT reset the streak: the failure mode this
    detector is built for is 'model spirals into reads without ever
    issuing a write', so a write that errored before reaching the
    filesystem is closer to a non-write than to progress. The
    edit_file-hard-stop logic (harness-a9f6) owns the 'tried to
    write but kept failing' case."""

    threshold: int = DEFAULT_NO_WRITE_STREAK_THRESHOLD
    _streak: int = 0
    _fired: bool = False

    @property
    def streak(self) -> int:
        """Current cumulative streak length. Public so callers can pass
        the count into `build_nudge_text` without reaching into private
        state."""
        return self._streak

    def observe(self, call: ToolCall, result: ToolResult) -> bool:
        """Update state for the just-completed tool call and return True
        the first time the streak crosses threshold. Returns False on
        every subsequent call this turn (one-shot)."""
        if call.name in _WRITE_TOOL_NAMES and result.success:
            self._streak = 0
            return False
        if self._fired:
            return False
        self._streak += 1
        if self._streak >= self.threshold:
            self._fired = True
            return True
        return False


# Phrasing mirrors `build_nudge_text` in repeat_detector.py: lead with a
# bracketed condition, then a concrete (a)(b)(c) menu of escape hatches.
# The model is conditioned by other catchers to react to this shape, so
# matching it gives this nudge the highest chance of producing a write
# on the next round rather than another grep.
_NUDGE_TEMPLATE = (
    "[NO-WRITE STREAK — {count} consecutive tool calls in IMPLEMENT "
    "without a single write. The assessment already named what to "
    "change. Your next reply MUST do one of:]\n"
    "(a) Call `edit_file` or `write_file` to make the actual change.\n"
    "(b) Call `submit_implementation_complete` if the work is already "
    "done (run `read_file` first to confirm the file matches your "
    "intent).\n"
    "(c) Stop and explain in your next reply what concrete obstacle is "
    "blocking the write — the phase will halt shortly otherwise."
)


def build_nudge_text(streak: int) -> str:
    """Compose the user-role nudge appended after the threshold-crossing
    tool call. `streak` is the count that just crossed — included
    verbatim so the model sees the magnitude of what it just did."""
    return _NUDGE_TEMPLATE.format(count=streak)


__all__ = [
    "DEFAULT_NO_WRITE_STREAK_THRESHOLD",
    "WRITE_TOOL_NAMES",
    "NoWriteStreakDetector",
    "build_nudge_text",
]
