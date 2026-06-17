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
from typing import ClassVar

from harness.tools.base import ToolCall, ToolResult

# Streak length that trips the nudge. Set above the conservative
# RepeatCounter default (3) so legitimate 'read three files before
# editing' doesn't false-positive, and below the default IMPLEMENT
# round budget so the nudge has rounds to actually land before the
# phase halts. One-shot per turn.
DEFAULT_NO_WRITE_STREAK_THRESHOLD: int = 4

# After the threshold-crossing fire, re-fire every this-many further
# non-write calls with escalated text (loop_run=135f0d99: the model
# ignored the single one-shot nudge, read four more rounds, then
# self-parked on tool-call exhaustion without a write). 2 keeps the
# IMPLEMENT budget (8 rounds) re-nudged at ~rounds 4/6/8 without spamming
# every round.
DEFAULT_NO_WRITE_REFIRE_EVERY: int = 2

# Tools that count as a write. Succeeding on any of these resets the
# streak. `fsm_turn._resolve_implement_outcome` imports this same set so
# what the detector counts and what the phase outcome checks stay in
# lock-step. stream_edit is here because the driver's IMPLEMENT roster
# offers it as an in-place edit path (awk/sed/cut/tr) — a productive
# stream_edit is real progress and must reset the streak, not trip it.
WRITE_TOOL_NAMES: frozenset[str] = frozenset({"edit_file", "write_file", "stream_edit"})
# Back-compat alias for the original private name.
_WRITE_TOOL_NAMES = WRITE_TOOL_NAMES

# Read-only exploration tools the IMPLEMENT read reservation locks once the
# round budget is down to its last `reserve` rounds. Writes (WRITE_TOOL_NAMES)
# and the phase-exit decisions always pass — the wall forces a write-or-decide,
# it never traps the model with no legal move.
READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {"read_file", "outline", "grep", "glob", "list_dir"}
)

# Reserve the last N IMPLEMENT rounds for non-read work.
DEFAULT_READ_RESERVE: int = 2


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
    refire_every: int = DEFAULT_NO_WRITE_REFIRE_EVERY
    _streak: int = 0
    _fires: int = 0

    @property
    def streak(self) -> int:
        """Current cumulative streak length. Public so callers can pass
        the count into `build_nudge_text` without reaching into private
        state."""
        return self._streak

    @property
    def fires(self) -> int:
        """How many times the nudge has fired this turn — drives the
        escalation tier `nudge()` renders."""
        return self._fires

    def observe(self, call: ToolCall, result: ToolResult) -> bool:
        """Update state for the just-completed tool call and return True
        whenever a nudge should fire: on the threshold-crossing call, then
        again every `refire_every` non-write calls past it. A successful
        write resets both the streak and the escalation tier."""
        if call.name in _WRITE_TOOL_NAMES and result.success:
            # Real progress: the spiral broke. Reset the streak AND the
            # escalation tier so a later relapse starts from tier 1.
            self._streak = 0
            self._fires = 0
            return False
        self._streak += 1
        if self._streak < self.threshold:
            return False
        # Escalate instead of latching silent: the model ignored the first
        # soft nudge (loop_run=135f0d99), so re-fire on a cadence with
        # progressively sharper text until a write lands or the phase halts.
        if (self._streak - self.threshold) % self.refire_every == 0:
            self._fires += 1
            return True
        return False

    def nudge(self) -> str:
        """The user-role nudge for this detector — paired interface with
        NoSubmitStreakDetector so the tool loop can call `.nudge()` on
        whichever stall detector a phase armed. Tier rises with `_fires`."""
        return build_nudge_text(self._streak, self._fires or 1)

    event_kind: ClassVar[str] = "no_write_streak_detected"


# Phrasing mirrors `build_nudge_text` in repeat_detector.py: lead with a
# bracketed condition, then a concrete (a)(b)(c) menu of escape hatches.
# The model is conditioned by other catchers to react to this shape, so
# matching it gives this nudge the highest chance of producing a write
# on the next round rather than another grep.
_NUDGE_TEMPLATE = (
    "[NO-WRITE STREAK — {count} consecutive tool calls in IMPLEMENT "
    "without a single SUCCESSFUL write. The assessment already named what "
    "to change. Your next reply MUST do one of:]\n"
    "(a) Make the change. If `edit_file` keeps failing to match its "
    "`old_string` (whitespace/format drift on a large file), switch tools: "
    "use `stream_edit` (sed/awk, matches by pattern not exact text) or "
    "`write_file` with the full new file content. Do NOT re-try the same "
    "edit_file call.\n"
    "(b) Call `submit_implementation_complete` if the work is already "
    "done (run `read_file` first to confirm the file matches your "
    "intent).\n"
    "(c) Stop and explain in your next reply what concrete obstacle is "
    "blocking the write — the phase will halt shortly otherwise."
)


# Escalated text for the second-and-later fire (loop_run=135f0d99): the
# soft menu already landed and was ignored, so this drops the optionality
# framing and makes the write-or-decide mandatory on the VERY NEXT call.
_ESCALATED_NUDGE_TEMPLATE = (
    "[NO-WRITE STREAK ESCALATING — {count} tool calls in IMPLEMENT, still "
    "ZERO successful writes; this is the {nth} time you've been told. STOP "
    "READING — the phase halts shortly and parks this bead UNMET. The "
    "assessment already names the change. Your VERY NEXT call MUST be one of:]\n"
    "(a) `edit_file` / `stream_edit` / `write_file` — make the change now. Do "
    "NOT read another file first.\n"
    "(b) `submit_implementation_complete` if it is already done.\n"
    "(c) `flag_blocked` with the concrete obstacle if you genuinely cannot "
    "proceed. Another read is not an option."
)


def _ordinal(n: int) -> str:
    """Small ordinal renderer for the escalation count (2 -> '2nd')."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_nudge_text(streak: int, fires: int = 1) -> str:
    """Compose the user-role nudge appended after a threshold-crossing or
    re-fire tool call. `streak` is the count that just crossed — included
    verbatim so the model sees the magnitude of what it just did. `fires`
    is the escalation tier: 1 (default) renders the soft (a)(b)(c) menu;
    2+ renders the mandatory escalated directive."""
    if fires <= 1:
        return _NUDGE_TEMPLATE.format(count=streak)
    return _ESCALATED_NUDGE_TEMPLATE.format(count=streak, nth=_ordinal(fires))


# --- ASSESS analog: read-without-submitting (harness follow-on) ------
#
# loop_run=ad30d9ad parked hewc/cw1m on "assess->halted (no assessment)":
# the model spent the whole ASSESS budget on outline/read/grep and never
# called submit_assessment, so the phase halted with no assessment. This
# is the ASSESS twin of the no-write streak — "looking instead of
# deciding" — and gets the same mid-phase nudge treatment.

# Tools that count as ASSESS progress: emitting the phase's decision.
# Succeeding on either resets the streak.
ASSESS_PROGRESS_TOOLS: frozenset[str] = frozenset({"submit_assessment", "flag_blocked"})

DEFAULT_NO_SUBMIT_STREAK_THRESHOLD: int = 3

_SUBMIT_NUDGE_TEMPLATE = (
    "[NO-SUBMIT STREAK — {count} consecutive read-only calls in ASSESS "
    "without submitting. You have enough context. Your next reply MUST "
    "do one of:]\n"
    "(a) Call `submit_assessment` with current_state / gap / approach — "
    "you do not need to read more files to write a gap analysis.\n"
    "(b) Call `flag_blocked` if the thing this bead asks you to verify or "
    "fix does not exist in the workspace yet (name the missing artifact).\n"
    "The phase halts when the round budget runs out, so submit now."
)


def build_submit_nudge_text(streak: int) -> str:
    """Nudge appended when the model has read N times in ASSESS without
    calling submit_assessment / flag_blocked."""
    return _SUBMIT_NUDGE_TEMPLATE.format(count=streak)


@dataclass
class NoSubmitStreakDetector:
    """ASSESS twin of NoWriteStreakDetector: counts tool calls that aren't
    the phase's decision action (submit_assessment / flag_blocked) and
    fires once when the streak crosses threshold. Same one-shot, same
    fire-on-first-cross contract; `.nudge()` returns the submit-now text.

    Constructed and armed only by the driver's ASSESS phase."""

    threshold: int = DEFAULT_NO_SUBMIT_STREAK_THRESHOLD
    _streak: int = 0
    _fired: bool = False

    @property
    def streak(self) -> int:
        return self._streak

    def observe(self, call: ToolCall, result: ToolResult) -> bool:
        if call.name in ASSESS_PROGRESS_TOOLS and result.success:
            self._streak = 0
            return False
        if self._fired:
            return False
        self._streak += 1
        if self._streak >= self.threshold:
            self._fired = True
            return True
        return False

    def nudge(self) -> str:
        return build_submit_nudge_text(self._streak)

    event_kind: ClassVar[str] = "no_submit_streak_detected"


# --- WRITE_TEST analog: testing-without-submitting (loop_run=3a0f6368) -
#
# loop_run=3a0f6368 turn 2 (vision-test-bun): the model wrote a genuine
# failing test (write_file + shell, exit=1, red output captured) and then
# burned the rest of the WRITE_TEST budget chasing edit_file instead of
# calling submit_failing_test — `write_test->halted (no test)` discarded
# real progress. Same stall shape as ASSESS's read-without-deciding, so
# it gets the same mid-phase nudge treatment.

# Tools that count as WRITE_TEST progress: emitting the phase's decision.
WRITE_TEST_PROGRESS_TOOLS: frozenset[str] = frozenset({"submit_failing_test", "skip_test_phase"})

# Lower than the IMPLEMENT threshold (4): the WRITE_TEST round budget is
# the tightest of the working phases (DEFAULT_PHASE_BUDGETS: 4), so the
# nudge must land while there are still rounds left to act on it. The
# legitimate pre-submit shape is write_file + shell (2 calls); a third
# non-submit call is already drift.
DEFAULT_NO_TEST_SUBMIT_STREAK_THRESHOLD: int = 3

_TEST_SUBMIT_NUDGE_TEMPLATE = (
    "[NO-SUBMIT STREAK — {count} consecutive tool calls in WRITE_TEST "
    "without submitting. Your next reply MUST do one of:]\n"
    "(a) If you already wrote a test and ran it red, call "
    "`submit_failing_test` with test_path, test_cmd, and the captured "
    "failure_output — that red output is this phase's deliverable.\n"
    "(b) If you cannot write a test for this issue, call "
    "`skip_test_phase` with a concrete reason.\n"
    "(c) If you are trying to modify source files: STOP — that is "
    "IMPLEMENT-phase work and those tools unlock only after you submit. "
    "The phase halts when the round budget runs out, so submit now."
)


def build_test_submit_nudge_text(streak: int) -> str:
    """Nudge appended when the model has worked N rounds in WRITE_TEST
    without calling submit_failing_test / skip_test_phase."""
    return _TEST_SUBMIT_NUDGE_TEMPLATE.format(count=streak)


@dataclass
class NoTestSubmitStreakDetector:
    """WRITE_TEST twin of NoSubmitStreakDetector: counts tool calls that
    aren't the phase's decision action (submit_failing_test /
    skip_test_phase) and fires once when the streak crosses threshold.
    Same one-shot, fire-on-first-cross contract.

    write_file / shell do NOT reset the streak — they're the expected
    pre-submit work, but the deliverable is the submit call, and the
    observed failure mode (loop_run=3a0f6368) had productive writes
    followed by a stall.

    Constructed and armed only by the driver's WRITE_TEST phase."""

    threshold: int = DEFAULT_NO_TEST_SUBMIT_STREAK_THRESHOLD
    _streak: int = 0
    _fired: bool = False

    @property
    def streak(self) -> int:
        return self._streak

    def observe(self, call: ToolCall, result: ToolResult) -> bool:
        if call.name in WRITE_TEST_PROGRESS_TOOLS and result.success:
            self._streak = 0
            return False
        if self._fired:
            return False
        self._streak += 1
        if self._streak >= self.threshold:
            self._fired = True
            return True
        return False

    def nudge(self) -> str:
        return build_test_submit_nudge_text(self._streak)

    event_kind: ClassVar[str] = "no_test_submit_streak_detected"


# --- IMPLEMENT read reservation: the wall behind the escalating nudge --
#
# loop_run=135f0d99 parked harness-l3tgq UNMET: the no-write nudge fired,
# the model ignored it, read several more rounds, then self-parked on
# "tool-call exhaustion" without ever attempting the write. The escalating
# nudge above is the louder warning; this is the hard backstop — once the
# IMPLEMENT round budget is down to its last `reserve` rounds, read-only
# tools are refused so the remaining budget can only go to a write or a
# phase-exit decision.

_RESERVATION_BLOCK_TEMPLATE = (
    "[READ BUDGET SPENT — {rounds_left} IMPLEMENT round(s) left, read-only "
    "tools are now locked: reading more cannot land the change in time, so "
    "`{tool}` was refused. The assessment already names what to change. Your "
    "next call must `edit_file` / `stream_edit` / `write_file`, "
    "`submit_implementation_complete`, or `flag_blocked`.]"
)


@dataclass
class ReadReservation:
    """Hard backstop paired with NoWriteStreakDetector's escalating nudge.

    Once the IMPLEMENT round budget is down to the last `reserve` rounds,
    `should_block` returns True for read-only exploration tools so the loop
    refuses them (via `block_result`) and the model must write or call a
    phase-exit tool. Writes and the exit decisions always pass — the wall
    forces a move, it never leaves the model with no legal call.

    Caller-supplied like the streak detectors: only the driver's IMPLEMENT
    phase arms it; every other caller passes None and sees no change."""

    reserve: int = DEFAULT_READ_RESERVE

    def should_block(self, call: ToolCall, rounds_left: int) -> bool:
        """True when only `reserve` (or fewer) rounds remain AND this call
        is a read-only exploration tool."""
        return rounds_left <= self.reserve and call.name in READ_ONLY_TOOL_NAMES

    def block_result(self, call: ToolCall, rounds_left: int) -> ToolResult:
        """The forcing FAIL substituted for a blocked read-only call."""
        return ToolResult(
            tool_name=call.name,
            output=_RESERVATION_BLOCK_TEMPLATE.format(
                rounds_left=max(rounds_left, 0), tool=call.name
            ),
            success=False,
            error="read_budget_reserved",
        )


__all__ = [
    "ASSESS_PROGRESS_TOOLS",
    "DEFAULT_NO_SUBMIT_STREAK_THRESHOLD",
    "DEFAULT_NO_TEST_SUBMIT_STREAK_THRESHOLD",
    "DEFAULT_NO_WRITE_REFIRE_EVERY",
    "DEFAULT_NO_WRITE_STREAK_THRESHOLD",
    "DEFAULT_READ_RESERVE",
    "READ_ONLY_TOOL_NAMES",
    "WRITE_TEST_PROGRESS_TOOLS",
    "WRITE_TOOL_NAMES",
    "NoSubmitStreakDetector",
    "NoTestSubmitStreakDetector",
    "NoWriteStreakDetector",
    "ReadReservation",
    "build_nudge_text",
    "build_submit_nudge_text",
    "build_test_submit_nudge_text",
]
