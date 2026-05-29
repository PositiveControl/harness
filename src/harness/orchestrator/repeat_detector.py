"""Per-turn repeat-call detection for the tool loop — harness-cna0.

Surfaced in loop run d4e01d68 turn 3 (harness-vjb6, §2 World map).
Model spent ~10 minutes doing four iterations of (edit_file game.js →
shell `node -e validate-row-widths…`) without the row-width validator
ever returning success. Each (edit_file, args) pair was technically
distinct because the old_string differed, so DuplicateCallHook
(exact-match) didn't fire. The pattern was visible at a coarser
granularity: same tool, same target file, same kind of work, no
progress.

This module supplies the coarser fingerprint and a small counter that
the tool loop consults after each tool execution. The first time a
fingerprint reaches the threshold within a turn, the loop appends a
user-role nudge telling the model it's stuck and listing three
escape hatches. Subsequent matches on the same fingerprint don't
re-fire — one nudge per fingerprint per turn is enough; spam would
just consume context.

Conservative threshold (3) and cumulative-per-turn counting: a real
iterative work pattern that legitimately edits the same file 5+ times
gets one nudge and the model can keep going if it's genuinely making
progress. False positives cost a short system-prompt insertion; false
negatives leave the model stuck (the failure mode we already had).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from harness.tools.base import ToolCall

# Number of same-fingerprint calls within a single turn that triggers
# the nudge for tools that don't have a tool-specific override. 3 is
# conservative for shell-verb-style repetition (e.g. running `node`
# four times to validate a single fix); too low for file-mutating
# tools where legitimate iterative work (writing a map row by row,
# multi-step edits to one file) routinely takes 4-5 calls.
DEFAULT_REPEAT_THRESHOLD: int = 3

# Per-tool threshold overrides. File-mutating tools get a higher
# threshold (5) — legitimate row-by-row implementation, multi-step
# edits, and split-up writes all expect 4-5 same-path calls before
# any nudge makes sense. The original d4e01d68 failure (4 edits with
# old==new on game.js) is still caught: DuplicateCallHook handles
# exact-args repeats; this detector backstops with the coarser
# fingerprint at threshold 5 (harness-qbu3).
_PER_TOOL_THRESHOLDS: Mapping[str, int] = {
    "edit_file": 5,
    "write_file": 5,
}


# Coarse-fingerprint extractors for the tools we care about.
#
# WHITELIST-ONLY: tools NOT in this map are exempt from the
# repeat-detector entirely — their fingerprint() returns None and the
# counter never increments. This guards against false positives on
# tools where each call is independent work (calc with different
# expressions, grep with different patterns, fetch_url with different
# URLs). Failure mode the detector targets is "same kind of work,
# repeated, no convergence" — a wide net here is wrong (harness-qbu3).
#
# For file-mutating tools the path is the obvious anchor: "the model
# keeps editing the same file" is the failure pattern (or, with the
# higher threshold, "the model keeps editing the same file WITHOUT
# making progress"). For `shell` the first whitespace-delimited token
# of the command captures the verb ("node", "grep", "pytest") without
# false-positiving on different sub-commands of the same binary —
# `node script_a` and `node script_b` share a fingerprint, which is
# exactly what we want in the d4e01d68 case (different validator
# scripts in successive iterations).
def _path_arg(args: Mapping[str, Any]) -> str | None:
    value = args.get("path")
    if isinstance(value, str) and value:
        return value
    return None


def _shell_cmd_verb(args: Mapping[str, Any]) -> str | None:
    cmd = args.get("cmd")
    if not isinstance(cmd, str) or not cmd.strip():
        return None
    # First whitespace-delimited token. Strips leading `cd <dir> && `
    # / `time ` / `env VAR=val ` etc. by walking until we hit a token
    # that doesn't look like a shell prelude — heuristic, kept simple.
    tokens = cmd.split()
    if not tokens:
        return None
    return tokens[0]


_FINGERPRINT_EXTRACTORS: Mapping[str, Any] = {
    "edit_file": _path_arg,
    "write_file": _path_arg,
    "read_file": _path_arg,
    "list_dir": _path_arg,
    "shell": _shell_cmd_verb,
}


def fingerprint(call: ToolCall) -> tuple[str, str] | None:
    """Coarse identity for repeat detection.

    Returns (tool_name, target) for tools where repetition signals
    stuckness. Returns None for tools where each call is independent
    work (calc, grep, glob, fetch_url, git_*, now, date_math, the
    search tools, meta-tools) — these are explicitly NOT in the
    extractor whitelist. RepeatCounter.record skips None outright.

    For whitelisted tools, an extractor that can't find its expected
    arg (e.g. edit_file with no path) still returns (name, "") — the
    call was malformed but it's still 'the same kind of work on no
    target', and repetition there is still a stuckness signal."""
    extractor = _FINGERPRINT_EXTRACTORS.get(call.name)
    if extractor is None:
        return None
    target = extractor(call.arguments) or ""
    return (call.name, target)


def threshold_for(call: ToolCall) -> int:
    """Per-tool threshold lookup. File-mutating tools (edit_file /
    write_file) use the elevated threshold (5); all others fall back
    to DEFAULT_REPEAT_THRESHOLD (3). Public so the RepeatCounter
    test surface can inspect what threshold a tool will trip at."""
    return _PER_TOOL_THRESHOLDS.get(call.name, DEFAULT_REPEAT_THRESHOLD)


# harness-4tphl: escalation multiplier over the nudge threshold. The
# nudge (at 1x threshold) is a soft "change approach" hint the model is
# free to ignore if it's genuinely progressing. But when the SAME
# fingerprint keeps firing well past that — 2x the threshold within one
# turn (shell-verb 3→6, edit/write 5→10) — the model is thrashing, not
# iterating, and every further round burns wall-clock against a wall it
# already declined to climb. At that point the counter raises a hard
# `escalated` flag the tool loop reads to end the turn early (forced
# wrap-up) instead of spinning to max_rounds. Run b085854e burned 13
# repeat_detected + 11 wrap_up_forced across one pass mostly thrashing
# on already-doomed parked issues; ending those turns sooner reclaims
# the rounds.
_ESCALATION_MULTIPLIER: int = 2


def escalation_threshold_for(call: ToolCall) -> int:
    """Count at which a fingerprint stops being a soft-nudge candidate
    and becomes a hard turn-ender. ``_ESCALATION_MULTIPLIER`` times the
    per-tool nudge threshold. Public for the RepeatCounter test surface."""
    return _ESCALATION_MULTIPLIER * threshold_for(call)


@dataclass
class RepeatCounter:
    """Per-turn counter; one instance per `run_tool_loop` invocation.

    Each `record(call)` increments the call's fingerprint counter and
    returns True iff this is the call that first reached the tool's
    threshold (see `threshold_for`). Subsequent records on the same
    fingerprint never return True again — the nudge is one-shot per
    fingerprint per turn so we don't spam the model after it's
    already been told to change approach.

    Calls whose fingerprint is None (non-whitelisted tools) are
    skipped entirely — the counter never increments for them
    (harness-qbu3). Threshold is per-tool by default; the constructor
    `threshold` arg only acts as the floor for tools without an
    explicit override, so most call sites should just instantiate
    with defaults."""

    threshold: int = DEFAULT_REPEAT_THRESHOLD
    _counts: dict[tuple[str, str], int] = field(default_factory=dict)
    _fired: set[tuple[str, str]] = field(default_factory=set)
    # harness-4tphl: set True once any fingerprint reaches its
    # escalation threshold (2x the nudge threshold). One-way latch — the
    # tool loop reads it to end the turn early. Inspect via `escalated`.
    _escalated: bool = False

    def record(self, call: ToolCall) -> bool:
        """Increment the fingerprint counter; return True the FIRST time
        the count hits the tool's threshold. Returns False on
        subsequent matches (nudge already fired) and False
        unconditionally when the call's fingerprint is None
        (tool not in the detection whitelist).

        Side effect (harness-4tphl): latches ``escalated`` True once any
        fingerprint reaches ``_ESCALATION_MULTIPLIER`` x its threshold —
        the signal the tool loop uses to stop a thrashing turn early."""
        key = fingerprint(call)
        if key is None:
            return False
        new_count = self._counts.get(key, 0) + 1
        self._counts[key] = new_count
        # Per-tool threshold falls back to the constructor's threshold
        # for tools without an override. Tests that want to tighten
        # detection still construct with a low value; production uses
        # the per-tool map for nuance.
        per_tool = _PER_TOOL_THRESHOLDS.get(call.name, self.threshold)
        if new_count >= _ESCALATION_MULTIPLIER * per_tool:
            self._escalated = True
        if new_count >= per_tool and key not in self._fired:
            self._fired.add(key)
            return True
        return False

    @property
    def escalated(self) -> bool:
        """True once some fingerprint thrashed past its escalation
        threshold this turn. The tool loop ends the turn (forced
        wrap-up) when this latches rather than spinning to max_rounds."""
        return self._escalated

    def count(self, call: ToolCall) -> int:
        """Inspect-only — current cumulative count for a call's
        fingerprint. Returns 0 when the tool is non-whitelisted
        (fingerprint is None)."""
        key = fingerprint(call)
        if key is None:
            return 0
        return self._counts.get(key, 0)


def build_nudge_text(call: ToolCall, count: int) -> str:
    """Compose the user-role nudge appended to the working thread.

    Format mirrors the duplicate-call rejection text but is broader:
    where DuplicateCallHook says 'this exact call was already rejected',
    this says 'this KIND of call has run N times — try something
    different.' Three escape hatches listed so the model has a clear
    next-action menu instead of guessing.

    For file-mutating tools (`edit_file` / `write_file`) we add a
    specific 'read the file fresh' suggestion — half the d4e01d68
    failures were 'model's mental model of the file diverged from
    what its edits actually wrote', and `read_file` is the cheap
    corrective."""
    # Tool name + target. Non-whitelisted tools shouldn't reach this
    # function (RepeatCounter never returns True for them), but fall
    # back to the bare name if a caller invokes it anyway (harness-qbu3).
    fp = fingerprint(call)
    if fp is None:
        name, target = call.name, ""
    else:
        name, target = fp
    target_clause = f" on `{target}`" if target else ""
    parts = [
        f"[STUCK — `{name}`{target_clause} has run {count} times this turn "
        "with no convergence. Change approach. Options:]",
        "(a) Regenerate the affected section in one shot instead of incremental edits.",
    ]
    if name in {"edit_file", "write_file"} and target:
        parts.append(
            f"(b) Run `read_file` on `{target}` to see its current state — "
            "your edits may have produced text different from what you expected."
        )
        parts.append("(c) Stop and explain in your next reply what's blocking progress.")
    else:
        parts.append("(b) Try a different tool that gives you fresh information.")
        parts.append("(c) Stop and explain in your next reply what's blocking progress.")
    return "\n".join(parts)


__all__ = [
    "DEFAULT_REPEAT_THRESHOLD",
    "RepeatCounter",
    "build_nudge_text",
    "escalation_threshold_for",
    "fingerprint",
    "threshold_for",
]
