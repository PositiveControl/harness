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
# the nudge. 3 is conservative — two repeats are routine
# (read-edit-read iteration), three signals fixation. Lower triggers
# false positives on legitimate iterative work; higher leaves the
# model stuck longer.
DEFAULT_REPEAT_THRESHOLD: int = 3


# Coarse-fingerprint extractors for the tools we care about. Each
# entry returns the "target" string for fingerprinting; a missing /
# unsuitable argument returns None and the call falls back to
# `(tool_name, "")` — still useful for tools without a natural target
# (e.g. `git_status`, `now`).
#
# For file-mutating tools the path is the obvious anchor: "the model
# keeps editing the same file" is the failure pattern. For `shell`
# the first whitespace-delimited token of the command captures the
# verb ("node", "grep", "pytest") without false-positiving on
# different sub-commands of the same binary — `node script_a` and
# `node script_b` share a fingerprint, which is exactly what we want
# in the d4e01d68 case (different validator scripts in successive
# iterations).
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
    "shell": _shell_cmd_verb,
}


def fingerprint(call: ToolCall) -> tuple[str, str]:
    """Coarse identity for repeat detection.

    Returns (tool_name, target). `target` is `""` when the tool isn't
    in the extractor map or when the extractor can't find a natural
    target — those calls still count toward `(name, "")` repetition,
    which catches "the model called `git_status` 5 times in this
    turn" patterns even though they have no path argument."""
    extractor = _FINGERPRINT_EXTRACTORS.get(call.name)
    if extractor is None:
        return (call.name, "")
    target = extractor(call.arguments) or ""
    return (call.name, target)


@dataclass
class RepeatCounter:
    """Per-turn counter; one instance per `run_tool_loop` invocation.

    Each `record(call)` increments the call's fingerprint counter and
    returns True iff this is the call that first reached `threshold`.
    Subsequent records on the same fingerprint never return True again —
    the nudge is one-shot per fingerprint per turn so we don't spam
    the model after it's already been told to change approach.

    Threshold is a constructor argument so the integration tests can
    override it without touching the module default."""

    threshold: int = DEFAULT_REPEAT_THRESHOLD
    _counts: dict[tuple[str, str], int] = field(default_factory=dict)
    _fired: set[tuple[str, str]] = field(default_factory=set)

    def record(self, call: ToolCall) -> bool:
        """Increment the fingerprint counter; return True the FIRST time
        the count hits `threshold`. Returns False on subsequent matches
        — the nudge has already fired for that fingerprint this turn."""
        key = fingerprint(call)
        new_count = self._counts.get(key, 0) + 1
        self._counts[key] = new_count
        if new_count >= self.threshold and key not in self._fired:
            self._fired.add(key)
            return True
        return False

    def count(self, call: ToolCall) -> int:
        """Inspect-only — current cumulative count for a call's
        fingerprint. Used by the nudge text to mention how many times
        the model has been called this way."""
        return self._counts.get(fingerprint(call), 0)


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
    name, target = fingerprint(call)
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
    "fingerprint",
]
