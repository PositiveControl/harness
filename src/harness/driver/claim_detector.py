"""Detect a model's "I'm done" success-claim in an executor reply.

Surfaced in loop run d4e01d68 (harness-pfvj). Small models on tough work
will sometimes:
  1. Do the work badly.
  2. Generate a confident final reply ("the issue has been resolved").
  3. Never invoke `bd close <id>` via shell.

The driver classifies that turn as "issue still open after turn" — a
true failure, but with no concrete reason the model can act on next
turn. The verify gate (harness-xfh2) only fires AFTER `bd close`, so it
doesn't trigger on the claim-and-stop path.

This module is the seam that lets `run_loop` recognize a turn as a
"pseudo-close" worth running the verify gate against, so the next-turn
handoff carries concrete proof the claim was wrong rather than a vague
"issue still open" hint.

Conservative by design: the pattern set is small and high-precision. A
false positive here just means we run the verify steps when we didn't
strictly have to (harmless — verify is idempotent). A false negative
means we fall back to the existing "issue still open" path (no
regression). When in doubt, don't add a pattern.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.model.adapter import ChatMessage

# The pattern set is intentionally short. Every regex must be:
#   (a) case-insensitive (small models capitalize unpredictably)
#   (b) word-boundary anchored where natural, to avoid matching inside
#       longer phrases that mean the opposite ("not yet resolved")
#   (c) drawn from the corpus of actual model claims surfaced in
#       executor logs, not invented from imagination
#
# Sources:
#   d4e01d68 turn 3 reply: "the issue has been resolved and the
#       acceptance criteria from the handoff have been met"
#   94534703 turns 1+2 reply: "the implementation now meets all
#       acceptance criteria", "validation confirms... satisfying all
#       the requirements" — present-tense active-voice variants that
#       the original past-tense regex set missed (harness-24pn)
#   prior driver runs (anecdotal): "all acceptance criteria satisfied",
#       "task is complete", "fix is in place"
_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "the issue has been resolved" — the d4e01d68 phrase verbatim.
    re.compile(r"\bissue\s+(has\s+been|is)\s+resolved\b", re.IGNORECASE),
    # "acceptance criteria ... met / satisfied" — generic completion claim.
    re.compile(
        r"\bacceptance\s+criteria\b[^.]*\b(met|satisfied|complete)\b",
        re.IGNORECASE,
    ),
    # "all (criteria|requirements) (met|satisfied|complete)" — same
    # shape without the "acceptance" qualifier.
    re.compile(
        r"\ball\s+(criteria|requirements)\b[^.]*\b(met|satisfied|complete)\b",
        re.IGNORECASE,
    ),
    # "the task / fix / change is complete" — generic completion claim,
    # word-boundary anchored on "complete" so "incomplete" doesn't match.
    re.compile(
        r"\b(task|fix|change|implementation)\s+is\s+complete\b",
        re.IGNORECASE,
    ),
    # "everything is in place / working / done" — looser completion claim.
    re.compile(
        r"\beverything\s+is\s+(in\s+place|working|done)\b",
        re.IGNORECASE,
    ),
    # Active-voice / present-tense variants (harness-24pn). Model says
    # "the implementation MEETS all acceptance criteria" (verb BEFORE the
    # criteria noun) instead of the past-tense "criteria ARE met" shape
    # the original patterns assumed. Same shape for satisfies/completes/
    # fulfills/addresses + ("acceptance criteria"|"the requirements"|
    # "the spec"|"all <N> criteria/requirements"). Verbs are 3rd-person
    # singular only (`meets`, not `meet`) — the bare infinitive shows
    # up in negated / subjunctive contexts ("does not meet", "if X
    # meets") that we don't want to false-positive on.
    re.compile(
        r"\b(meets|satisfies|completes|fulfills|addresses)\b[^.]*"
        r"\b(acceptance\s+criteria|requirements|criteria|spec)\b",
        re.IGNORECASE,
    ),
    # "satisfying / meeting / fulfilling all the requirements" —
    # present-participle voice, same family as above.
    re.compile(
        r"\b(satisfying|meeting|fulfilling|addressing)\s+"
        r"(?:all\s+)?(?:the\s+)?(acceptance\s+criteria|requirements|criteria)\b",
        re.IGNORECASE,
    ),
    # "validation confirms" / "tests pass" / "verification (passed|
    # succeeded)" — terminal verification-claim shapes. Model uses these
    # to assert the work is done (94534703 turn 2: "the validation
    # confirms that the grid now has 30 rows...").
    re.compile(
        r"\b(validation|verification|tests?)\s+"
        r"(confirms?|pass(?:es|ed)?|succeeded|succeeds)\b",
        re.IGNORECASE,
    ),
    # "fix applied successfully" / "change applied successfully" —
    # passive-voice completion claim. Often appears in a celebratory
    # `echo` shell call, but also leaks into prose. The echo path is
    # covered separately by detect_claim_in_shell_call.
    re.compile(
        r"\b(fix|change|patch|implementation|update)\s+(applied|works)\s+"
        r"(successfully|correctly|as\s+expected)\b",
        re.IGNORECASE,
    ),
)


# Celebratory shell-echo patterns (harness-24pn). Loop run 94534703 turns
# 1+2 both ended with the model running `echo "Fix applied successfully…"`
# as the terminal action instead of `bd close`. The model treats the
# echo as a finalization gesture — claim-without-close from the harness
# point of view. Match against the `cmd` string of the last shell tool
# call; presence of `echo` with a completion verb means the model
# thinks it's done.
#
# Tighter than the prose patterns: prose can mix completion language with
# qualifiers, but a shell command is short and direct. The pattern
# requires `echo` + a quoted (or unquoted-after-echo) span containing
# a strong completion verb. Specific verbs only — avoid "applied" /
# "done" / "complete" in isolation since they show up in legitimate
# echoes too (e.g. `echo "Step 3 complete"` would still match — but a
# turn that ENDS with that echo without `bd close` is still a
# claim-without-close worth routing through the verify gate).
_CELEBRATORY_ECHO_RE: re.Pattern[str] = re.compile(
    r"\becho\b[^|;&\n]*\b("
    r"successfully|"
    r"all\s+(?:done|fixed|resolved)|"
    r"task\s+(?:done|complete[d]?|finished)|"
    r"issue\s+(?:closed|resolved|fixed)|"
    r"(?:fix|change|patch|update|implementation)\s+(?:applied|complete[d]?|done)|"
    r"work\s+(?:complete[d]?|done|finished)"
    r")\b",
    re.IGNORECASE,
)


def detect_claim_signal(reply: str) -> bool:
    """Return True when `reply` looks like the model claimed success.

    Empty / whitespace-only replies return False — there's nothing to
    claim. The patterns are case-insensitive and operate on the raw
    text, so a wrapped or trailing reply ("here's the summary: the
    issue has been resolved.") still matches."""
    if not reply or not reply.strip():
        return False
    return any(p.search(reply) for p in _CLAIM_PATTERNS)


def last_shell_cmd_in_messages(messages: Sequence[ChatMessage]) -> str | None:
    """Walk `messages` in reverse and return the `cmd` argument of the
    most recent shell tool call (harness-24pn). Returns None if no
    shell call appeared in the message thread.

    Helper used by both the legacy single-turn driver and the FSM
    driver: each one accumulates a ToolLoopResult per turn or per
    phase, and the claim-without-close gate inspects the last shell
    cmd to detect `echo "Fix applied successfully"` finalization
    gestures.

    Robust to non-string cmd values (defensive — well-formed tool
    calls always carry a string) and to assistant messages without
    tool_calls (the common case)."""
    for msg in reversed(messages):
        if msg.role != "assistant" or not msg.tool_calls:
            continue
        for call in reversed(msg.tool_calls):
            if call.name != "shell":
                continue
            cmd = call.arguments.get("cmd")
            return cmd if isinstance(cmd, str) else None
    return None


def detect_claim_in_shell_call(cmd: str | None) -> bool:
    """Return True when a shell tool's `cmd` looks like a celebratory
    finalization echo — the model running `echo "Fix applied
    successfully"` as its terminal action instead of `bd close`
    (loop run 94534703, harness-24pn).

    Called by `run_loop`'s claim-without-close gate against the LAST
    shell tool call this turn. Composes with `detect_claim_signal` —
    either signal is enough to route the turn through the verify gate
    as a pseudo-close.

    Empty / None inputs return False — nothing to check."""
    if not cmd or not cmd.strip():
        return False
    return _CELEBRATORY_ECHO_RE.search(cmd) is not None


__all__ = [
    "detect_claim_in_shell_call",
    "detect_claim_signal",
    "last_shell_cmd_in_messages",
]
