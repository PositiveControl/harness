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


__all__ = ["detect_claim_signal"]
