"""Advisory vision-QA for drive browser smoke (harness-ke4hx.3).

Feeds a smoke screenshot + the bead's acceptance rubric to a VLM and
parses a PASS/FAIL verdict. PHASE 1 is ADVISORY: callers log the verdict,
they do NOT gate the drive on it. The point is to collect verdict-vs-
actual-outcome data and decide later whether the signal is trustworthy
enough to gate on (Phase 2).

Boundary: the VLM is reached through the ordinary model adapter
(`ModelAdapter.complete` + `ChatMessage.images`) — this module imports no
model SDK and no HTTP client. The adapter is resolved by
`harness.model.factory.make_vision_adapter`."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from harness.model.adapter import ChatMessage, image_from_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from harness.model.adapter import ModelAdapter

Verdict = Literal["pass", "fail", "unknown"]

# The model is asked to lead with the single word PASS or FAIL; the parser
# accepts the first PASS/FAIL token anywhere (tolerates "**PASS**",
# "Verdict: FAIL — …", a leading sentence) and treats anything with no such
# token as `unknown` rather than guessing.
_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
# Trim the reason so a runaway VLM reply can't bloat the verify log / JSONL.
_MAX_REASON_CHARS = 280

_PROMPT_TEMPLATE = """You are a strict QA reviewer inspecting a screenshot of a web app under test.

Acceptance criteria:
{rubric}

Judge ONLY what is visible in the screenshot. Do not assume behavior you cannot see.
Reply in exactly this format, leading with the single word PASS or FAIL:
PASS: <one short reason>
or
FAIL: <one short reason>"""


@dataclass(frozen=True)
class QaVerdict:
    """A parsed advisory verdict. `verdict` is the coarse signal; `reason`
    is the model's short justification; `raw` is the untouched reply (kept
    for the JSONL audit so a bad parse can be diagnosed later)."""

    verdict: Verdict
    reason: str
    raw: str


def build_qa_prompt(rubric: str) -> str:
    """Render the QA instruction around a bead's acceptance rubric."""
    return _PROMPT_TEMPLATE.format(rubric=rubric.strip() or "(no rubric provided)")


def parse_verdict(raw: str) -> QaVerdict:
    """Pull a PASS/FAIL verdict + reason out of a model reply. First
    PASS/FAIL token wins; no token → `unknown`. The reason is whatever
    follows the token on its line (leading `:`/`-`/`—`/whitespace
    stripped), truncated to keep logs bounded."""
    text = raw.strip()
    match = _VERDICT_RE.search(text)
    if match is None:
        return QaVerdict(verdict="unknown", reason=text[:_MAX_REASON_CHARS], raw=raw)
    verdict: Verdict = "pass" if match.group(1).upper() == "PASS" else "fail"
    tail = text[match.end() :]
    # Reason = remainder up to the first newline, minus leading punctuation.
    reason = tail.splitlines()[0] if tail.splitlines() else ""
    reason = reason.lstrip(" :\t-—").strip()
    return QaVerdict(verdict=verdict, reason=reason[:_MAX_REASON_CHARS], raw=raw)


def run_vision_qa(
    adapter: ModelAdapter,
    shot_path: Path | str,
    rubric: str,
    *,
    max_tokens: int = 64,
) -> QaVerdict:
    """Run one advisory QA judgment. Builds a single image+text turn, calls
    the adapter at temperature 0 (we want the most deterministic read we
    can get from an inherently non-deterministic judge), and parses the
    reply. Small `max_tokens` — the reply is PASS/FAIL + one sentence.

    Raises whatever the adapter raises (an unreachable endpoint surfaces as
    the adapter's RuntimeError); the CALLER is responsible for treating
    that as a skip, not a drive failure (advisory-only contract)."""
    message = ChatMessage(
        role="user",
        content=build_qa_prompt(rubric),
        images=(image_from_path(shot_path),),
    )
    raw = adapter.complete([message], max_tokens=max_tokens, temperature=0.0)
    return parse_verdict(raw)


def build_rubric(title: str, body: str = "") -> str:
    """Compose a QA rubric from a bead's title + (optional) body. Title
    is the headline acceptance signal; body adds detail when present."""
    rubric = title.strip()
    detail = (body or "").strip()
    if detail:
        rubric = f"{rubric}\n\n{detail}"
    return rubric or "(no rubric provided)"


def _append_jsonl(path: Path, record: dict[str, object], log: Callable[[str], None]) -> None:
    """Append one JSONL record. Best-effort — a write failure logs and is
    swallowed (the audit log is diagnostic, not a drive contract)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        log(f"vision-qa: audit-log write failed ({path}): {exc}")


def run_advisory_qa(
    adapter: ModelAdapter | None,
    shot_path: Path | str,
    rubric: str,
    *,
    gate_passed: bool,
    issue_id: str,
    jsonl_path: Path | str | None = None,
    log: Callable[[str], None] = print,
    max_tokens: int = 64,
) -> QaVerdict | None:
    """Run advisory vision-QA around a completed verify (harness-ke4hx.4).

    ADVISORY CONTRACT — this function NEVER raises and NEVER gates. It
    returns the verdict (for the caller to log) or None when QA could not
    run. The drive's pass/fail is whatever the caller already decided;
    nothing here changes it.

    Graceful degradation covers every "vision unavailable" path:
      - `adapter is None` (vision_base_url unset) → skip, return None
      - screenshot missing (smoke didn't capture, or capture failed) →
        skip, return None
      - the VLM call raised (endpoint down, timeout, bad response) →
        skip, return None

    On success it logs a one-line verdict and, when `jsonl_path` is set,
    appends an audit record carrying `gate_passed` + `agree` (whether the
    VLM verdict matched the real gate). That `agree` column is the
    precision/recall fuel for the Phase-2 gating decision (harness-ke4hx.5)."""
    if adapter is None:
        log("vision-qa: skipped (no vision adapter configured)")
        return None
    shot = Path(shot_path)
    if not shot.is_file():
        log(f"vision-qa: skipped (no screenshot at {shot})")
        return None
    try:
        verdict = run_vision_qa(adapter, shot, rubric, max_tokens=max_tokens)
    except Exception as exc:
        log(f"vision-qa: skipped (vision call failed: {exc})")
        return None

    agree = (verdict.verdict == "pass") == gate_passed
    gate_label = "pass" if gate_passed else "fail"
    log(
        f"vision-qa [{issue_id}]: {verdict.verdict.upper()} "
        f"(gate={gate_label}, agree={agree}) — {verdict.reason}"
    )
    if jsonl_path is not None:
        _append_jsonl(
            Path(jsonl_path),
            {
                "ts": datetime.now(UTC).isoformat(),
                "issue_id": issue_id,
                "shot": str(shot),
                "rubric": rubric,
                "verdict": verdict.verdict,
                "reason": verdict.reason,
                "gate_passed": gate_passed,
                "agree": agree,
            },
            log,
        )
    return verdict
