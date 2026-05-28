"""Post-drive critic (harness-dopb).

After a drive run drains its ready queue, the workspace can still carry
runtime/behavioral bugs that `node --check` and the smoke loader can't
detect — broken control wiring, mis-ordered draws, dead-code paths,
identifier shadowing that changes behavior. The critic compares the
artifact to the spec, identifies such bugs, and returns structured
``CriticFinding`` records the outer auto-iterate loop turns into beads.

Pure module: takes the adapter + already-resolved inputs, returns a
list of validated findings. The auto-iterate wrapper owns spec
resolution, workspace snapshotting, and bd writes. Keeps this layer
testable without bd or a filesystem.

Grounding contract (load-bearing):
  * Each finding MUST carry a ``<file>:<line>`` citation that resolves
    inside the supplied workspace snapshot.
  * Each finding MUST carry a ``spec_quote`` that appears verbatim in
    the supplied spec text (when a spec was supplied at all).
  * The finding's normalized title must not fuzzy-match any open bead
    title under the epic (dedupe against the operator's queue).

``_validate_finding`` enforces all three silently — invalid findings
are dropped rather than raised. The cost of a false negative (a real
bug we drop) is lower than the cost of a false positive (a fabricated
bug the next drive chases).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from harness.model.adapter import ChatMessage, ModelAdapter

# `<file>.<ext>:<line>` — e.g. ``game.js:701``. Loose on extension so
# .ts / .mjs / .cjs / .py all match. The line number can be any non-
# empty digit run; out-of-bounds checks happen in _validate_finding
# against the actual workspace snapshot.
_CITATION_RE = re.compile(r"\b([\w./-]+\.(?:js|mjs|cjs|ts|py|html|css))\s*:\s*(\d+)\b")
# Strip ```language fences if the model wraps its JSON.
_FENCE_STRIP = re.compile(r"```[a-zA-Z]*\n?|\n?```")
# First [...] block — used when the model wraps the JSON list in prose.
_JSON_ARRAY = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)+\]", re.DOTALL)
# Title-similarity threshold for dedupe. 0.85 = strong overlap on the
# core noun phrase but tolerant of minor wording differences.
_TITLE_DUPE_RATIO = 0.85
# Spec-quote length cap — the prompt asks for <=200 chars; we drop
# anything longer (the model overshot and the "verbatim" claim is
# suspect at that point). Lower bound 16 chars rejects 1-word quotes
# that match by accident.
_SPEC_QUOTE_MIN = 16
_SPEC_QUOTE_MAX = 240  # a bit of slack over the prompt's 200


@dataclass(frozen=True)
class CriticFinding:
    """One critic-proposed bug. Maps onto a bd create call in the
    auto-iterate layer. Fields are all required — `_validate_finding`
    refuses partial findings."""

    title: str
    description: str
    acceptance: str
    priority: int
    spec_quote: str
    evidence_path: str  # "<file>:<line>"


def _system_prompt() -> str:
    return (
        "You are a code critic. The workspace below was built to satisfy "
        "the SPEC. Find runtime/behavioral bugs that `node --check` and a "
        "headless smoke-load WOULD NOT catch — wrong logic, broken control "
        "wiring, mis-ordered draws, dead-code paths, off-by-one, identifier "
        "shadowing that changes behavior, state never updated.\n\n"
        "Each finding MUST include:\n"
        "  - title: ~10 words, action-shaped\n"
        "  - description: 1-3 sentences with a `<file>:<line>` citation\n"
        "  - acceptance: one testable sentence\n"
        "  - priority: integer 0..3 (0=critical / blocks gameplay, 3=cosmetic)\n"
        "  - spec_quote: verbatim quote from SPEC, 16-200 chars\n"
        '  - evidence_path: "<file>:<line>"\n\n'
        "A finding with no file:line citation OR no verbatim spec_quote is "
        "INVALID and will be dropped. Do not invent file paths or line "
        "numbers. Do not paraphrase the spec — copy text exactly.\n\n"
        "Output: a single JSON array. No prose before or after. If no bugs "
        "are found, emit an empty array `[]`."
    )


def _user_prompt(
    *,
    spec_text: str | None,
    workspace_snapshot: Mapping[str, str],
    closed_this_run: Sequence[str],
    open_under_epic: Sequence[str],
    max_findings: int,
) -> str:
    parts: list[str] = []
    if spec_text:
        parts.append("[SPEC]\n" + spec_text.strip())
    else:
        parts.append("[SPEC]\n(no spec supplied — leave spec_quote empty)")
    parts.append("[WORKSPACE]")
    for path in sorted(workspace_snapshot):
        text = workspace_snapshot[path]
        parts.append(f"=== {path} ===\n{text}")
    parts.append("[CLOSED THIS RUN]")
    parts.extend(f"- {t}" for t in closed_this_run) if closed_this_run else parts.append("- (none)")
    parts.append("[ALREADY OPEN UNDER EPIC]")
    parts.extend(f"- {t}" for t in open_under_epic) if open_under_epic else parts.append("- (none)")
    parts.append(
        f"Return at most {max_findings} findings as a JSON array. "
        f"Prioritize the bugs most likely to break gameplay."
    )
    return "\n\n".join(parts)


def _extract_json_array(text: str) -> str | None:
    """Pull the JSON array out of the model's response. Tolerant of
    fenced output, leading prose, and trailing prose."""
    stripped = _FENCE_STRIP.sub("", text).strip()
    # Try the whole text first — many models will emit a clean array.
    if stripped.startswith("["):
        return stripped
    m = _JSON_ARRAY.search(stripped)
    return m.group(0) if m else None


def _parse_findings(text: str) -> list[dict[str, Any]]:
    """Parse the model output into a list of raw finding dicts. Returns
    empty list on any parse failure — the caller treats no findings the
    same as 'critic returned nothing'."""
    block = _extract_json_array(text)
    if block is None:
        return []
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _normalize_title(title: str) -> str:
    """Lowercase, collapse runs of non-alphanumerics to a single space,
    strip. Used as the dedupe key — small wording / punctuation /
    capitalization differences shouldn't allow a dupe to land."""
    lowered = title.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _title_matches_any(title: str, open_titles: Sequence[str]) -> bool:
    """True iff `title` is a near-dupe of any title in `open_titles`.
    Uses SequenceMatcher.ratio over normalized forms — robust to
    re-orderings and minor word swaps."""
    needle = _normalize_title(title)
    if not needle:
        return False
    for other in open_titles:
        haystack = _normalize_title(other)
        if not haystack:
            continue
        if SequenceMatcher(None, needle, haystack).ratio() >= _TITLE_DUPE_RATIO:
            return True
    return False


def _citation_in_workspace(citation: str, workspace_snapshot: Mapping[str, str]) -> bool:
    """True iff `citation` ("<file>:<line>") resolves to an actual line
    in the workspace snapshot. Matches paths against the snapshot's
    keys; for multi-path citations (e.g. when description carries
    several) only the first must resolve."""
    m = _CITATION_RE.search(citation)
    if not m:
        return False
    path, line_str = m.group(1), m.group(2)
    text = workspace_snapshot.get(path)
    if text is None:
        # Tolerate basename matches — the model sometimes drops the
        # leading directory. e.g. "src/game.js" cited as "game.js".
        # Match against any snapshot key whose basename matches.
        candidates = [k for k in workspace_snapshot if k.endswith("/" + path) or k == path]
        if not candidates:
            return False
        text = workspace_snapshot[candidates[0]]
    try:
        line = int(line_str)
    except ValueError:
        return False
    return 1 <= line <= len(text.splitlines())


def _validate_finding(
    raw: dict[str, Any],
    *,
    spec_text: str | None,
    workspace_snapshot: Mapping[str, str],
    open_titles: Sequence[str],
) -> CriticFinding | None:
    """Apply the grounding contract. Returns a CriticFinding when the
    raw record passes every check; returns None (silently) on any
    failure. The caller drops Nones."""
    title = raw.get("title")
    description = raw.get("description")
    acceptance = raw.get("acceptance")
    priority_raw = raw.get("priority")
    spec_quote_raw = raw.get("spec_quote", "")
    evidence_path = raw.get("evidence_path")

    # Type checks first — anything malformed is out.
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None
    if not isinstance(acceptance, str) or not acceptance.strip():
        return None
    if not isinstance(evidence_path, str) or not evidence_path.strip():
        return None
    if not isinstance(priority_raw, int) or not 0 <= priority_raw <= 3:
        return None
    spec_quote = spec_quote_raw if isinstance(spec_quote_raw, str) else ""

    # Citation must resolve in the workspace snapshot. Check both the
    # explicit evidence_path field AND any citation embedded in the
    # description (the model sometimes only puts it in the prose).
    if not (
        _citation_in_workspace(evidence_path, workspace_snapshot)
        or _citation_in_workspace(description, workspace_snapshot)
    ):
        return None

    # Spec-quote gate. Skipped when no spec was supplied — without a
    # spec there's nothing to verbatim-match against.
    if spec_text:
        quote = spec_quote.strip()
        if not _SPEC_QUOTE_MIN <= len(quote) <= _SPEC_QUOTE_MAX:
            return None
        if quote not in spec_text:
            return None
    # When spec is None, accept any spec_quote (typically empty).

    # Dedupe against the operator's already-open queue.
    if _title_matches_any(title, open_titles):
        return None

    return CriticFinding(
        title=title.strip(),
        description=description.strip(),
        acceptance=acceptance.strip(),
        priority=priority_raw,
        spec_quote=spec_quote.strip(),
        evidence_path=evidence_path.strip(),
    )


def run_critic(
    *,
    adapter: ModelAdapter,
    spec_text: str | None,
    workspace_snapshot: Mapping[str, str],
    closed_this_run: Sequence[str],
    open_under_epic: Sequence[str],
    max_findings: int = 10,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> list[CriticFinding]:
    """Call the adapter with a CRITIC prompt, parse its output, and
    return the subset of findings that pass `_validate_finding`.

    Silent on every failure mode (adapter raises, model emits non-JSON,
    every finding is invalid) — the auto-iterate loop's contract is
    "critic returned no findings ⇒ this pass is empty". The same
    contract holds when the critic genuinely sees no bugs.

    `temperature=0.2` is low enough that the model commits to its
    grounded findings rather than improvising, high enough that it
    won't get stuck in a single failure mode across passes."""
    messages = [
        ChatMessage(role="system", content=_system_prompt()),
        ChatMessage(
            role="user",
            content=_user_prompt(
                spec_text=spec_text,
                workspace_snapshot=workspace_snapshot,
                closed_this_run=closed_this_run,
                open_under_epic=open_under_epic,
                max_findings=max_findings,
            ),
        ),
    ]
    try:
        raw = adapter.complete(messages, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        return []
    candidates = _parse_findings(raw)
    out: list[CriticFinding] = []
    for c in candidates[: max_findings * 2]:  # cap input before the validator
        finding = _validate_finding(
            c,
            spec_text=spec_text,
            workspace_snapshot=workspace_snapshot,
            open_titles=open_under_epic,
        )
        if finding is not None:
            out.append(finding)
        if len(out) >= max_findings:
            break
    return out


__all__ = ["CriticFinding", "run_critic"]
