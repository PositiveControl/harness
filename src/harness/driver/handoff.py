"""Deterministic handoff assembly — harness-2gut.

A handoff is the structured snapshot the loop driver injects into the
executor turn's system prompt at the start of every iteration. It
carries the load-bearing context the model needs to continue work
without re-reading the world: which bd issue is being driven, which
epic it belongs to, what files have changed since the loop began, which
bd issues have already closed in this run, and what `thought:*` beads
were written during the run.

Two layers:

  * `Handoff` (frozen dataclass) — the assembled snapshot. `render()`
    produces the `[SESSION HANDOFF]` block as a string. Pure value
    type; no I/O.
  * `build_handoff(state, current_bd_id, bd, git_root, prior=None)` —
    the assembler. Pulls `bd show` on the current issue + parent epic,
    runs `git diff --name-status <state.started_at_sha>..HEAD`, queries
    `bd.thoughts_in_loop_run`, and constructs a `Handoff`.

Truncation contract (per the epic spec):

  * decisions / observations: top 10 by recency.
  * open questions: ALL — load-bearing for what the next turn should
    resolve. Never truncated.
  * Render-time hard cap: ~1 000 tokens via a chars/4 heuristic. If
    over budget, drop the OLDEST decisions then observations one at a
    time until under cap. Current issue, files touched, closed list,
    open questions, and prior_attempt_failure are never dropped — if
    open questions push past budget, accept the bloat and let the
    operator tighten them. Closed-set bloat is the same story: it's a
    list of bd ids, not prose, so its growth is linear and bounded.

v0 limitation, per the epic discussion: only beads carrying a
`loop-run-<id>` label show up in decisions/observations/questions.
Driver-written `thought:session-state` beads carry it automatically;
model-written bd writes during executor turns do not (no tool-registry
shim yet). The v0 handoff therefore surfaces driver-written state plus
any externally-labelled beads; richer attribution is a future revision.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from harness.driver.bd import DriverBdError
from harness.driver.state import LoopRunState
from harness.store._bd_types import BeadsIssue

# Render-budget knobs. Public so tests can pin them and a future eval
# pass can tune them without monkeypatching internals.
TOKEN_BUDGET: int = 1000
CHARS_PER_TOKEN: int = 4
RENDER_CHAR_CAP: int = TOKEN_BUDGET * CHARS_PER_TOKEN

# Truncation floors (count-based, applied before the char-cap pass).
MAX_DECISIONS: int = 10
MAX_OBSERVATIONS: int = 10


class BdHandoffSource(Protocol):
    """Structural type for the bd surface `build_handoff` consumes.
    `DriverBd` satisfies this implicitly; tests pass a lightweight
    fake without inheriting from `DriverBd`. Keeps the handoff module
    free of test seams in its public signature while still letting
    mypy verify the contract."""

    def show(self, issue_id: str) -> BeadsIssue: ...

    def thoughts_in_loop_run(
        self,
        loop_run_id: str,
        *,
        types: Sequence[str] = ...,
    ) -> list[BeadsIssue]: ...


@dataclass(frozen=True)
class Handoff:
    """Assembled per-turn handoff snapshot.

    Field semantics:
      - `loop_run_id`: identifies the run that produced this handoff.
      - `epic_id`: the bd epic the loop is draining.
      - `current_issue`: pre-formatted block describing the bd issue
        the executor is being asked to drive this turn.
      - `parent_epic_summary`: short "<id> — <title>" line for the
        epic, or None if the epic couldn't be resolved (transient bd
        error — caller decided to render without it).
      - `files_touched`: tuple of "<status> <path>" lines from
        `git diff --name-status`. Status letters + path only, no diff
        body — keeps the block compact.
      - `closed_this_run`: ordered tuple of bd ids closed so far in
        this run. Pulled from LoopRunState, not bd.
      - `decisions`: top-10 thought:decision bodies (newest first).
      - `observations`: top-10 thought:observation bodies (newest
        first).
      - `open_questions`: ALL open thought:question bodies (newest
        first). Never truncated by count.
      - `prior_attempt_failure`: reason from the previous attempt at
        `current_issue`, when the executor is retrying. None on the
        first attempt.
      - `workspace_path`: absolute path of the executor's cwd. The
        executor's filesystem tools are sandboxed here; all read/write
        paths in tool calls are relative to this. Surfacing it up
        front saves 2-3 rounds per turn the model would otherwise burn
        on path-prefix discovery (harness-po0v).
      - `workspace_contents`: one-line-per-entry list of the top-level
        workspace items, prefixed by `f` (file) or `d` (directory).
        Mirrors the `list_dir` tool's output shape so the model has
        an immediate orientation without burning a tool call.
      - `targeted_fix`: when True, prepends a MODE banner instructing
        the executor to make minimal edits via edit_file and NOT call
        write_file on existing paths. Set by the caller (loop runner)
        when the bd issue is being retried in this run, or when the
        operator has annotated the issue's notes with "REGRESSION".
        Pairs with the WriteFileRedirectHook safety-shrink guard
        (harness-lefw 2026-05-21): two layered defenses against the
        small-model "rewrite from scratch" reflex on reopen.
    """

    loop_run_id: str
    epic_id: str
    current_issue: str
    parent_epic_summary: str | None
    files_touched: tuple[str, ...]
    closed_this_run: tuple[str, ...]
    decisions: tuple[str, ...]
    observations: tuple[str, ...]
    open_questions: tuple[str, ...]
    prior_attempt_failure: str | None
    workspace_path: Path | None = None
    workspace_contents: tuple[str, ...] = ()
    targeted_fix: bool = False
    # harness-kbnl: FSM-driven turn phase context. `phase` is the
    # current TurnPhase the executor is operating in (string value of
    # the enum, e.g. "assess"); rendered as `[PHASE: <name>]` so the
    # model knows which meta-tool is expected to advance the FSM.
    # `prior_assessment` and `prior_test_cmd` carry forward when a
    # turn halts mid-FSM (e.g. ASSESS completed but IMPLEMENT didn't
    # finish) so the next attempt resumes with the work already
    # captured. None values render to nothing — handoffs without
    # FSM context behave identically to the pre-FSM shape.
    phase: str | None = None
    phase_instructions: str | None = None
    prior_assessment: dict[str, object] | None = None
    prior_test_cmd: str | None = None
    # harness-d8e3: forbidden_patterns surfaced into the handoff so the
    # model knows up-front which placeholder strings the harness will
    # reject at close time. Without this, the model writes "// TODO"
    # comments, calls `bd close`, the close succeeds, then the
    # forbidden-pattern audit fails the turn — and the model never
    # learned in time to fix it. Empty tuple means the operator
    # disabled the check (or no forbidden_patterns set in LoopConfig);
    # the block doesn't render.
    forbidden_patterns: tuple[str, ...] = ()

    def render(self) -> str:
        """Produce the `[SESSION HANDOFF]` block. Applies the render-
        time char cap by dropping the OLDEST decisions then
        observations one at a time until under budget. Open questions,
        the current issue, files touched, closed list, and the
        prior-attempt block are never dropped."""
        decisions = list(self.decisions)
        observations = list(self.observations)
        rendered = self._render_with(decisions, observations)
        while len(rendered) > RENDER_CHAR_CAP and (decisions or observations):
            # Drop oldest from whichever list is longer; tie-break to
            # decisions. Lists are newest-first, so OLDEST is the tail.
            if len(decisions) >= len(observations) and decisions:
                decisions.pop()
            elif observations:
                observations.pop()
            else:
                break
            rendered = self._render_with(decisions, observations)
        return rendered

    def _render_with(self, decisions: Sequence[str], observations: Sequence[str]) -> str:
        parts: list[str] = [
            f"[SESSION HANDOFF — loop_run={self.loop_run_id} epic={self.epic_id}]",
            "",
        ]
        if self.phase is not None:
            parts.extend([f"[PHASE: {self.phase.upper()}]", ""])
            if self.phase_instructions:
                parts.append(self.phase_instructions)
                parts.append("")
        if self.prior_assessment:
            parts.append("[PRIOR ASSESSMENT (from earlier turn in this run)]")
            for key in ("current_state", "gap", "approach"):
                value = self.prior_assessment.get(key)
                if isinstance(value, str) and value:
                    parts.append(f"  {key}: {value}")
            parts.append("")
        if self.prior_test_cmd:
            parts.extend(
                [
                    "[PRIOR TEST CMD (from WRITE_TEST phase)]",
                    f"  {self.prior_test_cmd}",
                    "",
                ]
            )
        if self.targeted_fix:
            parts.extend(
                [
                    "[MODE: TARGETED-FIX]",
                    "The artifact already exists from a prior implementation in this",
                    "workspace. Read the file FIRST and make a minimal edit via",
                    "edit_file. Do NOT call write_file on an existing path; that",
                    "wipes prior work — including sections this issue does not own.",
                    "If you believe a full rewrite is required, close this issue with",
                    'reason="rewrite-required" and stop — the operator will decide.',
                    "",
                ]
            )
        if self.prior_attempt_failure:
            parts.extend(
                [
                    "[PRIOR ATTEMPT FAILED]",
                    self.prior_attempt_failure,
                    "",
                ]
            )
        if self.forbidden_patterns:
            pattern_list = ", ".join(repr(p) for p in self.forbidden_patterns)
            parts.extend(
                [
                    "[FORBIDDEN PATTERNS]",
                    f"The harness rejects any close that introduces {pattern_list}",
                    "into any workspace file (post-close audit, harness-k52f).",
                    "Do NOT leave placeholder comments with these strings — write",
                    "the actual implementation or leave the file alone.",
                    "",
                ]
            )
        parts.extend(
            [
                "Current issue:",
                self.current_issue,
                "",
            ]
        )
        if self.parent_epic_summary:
            parts.extend(["Parent epic:", self.parent_epic_summary, ""])
        if self.workspace_path is not None:
            parts.append(
                f"Workspace (your cwd; tool paths are relative to this): {self.workspace_path}"
            )
            parts.append("Contents:")
            if self.workspace_contents:
                parts.extend(f"  {line}" for line in self.workspace_contents)
            else:
                parts.append("  (empty)")
            parts.append("")
        parts.append("Files touched this loop run:")
        if self.files_touched:
            parts.extend(f"  {line}" for line in self.files_touched)
        else:
            parts.append("  (none yet)")
        parts.append("")
        parts.append("Closed this loop run:")
        if self.closed_this_run:
            parts.append("  " + ", ".join(self.closed_this_run))
        else:
            parts.append("  (none yet)")
        parts.append("")
        parts.append("Decisions:")
        if decisions:
            parts.extend(f"  - {line}" for line in decisions)
        else:
            parts.append("  (none recorded this run)")
        parts.append("")
        parts.append("Observations:")
        if observations:
            parts.extend(f"  - {line}" for line in observations)
        else:
            parts.append("  (none recorded this run)")
        parts.append("")
        parts.append("Open questions:")
        if self.open_questions:
            parts.extend(f"  - {line}" for line in self.open_questions)
        else:
            parts.append("  (none open)")
        parts.append("")
        parts.append("[END HANDOFF]")
        return "\n".join(parts)


# --- builder ---------------------------------------------------------


def build_handoff(
    state: LoopRunState,
    current_bd_id: str,
    bd: BdHandoffSource,
    *,
    git_root: Path,
    prior_attempt_failure: str | None = None,
    workspace: Path | None = None,
    targeted_fix: bool = False,
    phase: str | None = None,
    phase_instructions: str | None = None,
    prior_assessment: dict[str, object] | None = None,
    prior_test_cmd: str | None = None,
    forbidden_patterns: tuple[str, ...] = (),
) -> Handoff:
    """Assemble a `Handoff` for the next executor turn.

    Failure modes are defensive: a transient bd or git error for the
    PARENT epic or the files-touched query degrades gracefully (the
    field becomes None / empty), since those are context, not the
    contract. A failure to resolve `current_bd_id` raises — that's the
    issue the executor is about to drive, and operating without it
    would corrupt the turn.

    `workspace`, when supplied, is rendered into the handoff so the
    model knows its cwd (saves rounds of path-prefix discovery —
    harness-po0v). Defaults to None for back-compat with callers
    (e.g. tests) that don't care about workspace orientation.

    `targeted_fix` (harness-lefw): when True, the rendered handoff
    leads with a "MODE: TARGETED-FIX" banner that explicitly forbids
    write_file on existing paths and steers the model toward
    edit_file. Caller (loop.run_loop) sets True when the bd issue
    was retried (`attempt_counts[id] > 0`) or its notes contain the
    operator's REGRESSION marker. Pairs with the wired
    WriteFileRedirectHook safety-shrink guard."""
    issue = bd.show(current_bd_id)
    current_issue = _format_issue(issue)

    parent_epic_summary = _try_epic_summary(bd, state.epic_id)
    files_touched = _git_diff_name_status(git_root, state.started_at_sha)
    workspace_contents = _list_workspace_top_level(workspace) if workspace is not None else ()

    thoughts = bd.thoughts_in_loop_run(state.loop_run_id)
    decisions = tuple(
        _short_body(issue) for issue in thoughts if _has_label(issue, "thought:decision")
    )[:MAX_DECISIONS]
    observations = tuple(
        _short_body(issue) for issue in thoughts if _has_label(issue, "thought:observation")
    )[:MAX_OBSERVATIONS]
    open_questions = tuple(
        _short_body(issue)
        for issue in thoughts
        if _has_label(issue, "thought:question") and issue.status == "open"
    )

    return Handoff(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        current_issue=current_issue,
        parent_epic_summary=parent_epic_summary,
        files_touched=files_touched,
        closed_this_run=tuple(state.closed_this_run),
        decisions=decisions,
        observations=observations,
        open_questions=open_questions,
        prior_attempt_failure=prior_attempt_failure,
        workspace_path=workspace.resolve() if workspace is not None else None,
        workspace_contents=workspace_contents,
        targeted_fix=targeted_fix,
        phase=phase,
        phase_instructions=phase_instructions,
        prior_assessment=prior_assessment,
        prior_test_cmd=prior_test_cmd,
        forbidden_patterns=forbidden_patterns,
    )


def _list_workspace_top_level(workspace: Path) -> tuple[str, ...]:
    """One line per top-level entry in `workspace`. `ls -F` format —
    just the name, with a trailing `/` for directories. Skips hidden
    entries (anything starting with `.`) so the handoff isn't
    polluted by `.harness/` / `.git/` / `.beads/`.

    Format choice (harness-cb7c): the original `f <name>` / `d <name>/`
    type-prefixed shape kept getting misparsed by small models as
    `f/<name>` / `d/<name>` paths — Mark's 2026-05-21 GTA2 §6 retry
    halted twice when the model emitted `read_file path=d/gta/game.js`
    interpreting `d gta/` as a path with a `d/` directory prefix.
    Plain `ls -F` output is what the model has seen in training and
    parses cleanly.

    Best-effort: returns `()` on IO error rather than raising — a
    missing or unreadable workspace shouldn't crash the handoff."""
    try:
        entries = sorted(workspace.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return ()
    lines: list[str] = []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            lines.append(entry.name)
    return tuple(lines)


# --- helpers ---------------------------------------------------------


def _format_issue(issue: BeadsIssue) -> str:
    """Compact human/model-readable issue block. Pulls title +
    description + acceptance criteria + notes from `issue.raw`.
    All four are kept verbatim — the executor needs the full
    acceptance criteria to know when to close, descriptions and
    notes are part of the contract (operators add notes when
    reopening an issue with violation feedback)."""
    raw = issue.raw
    lines: list[str] = [
        f"{issue.id} (P{issue.priority} {issue.issue_type})",
        f"Title: {issue.title}",
    ]
    description = (raw.get("description") or "").strip()
    if description:
        lines.extend(["Description:", description])
    acceptance = (raw.get("acceptance_criteria") or "").strip()
    if acceptance:
        lines.extend(["Acceptance:", acceptance])
    notes = (raw.get("notes") or "").strip()
    if notes:
        lines.extend(
            [
                "Notes (operator-supplied; read these — they often carry retry feedback):",
                notes,
            ]
        )
    return "\n".join(lines)


def _try_epic_summary(bd: BdHandoffSource, epic_id: str) -> str | None:
    """Short summary for the parent epic. Swallows bd errors so a
    transient lookup failure doesn't crash the whole turn — the parent
    epic is context, not contract."""
    try:
        epic = bd.show(epic_id)
    except DriverBdError:
        return None
    return f"{epic.id} — {epic.title}"


def _git_diff_name_status(git_root: Path, base_sha: str) -> tuple[str, ...]:
    """`git diff --name-status <base_sha>..HEAD`. Returns one
    "<status>\t<path>" line per change, in git's natural order. On any
    git error (no such ref, not a repo, etc.) returns an empty tuple —
    the handoff renders a "(none yet)" placeholder."""
    try:
        result = subprocess.run(  # noqa: S603 — fixed git args, no shell
            ["git", "diff", "--name-status", f"{base_sha}..HEAD"],  # noqa: S607 — git on PATH is expected
            cwd=git_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ()
    if result.returncode != 0:
        return ()
    stdout = (result.stdout or "").strip()
    if not stdout:
        return ()
    return tuple(line.strip() for line in stdout.splitlines() if line.strip())


def _has_label(issue: BeadsIssue, label: str) -> bool:
    return label in issue.labels


def _short_body(issue: BeadsIssue) -> str:
    """One-line summary of a thought bead for the handoff list.

    Uses the title if present (the driver writes structured titles like
    "session-state <id> (success)"); falls back to the first non-empty
    line of the description. Tight by design — the handoff is a survey
    of state, not a transcript."""
    if issue.title:
        return issue.title
    description = (issue.raw.get("description") or "").strip()
    for line in description.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return issue.id


__all__ = [
    "CHARS_PER_TOKEN",
    "MAX_DECISIONS",
    "MAX_OBSERVATIONS",
    "RENDER_CHAR_CAP",
    "TOKEN_BUDGET",
    "BdHandoffSource",
    "Handoff",
    "build_handoff",
]
