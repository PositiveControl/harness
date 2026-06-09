"""Meta-tools that signal turn-phase transitions — harness-kbnl.

The driver's TurnFSM transitions on PhaseOutcome events. These tools
are the model's vocabulary for emitting those outcomes:

  - `submit_assessment(current_state, gap, approach, tdd_applicable)`
      → emits the ASSESS phase's outcome
  - `skip_test_phase(reason)`
      → in WRITE_TEST, emits a tdd-skip outcome (operator escape hatch
        when the issue doesn't admit a unit test)
  - `submit_failing_test(test_path, test_cmd, failure_output)`
      → emits the WRITE_TEST phase's outcome; the test_cmd
        auto-feeds into VERIFY
  - `submit_implementation_complete(summary)`
      → emits the IMPLEMENT phase's outcome

Design notes:

  - Each tool is a dataclass with a `captured` list field. The
    driver inspects that field after the phase's `run_tool_loop`
    returns to construct the PhaseOutcome — simpler than threading
    a callback through the loop.
  - Read-tier (no write side effects, no file modifications). Just
    structured intent the FSM consumes.
  - Validation lives in the `call()` method: empty / whitespace-only
    args raise ValueError. The orchestrator surfaces the error to
    the model so it can retry with real content; a vacuous
    assessment isn't accepted.
  - Schema-cheap (~100-250 tokens each). The full meta-tool set
    adds <800 tokens to the schema budget — well within the
    coding profile's headroom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.tools.base import ToolSpec

# Validation threshold for the structured-text fields. Below this the
# tool refuses — too short to be a real assessment / summary.
# 20 chars catches "ok" / "done" / "implemented" without being so
# strict that legitimate short answers get rejected. The model gets
# the error back as a tool result and can resubmit.
_MIN_FIELD_CHARS: int = 20


def _require_nonempty(name: str, value: str, *, min_chars: int = _MIN_FIELD_CHARS) -> str:
    """Trim + length-check a single string field. Raises ValueError
    on empty / whitespace-only / too-short input. Returns the trimmed
    value so callers can store the clean form.

    `min_chars` defaults to the substantive-prose floor; identifier-
    shaped fields (paths, commands) pass a lower value since they're
    legitimately short."""
    cleaned = value.strip() if value else ""
    if not cleaned:
        raise ValueError(f"{name!r} must be a non-empty string")
    if len(cleaned) < min_chars:
        raise ValueError(
            f"{name!r} too short ({len(cleaned)} chars); supply at least "
            f"{min_chars} characters of actual content"
        )
    return cleaned


# Floor for identifier-shaped fields (paths, commands). Just guards
# against empty / 1-char inputs; the substantive validation lives
# in the prose-shaped fields (current_state, gap, approach,
# failure_output, reason, summary).
_MIN_IDENTIFIER_CHARS: int = 3


@dataclass
class SubmitAssessmentTool:
    """Records the model's gap analysis at the end of the ASSESS phase.

    Required fields:
      - `current_state`: what the file/system looks like right now,
        in the model's words. Pinning this surfaces misreads of the
        file before they become bad edits.
      - `gap`: difference between current state and acceptance
        criteria. The actionable delta.
      - `approach`: how the model plans to close the gap. Sets up
        the IMPLEMENT phase with a written game-plan the FSM can
        echo back into the handoff if a retry is needed.

    Optional:
      - `tdd_applicable`: defaults True. Setting False routes the
        FSM directly from ASSESS to IMPLEMENT, skipping WRITE_TEST.
        The model must justify the skip via the assessment text
        (e.g. "this is a UI tweak with no unit-testable behavior")."""

    captured: list[dict[str, Any]] = field(default_factory=list)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="submit_assessment",
            description=(
                "Required exit point of the ASSESS phase. Submit your gap "
                "analysis: where the artifact stands now, what differs "
                "from the acceptance criteria, and how you plan to close "
                "the gap. Set tdd_applicable=false ONLY when the issue "
                "genuinely admits no unit test (e.g. UI tweak, doc edit) — "
                "supply your reason in the approach field."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "current_state": {
                        "type": "string",
                        "description": (
                            "What the artifact looks like right now. "
                            "Be concrete — name specific functions / "
                            "data structures / observable behaviors. "
                            f"Min {_MIN_FIELD_CHARS} chars."
                        ),
                    },
                    "gap": {
                        "type": "string",
                        "description": (
                            "Specific delta between current state and the "
                            "acceptance criteria. The actionable "
                            f"difference. Min {_MIN_FIELD_CHARS} chars."
                        ),
                    },
                    "approach": {
                        "type": "string",
                        "description": (
                            "How you plan to close the gap. Concrete steps. "
                            f"Min {_MIN_FIELD_CHARS} chars."
                        ),
                    },
                    "tdd_applicable": {
                        "type": "boolean",
                        "description": (
                            "Default true. Set false ONLY when the issue "
                            "doesn't admit a unit test; justify in approach."
                        ),
                    },
                    "premise_mismatch": {
                        "type": "boolean",
                        "description": (
                            "Default false. Set TRUE only when the issue's "
                            "spec targets a DIFFERENT artifact, project, or "
                            "language than this workspace — e.g. the spec "
                            "names a Python 'snake_game.py' with food/score "
                            "but the workspace is a JavaScript driving game. "
                            "Setting it PARKS the bead for operator review "
                            "instead of forcing an unrelated change. Explain "
                            "the mismatch in `gap`. Do NOT set it merely "
                            "because the work is hard, or because this bead's "
                            "OWN deliverable is absent (that absence is your "
                            "task) — only when the spec is grounded against a "
                            "different codebase than the one in front of you."
                        ),
                    },
                    "already_satisfied": {
                        "type": "boolean",
                        "description": (
                            "Default false. Set TRUE only when the workspace "
                            "ALREADY fully satisfies this bead's acceptance and "
                            "NO change is needed — e.g. a scaffold/skeleton bead "
                            "whose files + structure already exist from earlier "
                            "work. Routes straight to CLOSE (a close-time verify "
                            "still gates the close, so a wrong claim fails there, "
                            "not silently). Honored only for structural "
                            "(scaffold/declaration) beads; ignored otherwise. Do "
                            "NOT set it to dodge work that genuinely remains — "
                            "say so in `gap` and implement it instead."
                        ),
                    },
                },
                "required": ["current_state", "gap", "approach"],
            },
            tier="read",
            display_name="Submit assessment",
        )

    def call(
        self,
        *,
        current_state: str,
        gap: str,
        approach: str,
        tdd_applicable: bool = True,
        premise_mismatch: bool = False,
        already_satisfied: bool = False,
    ) -> str:
        cs = _require_nonempty("current_state", current_state)
        gp = _require_nonempty("gap", gap)
        ap = _require_nonempty("approach", approach)
        self.captured.append(
            {
                "current_state": cs,
                "gap": gp,
                "approach": ap,
                "tdd_applicable": bool(tdd_applicable),
                "premise_mismatch": bool(premise_mismatch),
                "already_satisfied": bool(already_satisfied),
            }
        )
        if premise_mismatch:
            return "assessment recorded (premise mismatch flagged — bead will be parked)"
        if already_satisfied:
            return "assessment recorded (already satisfied — routing to close, verify still gates)"
        tdd_note = "" if tdd_applicable else " (TDD skipped — see approach for reason)"
        return f"assessment recorded{tdd_note}"

    def latest(self) -> dict[str, Any] | None:
        """Most recent capture, or None when no assessment was
        submitted this phase. The driver uses this to construct the
        PhaseOutcome after run_tool_loop returns."""
        return self.captured[-1] if self.captured else None


@dataclass
class FlagBlockedTool:
    """ASSESS escape hatch: the bead's PREMISE is unmet — a precondition
    that a DIFFERENT, upstream bead was supposed to land never arrived
    (harness-u1il5 class). Calling this routes ASSESS straight to a
    parked-and-flagged halt instead of burning the retry budget on a
    futile premise.

    The trap this guards against (harness-pmz3i): a build/create bead's
    own deliverable never exists before the bead runs — that absence IS
    the task, not a block. flag_blocked is ONLY for an absent UPSTREAM
    precondition, never for the artifact THIS bead is asked to produce.

    Deliberately narrow so the model can't use it to dodge hard work:
      - `missing` must name a CONCRETE, operator-checkable absent artifact
        (a function / file / symbol / keycode the bead presupposes) that
        another bead owns — not this bead's own deliverable, not a vague
        "this is hard."
      - `reason` explains why that upstream absence blocks the bead.
    The outcome is a PARK (operator review with the `blocked` flag), never
    a close — so a wrong "blocked" costs only an operator glance, while a
    right one saves three futile attempts."""

    captured: list[dict[str, Any]] = field(default_factory=list)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="flag_blocked",
            description=(
                "Use ONLY when this bead cannot be done because a concrete "
                "UPSTREAM precondition is absent — a function / file / "
                "symbol / input handler that a DIFFERENT, earlier bead was "
                "supposed to create, but didn't. This PARKS the bead for "
                "operator review; it does NOT close it and is NOT a way to "
                "skip difficult work. Do NOT flag when the absent thing is "
                "what THIS bead is asked to build/create — that absence is "
                "your task; implement it. Do NOT flag when the thing exists "
                "and is merely hard to change — do the work. Name the "
                "missing upstream artifact concretely."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "missing": {
                        "type": "string",
                        "description": (
                            "The concrete absent UPSTREAM precondition — a "
                            "named function, file, symbol, or keycode the "
                            "bead presupposes, owned by an earlier bead, NOT "
                            "present in the workspace. Must NOT be this "
                            "bead's own deliverable. Operator-checkable. "
                            f"Min {_MIN_IDENTIFIER_CHARS} chars."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why that absence blocks this bead — e.g. 'the "
                            "bead verifies fire-key gating but no fire "
                            "handler exists to gate'. "
                            f"Min {_MIN_FIELD_CHARS} chars."
                        ),
                    },
                },
                "required": ["missing", "reason"],
            },
            tier="read",
            display_name="Flag blocked (premise unmet)",
        )

    def call(self, *, missing: str, reason: str) -> str:
        ms = _require_nonempty("missing", missing, min_chars=_MIN_IDENTIFIER_CHARS)
        rs = _require_nonempty("reason", reason)
        self.captured.append({"missing": ms, "reason": rs})
        return f"flagged blocked: {ms}"

    def latest(self) -> dict[str, str] | None:
        """Most recent capture, or None when flag_blocked wasn't called
        this phase. The driver checks this BEFORE submit_assessment so a
        premise-unmet signal short-circuits the normal ASSESS exit."""
        return self.captured[-1] if self.captured else None


@dataclass
class SkipTestPhaseTool:
    """Operator escape hatch from WRITE_TEST. Captures a non-empty
    reason for skipping the failing-test gate; the FSM transitions
    WRITE_TEST → IMPLEMENT. Use sparingly — the default expectation
    is that real engineering work admits a test."""

    captured: list[dict[str, str]] = field(default_factory=list)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="skip_test_phase",
            description=(
                "Skip the WRITE_TEST phase when the issue does not admit "
                "a unit test. Supply a non-empty reason. The driver "
                "records the skip in the turn audit log. Use only when "
                "you've considered TDD and determined it's not applicable."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Concrete justification for skipping TDD. "
                            f"Min {_MIN_FIELD_CHARS} chars."
                        ),
                    },
                },
                "required": ["reason"],
            },
            tier="read",
            display_name="Skip test phase",
        )

    def call(self, *, reason: str) -> str:
        r = _require_nonempty("reason", reason)
        self.captured.append({"reason": r})
        return "test phase skipped"

    def latest(self) -> dict[str, str] | None:
        return self.captured[-1] if self.captured else None


@dataclass
class SubmitFailingTestTool:
    """Records a red test at the end of WRITE_TEST. The test_cmd is
    captured and auto-fed into VERIFY — the same command that proves
    the test currently fails will later be the one that proves the
    implementation makes it pass.

    failure_output should be a short captured tail of the test
    runner's stderr / stdout demonstrating the failure mode. The
    driver doesn't re-execute the test here (it trusts the model's
    captured output) — VERIFY is where the real re-execution
    happens against the implementation.
    """

    captured: list[dict[str, str]] = field(default_factory=list)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="submit_failing_test",
            description=(
                "Required exit point of the WRITE_TEST phase. Supply the "
                "test file path, the command to run it, and the captured "
                "failure output proving it currently fails red. The "
                "test_cmd will be carried into the VERIFY phase to "
                "confirm the implementation makes it pass."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "test_path": {
                        "type": "string",
                        "description": "Workspace-relative path to the test file.",
                    },
                    "test_cmd": {
                        "type": "string",
                        "description": (
                            "Shell command that runs the test (e.g. "
                            "'pytest tests/test_x.py::test_y -v'). Must "
                            "exit non-zero now and exit zero after "
                            "implementation."
                        ),
                    },
                    "failure_output": {
                        "type": "string",
                        "description": (
                            "Captured stderr/stdout tail proving the "
                            "test currently fails. Trimmed to the "
                            f"failure assertion. Min {_MIN_FIELD_CHARS} chars."
                        ),
                    },
                },
                "required": ["test_path", "test_cmd", "failure_output"],
            },
            tier="read",
            display_name="Submit failing test",
        )

    def call(self, *, test_path: str, test_cmd: str, failure_output: str) -> str:
        # Path + command are identifier-shaped (legitimately short);
        # failure_output is the load-bearing evidence and uses the
        # prose-shaped floor.
        tp = _require_nonempty("test_path", test_path, min_chars=_MIN_IDENTIFIER_CHARS)
        tc = _require_nonempty("test_cmd", test_cmd, min_chars=_MIN_IDENTIFIER_CHARS)
        fo = _require_nonempty("failure_output", failure_output)
        self.captured.append({"test_path": tp, "test_cmd": tc, "failure_output": fo})
        return f"failing test recorded: {tp}"

    def latest(self) -> dict[str, str] | None:
        return self.captured[-1] if self.captured else None


@dataclass
class SubmitImplementationCompleteTool:
    """Records the model's "I'm done with IMPLEMENT" signal. Carries
    a short summary of what changed — the FSM passes it into the
    handoff if VERIFY fails and the loop re-enters IMPLEMENT."""

    captured: list[dict[str, str]] = field(default_factory=list)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="submit_implementation_complete",
            description=(
                "Required exit point of the IMPLEMENT phase. Supply a "
                "short summary of the changes you made. The driver "
                "moves to VERIFY next — if the test or verify steps "
                "fail there, you'll re-enter IMPLEMENT with this "
                "summary in the handoff."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Short, concrete description of the changes "
                            f"made. Min {_MIN_FIELD_CHARS} chars."
                        ),
                    },
                },
                "required": ["summary"],
            },
            tier="read",
            display_name="Submit implementation complete",
        )

    def call(self, *, summary: str) -> str:
        s = _require_nonempty("summary", summary)
        self.captured.append({"summary": s})
        return "implementation complete recorded"

    def latest(self) -> dict[str, str] | None:
        return self.captured[-1] if self.captured else None


__all__ = [
    "FlagBlockedTool",
    "SkipTestPhaseTool",
    "SubmitAssessmentTool",
    "SubmitFailingTestTool",
    "SubmitImplementationCompleteTool",
]
