"""LoopRunState — crash-safe per-loop-run snapshot (harness-ej42).

A `harness loop` invocation owns one `LoopRunState`. It tracks which bd
issues have closed this run, attempt counts per issue (for the
retry-once-then-halt rule), the failure reason carried into a retry
turn, and the turn budget. The struct serialises to a single JSON file
under `<workspace>/.harness/loop_runs/<id>.json` via atomic write so a
crash between turns can't tear the snapshot — `Path.replace()` is
atomic on POSIX, and the load path treats a missing file as "no prior
run" rather than an error.

Why mutable (not frozen): the loop increments `turns_used`, appends to
`closed_this_run`, and writes into `attempt_counts` after every turn —
`dataclasses.replace()` for every field bump would be more ceremony
than value. The save() call after each mutation is the explicit barrier
that crash-safety actually relies on, not immutability.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Filesystem layout under the workspace root. Public so the CLI can list
# runs without having to load every file.
_LOOP_RUNS_SUBDIR = (".harness", "loop_runs")


@dataclass
class LoopRunState:
    """One snapshot of an in-progress loop run.

    Field semantics:
      - `loop_run_id`: 8-char hex prefix of a uuid4. Filename stem.
      - `started_at_sha`: git rev-parse HEAD at run start. Anchors
        the "files touched this run" diff for the handoff builder.
      - `started_at`: wallclock at run start. Serialised as ISO-8601.
      - `epic_id`: bd id of the epic this loop is draining.
      - `closed_this_run`: ordered list of bd ids closed, in close
        order. Both the run log and the handoff consume this.
      - `attempt_counts`: map of bd-id → attempts started this run.
        The retry-once-then-halt rule reads this; once an issue
        closes the entry stays for audit, since attempt_counts
        also doubles as "every issue this run touched."
      - `last_failure`: map of bd-id → failure reason from the
        previous attempt. Populated when attempt #1 failed; cleared
        on close or on halt.
      - `max_turns`: turn budget for this run (CLI default 20).
      - `turns_used`: turns consumed so far this run. Includes
        retries — a single bd issue can consume two turns before
        the halt-on-second-failure rule kicks in.
    """

    loop_run_id: str
    started_at_sha: str
    started_at: datetime
    epic_id: str
    max_turns: int
    closed_this_run: list[str] = field(default_factory=list)
    attempt_counts: dict[str, int] = field(default_factory=dict)
    last_failure: dict[str, str] = field(default_factory=dict)
    turns_used: int = 0
    # harness-kbnl: per-issue FSM resume state. When a turn halts mid-FSM
    # (e.g. ASSESS completed but IMPLEMENT didn't finish), the next
    # attempt picks up with the assessment text + last known phase in
    # the handoff. Plain strings (not enums) so JSON round-trip is
    # trivial and the field stays operator-readable in the .json
    # dump — pretty `cat .harness/loop_runs/<id>.json` shows the
    # state without consumers needing the TurnPhase enum to interpret
    # values. `last_assessment` stores the most recent
    # submit_assessment payload (current_state, gap, approach, etc.)
    # so the handoff can echo it back; `last_test_cmd` stores the
    # WRITE_TEST artifact so VERIFY knows how to re-run it.
    last_turn_phase: dict[str, str] = field(default_factory=dict)
    last_assessment: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_test_cmd: dict[str, str] = field(default_factory=dict)
    # harness-smplj: per-issue last VERIFY test-step failure tail, carried
    # ACROSS attempts. The intra-turn gate-suspect detector compares
    # consecutive verify failures within one turn; when each turn instead
    # halts at the hs50i verify-retry ceiling first, the byte-identical
    # signal never spans two attempts. Persisting the tail lets attempt
    # N+1's first verify fail trip gate-suspect against attempt N's tail.
    last_test_fail_tail: dict[str, str] = field(default_factory=dict)
    # harness-6zjjm: bd ids whose most recent turn halted gate-suspect (the
    # carried test was dropped as implementation-insensitive). Surfaced on
    # LoopResult so an outer auto_iterate pass revives the park once — the
    # gate DROP is the change signal, but s0el9's gate-file fingerprint
    # can't see it (no test_cmd → workspace fingerprint, unchanged between
    # read-only critic passes). Discarded when the issue closes or a later
    # turn doesn't halt gate-suspect.
    gate_suspect_ids: list[str] = field(default_factory=list)
    # harness-zcrd: bd ids the drive parked after max-attempts
    # exhaustion. Filtered out of subsequent `ready_under_epic` results
    # so the drive doesn't re-pick them within the same run. Each entry
    # is also flagged via `bd flag_human` so the operator sees it in
    # `bd human list`. Resume reloads the set; clear manually
    # (state file edit or new loop_run_id) to retry a parked issue.
    parked_issues: list[str] = field(default_factory=list)

    # --- construction -------------------------------------------------

    @classmethod
    def fresh(cls, epic_id: str, max_turns: int, started_at_sha: str) -> LoopRunState:
        """Build a brand-new state with an 8-char run id and the supplied
        HEAD sha. The caller resolves the sha (via `git rev-parse HEAD`
        or equivalent) so this module stays free of subprocess concerns
        and tests don't need a real git tree."""
        return cls(
            loop_run_id=uuid.uuid4().hex[:8],
            started_at_sha=started_at_sha,
            started_at=datetime.now(UTC),
            epic_id=epic_id,
            max_turns=max_turns,
        )

    # --- persistence --------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomically persist the state to `path`. Creates the parent
        directory if absent. Strategy: serialise to JSON, write to a
        sibling `<path>.tmp`, then `Path.replace()` to swap. On POSIX
        `replace` is atomic — a crash between write and replace leaves
        the previous file (if any) untouched, and a half-written tmp
        sibling is reaped on the next save (same name)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._to_dict(), indent=2, sort_keys=True))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> LoopRunState:
        """Rehydrate from `path`. Raises FileNotFoundError if absent —
        the caller (CLI `--resume`) decides whether that's an error or
        a "no such run" message."""
        raw = json.loads(path.read_text())
        return cls._from_dict(raw)

    # --- filesystem layout --------------------------------------------

    @staticmethod
    def state_dir(workspace_root: Path) -> Path:
        """`<workspace_root>/.harness/loop_runs/`. The driver writes
        state files here and the CLI `--list-runs` walks it. Workspace-
        rooted (not user-home-rooted) so loop state correlates with the
        git tree the run was operating against — moving the repo moves
        the state with it."""
        return workspace_root.joinpath(*_LOOP_RUNS_SUBDIR)

    @staticmethod
    def state_path(workspace_root: Path, loop_run_id: str) -> Path:
        return LoopRunState.state_dir(workspace_root) / f"{loop_run_id}.json"

    # --- serialisation internals --------------------------------------

    def _to_dict(self) -> dict[str, Any]:
        # asdict() handles the primitives; only datetime needs custom
        # coercion. ISO-8601 with UTC offset round-trips cleanly through
        # datetime.fromisoformat() on Python 3.11+.
        payload = asdict(self)
        payload["started_at"] = self.started_at.isoformat()
        return payload

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> LoopRunState:
        started_at_raw = raw["started_at"]
        started_at = datetime.fromisoformat(started_at_raw)
        return cls(
            loop_run_id=str(raw["loop_run_id"]),
            started_at_sha=str(raw["started_at_sha"]),
            started_at=started_at,
            epic_id=str(raw["epic_id"]),
            max_turns=int(raw["max_turns"]),
            closed_this_run=list(raw.get("closed_this_run", [])),
            attempt_counts=dict(raw.get("attempt_counts", {})),
            last_failure=dict(raw.get("last_failure", {})),
            turns_used=int(raw.get("turns_used", 0)),
            last_turn_phase=dict(raw.get("last_turn_phase", {})),
            last_assessment=dict(raw.get("last_assessment", {})),
            last_test_cmd=dict(raw.get("last_test_cmd", {})),
            last_test_fail_tail=dict(raw.get("last_test_fail_tail", {})),
            gate_suspect_ids=list(raw.get("gate_suspect_ids", [])),
            parked_issues=list(raw.get("parked_issues", [])),
        )
