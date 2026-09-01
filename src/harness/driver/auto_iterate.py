"""Drive + critic auto-iterate (harness-3f8e).

Wraps `run_loop` in an outer convergence loop:

    while not converged:
        drive_result = run_loop(...)             # drain ready queue
        if drive halted: stop
        findings = run_critic(...)               # propose follow-up bugs
        for f in findings:
            bd create + dep_add to auto-block the epic
        if findings was empty for N consecutive passes: stop

The critic itself lives in ``driver.critic``; this module owns spec
resolution (``--spec`` flag OR ``plan-source:<file>`` epic label),
workspace snapshotting (source files only — skip ``node_modules``,
``.git``, ``.harness``, vendored stuff), and the bd writes.

Convergence is conservative: ``convergence_streak`` consecutive empty
critic passes (default 2) before declaring done. One empty critic pass
isn't enough — a transient model whiff would false-converge. A hard
``max_passes`` cap (default 8) terminates regardless.

An empty critic pass only counts toward convergence when the drive made
progress — closed something, or drained the ready queue by running turns
(harness-eh07). A ``success`` drive that ran zero turns means the ready
queue was empty from the start (e.g. every child already closed); that's
``no_work`` (a no-op run), not convergence, and exits non-zero so a
wrapper doesn't mistake it for a finish.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from harness.driver.bd import DriverBd, DriverBdError
from harness.driver.critic import (
    CriticAdapterError,
    CriticFinding,
    budget_snapshot,
    critic_char_budget,
    run_critic,
)
from harness.driver.fsm_executor import _test_cmd_script
from harness.driver.loop import LoopConfig, LoopResult, ambient_vllm_trace, run_loop
from harness.model.adapter import ModelAdapter

# Source-file suffixes we sample into the critic's workspace snapshot.
# JS/TS for browser-game workspaces, Python for tool/script workspaces.
# HTML/CSS too because the critic occasionally needs to cite the entry
# page (e.g. "the script tag at index.html:25 loads game.js but
# canvas#gameCanvas is missing").
_SOURCE_SUFFIXES: frozenset[str] = frozenset({".js", ".mjs", ".cjs", ".ts", ".py", ".html", ".css"})
# Directories we never recurse into. Mirrors plan_linter's exclude list
# plus a few common JS-build directories.
_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".harness",
        "node_modules",
        "__pycache__",
        "venv",
        ".venv",
        "dist",
        "build",
        # Generated reports / tool caches (harness-zk3c). These carry no
        # source the critic should review and — being large generated
        # HTML/JS/CSS — blew the context window (snake htmlcov was ~115 KB
        # of a ~120 KB snapshot, vs ~6 KB of actual code).
        "htmlcov",
        "coverage",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nyc_output",
        ".next",
        ".svelte-kit",
        "site-packages",
        "target",
    }
)
# Bytes-per-file ceiling on the workspace snapshot. A single file over
# this is included whole (we don't truncate mid-file — line numbers
# would shift), but the per-file cap stops a vendored bundle hidden
# under a non-excluded path from blowing the model's context window.
# 256 KB is comfortable for hand-written game code; vendor bundles
# are typically multi-MB and get dropped.
_PER_FILE_BYTES_CAP = 256 * 1024


@dataclass(frozen=True)
class AutoIterateConfig:
    """Outer-loop knobs. `loop_config` is forwarded verbatim to every
    `run_loop` invocation — same model, same workspace, same retry
    budget, etc. across passes."""

    loop_config: LoopConfig
    spec_path: Path | None = None
    max_passes: int = 8
    convergence_streak: int = 2
    critic_max_findings: int = 10
    # Adapter call cost knobs surfaced for ops tuning. Defaults
    # match the critic module's own defaults.
    critic_max_tokens: int = 4096
    critic_temperature: float = 0.2
    # Grounding-verify gate (harness-hdwp). On by default: each finding
    # that clears the deterministic gates is re-checked against the real
    # code at its citation before it is filed. Disable only to reproduce
    # the pre-hdwp behavior for attribution/measurement.
    critic_verify_grounding: bool = True
    # Slice-scoped critique (harness-a0yj). On by default: critique one
    # symbol-aligned slice at a time so the model is anchored to bounded
    # real code. When on, the whole-file context budget (harness-zk3c) is
    # skipped — slicing bounds each call, and the gates still validate
    # against the full snapshot. Set False for the legacy whole-file path.
    critic_slice_mode: bool = True


@dataclass
class AutoIterateResult:
    """Outcome of `run_auto_iterate`. `drive_results` is the per-pass
    LoopResult list (oldest first); `critic_findings_total` counts the
    findings that were actually filed as beads across every pass."""

    passes_run: int
    drive_results: list[LoopResult]
    critic_findings_total: int
    exit_reason: Literal[
        "converged", "passes_exhausted", "drive_halted", "critic_failed", "stuck", "no_work"
    ]
    filed_beads: list[str] = field(default_factory=list)
    # Whether the critic had a spec to ground against this run. False means
    # it ran artifact-only (no spec_quote grounding) — surfaced so a no-op
    # run isn't read as a clean finish (harness-d3cs).
    spec_resolved: bool = False


def _resolve_spec(config: AutoIterateConfig, bd: DriverBd) -> str | None:
    """Spec resolution order: explicit ``--spec`` path wins; else read
    the epic's ``plan-source:<file>`` label and look up the file under
    the workspace. Returns None when no spec is available (the critic
    then runs in artifact-only mode — citations still required, spec
    quote tolerated empty)."""
    if config.spec_path is not None:
        try:
            return config.spec_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    try:
        epic = bd.show(config.loop_config.epic_id)
    except DriverBdError:
        return None
    for label in epic.labels:
        if not label.startswith("plan-source:"):
            continue
        candidate = config.loop_config.workspace / label[len("plan-source:") :]
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
        # Plan source might live outside the workspace (e.g. the
        # harness repo's scratch/gta/spec/ tree). Try relative to the
        # workspace's parents up to git_root before giving up.
        for parent in config.loop_config.workspace.parents:
            candidate = parent / label[len("plan-source:") :]
            if candidate.is_file():
                try:
                    return candidate.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    return None
        # Last resort: the label often carries only a bare filename while
        # the spec lives in a sibling/hidden dir the exact-path attempts
        # never reach — e.g. label `plan-source:gta2-spec.md` but the file
        # at <workspace>/../.artifacts/spec/gta2-spec.md. `.artifacts` is
        # hidden, so the source walk skips it too. Search the workspace
        # subtree and its immediate parent for the basename (harness-s1rv).
        basename = Path(label[len("plan-source:") :]).name
        found = _search_for_basename(
            (config.loop_config.workspace, config.loop_config.workspace.parent), basename
        )
        if found is not None:
            try:
                return found.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
    return None


def _search_for_basename(roots: tuple[Path, ...], basename: str) -> Path | None:
    """Bounded breadth-first search for `basename` under each root. Skips
    `_EXCLUDE_DIRS` (so a vendored bundle never matches) but — unlike the
    source walk — descends into hidden dirs, since specs commonly live in
    `.artifacts/`. Depth- and visit-capped so a huge parent tree can't turn
    spec resolution into a full-disk crawl. Returns the shallowest match
    (ties broken lexicographically) for determinism."""
    max_depth = 6
    max_dirs = 4096
    for root in roots:
        visited = 0
        # (dir, depth) frontier; sort siblings for a stable match order.
        frontier: list[tuple[Path, int]] = [(root, 0)]
        while frontier:
            cur, depth = frontier.pop(0)
            visited += 1
            if visited > max_dirs:
                break
            try:
                entries = sorted(cur.iterdir())
            except OSError:
                continue
            subdirs: list[tuple[Path, int]] = []
            for entry in entries:
                if entry.is_dir():
                    if entry.name in _EXCLUDE_DIRS or depth + 1 > max_depth:
                        continue
                    subdirs.append((entry, depth + 1))
                elif entry.name == basename and entry.is_file():
                    return entry
            frontier.extend(subdirs)
    return None


def _iter_source_files(workspace: Path) -> Iterator[Path]:
    """Walk `workspace`, yielding source files. Skips _EXCLUDE_DIRS at
    every depth and any directory whose name starts with `.` (hidden)."""
    stack: list[Path] = [workspace]
    while stack:
        cur = stack.pop()
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in _EXCLUDE_DIRS or entry.name.startswith("."):
                    continue
                stack.append(entry)
                continue
            if entry.suffix in _SOURCE_SUFFIXES:
                yield entry


def _snapshot_source_files(workspace: Path) -> dict[str, str]:
    """Build the critic's workspace snapshot: rel-path -> text. Files
    over `_PER_FILE_BYTES_CAP` are dropped (logged via stderr is not
    appropriate here — the auto-iterate caller controls logging).
    Sorted by path so the snapshot is stable across calls."""
    out: dict[str, str] = {}
    workspace = workspace.resolve()
    for entry in _iter_source_files(workspace):
        try:
            size = entry.stat().st_size
        except OSError:
            continue
        if size > _PER_FILE_BYTES_CAP:
            continue
        try:
            text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = entry.resolve().relative_to(workspace).as_posix()
        out[rel] = text
    return out


def _file_critic_finding(
    bd: DriverBd, epic_id: str, loop_run_id: str, finding: CriticFinding
) -> str:
    """File one CriticFinding as a bd issue under `epic_id`. Embeds the
    spec quote and evidence path in the description so the bead carries
    the same audit trail the critic produced. Returns the new bead id."""
    description = (
        f"{finding.description}\n\n"
        f"evidence: {finding.evidence_path}\n"
        f"code-quote: {finding.code_quote!r}\n"
        f"spec-quote: {finding.spec_quote!r}"
    )
    labels = (f"critic:{loop_run_id}",)
    new_id = bd.create_with_labels(
        title=finding.title,
        description=description,
        issue_type="bug",
        priority=finding.priority,
        labels=labels,
        acceptance=finding.acceptance,
    )
    # Auto-block the epic so the next drive picks the bead up.
    bd.dep_add(epic_id, new_id)
    return new_id


def _workspace_fingerprint(workspace: Path) -> str:
    """Content fingerprint of the workspace's source files (harness-64jge).

    Stands in for a gate fingerprint when a parked issue has no test_cmd —
    the verify-parked class (smoke gate, workspace-typed verify steps),
    where "the gate" is the artifact itself. Built from the same source-file
    set the critic snapshots, so .harness/ scratch and oversized files never
    perturb it. An operator repair (or any inter-pass edit) changes the hash
    and revives the issue for one re-attempt."""
    digest = hashlib.sha256()
    for rel, text in sorted(_snapshot_source_files(workspace).items()):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _gate_fingerprint(workspace: Path, test_cmd: str | None) -> str | None:
    """Content fingerprint of the gate file a parked issue was stuck on
    (harness-s0el9).

    Resolves the script path out of `test_cmd` (the same extraction the FSM
    uses to tell a real gate from a no-op) and hashes its bytes.

    harness-64jge: a park with NO test_cmd at all — verify_failed on a
    smoke / workspace-typed gate, or write_test->halted — used to return
    None, which `_revive_repaired_gates` treats as "never revive". That
    stranded the whole verify-parked class forever (loop_run=f9e09713 ran
    0 turns against two such parks). Those now fall back to the workspace
    source fingerprint: the failing gate's subject IS the workspace, so an
    inter-pass repair reads as a gate change. None remains only for a
    test_cmd whose script can't be resolved or read — there's a claimed
    gate file but nothing to fingerprint, so conservative no-revive holds.
    A None->hash transition (a gate that appeared) reads as a change, same
    as hash->hash'.
    """
    if not test_cmd:
        return _workspace_fingerprint(workspace)
    script = _test_cmd_script(test_cmd)
    if script is None:
        return None
    path = workspace / script
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _revive_repaired_gates(
    workspace: Path,
    carried_skip: set[str],
    skip_gate_fp: dict[str, str | None],
    skip_gate_cmd: dict[str, str],
) -> list[str]:
    """Drop carried-skip ids whose gate file changed since they parked
    (harness-s0el9), so the next pass re-drives them once.

    Mutates `carried_skip`, `skip_gate_fp`, and `skip_gate_cmd` in place: an
    id whose current gate fingerprint differs from the one captured at park
    time is removed from all three and returned. Ids with a None baseline (no
    resolvable gate at park) are never revived — there's nothing to detect a
    repair against, so leaving them skipped is the conservative default
    (preserves harness-6y2dc's no-wheel-spin guarantee). If a revived issue
    re-parks, its fresh park re-snapshots the fingerprint, so an
    unchanged-but-still-red gate is only re-driven once, not every pass."""
    revived: list[str] = []
    for issue_id in sorted(carried_skip):
        baseline = skip_gate_fp.get(issue_id)
        if baseline is None:
            continue
        if _gate_fingerprint(workspace, skip_gate_cmd.get(issue_id)) != baseline:
            revived.append(issue_id)
    for issue_id in revived:
        carried_skip.discard(issue_id)
        skip_gate_fp.pop(issue_id, None)
        skip_gate_cmd.pop(issue_id, None)
    return revived


def run_auto_iterate(
    adapter: ModelAdapter, bd: DriverBd, config: AutoIterateConfig
) -> AutoIterateResult:
    """Drive-then-critic-then-drive until convergence or exhaustion.

    Each pass:
      1. ``run_loop(loop_config)`` drains the ready queue.
      2. If the drive halted/interrupted, stop and return
         ``drive_halted`` — critic isn't consulted; the operator
         deals with the halt first.
      3. Snapshot source files + load spec, then call ``run_critic``.
      4. File each validated finding as a new bead, auto-blocked
         against the epic.
      5. Empty findings → bump the streak. ``convergence_streak``
         empty passes in a row → ``converged``.
      6. Otherwise continue. Hard cap at ``max_passes``.
    """
    epic_id = config.loop_config.epic_id
    drive_results: list[LoopResult] = []
    filed_beads: list[str] = []
    findings_total = 0
    empty_streak = 0
    # harness-6y2dc: union of every prior pass's parked issues. Each
    # pass runs with a fresh LoopRunState (resume cleared below), so
    # without this carry a pass re-drives issues that already parked
    # last pass — `bd ready` still lists parked-but-open issues, and
    # the workspace is unchanged between passes (critic only reads +
    # files beads). Carrying them as skip_issue_ids lets the next pass
    # exit fast (partial) instead of burning its whole budget
    # re-discovering the same issues can't close.
    # harness-64jge: seeded with the operator's skip_issue_ids — the
    # later replace() overwrites the LoopConfig field wholesale, so an
    # unseeded set silently dropped operator skips from pass 1 onward.
    # Operator ids get no gate fingerprint (below), so the revive pass
    # never un-skips them — only drive-parked ids earn revival.
    carried_skip: set[str] = set(config.loop_config.skip_issue_ids)
    # harness-s0el9: per-carried-id gate fingerprint + the cmd that produced
    # it, captured when the issue parked. A bugged gate (loop_run=e7644fa7:
    # gate grepped a literal token the correct idiomatic code never emits)
    # parks an issue that a between-pass gate repair would unblock — but
    # carried_skip alone strands it forever and the epic exits `stuck` on a
    # now-passing gate. Re-fingerprinting before each pass lets a changed gate
    # revive the issue for one re-attempt.
    skip_gate_fp: dict[str, str | None] = {}
    skip_gate_cmd: dict[str, str] = {}
    # harness-6zjjm: a gate-suspect park dropped its carried test, so it has
    # no gate file for the s0el9 fingerprint to detect a change on — the
    # DROP itself is the re-author signal. Revive each such park EXACTLY
    # once on the next pass. `gate_suspect_revive` arms the next pass;
    # `gate_suspect_seen` bounds it to one revival per id ever — a revived
    # issue that re-parks (gate-suspect again or otherwise) is carried
    # normally, no wheel-spin.
    gate_suspect_revive: set[str] = set()
    gate_suspect_seen: set[str] = set()
    workspace = config.loop_config.workspace
    spec_text = _resolve_spec(config, bd)
    spec_resolved = spec_text is not None
    if not spec_resolved:
        # The critic falls back to artifact-only mode (no spec_quote
        # grounding). Surface it loudly — a silently spec-blind critic on a
        # fully-drained epic finds nothing and looks like a clean pass
        # (harness-s1rv).
        print(
            "auto-iterate: no spec resolved — critic running artifact-only "
            "(no spec grounding). Pass --spec PATH or fix the epic's "
            "plan-source:<file> label.",
            file=sys.stderr,
        )

    for pass_index in range(config.max_passes):
        # Inner drive. Pass `resume_from` only on the FIRST iteration;
        # subsequent passes always start a fresh loop_run so closed
        # critic-filed beads don't get tangled with the prior run's
        # state file. (run_loop's own state machine handles fresh runs
        # cleanly when resume_from is None.)
        loop_config = config.loop_config if pass_index == 0 else _clear_resume(config.loop_config)
        # harness-s0el9: before re-applying the skip-list, revive any carried
        # id whose gate file changed since it parked — a repaired gate must
        # un-strand the issue instead of carrying it skipped forever.
        for revived_id in _revive_repaired_gates(
            workspace, carried_skip, skip_gate_fp, skip_gate_cmd
        ):
            print(
                f"auto-iterate: gate for {revived_id} changed since it parked — "
                "re-driving once (harness-s0el9)",
                file=sys.stderr,
            )
        # harness-6zjjm: revive gate-suspect parks once — the dropped test
        # means s0el9's fingerprint can't fire, but the re-author still
        # needs an attempt. Drop them from carried_skip (+ stale fingerprint
        # state) so this pass re-drives them; the arm is one-shot.
        for revived_id in sorted(gate_suspect_revive):
            carried_skip.discard(revived_id)
            skip_gate_fp.pop(revived_id, None)
            skip_gate_cmd.pop(revived_id, None)
            print(
                f"auto-iterate: {revived_id} parked gate-suspect (test dropped) — "
                "re-driving once to re-author the gate (harness-6zjjm)",
                file=sys.stderr,
            )
        gate_suspect_revive.clear()
        # harness-64jge: when every ready issue under the epic is carried-
        # skipped and nothing was revived, the pass is a foregone no-op —
        # run_loop would spin up full scaffolding (workspace snapshot,
        # last-green baseline, vllm trace) to execute 0 turns and exit
        # partial (loop_run=f9e09713). Report stuck immediately instead.
        # First-pass runs are exempt: pass 0's skip set is operator-
        # supplied, and the operator may still want the critic's read on
        # the artifact. A bd error skips the check — the pass itself
        # handles bd failure with a proper halt.
        if pass_index > 0 and carried_skip:
            try:
                ready_ids = {issue.id for issue in bd.ready_under_epic(epic_id)}
            except DriverBdError:
                ready_ids = set()
            if ready_ids and ready_ids <= carried_skip:
                print(
                    f"auto-iterate: every ready issue is carried-skipped with no "
                    f"gate repair ({', '.join(sorted(ready_ids))}) — skipping "
                    f"pass {pass_index + 1}, exiting stuck (harness-64jge)",
                    file=sys.stderr,
                )
                return AutoIterateResult(
                    passes_run=pass_index,
                    drive_results=drive_results,
                    critic_findings_total=findings_total,
                    exit_reason="stuck",
                    filed_beads=filed_beads,
                    spec_resolved=spec_resolved,
                )
        # harness-6y2dc: feed prior passes' parked ids forward so this
        # pass skips them instead of re-driving from cold.
        if carried_skip:
            loop_config = replace(loop_config, skip_issue_ids=frozenset(carried_skip))
        drive_result = run_loop(adapter, bd, loop_config)
        drive_results.append(drive_result)
        # harness-s0el9: carry newly-parked ids forward AND snapshot each
        # one's gate fingerprint, so a between-pass gate repair can revive it
        # next pass. Re-parked revived issues re-snapshot here, bounding the
        # re-drive to once per actual gate change.
        for pid in drive_result.parked_issues:
            carried_skip.add(pid)
            gate_cmd = drive_result.parked_test_cmds.get(pid)
            skip_gate_fp[pid] = _gate_fingerprint(workspace, gate_cmd)
            if gate_cmd is not None:
                skip_gate_cmd[pid] = gate_cmd
        # harness-6zjjm: arm a one-shot revive for each NEW gate-suspect
        # park. `gate_suspect_seen` ensures a re-parked issue (it already
        # had its one revive) is carried normally — no wheel-spin.
        for pid in drive_result.gate_suspect_parks:
            if pid not in gate_suspect_seen:
                gate_suspect_seen.add(pid)
                gate_suspect_revive.add(pid)

        if drive_result.exit_reason in {"halted", "interrupted"}:
            return AutoIterateResult(
                passes_run=pass_index + 1,
                drive_results=drive_results,
                critic_findings_total=findings_total,
                exit_reason="drive_halted",
                filed_beads=filed_beads,
                spec_resolved=spec_resolved,
            )

        # Critic pass. Spec is loaded once (above) — passes don't
        # change the spec. Workspace snapshot IS recomputed per pass
        # (the drive just wrote to it).
        snapshot = _snapshot_source_files(config.loop_config.workspace)
        # In slice mode (harness-a0yj) each critic call sees one bounded
        # slice, so the whole-file context budget would only shrink what the
        # gates validate against — skip it. In whole-file mode, budget the
        # snapshot against the model context window so a large workspace
        # doesn't overflow and fail the pass (harness-zk3c). Trims are
        # surfaced — never silent.
        if not config.critic_slice_mode:
            char_budget = critic_char_budget(
                context_window=adapter.context_window,
                max_tokens=config.critic_max_tokens,
                spec_text=spec_text,
            )
            snapshot, trim_notes = budget_snapshot(snapshot, char_budget=char_budget)
            for note in trim_notes:
                print(f"critic: context budget — {note}", file=sys.stderr)
        dedup_titles = _dedup_titles_under_epic(bd, epic_id)
        try:
            # harness-5t0a: wrap the critic in the same ambient trace as the
            # drive, scoped to this pass's loop_run_id, so an env-unset run
            # records the critic generation + verify calls in
            # .harness/loop_runs/<id>.vllm_trace.jsonl alongside the drive
            # turns. An explicit HARNESS_VLLM_TRACE still wins (the context
            # is a no-op when the env is already set).
            with ambient_vllm_trace(config.loop_config.workspace, drive_result.loop_run_id):
                findings = run_critic(
                    adapter=adapter,
                    spec_text=spec_text,
                    workspace_snapshot=snapshot,
                    closed_this_run=tuple(drive_result.closed),
                    open_under_epic=dedup_titles,
                    max_findings=config.critic_max_findings,
                    max_tokens=config.critic_max_tokens,
                    temperature=config.critic_temperature,
                    verify_grounding=config.critic_verify_grounding,
                    slice_mode=config.critic_slice_mode,
                )
        except CriticAdapterError:
            # The model was never reached this pass. Surface it — do NOT
            # let a silent [] feed the convergence streak (harness-fote).
            return AutoIterateResult(
                passes_run=pass_index + 1,
                drive_results=drive_results,
                critic_findings_total=findings_total,
                exit_reason="critic_failed",
                filed_beads=filed_beads,
                spec_resolved=spec_resolved,
            )

        if not findings:
            # Convergence requires actual progress (harness-dqoy,
            # harness-eh07). An empty critic pass only counts toward "done"
            # if the drive either closed something this pass or drained the
            # ready queue by actually running turns. The subtlety: a
            # "success" with turns_used == 0 means the ready queue was EMPTY
            # FROM THE START (e.g. every child of the epic was already
            # closed) — nothing was drained, so the critic trivially finds
            # nothing against an unchanged artifact. That's a no-op run, not
            # convergence. When this pass made no progress, terminate with
            # the flavor that fits:
            #   - success + turns=0 + closed=0 -> "no_work": the epic had no
            #     ready work to begin with. Non-zero exit so a wrapper
            #     doesn't read the no-op as a real finish.
            #   - exhausted / partial          -> "stuck": the drive ran but
            #     stalled (max_turns hit, or everything parked) without
            #     closing anything — the artifact didn't change.
            made_progress = bool(drive_result.closed)
            drained_by_work = drive_result.exit_reason == "success" and drive_result.turns_used > 0
            if not (made_progress or drained_by_work):
                empty_queue = (
                    drive_result.exit_reason == "success"
                    and drive_result.turns_used == 0
                    and not drive_result.closed
                )
                return AutoIterateResult(
                    passes_run=pass_index + 1,
                    drive_results=drive_results,
                    critic_findings_total=findings_total,
                    exit_reason="no_work" if empty_queue else "stuck",
                    filed_beads=filed_beads,
                    spec_resolved=spec_resolved,
                )
            empty_streak += 1
            if empty_streak >= config.convergence_streak:
                return AutoIterateResult(
                    passes_run=pass_index + 1,
                    drive_results=drive_results,
                    critic_findings_total=findings_total,
                    exit_reason="converged",
                    filed_beads=filed_beads,
                    spec_resolved=spec_resolved,
                )
            continue

        empty_streak = 0
        for finding in findings:
            try:
                new_id = _file_critic_finding(bd, epic_id, drive_result.loop_run_id, finding)
            except DriverBdError:
                # bd write failed mid-batch — skip this finding, keep
                # going. A later pass will see the same workspace and
                # propose it again.
                continue
            filed_beads.append(new_id)
            findings_total += 1

    return AutoIterateResult(
        passes_run=config.max_passes,
        drive_results=drive_results,
        critic_findings_total=findings_total,
        exit_reason="passes_exhausted",
        filed_beads=filed_beads,
        spec_resolved=spec_resolved,
    )


def _clear_resume(loop_config: LoopConfig) -> LoopConfig:
    """Return a LoopConfig with `resume_from=None` so a follow-up pass
    starts a fresh loop_run. Other fields preserved."""
    return replace(loop_config, resume_from=None)


def _dedup_titles_under_epic(bd: DriverBd, epic_id: str) -> tuple[str, ...]:
    """Titles of every child bead under `epic_id`, **open or closed**.
    Fed to the critic so its dedupe gate drops near-dupes BEFORE they're
    filed — including refiles of bugs that were already fixed/closed on
    a prior pass (harness-hdwp). The 2026-05-29 run refiled closed
    harness-77ht as harness-6rai precisely because dedup only looked at
    open beads, and re-rolled the same themes every pass because each
    pass closed the prior pass's findings out of the open set.

    Soft on bd errors — a stale title list just means the critic might
    propose a dupe; the fuzz-match still catches strong overlaps and the
    grounding-verify gate catches fabricated ones. Better than failing
    the whole auto-iterate run on a transient bd hiccup."""
    try:
        epic = bd.show(epic_id)
    except DriverBdError:
        return ()
    titles: list[str] = []
    for dep in epic.raw.get("dependencies") or []:
        dep_id = dep.get("id")
        if not isinstance(dep_id, str):
            continue
        try:
            child = bd.show(dep_id)
        except DriverBdError:
            continue
        titles.append(child.title)
    return tuple(titles)


__all__ = [
    "AutoIterateConfig",
    "AutoIterateResult",
    "run_auto_iterate",
]
