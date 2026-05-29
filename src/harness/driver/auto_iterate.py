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
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from harness.driver.bd import DriverBd, DriverBdError
from harness.driver.critic import CriticAdapterError, CriticFinding, run_critic
from harness.driver.loop import LoopConfig, LoopResult, run_loop
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


@dataclass
class AutoIterateResult:
    """Outcome of `run_auto_iterate`. `drive_results` is the per-pass
    LoopResult list (oldest first); `critic_findings_total` counts the
    findings that were actually filed as beads across every pass."""

    passes_run: int
    drive_results: list[LoopResult]
    critic_findings_total: int
    exit_reason: Literal["converged", "passes_exhausted", "drive_halted", "critic_failed"]
    filed_beads: list[str] = field(default_factory=list)


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
    spec_text = _resolve_spec(config, bd)

    for pass_index in range(config.max_passes):
        # Inner drive. Pass `resume_from` only on the FIRST iteration;
        # subsequent passes always start a fresh loop_run so closed
        # critic-filed beads don't get tangled with the prior run's
        # state file. (run_loop's own state machine handles fresh runs
        # cleanly when resume_from is None.)
        loop_config = config.loop_config if pass_index == 0 else _clear_resume(config.loop_config)
        drive_result = run_loop(adapter, bd, loop_config)
        drive_results.append(drive_result)

        if drive_result.exit_reason in {"halted", "interrupted"}:
            return AutoIterateResult(
                passes_run=pass_index + 1,
                drive_results=drive_results,
                critic_findings_total=findings_total,
                exit_reason="drive_halted",
                filed_beads=filed_beads,
            )

        # Critic pass. Spec is loaded once (above) — passes don't
        # change the spec. Workspace snapshot IS recomputed per pass
        # (the drive just wrote to it).
        snapshot = _snapshot_source_files(config.loop_config.workspace)
        dedup_titles = _dedup_titles_under_epic(bd, epic_id)
        try:
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
            )

        if not findings:
            empty_streak += 1
            if empty_streak >= config.convergence_streak:
                return AutoIterateResult(
                    passes_run=pass_index + 1,
                    drive_results=drive_results,
                    critic_findings_total=findings_total,
                    exit_reason="converged",
                    filed_beads=filed_beads,
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
    )


def _clear_resume(loop_config: LoopConfig) -> LoopConfig:
    """Return a LoopConfig with `resume_from=None` so a follow-up pass
    starts a fresh loop_run. Other fields preserved."""
    from dataclasses import replace

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
