"""Executor loop — harness-ml66.

Drives a bd epic to closure by iterating `bd ready --under-epic`, one
issue per turn, until the epic empties or a halt condition fires.
Bypasses `harness chat` and calls `orchestrator.tool_loop.run_tool_loop`
directly so the executor turn skips voice retrieval, persona rewrite,
and chat-history tracking — none of which apply to agent-to-agent
work.

Per-iteration flow:

  1. SIGINT check — clean exit with a session-state bead + state save.
  2. Turn-budget check — exit_exhausted if `turns_used >= max_turns`.
  3. `bd.ready_under_epic(epic_id)` — exit_success if empty.
  4. Take the top of ready; increment attempt counter for that id.
  5. Build the handoff. If `dry_run`, print and return without calling
     the model.
  6. Assemble messages: `[system=character.system_prompt(...) + handoff,
     user="Drive this bd issue to closure..."]`. No voice samples
     (stripped prompt, per harness-ml66).
  7. Call `run_tool_loop` with the coding-profile registry (filesystem
     + git + reckon + discovery + fetch_url; store-dependent tools
     skipped for v0).
  8. Post-turn outcome check: `bd.show(current_id).status == "closed"`
     AND the reply isn't the fabrication-fallback sentinel.
  9. On SUCCESS — record close, write a session-state bead, log,
     continue.
  10. On FIRST FAILURE — stash the reason in `state.last_failure`,
      save, log, retry the same issue.
  11. On SECOND FAILURE — `bd.flag_human`, write a session-state bead
      with `status=halted`, save, log, exit_halted.

Outcome detection is intentionally NOT scan-based on tool calls —
the executor's coding profile gives the model `shell`, so it'll close
via `bd close <id>` via shell rather than a structured `bd_close`
tool. Post-turn `bd.show` is mechanical and doesn't care which path
the model took.

State persists between turns to `<workspace>/.harness/loop_runs/<id>.json`
(crash-safe atomic write via `LoopRunState.save`). Progress log mirrors
the bead writes to a plain-text file so `tail -f` works during a long
run.

Resume contract (`--resume <loop_run_id>`): load the state file, pick
the next ready issue per the same `ready_under_epic` query, continue.
Any in-progress attempt count is preserved — a SIGINT mid-attempt
resumes on attempt #2 (operator-visible via the progress log).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import tarfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from harness.character import Character
from harness.driver.bd import DriverBd, DriverBdError
from harness.driver.handoff import Handoff, build_handoff
from harness.driver.state import LoopRunState
from harness.model.adapter import ChatMessage, ModelAdapter
from harness.orchestrator import ToolLoopEvent, ToolLoopResult, run_tool_loop
from harness.orchestrator.hook_wiring import make_write_file_redirect_hook
from harness.orchestrator.hooks import (
    EXHAUSTED_FABRICATION_FALLBACK,
    HookPipeline,
    default_hook_pipeline,
)
from harness.tools import (
    CalcTool,
    DateMathTool,
    EditFileTool,
    FetchUrlTool,
    GitDiffTool,
    GitLogTool,
    GitStatusTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    LoadToolTool,
    NowTool,
    ReadFileTool,
    ShellTool,
    Tool,
    ToolCatalog,
    ToolRegistry,
    ToolSearchTool,
    WriteFileTool,
    seed_builtins_into,
)

# The user message stays fixed across the run — the handoff carries the
# noun, this is the verb. Mentions `bd close` explicitly so the model
# knows shelling to bd is how it signals completion (v0 has no
# structured bd_close tool — that's a future revision).
EXECUTOR_USER_MESSAGE = (
    "Drive the bd issue described in the session handoff to closure. "
    "Run `bd close <issue-id>` via shell when the acceptance criteria are met. "
    "Do not invent acceptance criteria the issue doesn't list."
)


@dataclass
class LoopConfig:
    """Configuration for one `run_loop` invocation.

    - `epic_id`: bd epic this run drains. Ignored when `resume_from` is
      set (state file carries it).
    - `workspace`: workspace root. Used as `bd_dir` for `DriverBd`,
      `cwd` for git diff, sandbox root for filesystem tools, and
      anchor for `.harness/loop_runs/*.json`.
    - `character`: loaded `Character` used for the (stripped) system
      prompt. `character.system_prompt(include_samples=())` strips
      voice few-shot — identity + values + style rules survive.
    - `max_turns`: hard cap on iterations. Default 20.
    - `profile`: pinned to "coding" today; field present so future
      revisions can swap without changing the run_loop signature.
    - `resume_from`: loop_run_id of an existing state file. When set,
      `epic_id` and `max_turns` are ignored — the state file owns
      those values.
    - `dry_run`: assemble the handoff and print it, then return
      without running the model. Useful for verifying handoff shape
      before burning real turns.
    - `log_path`: progress log file. Defaults to
      `<workspace>/.harness/loop_runs/<loop_run_id>.log`.
    """

    epic_id: str
    workspace: Path
    character: Character
    max_turns: int = 20
    profile: str = "coding"
    resume_from: str | None = None
    dry_run: bool = False
    log_path: Path | None = None
    # When set, mirror per-turn tool-loop events to this callable (in
    # addition to the existing file log). Used by the CLI's --verbose
    # flag to tee events to stderr. harness-9bpt.
    extra_observer: Callable[[ToolLoopEvent], None] | None = None
    # Forbidden substrings checked in workspace files modified during
    # this run (harness-k52f). After a turn closes its bd issue, if any
    # of these strings appear in any file modified since loop start,
    # the turn fails with the violation list as the reason — fed back
    # to the model via prior_attempt_failure on retry. Default catches
    # the spec-opening "no TODO / no future improvement" rule that
    # cross-cuts most workplan specs. Set to `()` to disable.
    forbidden_patterns: tuple[str, ...] = ("TODO", "FIXME", "XXX", "HACK")
    # Per-turn round budget for the inner run_tool_loop call
    # (harness-pipb). The orchestrator's default (8) is too tight for
    # executor turns that do multi-edit work — read spec, read
    # existing file, edit + verify + edit again — leading to
    # `wrap_up_forced` firing before the model finishes. 12 gives
    # ~50% more headroom while keeping each turn bounded (12 rounds *
    # ~10s/round ≈ 2 min upper bound on M4 Pro).
    executor_max_rounds: int = 12
    # Tar+gzip the workspace into
    # .harness/loop_runs/<id>_workspace.tar.gz before the first turn
    # fires (harness-9ijr). Default on — gitignored workspaces are
    # otherwise unrecoverable when a runaway model rewrites a file
    # (today's GTA2 §1 regression that prompted this guard). Resume
    # runs (`resume_from` set) skip the snapshot — the original run's
    # snapshot is the operator's restore point and overwriting it
    # would defeat the purpose. Set False via --no-snapshot for
    # callers who know what they're doing (large workspaces over the
    # cap, fully-tracked git trees, throwaway scratch sessions).
    snapshot: bool = True


@dataclass
class LoopResult:
    """What `run_loop` returns to the caller.

    `exit_reason` matches the lifecycle event names in the progress log:
      - "success":     ready_under_epic emptied; epic complete.
      - "exhausted":   turns_used reached max_turns.
      - "halted":      second failure on the same issue; bd-human flagged.
      - "interrupted": SIGINT mid-loop.
      - "dry_run":     --dry-run; one handoff printed, no turn ran.
    """

    loop_run_id: str
    epic_id: str
    closed: list[str]
    halted_on: str | None
    turns_used: int
    exit_reason: Literal["success", "halted", "exhausted", "interrupted", "dry_run"]
    handoffs: list[Handoff] = field(default_factory=list)


# --- top-level entry --------------------------------------------------


def run_loop(adapter: ModelAdapter, bd: DriverBd, config: LoopConfig) -> LoopResult:
    """Run the executor loop until success / halt / exhaustion / interrupt."""
    state = _load_or_init_state(config)
    log_path = _resolve_log_path(config, state)
    log = _open_log(log_path)
    log(
        f"loop_run={state.loop_run_id} epic={state.epic_id} max_turns={state.max_turns} "
        f"resumed={'yes' if config.resume_from else 'no'}"
    )

    # harness-9ijr: snapshot the workspace once per fresh run BEFORE
    # the first turn fires. Resume runs inherit the original
    # snapshot — overwriting it would lose the operator's recovery
    # point. SnapshotTooBigError aborts the run; the operator chooses
    # between narrowing --workspace and passing --no-snapshot.
    if config.snapshot and config.resume_from is None:
        snapshot_path = _snapshot_workspace(config.workspace, state.loop_run_id)
        log(f"workspace snapshot: {snapshot_path}")

    with _sigint_guard() as interrupted:
        while True:
            if interrupted.is_set():
                return _exit_interrupted(bd, state, log)
            if state.turns_used >= state.max_turns:
                return _exit_exhausted(state, log)

            try:
                ready = bd.ready_under_epic(state.epic_id)
            except DriverBdError as exc:
                log(f"bd ready_under_epic failed: {exc}; halting")
                return _exit_halted(bd, state, current_id=state.epic_id, reason=str(exc), log=log)
            if not ready:
                return _exit_success(state, log)

            current = ready[0]
            attempt = state.attempt_counts.get(current.id, 0) + 1
            state.attempt_counts[current.id] = attempt
            prior_failure = state.last_failure.get(current.id) if attempt > 1 else None
            # harness-lefw: signal targeted-fix mode when the loop has
            # already touched this issue OR the operator left a
            # "REGRESSION" marker in notes (their convention when
            # reopening a previously-closed issue). _issue_has_regression
            # tolerates a missing bd lookup — bd.show is called again
            # inside build_handoff and a transient miss there raises.
            targeted_fix = attempt > 1 or _issue_has_regression(bd, current.id)

            handoff = build_handoff(
                state,
                current.id,
                bd,
                git_root=config.workspace,
                prior_attempt_failure=prior_failure,
                workspace=config.workspace,
                targeted_fix=targeted_fix,
            )

            if config.dry_run:
                log(f"dry_run: handoff for {current.id} (attempt {attempt}) — exiting")
                # Stash the handoff so callers (eg. tests) can inspect.
                return LoopResult(
                    loop_run_id=state.loop_run_id,
                    epic_id=state.epic_id,
                    closed=list(state.closed_this_run),
                    halted_on=None,
                    turns_used=state.turns_used,
                    exit_reason="dry_run",
                    handoffs=[handoff],
                )

            turn_observer = _make_turn_observer(
                log_path, state.turns_used + 1, config.extra_observer
            )
            turn_success, turn_reason = _run_executor_turn(
                adapter=adapter,
                character=config.character,
                handoff=handoff,
                workspace=config.workspace,
                observe=turn_observer,
                max_rounds=config.executor_max_rounds,
            )
            state.turns_used += 1

            success, reason = _classify_post_turn(
                bd,
                current.id,
                turn_success,
                turn_reason,
                workspace=config.workspace,
                started_at=state.started_at,
                forbidden_patterns=config.forbidden_patterns,
            )

            if success:
                _on_success(bd, state, current.id, log)
                _save_state(state, config.workspace)
                continue

            log(f"turn {state.turns_used}: {current.id} attempt={attempt} FAIL ({reason})")

            if attempt == 1:
                state.last_failure[current.id] = reason
                _save_state(state, config.workspace)
                continue

            # Second consecutive failure — halt.
            return _exit_halted(bd, state, current_id=current.id, reason=reason, log=log)


# --- per-turn -----------------------------------------------------


# Observer type for the executor's per-turn tool-loop event stream
# (harness-9bpt). Same shape as planner's PlannerObserver.
ExecutorObserver = Callable[[ToolLoopEvent], None]


def _make_turn_observer(
    log_path: Path,
    turn_index: int,
    extra: ExecutorObserver | None,
) -> ExecutorObserver:
    """Build an observer that:
      1. Appends one line per ToolLoopEvent to `log_path` (prefixed
         with `turn N |` so the operator can grep per-turn slices).
      2. Optionally forwards to `extra` — used by the CLI's --verbose
         flag to mirror to stderr.

    Best-effort IO: log write failures don't crash the turn. Observer
    failures in `extra` are suppressed for the same reason (matches
    the planner's _compose_observers contract)."""

    def emit(event: ToolLoopEvent) -> None:
        line = f"turn {turn_index} | {_format_executor_event(event)}"
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        if extra is not None:
            with contextlib.suppress(Exception):
                extra(event)

    return emit


def _format_executor_event(event: ToolLoopEvent) -> str:
    """Same shape as planner's _format_event — timestamp + kind + most
    relevant payload. Kept separate so the driver module doesn't import
    from planner."""
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    parts: list[str] = [ts, event.kind]
    if event.call is not None:
        args_preview = repr(event.call.arguments)
        if len(args_preview) > 200:
            args_preview = args_preview[:200] + "...]"
        parts.append(f"call={event.call.name} args={args_preview}")
    if event.result is not None:
        success = "ok" if event.result.success else f"FAIL[{event.result.error}]"
        output_preview = event.result.output.replace("\n", " ")[:160]
        parts.append(f"result={success} output={output_preview!r}")
    if event.delta:
        parts.append(f"delta={event.delta}")
    if event.catcher:
        parts.append(f"catcher={event.catcher}")
    return " | ".join(parts)


def _run_executor_turn(
    *,
    adapter: ModelAdapter,
    character: Character,
    handoff: Handoff,
    workspace: Path,
    observe: ExecutorObserver | None = None,
    max_rounds: int = 12,
) -> tuple[bool, str]:
    """Run one executor turn. Returns (succeeded, reason).

    `succeeded` is computed against the post-turn ToolLoopResult only —
    the caller is responsible for the post-turn `bd.show` outcome check
    (success requires BOTH a clean reply AND the bd issue actually
    being closed). This split keeps the test surface small: the turn
    runner is a pure function of its inputs, and the caller composes
    the bd check on top.

    `observe`, when set, receives every `ToolLoopEvent` from the inner
    `run_tool_loop` — same shape as the planner's observer
    (harness-9bpt). The caller is responsible for writing to a log
    file / stderr; this function is just the seam."""
    registry = _build_executor_registry(workspace)
    # `include_samples=()` strips the voice few-shot block. Identity +
    # values + style rules from `core.yaml` survive — the executor still
    # benefits from "say 'Don't know' plainly" and friends, but isn't
    # padded with examples it doesn't need.
    base_prompt = character.system_prompt(include_samples=())
    system_prompt = f"{base_prompt}\n\n{handoff.render()}"
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=EXECUTOR_USER_MESSAGE),
    ]
    # harness-lefw: wire WriteFileRedirectHook so the safety-shrink guard
    # catches "rewrite-from-scratch" wipes on reopened issues (the driver
    # previously ran with module-default hooks, which left this nullable
    # parameter at None — chat sessions in cli_classic.py have always had
    # it wired). Closures bind to the executor's workspace + registry.
    write_file_redirect_hook = make_write_file_redirect_hook(
        registry=registry,
        workspace_path=workspace,
    )
    hooks: HookPipeline = default_hook_pipeline(
        write_file_redirect_hook=write_file_redirect_hook,
    )
    result: ToolLoopResult = run_tool_loop(
        adapter,  # type: ignore[arg-type]  # narrower _ToolCapableAdapter, checked at runtime
        messages,
        registry,
        hooks=hooks,
        observe=observe,
        max_rounds=max_rounds,
    )
    if result.content.strip() == EXHAUSTED_FABRICATION_FALLBACK.strip():
        return False, "fabrication_fallback fired"
    return True, ""


def _classify_post_turn(
    bd: DriverBd,
    current_id: str,
    turn_success: bool,
    turn_reason: str,
    *,
    workspace: Path | None = None,
    started_at: datetime | None = None,
    forbidden_patterns: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Combine the turn outcome with the post-turn bd state.

    Issue must actually be closed for the turn to count as a real win —
    a clean reply with the issue still open means the model didn't
    finish the work, regardless of how confidently it claimed to.

    Forbidden-pattern verification (harness-k52f): when
    `forbidden_patterns` is non-empty and `workspace` + `started_at`
    are supplied, the function scans workspace files modified since
    `started_at` for any of the patterns. Any hit fails the turn
    even if the bd issue closed — small models will happily close
    after writing 'TODO' comments that violate spec-opening rules."""
    if not turn_success:
        return False, turn_reason
    try:
        issue = bd.show(current_id)
    except DriverBdError as exc:
        return False, f"post-turn bd.show failed: {exc}"
    if issue.status != "closed":
        return False, f"issue still {issue.status} after turn"
    # bd issue closed; now check forbidden patterns in the workspace.
    if forbidden_patterns and workspace is not None and started_at is not None:
        violations = _find_violations(workspace, started_at, forbidden_patterns)
        if violations:
            joined = "; ".join(violations[:5])
            extra = f" (+{len(violations) - 5} more)" if len(violations) > 5 else ""
            return False, f"closed but forbidden-pattern hits: {joined}{extra}"
    return True, ""


def _find_violations(
    workspace: Path,
    started_at: datetime,
    forbidden_patterns: tuple[str, ...],
) -> list[str]:
    """Walk `workspace` for files modified after `started_at` and check
    each for any of `forbidden_patterns`. Returns one entry per
    (path, pattern) hit. Empty = no violations (harness-k52f).

    Skips:
      - hidden dirs (any path component starting with `.`)
      - files > 1MB (binary-shaped, not worth scanning)
      - files that aren't decodable as UTF-8 text
    Best-effort: IO errors are silently skipped — the verification is
    a safety net, not a contract gate."""
    if not forbidden_patterns:
        return []
    started_ts = started_at.timestamp()
    violations: list[str] = []
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(workspace).parts
        if any(p.startswith(".") for p in rel_parts):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime < started_ts:
            continue
        if stat.st_size > 1024 * 1024:
            continue
        try:
            content = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(workspace)
        for pattern in forbidden_patterns:
            if pattern in content:
                violations.append(f"{rel}: contains {pattern!r}")
    return violations


# --- targeted-fix detection ---------------------------------------


# Operator convention when reopening a previously-closed bd issue: lead
# the notes with "REGRESSION YYYY-MM-DD:" so the driver can recognize
# the retry as a targeted fix. Case-sensitive substring match — the
# convention is established (e.g. harness-90j0 / harness-vjb6 reopened
# on 2026-05-21 both carry the literal "REGRESSION 2026-05-21:"). False
# positives (notes that happen to mention regressions in passing) are
# mostly harmless: the worst case is an extra MODE banner the model
# reads and complies with.
_REGRESSION_MARKER: str = "REGRESSION"


def _issue_has_regression(bd: DriverBd, issue_id: str) -> bool:
    """Best-effort check: does the bd issue's notes contain the
    operator's regression marker? Swallows DriverBdError so a transient
    bd hiccup degrades to 'no marker found' — build_handoff is about to
    call bd.show again anyway and will raise loudly on a real lookup
    failure. Pure read; no mutation."""
    try:
        issue = bd.show(issue_id)
    except DriverBdError:
        return False
    notes = issue.raw.get("notes") or ""
    return _REGRESSION_MARKER in notes


# --- tool registry ------------------------------------------------


def _build_executor_registry(workspace: Path) -> ToolRegistry:
    """Build the executor's tool registry. v0: the coding-profile
    subset that doesn't depend on stores / character. Store-dependent
    tools (search_memory, search_facts) are skipped — they need
    retrieval setup the driver doesn't provision. introspect /
    spawn_subagent are also skipped (need adapter+character wiring;
    can be added in a future revision).

    All filesystem tools sandboxed to `workspace`. The shell tool runs
    with `cwd=workspace`, so `bd close <id>` works as long as the
    workspace is a bd project root (the same constraint the rest of
    the harness already enforces)."""
    catalog = ToolCatalog()
    seed_builtins_into(catalog, now_iso=datetime.now(UTC).isoformat(timespec="seconds"))
    registry = ToolRegistry(catalog=catalog)

    builders: dict[str, Tool] = {
        "read_file": ReadFileTool(root=workspace),
        "list_dir": ListDirTool(root=workspace),
        "grep": GrepTool(root=workspace),
        "glob": GlobTool(root=workspace),
        "edit_file": EditFileTool(root=workspace),
        "write_file": WriteFileTool(root=workspace),
        "shell": ShellTool(cwd=workspace),
        "git_status": GitStatusTool(root=workspace),
        "git_diff": GitDiffTool(root=workspace),
        "git_log": GitLogTool(root=workspace),
        "fetch_url": FetchUrlTool(),
        "now": NowTool(),
        "date_math": DateMathTool(),
        "calc": CalcTool(),
    }
    for tool in builders.values():
        registry.register(tool)
    # tool_search / load_tool need a reference to the registry they
    # operate on, so they construct AFTER the static tools are in
    # place. `load_tool`'s builder map is empty — the executor exposes
    # its full set upfront, so lazy-loading at runtime isn't needed.
    registry.register(ToolSearchTool(catalog=catalog, registry=registry))
    registry.register(LoadToolTool(catalog=catalog, registry=registry, builders={}))
    return registry


# --- state / log -------------------------------------------------


def _load_or_init_state(config: LoopConfig) -> LoopRunState:
    if config.resume_from is not None:
        path = LoopRunState.state_path(config.workspace, config.resume_from)
        state = LoopRunState.load(path)
        # Override the loaded state's max_turns with the config value
        # (harness-iai4). The CLI's --max-turns is user intent — saved
        # state's stale value would silently make a "resume with a
        # higher cap" run instantly exhaust. Persist immediately so a
        # subsequent reload reflects the new cap.
        if state.max_turns != config.max_turns:
            state.max_turns = config.max_turns
            _save_state(state, config.workspace)
        return state
    sha = _git_head_sha(config.workspace)
    state = LoopRunState.fresh(
        epic_id=config.epic_id,
        max_turns=config.max_turns,
        started_at_sha=sha,
    )
    _save_state(state, config.workspace)
    return state


def _save_state(state: LoopRunState, workspace: Path) -> None:
    state.save(LoopRunState.state_path(workspace, state.loop_run_id))


def _git_head_sha(workspace: Path) -> str:
    """`git rev-parse HEAD` from the workspace. Returns "unknown" on
    any error — the handoff still functions with a degenerate sha
    (git diff just returns nothing), so this is non-fatal."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 — git on PATH is expected
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _resolve_log_path(config: LoopConfig, state: LoopRunState) -> Path:
    if config.log_path is not None:
        return config.log_path
    return LoopRunState.state_dir(config.workspace) / f"{state.loop_run_id}.log"


def _open_log(log_path: Path) -> _LogWriter:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return _LogWriter(log_path)


class _LogWriter:
    """Tiny append-only log writer. One line per call. Timestamps each
    line so `tail -f` is informative. Best-effort — IO errors don't
    crash the loop (a crashed log isn't worse than no log)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def __call__(self, message: str) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}\n"
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            # Log a warning to stderr but don't crash the loop — a
            # missing log is annoying, not corrupting.
            logging.getLogger(__name__).warning("loop log write failed: %s", exc)


# --- exit paths --------------------------------------------------


def _exit_success(state: LoopRunState, log: _LogWriter) -> LoopResult:
    log(f"loop_run={state.loop_run_id} SUCCESS (epic empty)")
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=None,
        turns_used=state.turns_used,
        exit_reason="success",
    )


def _exit_exhausted(state: LoopRunState, log: _LogWriter) -> LoopResult:
    log(
        f"loop_run={state.loop_run_id} EXHAUSTED "
        f"(turns_used={state.turns_used} max={state.max_turns})"
    )
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=None,
        turns_used=state.turns_used,
        exit_reason="exhausted",
    )


def _exit_halted(
    bd: DriverBd,
    state: LoopRunState,
    *,
    current_id: str,
    reason: str,
    log: _LogWriter,
) -> LoopResult:
    log(f"loop_run={state.loop_run_id} HALTED on {current_id}: {reason}")
    with contextlib.suppress(DriverBdError):
        bd.flag_human(current_id, reason=f"loop halted: {reason}")
    with contextlib.suppress(DriverBdError):
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=current_id,
            status="halted",
            body=reason,
        )
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=current_id,
        turns_used=state.turns_used,
        exit_reason="halted",
    )


def _exit_interrupted(bd: DriverBd, state: LoopRunState, log: _LogWriter) -> LoopResult:
    log(f"loop_run={state.loop_run_id} INTERRUPTED")
    with contextlib.suppress(DriverBdError):
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=state.epic_id,
            status="interrupted",
            body=f"SIGINT received at turns_used={state.turns_used}",
        )
    return LoopResult(
        loop_run_id=state.loop_run_id,
        epic_id=state.epic_id,
        closed=list(state.closed_this_run),
        halted_on=None,
        turns_used=state.turns_used,
        exit_reason="interrupted",
    )


def _on_success(bd: DriverBd, state: LoopRunState, current_id: str, log: _LogWriter) -> None:
    state.closed_this_run.append(current_id)
    state.last_failure.pop(current_id, None)
    log(f"turn {state.turns_used}: {current_id} CLOSED")
    with contextlib.suppress(DriverBdError):
        bd.write_session_state(
            loop_run_id=state.loop_run_id,
            current_issue_id=current_id,
            status="success",
            body=f"closed at turn {state.turns_used}",
        )


# --- signal handling ----------------------------------------------


class _InterruptFlag:
    """Set on SIGINT; checked at the top of every loop iteration."""

    def __init__(self) -> None:
        self._flag = False

    def set(self) -> None:
        self._flag = True

    def is_set(self) -> bool:
        return self._flag


@contextlib.contextmanager
def _sigint_guard() -> Iterator[_InterruptFlag]:
    """Install a SIGINT handler that flips the interrupt flag, restore
    the previous handler on exit. Safe to nest under outer handlers —
    the previous handler is restored on context exit."""
    flag = _InterruptFlag()

    def handler(_signum: int, _frame: object) -> None:
        flag.set()

    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGINT, previous)


# --- workspace snapshot (harness-9ijr) ----------------------------


# Names that are pruned from the snapshot walk (entire subtree
# skipped). Targets the high-cost / not-our-state directories
# operators routinely have in workspaces. `.harness` is special:
# the snapshot tar lives INSIDE this dir, so excluding it prevents
# the tar from trying to archive itself.
DEFAULT_SNAPSHOT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".harness", ".git", "node_modules", ".venv", "__pycache__"}
)

# File-suffix exclusions. `.pyc` byte-compiled artifacts and OS
# scratch files don't help recovery and bloat the tar; skipping is
# pure win. Tuple matches `str.endswith`'s signature.
DEFAULT_SNAPSHOT_EXCLUDE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")

# Pre-tar uncompressed size cap. 100MB matches the spec — large
# enough to capture realistic scratch workspaces (the GTA2 case is
# <1MB), small enough to force operators to think before they
# snapshot a node_modules-laden tree.
SNAPSHOT_SIZE_CAP_BYTES: int = 100 * 1024 * 1024


class SnapshotTooBigError(RuntimeError):
    """Raised when the pre-tar workspace size exceeds the cap. The
    message names the cap so operators know the threshold to clear
    or the flag to bypass it."""


def _snapshot_path(workspace: Path, loop_run_id: str) -> Path:
    """`.harness/loop_runs/<id>_workspace.tar.gz` under workspace."""
    return workspace / ".harness" / "loop_runs" / f"{loop_run_id}_workspace.tar.gz"


def _iter_snapshot_files(workspace: Path) -> Iterator[Path]:
    """Walk workspace yielding files to include in the snapshot.

    Uses os.walk with followlinks=False to avoid infinite loops on
    self-referential symlinks. Prunes DEFAULT_SNAPSHOT_EXCLUDE_DIRS
    in-place so we never recurse into them (cheap — saves the cost
    of statting every node_modules file). Skips files whose basename
    ends with DEFAULT_SNAPSHOT_EXCLUDE_SUFFIXES."""
    for root, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = [d for d in dirs if d not in DEFAULT_SNAPSHOT_EXCLUDE_DIRS]
        for fname in files:
            if fname.endswith(DEFAULT_SNAPSHOT_EXCLUDE_SUFFIXES):
                continue
            yield Path(root) / fname


def _snapshot_workspace(
    workspace: Path,
    loop_run_id: str,
    *,
    size_cap_bytes: int = SNAPSHOT_SIZE_CAP_BYTES,
) -> Path:
    """tar+gzip `workspace` to `.harness/loop_runs/<id>_workspace.tar.gz`.

    Returns the snapshot path on success. Raises SnapshotTooBigError when
    pre-tar uncompressed total exceeds size_cap_bytes — caller MUST
    abort the run rather than continue without a recoverable
    starting point.

    Pre-walk size accounting runs first; only after the cap check
    passes does the tar open. This means an over-cap workspace
    leaves no partial tar behind. Files that vanish between the
    size walk and the tar pass (unlikely in practice; the loop's
    workspace shouldn't be churning) are silently skipped from the
    tar — `tarfile.add` raises but we don't catch; let it surface
    so the operator sees the race condition."""
    total_bytes = 0
    files_to_include: list[Path] = []
    for path in _iter_snapshot_files(workspace):
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
        if total_bytes > size_cap_bytes:
            mb = size_cap_bytes // (1024 * 1024)
            raise SnapshotTooBigError(
                f"workspace size > {mb}MB cap (rooted at {workspace}). "
                f"Pass --no-snapshot to skip the snapshot guard, or "
                f"narrow --workspace to a smaller subtree."
            )
        files_to_include.append(path)

    snapshot_path = _snapshot_path(workspace, loop_run_id)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snapshot_path, "w:gz") as tf:
        for path in files_to_include:
            arcname = path.relative_to(workspace)
            tf.add(path, arcname=str(arcname), recursive=False)
    return snapshot_path


__all__ = [
    "DEFAULT_SNAPSHOT_EXCLUDE_DIRS",
    "DEFAULT_SNAPSHOT_EXCLUDE_SUFFIXES",
    "EXECUTOR_USER_MESSAGE",
    "SNAPSHOT_SIZE_CAP_BYTES",
    "LoopConfig",
    "LoopResult",
    "SnapshotTooBigError",
    "run_loop",
]
